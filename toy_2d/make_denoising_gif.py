from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from toy_2d.benchmark import build_fig_solver, load_model_from_checkpoint
from toy_2d.datasets import build_distribution
from toy_2d.inverse_problem import ConditionalReferenceSampler, LinearObservationModel
from toy_2d.solvers import DPSSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a GIF comparing DPS and FIG denoising trajectories.")
    parser.add_argument("--dataset", type=str, default="two_moons", choices=["two_moons", "eight_gaussians"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="toy_2d_outputs/final_runs/animations")
    parser.add_argument("--sigma_noise", type=float, default=0.5)
    parser.add_argument("--fig_variant", type=str, default="fig_plus", choices=["fig_fixed", "fig_snr", "fig_plus"])
    parser.add_argument("--secondary_fig_variant", type=str, default=None, choices=["fig_fixed", "fig_snr", "fig_plus"])
    parser.add_argument("--fig_k", type=int, default=5)
    parser.add_argument("--fig_w", type=float, default=0.0)
    parser.add_argument("--fig_lr", type=float, default=0.5)
    parser.add_argument("--fig_mix_coef", type=float, default=0.95)
    parser.add_argument("--dps_zeta", type=float, default=0.05)
    parser.add_argument("--num_samples", type=int, default=192)
    parser.add_argument("--background_samples", type=int, default=2000)
    parser.add_argument("--reference_samples", type=int, default=1200)
    parser.add_argument("--reference_pool_size", type=int, default=50000)
    parser.add_argument("--fixed_measurement", type=float, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _frame_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf.copy()


def _plot_panel(ax, background, reference, current, x_true, y, observation_model, limits, title, color):
    ax.scatter(background[:, 0], background[:, 1], s=8, c="lightgray", alpha=0.30, edgecolors="none")
    if reference is not None:
        ax.scatter(reference[:, 0], reference[:, 1], s=8, c="#2ca02c", alpha=0.22, edgecolors="none")
    ax.scatter(current[:, 0], current[:, 1], s=14, c=color, alpha=0.78, edgecolors="none")
    line = observation_model.project_line(y.cpu(), x_range=(limits[0], limits[1]))
    ax.plot(line[:, 0], line[:, 1], color="black", linestyle="--", linewidth=1.3)
    if x_true is not None:
        ax.scatter([x_true[0]], [x_true[1]], s=80, c="gold", edgecolors="black", zorder=10)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(alpha=0.2)


def _solver_label(name: str) -> str:
    labels = {
        "dps": "DPS",
        "fig_fixed": "FIG (fixed step)",
        "fig_snr": "FIG (SNR step)",
        "fig_plus": "FIG+",
    }
    return labels.get(name, name)


def _measurement_tag(value: float | None) -> str:
    if value is None:
        return "random_obs"
    return f"x1_{value:+.2f}".replace("+", "p").replace("-", "m")


def _save_frame_strip(
    trajectories: list[tuple[str, list[torch.Tensor], str]],
    background: np.ndarray,
    reference: np.ndarray | None,
    x_true: np.ndarray | None,
    y: torch.Tensor,
    observation_model,
    limits: tuple[float, float, float, float],
    output_path: Path,
):
    indices = [0, len(trajectories[0][1]) // 2, len(trajectories[0][1]) - 1]
    fig, axes = plt.subplots(len(trajectories), 3, figsize=(12, 3.5 * len(trajectories)))
    if len(trajectories) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (solver_name, solver_traj, color) in enumerate(trajectories):
        for col_idx, frame_idx in enumerate(indices):
            points = solver_traj[frame_idx].numpy()
            step_label = f"Frame {frame_idx}/{len(solver_traj)-1}"
            _plot_panel(
                axes[row_idx, col_idx],
                background,
                reference,
                points,
                x_true,
                y,
                observation_model,
                limits,
                f"{_solver_label(solver_name)} - {step_label}",
                color,
            )
    fig.suptitle("Representative denoising stages", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution = build_distribution(args.dataset)
    model, schedule, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    observation_model = LinearObservationModel(matrix=torch.tensor([[1.0, 0.0]]), sigma_noise=args.sigma_noise, device=device)
    reference_sampler = ConditionalReferenceSampler(
        distribution=distribution,
        observation_model=observation_model,
        pool_size=args.reference_pool_size,
        device=device,
    )

    background = distribution.sample(args.background_samples, device="cpu").numpy()
    if args.fixed_measurement is None:
        x_true = distribution.sample(1, device=device).squeeze(0)
        y = observation_model.observe(x_true.unsqueeze(0)).squeeze(0)
    else:
        x_true = None
        y = torch.tensor([args.fixed_measurement], dtype=torch.float32, device=device)
    reference = reference_sampler.sample(y, num_samples=args.reference_samples).cpu().numpy()
    x_init = torch.randn(args.num_samples, 2, device=device)

    dps_solver = DPSSolver(model, schedule, observation_model, zeta=args.dps_zeta)
    primary_fig_solver = build_fig_solver(
        variant=args.fig_variant,
        model=model,
        schedule=schedule,
        observation_model=observation_model,
        fig_k=args.fig_k,
        fig_w=args.fig_w,
        fig_lr=args.fig_lr,
        fig_mix_coef=args.fig_mix_coef,
    )
    secondary_fig_solver = None
    if args.secondary_fig_variant is not None:
        secondary_fig_solver = build_fig_solver(
            variant=args.secondary_fig_variant,
            model=model,
            schedule=schedule,
            observation_model=observation_model,
            fig_k=args.fig_k,
            fig_w=args.fig_w,
            fig_lr=args.fig_lr,
            fig_mix_coef=args.fig_mix_coef,
        )

    _, dps_traj = dps_solver.sample(
        y,
        num_samples=args.num_samples,
        device=device,
        initial_x=x_init,
        return_trajectory=True,
    )
    _, primary_fig_traj = primary_fig_solver.sample(
        y,
        num_samples=args.num_samples,
        device=device,
        initial_x=x_init,
        return_trajectory=True,
    )
    trajectory_specs = [
        ("dps", dps_traj, "#1f77b4"),
        (args.fig_variant, primary_fig_traj, "#d62728"),
    ]
    if secondary_fig_solver is not None:
        _, secondary_fig_traj = secondary_fig_solver.sample(
            y,
            num_samples=args.num_samples,
            device=device,
            initial_x=x_init,
            return_trajectory=True,
        )
        trajectory_specs.append((args.secondary_fig_variant, secondary_fig_traj, "#9467bd"))

    limits = distribution.default_plot_limits()
    frames = []
    x_true_np = x_true.detach().cpu().numpy() if x_true is not None else None
    num_panels = len(trajectory_specs)
    for frame_idx in range(len(dps_traj)):
        fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 5))
        if num_panels == 1:
            axes = [axes]
        for ax, (solver_name, solver_traj, color) in zip(axes, trajectory_specs):
            _plot_panel(
                ax,
                background,
                reference,
                solver_traj[frame_idx].numpy(),
                x_true_np,
                y.detach().cpu(),
                observation_model,
                limits,
                f"{_solver_label(solver_name)} | step {frame_idx}/{len(dps_traj)-1}",
                color,
            )
        fig.suptitle(
            (
                f"{args.dataset} | sigma_n={args.sigma_noise:.2f} | "
                f"{_solver_label(args.fig_variant)}(K={args.fig_k}, w={args.fig_w:.1f}) | "
                f"x1={float(y.item()):.2f}"
            ),
            fontsize=14,
        )
        fig.tight_layout()
        frames.append(_frame_to_array(fig))
        plt.close(fig)

    stem_parts = [
        args.dataset,
        f"sigma_{args.sigma_noise:.2f}",
        _measurement_tag(args.fixed_measurement),
        "dps",
        args.fig_variant,
    ]
    if args.secondary_fig_variant is not None:
        stem_parts.append(args.secondary_fig_variant)
    comparison_tag = "_vs_".join(stem_parts[2:])
    file_stem = f"{args.dataset}_sigma_{args.sigma_noise:.2f}_{comparison_tag}"
    gif_path = output_dir / f"{file_stem}.gif"
    imageio.mimsave(gif_path, frames, fps=args.fps, loop=0)

    strip_path = output_dir / f"{file_stem}_strip.png"
    _save_frame_strip(
        trajectories=trajectory_specs,
        background=background,
        reference=reference,
        x_true=x_true_np,
        y=y.detach().cpu(),
        observation_model=observation_model,
        limits=limits,
        output_path=strip_path,
    )

    print(f"Saved GIF to {gif_path}")
    print(f"Saved frame strip to {strip_path}")


if __name__ == "__main__":
    main()
