from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from toy_2d.datasets import build_distribution
from toy_2d.diffusion import DiffusionSchedule
from toy_2d.inverse_problem import ConditionalReferenceSampler, LinearObservationModel
from toy_2d.metrics import ground_truth_mse, measurement_mse, posterior_mean_mse, sliced_wasserstein_distance
from toy_2d.model import EpsilonMLP
from toy_2d.plotting import save_conditional_comparison_plot
from toy_2d.solvers import DPSSolver, FIGDiffusionSolver, UnconditionalDDIMSampler


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DPS and FIG-Diffusion on toy 2D inverse problems.")
    parser.add_argument("--dataset", type=str, choices=["two_moons", "eight_gaussians"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="toy_2d_outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_test_observations", type=int, default=32)
    parser.add_argument("--num_solver_samples", type=int, default=512)
    parser.add_argument("--num_reference_samples", type=int, default=512)
    parser.add_argument("--reference_pool_size", type=int, default=50_000)
    parser.add_argument("--noise_levels", type=str, default="0.05,0.5,1.0")
    parser.add_argument("--fig_k_values", type=str, default="1,3,5")
    parser.add_argument("--fig_w_values", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--fig_lr", type=float, default=0.5)
    parser.add_argument("--fig_variants", type=str, default="fig_snr,fig_plus")
    parser.add_argument("--fig_mix_coef", type=float, default=0.95)
    parser.add_argument("--dps_zeta", type=float, default=0.05)
    parser.add_argument("--plot_count", type=int, default=4)
    parser.add_argument("--observation_matrix", type=str, default="1.0,0.0")
    return parser.parse_args()


def parse_observation_matrix(value: str) -> torch.Tensor:
    entries = [float(item) for item in value.split(",") if item]
    if len(entries) != 2:
        raise ValueError("For now --observation_matrix must contain exactly two comma-separated values.")
    return torch.tensor([entries], dtype=torch.float32)


def parse_string_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_fig_solver(
    variant: str,
    model,
    schedule,
    observation_model,
    fig_k: int,
    fig_w: float,
    fig_lr: float,
    fig_mix_coef: float,
) -> FIGDiffusionSolver:
    if variant == "fig_fixed":
        return FIGDiffusionSolver(
            model=model,
            schedule=schedule,
            observation_model=observation_model,
            correction_steps=fig_k,
            w=fig_w,
            lr=fig_lr,
            use_snr_weighting=False,
            use_fig_plus=False,
            mix_coef=fig_mix_coef,
        )
    if variant == "fig_snr":
        return FIGDiffusionSolver(
            model=model,
            schedule=schedule,
            observation_model=observation_model,
            correction_steps=fig_k,
            w=fig_w,
            lr=fig_lr,
            use_snr_weighting=True,
            use_fig_plus=False,
            mix_coef=fig_mix_coef,
        )
    if variant == "fig_plus":
        return FIGDiffusionSolver(
            model=model,
            schedule=schedule,
            observation_model=observation_model,
            correction_steps=fig_k,
            w=fig_w,
            lr=fig_lr,
            use_snr_weighting=True,
            use_fig_plus=True,
            mix_coef=fig_mix_coef,
        )
    raise ValueError(f"Unknown FIG variant: {variant}")


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["extra_config"]["model"]
    schedule_config = checkpoint["schedule_config"]

    model = EpsilonMLP(
        input_dim=2,
        hidden_dim=model_config["hidden_dim"],
        time_dim=model_config["time_dim"],
        num_layers=model_config["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    schedule = DiffusionSchedule(device=device, **schedule_config)
    return model, schedule, checkpoint


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict], group_keys: list[str], metric_keys: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary_rows = []
    for group_values, group_rows in grouped.items():
        summary = {key: value for key, value in zip(group_keys, group_values)}
        for metric_key in metric_keys:
            values = torch.tensor([float(row[metric_key]) for row in group_rows], dtype=torch.float32)
            summary[f"{metric_key}_mean"] = values.mean().item()
            summary[f"{metric_key}_std"] = values.std(unbiased=False).item()
        summary_rows.append(summary)
    return sorted(
        summary_rows,
        key=lambda row: (
            row["solver"],
            row["sigma_noise"],
            row["fig_k"],
            row["fig_w"],
        ),
    )


def write_markdown_report(path: Path, summary_rows: list[dict], args: argparse.Namespace):
    lines = [
        "# Toy 2D Benchmark Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Test observations: `{args.num_test_observations}`",
        f"- Solver samples per observation: `{args.num_solver_samples}`",
        f"- FIG variants: `{args.fig_variants}`",
        "",
        "## Aggregated Metrics",
        "",
        "| Solver | Sigma_n | FIG K | FIG w | Posterior Mean MSE | SW Distance | GT MSE | Measurement MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            "| {solver} | {sigma_noise:.3f} | {fig_k} | {fig_w:.3f} | {posterior_mean_mse_mean:.6f} | {swd_mean:.6f} | {gt_mse_mean:.6f} | {measurement_mse_mean:.6f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    output_dir = Path(args.output_dir) / args.dataset / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "benchmark_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    distribution = build_distribution(args.dataset)
    model, schedule, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)
    unconditional_sampler = UnconditionalDDIMSampler(model, schedule)

    observation_matrix = parse_observation_matrix(args.observation_matrix).to(device)
    noise_levels = parse_float_list(args.noise_levels)
    fig_k_values = parse_int_list(args.fig_k_values)
    fig_w_values = parse_float_list(args.fig_w_values)
    fig_variants = parse_string_list(args.fig_variants)

    test_points = distribution.sample(args.num_test_observations, device=device)
    unconditional_samples = unconditional_sampler.sample(args.num_solver_samples, device=device)

    raw_rows = []
    metric_keys = ["posterior_mean_mse", "swd", "gt_mse", "measurement_mse"]

    for sigma_noise in noise_levels:
        observation_model = LinearObservationModel(
            matrix=observation_matrix,
            sigma_noise=sigma_noise,
            device=device,
        )
        reference_sampler = ConditionalReferenceSampler(
            distribution=distribution,
            observation_model=observation_model,
            pool_size=args.reference_pool_size,
            device=device,
        )
        dps_solver = DPSSolver(
            model=model,
            schedule=schedule,
            observation_model=observation_model,
            zeta=args.dps_zeta,
        )

        for obs_idx, x_true in enumerate(test_points):
            y = observation_model.observe(x_true.unsqueeze(0)).squeeze(0)
            reference_samples = reference_sampler.sample(y, num_samples=args.num_reference_samples)
            dps_samples = dps_solver.sample(y, num_samples=args.num_solver_samples, device=device)

            raw_rows.append(
                {
                    "dataset": args.dataset,
                    "solver": "dps",
                    "sigma_noise": sigma_noise,
                    "fig_k": 0,
                    "fig_w": 0.0,
                    "observation_idx": obs_idx,
                    "posterior_mean_mse": posterior_mean_mse(dps_samples, reference_samples),
                    "swd": sliced_wasserstein_distance(dps_samples, reference_samples),
                    "gt_mse": ground_truth_mse(dps_samples, x_true),
                    "measurement_mse": measurement_mse(dps_samples, observation_model, y.unsqueeze(0)),
                }
            )

            best_fig_samples = None
            best_fig_config = None
            best_fig_score = None
            for variant in fig_variants:
                for fig_k in fig_k_values:
                    for fig_w in fig_w_values:
                        fig_solver = build_fig_solver(
                            variant=variant,
                            model=model,
                            schedule=schedule,
                            observation_model=observation_model,
                            fig_k=fig_k,
                            fig_w=fig_w,
                            fig_lr=args.fig_lr,
                            fig_mix_coef=args.fig_mix_coef,
                        )
                        fig_samples = fig_solver.sample(y, num_samples=args.num_solver_samples, device=device)
                        row = {
                            "dataset": args.dataset,
                            "solver": variant,
                            "sigma_noise": sigma_noise,
                            "fig_k": fig_k,
                            "fig_w": fig_w,
                            "observation_idx": obs_idx,
                            "posterior_mean_mse": posterior_mean_mse(fig_samples, reference_samples),
                            "swd": sliced_wasserstein_distance(fig_samples, reference_samples),
                            "gt_mse": ground_truth_mse(fig_samples, x_true),
                            "measurement_mse": measurement_mse(fig_samples, observation_model, y.unsqueeze(0)),
                        }
                        raw_rows.append(row)
                        if best_fig_score is None or row["posterior_mean_mse"] < best_fig_score:
                            best_fig_score = row["posterior_mean_mse"]
                            best_fig_samples = fig_samples
                            best_fig_config = (variant, fig_k, fig_w)

            if obs_idx < args.plot_count and best_fig_samples is not None:
                plot_path = output_dir / "plots" / f"sigma_{sigma_noise:.2f}_obs_{obs_idx:03d}.png"
                plot_title = (
                    f"{args.dataset} | sigma_n={sigma_noise:.2f} | "
                    f"{best_fig_config[0]}(K={best_fig_config[1]}, w={best_fig_config[2]:.2f})"
                )
                save_conditional_comparison_plot(
                    output_path=plot_path,
                    unconditional_samples=unconditional_samples,
                    dps_samples=dps_samples,
                    fig_samples=best_fig_samples,
                    x_true=x_true,
                    y=y,
                    observation_model=observation_model,
                    plot_limits=distribution.default_plot_limits(),
                    title=plot_title,
                )
                print(f"[plot] saved {plot_path}")

    raw_csv_path = output_dir / "benchmark_raw_metrics.csv"
    write_csv(raw_csv_path, raw_rows)

    summary_rows = summarize_rows(
        raw_rows,
        group_keys=["dataset", "solver", "sigma_noise", "fig_k", "fig_w"],
        metric_keys=metric_keys,
    )
    summary_csv_path = output_dir / "benchmark_summary_metrics.csv"
    write_csv(summary_csv_path, summary_rows)

    report_path = output_dir / "benchmark_report.md"
    write_markdown_report(report_path, summary_rows, args)

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Training config: {checkpoint['extra_config'].get('training', {})}")
    print(f"Saved raw metrics to {raw_csv_path}")
    print(f"Saved summary metrics to {summary_csv_path}")
    print(f"Saved markdown report to {report_path}")


if __name__ == "__main__":
    main()
