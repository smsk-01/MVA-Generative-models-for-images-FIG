from __future__ import annotations

import argparse
import csv
import math
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages

from toy_2d.benchmark import load_model_from_checkpoint
from toy_2d.datasets import build_distribution
from toy_2d.inverse_problem import LinearObservationModel
from toy_2d.plotting import save_conditional_comparison_plot
from toy_2d.solvers import DPSSolver, FIGDiffusionSolver, UnconditionalDDIMSampler


SIGMAS = [0.05, 0.5, 1.0]
FIG_K_VALUES = [1, 3, 5]
FIG_W_VALUES = [0.0, 0.5, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final English PDF report for toy 2D experiments.")
    parser.add_argument("--root_dir", type=str, default="toy_2d_outputs/final_runs")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _read_summary_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["sigma_noise", "fig_k", "fig_w"]:
        df[col] = df[col].astype(float)
    return df


def _best_fig_rows(df: pd.DataFrame) -> pd.DataFrame:
    fig_df = df[df["solver"] == "fig_diffusion"].copy()
    rows = []
    for sigma in SIGMAS:
        sub = fig_df[np.isclose(fig_df["sigma_noise"], sigma)]
        best = sub.sort_values("posterior_mean_mse_mean", ascending=True).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows)


def _dps_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sigma in SIGMAS:
        sub = df[(df["solver"] == "dps") & (np.isclose(df["sigma_noise"], sigma))]
        rows.append(sub.iloc[0])
    return pd.DataFrame(rows)


def _summary_table_rows(df: pd.DataFrame) -> list[list[str]]:
    dps_df = _dps_rows(df)
    best_fig_df = _best_fig_rows(df)
    rows = []
    for sigma in SIGMAS:
        dps = dps_df[np.isclose(dps_df["sigma_noise"], sigma)].iloc[0]
        fig = best_fig_df[np.isclose(best_fig_df["sigma_noise"], sigma)].iloc[0]
        rows.append(
            [
                f"{sigma:.2f}",
                f"{dps['posterior_mean_mse_mean']:.3f}",
                f"{dps['swd_mean']:.3f}",
                f"({int(fig['fig_k'])}, {fig['fig_w']:.1f})",
                f"{fig['posterior_mean_mse_mean']:.3f}",
                f"{fig['swd_mean']:.3f}",
                f"{fig['measurement_mse_mean']:.2e}",
            ]
        )
    return rows


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_metric_trend_figure(df: pd.DataFrame, dataset: str, output_path: Path) -> Path:
    dps_df = _dps_rows(df)
    best_fig_df = _best_fig_rows(df)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("posterior_mean_mse_mean", "Posterior Mean MSE"),
        ("swd_mean", "Sliced Wasserstein"),
        ("measurement_mse_mean", "Measurement MSE"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        ax.plot(dps_df["sigma_noise"], dps_df[metric], marker="o", linewidth=2, label="DPS")
        ax.plot(best_fig_df["sigma_noise"], best_fig_df[metric], marker="s", linewidth=2, label="Best FIG")
        ax.set_title(title)
        ax.set_xlabel(r"Measurement Noise $\sigma_n$")
        ax.set_xticks(SIGMAS)
        ax.grid(alpha=0.3)
        if metric == "measurement_mse_mean":
            ax.set_yscale("log")
        if metric == "posterior_mean_mse_mean":
            for _, row in best_fig_df.iterrows():
                ax.annotate(
                    f"K={int(row['fig_k'])}, w={row['fig_w']:.1f}",
                    (row["sigma_noise"], row[metric]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )
    axes[0].legend(loc="best")
    fig.suptitle(f"{dataset.replace('_', ' ').title()}: DPS vs Best FIG Across Noise Levels", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_ablation_heatmaps(df: pd.DataFrame, dataset: str, output_path: Path) -> Path:
    fig_df = df[df["solver"] == "fig_diffusion"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    metrics = [("posterior_mean_mse_mean", "Posterior Mean MSE"), ("swd_mean", "Sliced Wasserstein")]
    for row_idx, (metric, row_title) in enumerate(metrics):
        for col_idx, sigma in enumerate(SIGMAS):
            ax = axes[row_idx, col_idx]
            sub = fig_df[np.isclose(fig_df["sigma_noise"], sigma)]
            pivot = (
                sub.pivot_table(index="fig_k", columns="fig_w", values=metric, aggfunc="first")
                .reindex(index=FIG_K_VALUES, columns=FIG_W_VALUES)
            )
            image = ax.imshow(pivot.values, cmap="viridis")
            ax.set_title(f"$\\sigma_n$ = {sigma:.2f}")
            ax.set_xticks(range(len(FIG_W_VALUES)))
            ax.set_xticklabels([f"{w:.1f}" for w in FIG_W_VALUES])
            ax.set_yticks(range(len(FIG_K_VALUES)))
            ax.set_yticklabels([str(k) for k in FIG_K_VALUES])
            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\nK")
            else:
                ax.set_ylabel("K")
            ax.set_xlabel("w")
            for i, k in enumerate(FIG_K_VALUES):
                for j, w in enumerate(FIG_W_VALUES):
                    value = pivot.loc[k, w]
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{dataset.replace('_', ' ').title()}: FIG Ablation Over K and w", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_representative_plot(
    dataset: str,
    checkpoint_path: Path,
    sigma_noise: float,
    fig_k: int,
    fig_w: float,
    output_path: Path,
    device: str,
    seed: int,
    num_samples: int,
):
    torch.manual_seed(seed)
    distribution = build_distribution(dataset)
    model, schedule, _ = load_model_from_checkpoint(str(checkpoint_path), device=torch.device(device))
    observation_model = LinearObservationModel(matrix=torch.tensor([[1.0, 0.0]]), sigma_noise=sigma_noise, device=device)

    unconditional_sampler = UnconditionalDDIMSampler(model, schedule)
    dps_solver = DPSSolver(model, schedule, observation_model, zeta=0.05)
    fig_solver = FIGDiffusionSolver(model, schedule, observation_model, correction_steps=fig_k, w=fig_w, lr=0.5)

    x_true = distribution.sample(1, device=device).squeeze(0)
    y = observation_model.observe(x_true.unsqueeze(0)).squeeze(0)

    unconditional_samples = unconditional_sampler.sample(num_samples=num_samples, device=device)
    dps_samples = dps_solver.sample(y, num_samples=num_samples, device=device)
    fig_samples = fig_solver.sample(y, num_samples=num_samples, device=device)

    title = f"{dataset} | sigma_n={sigma_noise:.2f} | FIG(K={fig_k}, w={fig_w:.1f})"
    save_conditional_comparison_plot(
        output_path=output_path,
        unconditional_samples=unconditional_samples,
        dps_samples=dps_samples,
        fig_samples=fig_samples,
        x_true=x_true,
        y=y,
        observation_model=observation_model,
        plot_limits=distribution.default_plot_limits(),
        title=title,
    )
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def make_qualitative_assets(root_dir: Path, device: str, num_samples: int, seed: int) -> dict[str, list[Path]]:
    assets_dir = _ensure_dir(root_dir / "report_assets" / "qualitative")
    output: dict[str, list[Path]] = {}
    for dataset in ["two_moons", "eight_gaussians"]:
        summary_df = _read_summary_csv(root_dir / dataset / "benchmark" / "benchmark_summary_metrics.csv")
        best_fig_df = _best_fig_rows(summary_df)
        dataset_paths = []
        checkpoint_path = root_dir / dataset / "ddpm_toy_2d.pt"
        for idx, sigma in enumerate(SIGMAS):
            row = best_fig_df[np.isclose(best_fig_df["sigma_noise"], sigma)].iloc[0]
            out_path = assets_dir / f"{dataset}_sigma_{sigma:.2f}.png"
            generate_representative_plot(
                dataset=dataset,
                checkpoint_path=checkpoint_path,
                sigma_noise=float(sigma),
                fig_k=int(row["fig_k"]),
                fig_w=float(row["fig_w"]),
                output_path=out_path,
                device=device,
                seed=seed + idx + (100 if dataset == "eight_gaussians" else 0),
                num_samples=num_samples,
            )
            dataset_paths.append(out_path)
        output[dataset] = dataset_paths
    return output


def _page_with_text(pdf: PdfPages, title: str, paragraphs: list[str], footer: str | None = None):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.08, 0.95, title, fontsize=20, fontweight="bold", va="top")
    y = 0.90
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, width=100, break_long_words=False)
        fig.text(0.08, y, wrapped, fontsize=11.5, va="top", linespacing=1.5)
        y -= 0.035 * (wrapped.count("\n") + 2)
    if footer:
        fig.text(0.08, 0.04, footer, fontsize=9, color="dimgray")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_with_table_and_figure(
    pdf: PdfPages,
    title: str,
    intro: str,
    table_rows: list[list[str]],
    table_columns: list[str],
    figure_path: Path,
):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.08, 0.96, title, fontsize=18, fontweight="bold", va="top")
    fig.text(0.08, 0.92, textwrap.fill(intro, width=100), fontsize=11.5, va="top", linespacing=1.4)

    table_ax = fig.add_axes([0.08, 0.66, 0.84, 0.18])
    table_ax.axis("off")
    table = table_ax.table(cellText=table_rows, colLabels=table_columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    image_ax = fig.add_axes([0.08, 0.08, 0.84, 0.52])
    image_ax.axis("off")
    image_ax.imshow(mpimg.imread(figure_path))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_with_two_figures(pdf: PdfPages, title: str, intro: str, figure_paths: list[Path], captions: list[str]):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.08, 0.96, title, fontsize=18, fontweight="bold", va="top")
    fig.text(0.08, 0.92, textwrap.fill(intro, width=100), fontsize=11.5, va="top", linespacing=1.4)

    slots = [(0.08, 0.49, 0.84, 0.30), (0.08, 0.11, 0.84, 0.30)]
    for (x, y, w, h), image_path, caption in zip(slots, figure_paths, captions):
        image_ax = fig.add_axes([x, y, w, h])
        image_ax.axis("off")
        image_ax.imshow(mpimg.imread(image_path))
        fig.text(x, y - 0.025, textwrap.fill(caption, width=95), fontsize=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_with_three_figures(pdf: PdfPages, title: str, intro: str, figure_paths: list[Path], captions: list[str]):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.08, 0.96, title, fontsize=18, fontweight="bold", va="top")
    fig.text(0.08, 0.92, textwrap.fill(intro, width=100), fontsize=11.5, va="top", linespacing=1.4)

    slots = [(0.08, 0.67, 0.84, 0.18), (0.08, 0.39, 0.84, 0.18), (0.08, 0.11, 0.84, 0.18)]
    for (x, y, w, h), image_path, caption in zip(slots, figure_paths, captions):
        image_ax = fig.add_axes([x, y, w, h])
        image_ax.axis("off")
        image_ax.imshow(mpimg.imread(image_path))
        fig.text(x, y - 0.025, textwrap.fill(caption, width=95), fontsize=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _make_title_page(pdf: PdfPages):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.08, 0.86, "DPS vs FIG-Diffusion", fontsize=26, fontweight="bold")
    fig.text(0.08, 0.81, "A Complete 2D Inverse-Problem Study on Two-Moons and Eight-Gaussians", fontsize=16)
    fig.text(0.08, 0.74, "Final English PDF Report", fontsize=14)
    fig.text(
        0.08,
        0.64,
        textwrap.fill(
            "This document summarizes the full experimental pipeline, from training small DDPM priors in 2D to "
            "evaluating conditional samplers for linear inverse problems under several noise levels. The emphasis "
            "is on understanding when FIG-style interpolant guidance improves measurement consistency and when it "
            "hurts the quality of the recovered posterior distribution.",
            width=90,
        ),
        fontsize=12,
        linespacing=1.5,
    )
    fig.text(
        0.08,
        0.10,
        "Generated from workspace experiment outputs in toy_2d_outputs/final_runs",
        fontsize=10,
        color="dimgray",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf_report(root_dir: Path, device: str, num_samples: int, seed: int):
    assets_dir = _ensure_dir(root_dir / "report_assets")
    pdf_path = root_dir / "FIG_DPS_Toy2D_Report_English.pdf"

    qualitative_paths = make_qualitative_assets(root_dir=root_dir, device=device, num_samples=num_samples, seed=seed)

    trend_paths = {}
    heatmap_paths = {}
    summary_dfs = {}
    for dataset in ["two_moons", "eight_gaussians"]:
        df = _read_summary_csv(root_dir / dataset / "benchmark" / "benchmark_summary_metrics.csv")
        summary_dfs[dataset] = df
        trend_paths[dataset] = make_metric_trend_figure(df, dataset, assets_dir / f"{dataset}_metric_trends.png")
        heatmap_paths[dataset] = make_ablation_heatmaps(df, dataset, assets_dir / f"{dataset}_ablation_heatmaps.png")

    with PdfPages(pdf_path) as pdf:
        _make_title_page(pdf)

        _page_with_text(
            pdf,
            "Executive Summary",
            [
                "We trained lightweight DDPM priors on the two-moons and eight-gaussians toy datasets and used them "
                "as generative priors for a linear inverse problem where only the x-coordinate is observed. We then "
                "compared two conditional samplers: DPS, which performs gradient guidance through Tweedie x0 estimates, "
                "and FIG-Diffusion, which alternates an unconditional DDIM step with gradient corrections toward a "
                "time-dependent measurement interpolant.",
                "Across both datasets and all tested measurement-noise levels (0.05, 0.5, and 1.0), DPS achieved the "
                "best posterior-quality metrics: lower posterior mean MSE and lower sliced Wasserstein distance. "
                "FIG-Diffusion, however, consistently achieved much lower measurement error, often pushing samples "
                "almost exactly onto the observation constraint. In this 2D setup, that stronger measurement adherence "
                "came at the expense of diversity and posterior faithfulness.",
                "The main practical conclusion is that the FIG-style correction behaved here as an aggressive projector "
                "toward the measurement line. This is useful if strict data consistency is the main target, but it did "
                "not outperform DPS as a posterior sampler on the toy inverse problems considered in this report.",
            ],
        )

        _page_with_text(
            pdf,
            "Experimental Setup",
            [
                "Data. Two distributions were used: Two-Moons and Eight-Gaussians. Both are non-Gaussian and "
                "multimodal, which makes them useful sanity checks for posterior collapse and mode preservation.",
                "Prior model. Each dataset was modeled with a small epsilon-prediction MLP using sinusoidal timestep "
                "embeddings, four hidden layers, hidden dimension 128, and a cosine DDPM schedule with 64 diffusion "
                "steps. Training used 8000 optimization steps, batch size 1024, learning rate 1e-3, and the Apple "
                "MPS backend.",
                "Inverse problem. The observation model was y = A x* + n with A = [1, 0], so the first coordinate is "
                "observed and the second coordinate is hidden. Gaussian measurement noise was evaluated at sigma_n = "
                "0.05, 0.5, and 1.0.",
                "Evaluation. For each dataset and each sigma_n, we sampled 32 test observations. For every observation, "
                "each solver generated 256 conditional samples. A reference conditional distribution was approximated by "
                "importance reweighting 50,000 samples from the true toy distribution with the Gaussian likelihood. "
                "Metrics include posterior mean MSE, sliced Wasserstein distance, ground-truth MSE, and measurement MSE.",
            ],
        )

        _page_with_text(
            pdf,
            "Algorithmic Comparison",
            [
                "DPS. At each reverse step t, we estimate x0_hat from the current noisy sample x_t using the diffusion "
                "model and Tweedie's formula: x0_hat = (x_t - sqrt(1-alpha_bar_t) * eps_theta(x_t, t)) / sqrt(alpha_bar_t). "
                "We then compute a measurement loss || y - A x0_hat ||^2 and backpropagate it to x_t before applying the "
                "unconditional DDIM step. In words, DPS nudges the current latent state by looking at how well its "
                "predicted clean reconstruction matches the measurement.",
                "FIG-Diffusion. FIG first performs the unconditional DDIM update to obtain the next sample state. It then "
                "creates a time-dependent measurement interpolant y_t = sqrt(alpha_bar_t) y + w sqrt(1-alpha_bar_t) A eps "
                "and applies K gradient steps that minimize the Gaussian negative log-likelihood between A x and this "
                "interpolant. In words, FIG enforces data consistency directly on the updated sample through a moving "
                "target in measurement space.",
                "Hyperparameter grid. DPS used zeta = 0.05. FIG-Diffusion was evaluated over K in {1, 3, 5} and "
                "w in {0.0, 0.5, 1.0}. The FIG correction step size was set to 0.5 and normalized internally by the "
                "measurement variance and the operator norm of A.",
            ],
        )

        for dataset in ["two_moons", "eight_gaussians"]:
            intro = (
                f"The table below compares DPS with the best FIG configuration at each noise level, where 'best' means "
                f"the lowest posterior mean MSE among all tested FIG settings. The figure underneath shows how DPS and "
                f"the best FIG trajectory evolve across measurement noise for posterior quality, sliced Wasserstein "
                f"distance, and measurement consistency."
            )
            _page_with_table_and_figure(
                pdf,
                title=f"{dataset.replace('_', ' ').title()}: Quantitative Summary",
                intro=intro,
                table_rows=_summary_table_rows(summary_dfs[dataset]),
                table_columns=[
                    "sigma_n",
                    "DPS PM-MSE",
                    "DPS SWD",
                    "Best FIG (K,w)",
                    "FIG PM-MSE",
                    "FIG SWD",
                    "FIG Meas-MSE",
                ],
                figure_path=trend_paths[dataset],
            )

            _page_with_two_figures(
                pdf,
                title=f"{dataset.replace('_', ' ').title()}: Ablation Over K and w",
                intro=(
                    "These heatmaps show the full FIG-Diffusion ablation. The top row reports posterior mean MSE and "
                    "the bottom row reports sliced Wasserstein distance. Each column corresponds to a different "
                    "measurement noise level. Lower values are better."
                ),
                figure_paths=[heatmap_paths[dataset], trend_paths[dataset]],
                captions=[
                    "Full FIG ablation across the correction count K and the interpolant scaling parameter w.",
                    "Noise-wise comparison between DPS and the best FIG setting selected independently for each sigma_n.",
                ],
            )

            qualitative_intro = (
                "The three panels below show representative qualitative comparisons between unconditional prior samples "
                "(gray), DPS samples (blue), and FIG-Diffusion samples (red). Each plot uses the best FIG setting "
                "for posterior mean MSE at that noise level. The dashed line is the observation constraint, and the "
                "yellow marker is the ground-truth point."
            )
            captions = [
                f"Low-noise case (sigma_n = 0.05). This case is the easiest, so mode preservation is especially visible.",
                f"Medium-noise case (sigma_n = 0.50). This case highlights the trade-off between posterior spread and strict measurement matching.",
                f"High-noise case (sigma_n = 1.00). This case stresses robustness when the observation is weakly informative.",
            ]
            _page_with_three_figures(
                pdf,
                title=f"{dataset.replace('_', ' ').title()}: Qualitative Comparisons",
                intro=qualitative_intro,
                figure_paths=qualitative_paths[dataset],
                captions=captions,
            )

        _page_with_text(
            pdf,
            "Discussion",
            [
                "Why did DPS win here? In both toy datasets, the posterior remains highly structured and often "
                "multimodal because only one coordinate is observed. DPS preserves more of that structure by guiding "
                "through the model's x0 prediction instead of forcing the current sample directly onto the measurement "
                "constraint. This seems especially important for Eight-Gaussians, where several clusters remain plausible "
                "after conditioning on the observed x-coordinate.",
                "Why is FIG still interesting? FIG-Diffusion produces dramatically smaller measurement errors, sometimes "
                "close to numerical zero. So even though it underperformed as a posterior sampler in this study, it is "
                "clearly effective at enforcing observation consistency. In a different regime, for example with a "
                "better prior, a different step-size schedule, or harder operators, that mechanism could still prove "
                "useful.",
                "Role of K and w. Larger K often improved FIG when the noise was low, because repeated corrections "
                "helped align samples with the measurement. However, the interpolation scaling w was rarely helped by "
                "large values in our runs. Empirically, w = 0.0 or 0.5 was usually preferable to w = 1.0.",
            ],
        )

        _page_with_text(
            pdf,
            "Limitations and Next Steps",
            [
                "This report should be interpreted as a careful toy-study, not as a definitive refutation of the "
                "FIG paper. The current implementation is deliberately lightweight: a small MLP prior, a single linear "
                "operator, a short DDPM schedule, and a simple adaptation of FIG-Diffusion to 2D data.",
                "The most valuable next experiment would be a dedicated tuning sweep over the FIG correction strength. "
                "In the current runs, FIG clearly over-enforced the measurement line in several settings. A smaller or "
                "time-dependent correction step might improve the balance between consistency and diversity.",
                "Other useful extensions would be: using deeper score networks, increasing training time, adding more "
                "operators than A = [1, 0], generating all qualitative figures from the exact best FIG configuration "
                "used in the numerical table, and comparing against a posterior with an analytical reference on a toy "
                "Gaussian prior for calibration.",
            ],
            footer=f"PDF saved to: {pdf_path}",
        )

    return pdf_path


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    pdf_path = build_pdf_report(root_dir=root_dir, device=args.device, num_samples=args.num_samples, seed=args.seed)
    print(f"Saved PDF report to {pdf_path}")


if __name__ == "__main__":
    main()
