from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_SPECS = [
    ("posterior_mean_mse_mean", "Posterior Mean MSE", True),
    ("swd_mean", "Sliced Wasserstein Distance", True),
    ("measurement_mse_mean", "Measurement MSE", True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-style summary assets from benchmark CSV files.")
    parser.add_argument("--root_dir", type=str, default="toy_2d_outputs/paper_style")
    return parser.parse_args()


def _load_summary(root_dir: Path, dataset: str) -> pd.DataFrame:
    path = root_dir / dataset / "benchmark" / "benchmark_summary_metrics.csv"
    df = pd.read_csv(path)
    for col in ["sigma_noise", "fig_k", "fig_w"]:
        df[col] = df[col].astype(float)
    return df


def _best_rows(df: pd.DataFrame, solver: str) -> pd.DataFrame:
    rows = []
    sub = df[df["solver"] == solver]
    for sigma in sorted(sub["sigma_noise"].unique()):
        sigma_rows = sub[sub["sigma_noise"] == sigma].sort_values("posterior_mean_mse_mean")
        rows.append(sigma_rows.iloc[0])
    return pd.DataFrame(rows)


def _dps_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["solver"] == "dps"].sort_values("sigma_noise").reset_index(drop=True)


def _save_summary(root_dir: Path, summaries: list[dict]):
    summary_df = pd.DataFrame(summaries)
    csv_path = root_dir / "corrected_best_summary.csv"
    md_path = root_dir / "corrected_best_summary.md"
    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(summary_df.to_markdown(index=False))


def _make_trend_plot(root_dir: Path, dataset: str, dps_df: pd.DataFrame, fig_df: pd.DataFrame, fig_plus_df: pd.DataFrame):
    out_path = root_dir / "report_assets" / f"{dataset}_paper_metric_trends.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (metric, title, use_log) in zip(axes, METRIC_SPECS):
        ax.plot(dps_df["sigma_noise"], dps_df[metric], marker="o", linewidth=2, label="DPS")
        ax.plot(fig_df["sigma_noise"], fig_df[metric], marker="s", linewidth=2, label="Best FIG")
        ax.plot(fig_plus_df["sigma_noise"], fig_plus_df[metric], marker="^", linewidth=2, label="Best FIG+")
        if use_log:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(r"Measurement noise $\sigma_n$")
        ax.set_xticks(sorted(dps_df["sigma_noise"].unique()))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Metric value")
    axes[0].legend(loc="best")
    fig.suptitle(f"{dataset.replace('_', ' ').title()}: DPS vs best FIG / FIG+", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    summary_rows = []

    for dataset in ["two_moons", "eight_gaussians"]:
        df = _load_summary(root_dir, dataset)
        dps_df = _dps_rows(df)
        fig_df = _best_rows(df, "fig_snr")
        fig_plus_df = _best_rows(df, "fig_plus")

        for _, row in dps_df.iterrows():
            summary_rows.append(
                {
                    "dataset": dataset,
                    "solver": "dps",
                    "sigma_noise": row["sigma_noise"],
                    "fig_k": 0,
                    "fig_w": 0.0,
                    "posterior_mean_mse_mean": row["posterior_mean_mse_mean"],
                    "swd_mean": row["swd_mean"],
                    "measurement_mse_mean": row["measurement_mse_mean"],
                }
            )
        for _, row in fig_df.iterrows():
            summary_rows.append(
                {
                    "dataset": dataset,
                    "solver": "fig_snr",
                    "sigma_noise": row["sigma_noise"],
                    "fig_k": int(row["fig_k"]),
                    "fig_w": row["fig_w"],
                    "posterior_mean_mse_mean": row["posterior_mean_mse_mean"],
                    "swd_mean": row["swd_mean"],
                    "measurement_mse_mean": row["measurement_mse_mean"],
                }
            )
        for _, row in fig_plus_df.iterrows():
            summary_rows.append(
                {
                    "dataset": dataset,
                    "solver": "fig_plus",
                    "sigma_noise": row["sigma_noise"],
                    "fig_k": int(row["fig_k"]),
                    "fig_w": row["fig_w"],
                    "posterior_mean_mse_mean": row["posterior_mean_mse_mean"],
                    "swd_mean": row["swd_mean"],
                    "measurement_mse_mean": row["measurement_mse_mean"],
                }
            )

        _make_trend_plot(root_dir, dataset, dps_df, fig_df, fig_plus_df)

    _save_summary(root_dir, summary_rows)


if __name__ == "__main__":
    main()
