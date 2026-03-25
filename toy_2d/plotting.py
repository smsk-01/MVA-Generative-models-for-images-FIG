from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _scatter(ax, points: torch.Tensor, color: str, label: str, alpha: float = 0.6, size: int = 10):
    ax.scatter(points[:, 0].cpu(), points[:, 1].cpu(), s=size, c=color, label=label, alpha=alpha, edgecolors="none")


def save_conditional_comparison_plot(
    output_path: str | Path,
    unconditional_samples: torch.Tensor,
    dps_samples: torch.Tensor,
    fig_samples: torch.Tensor,
    x_true: torch.Tensor,
    y: torch.Tensor,
    observation_model,
    plot_limits: tuple[float, float, float, float],
    title: str,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter(ax, unconditional_samples, color="lightgray", label="Unconditional prior", alpha=0.45, size=10)
    _scatter(ax, dps_samples, color="#1f77b4", label="DPS", alpha=0.75, size=14)
    _scatter(ax, fig_samples, color="#d62728", label="FIG-Diffusion", alpha=0.75, size=14)

    obs_line = observation_model.project_line(y, x_range=(plot_limits[0], plot_limits[1]))
    ax.plot(obs_line[:, 0].cpu(), obs_line[:, 1].cpu(), color="black", linestyle="--", linewidth=1.5, label="Observation constraint")
    ax.scatter([x_true[0].item()], [x_true[1].item()], color="gold", edgecolors="black", s=80, label=r"$x^\star$")

    ax.set_xlim(plot_limits[0], plot_limits[1])
    ax.set_ylim(plot_limits[2], plot_limits[3])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

