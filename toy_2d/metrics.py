from __future__ import annotations

import torch


def posterior_mean_mse(samples: torch.Tensor, reference_samples: torch.Tensor) -> float:
    return ((samples.mean(dim=0) - reference_samples.mean(dim=0)) ** 2).mean().item()


def ground_truth_mse(samples: torch.Tensor, x_true: torch.Tensor) -> float:
    return ((samples.mean(dim=0) - x_true.reshape(-1)) ** 2).mean().item()


def measurement_mse(samples: torch.Tensor, observation_model, y: torch.Tensor) -> float:
    return observation_model.squared_error(samples, y.expand(samples.shape[0], -1)).mean().item()


def sliced_wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    num_projections: int = 128,
    p: int = 2,
) -> float:
    if x.shape != y.shape:
        n = min(x.shape[0], y.shape[0])
        x = x[:n]
        y = y[:n]

    projections = torch.randn(num_projections, x.shape[1], device=x.device)
    projections = projections / projections.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    proj_x = torch.sort(x @ projections.t(), dim=0).values
    proj_y = torch.sort(y @ projections.t(), dim=0).values
    distance = torch.mean(torch.abs(proj_x - proj_y) ** p, dim=0)
    return torch.mean(distance).pow(1.0 / p).item()

