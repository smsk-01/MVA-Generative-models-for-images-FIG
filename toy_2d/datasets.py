from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


class ToyDistribution:
    name: str

    def sample(self, num_samples: int, device: torch.device | str = "cpu") -> torch.Tensor:
        raise NotImplementedError

    def default_plot_limits(self) -> tuple[float, float, float, float]:
        raise NotImplementedError


@dataclass
class TwoMoonsConfig:
    noise_std: float = 0.08
    scale: float = 1.6


class TwoMoons(ToyDistribution):
    name = "two_moons"

    def __init__(self, noise_std: float = 0.08, scale: float = 1.6):
        self.config = TwoMoonsConfig(noise_std=noise_std, scale=scale)

    def sample(self, num_samples: int, device: torch.device | str = "cpu") -> torch.Tensor:
        num_first = num_samples // 2
        num_second = num_samples - num_first

        theta_first = torch.rand(num_first, device=device) * torch.pi
        theta_second = torch.rand(num_second, device=device) * torch.pi

        first = torch.stack([torch.cos(theta_first), torch.sin(theta_first)], dim=-1)
        second = torch.stack([1.0 - torch.cos(theta_second), 1.0 - torch.sin(theta_second) - 0.5], dim=-1)

        points = torch.cat([first, second], dim=0)
        noise = torch.randn_like(points) * self.config.noise_std
        points = points + noise

        center = torch.tensor([0.5, 0.25], device=device)
        points = (points - center) * self.config.scale
        return points

    def default_plot_limits(self) -> tuple[float, float, float, float]:
        return (-3.0, 3.0, -2.5, 2.5)


@dataclass
class EightGaussiansConfig:
    radius: float = 2.0
    std: float = 0.12


class EightGaussians(ToyDistribution):
    name = "eight_gaussians"

    def __init__(self, radius: float = 2.0, std: float = 0.12):
        self.config = EightGaussiansConfig(radius=radius, std=std)
        angles = torch.arange(8) * (torch.pi / 4.0)
        self.centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) * radius

    def sample(self, num_samples: int, device: torch.device | str = "cpu") -> torch.Tensor:
        centers = self.centers.to(device)
        component_ids = torch.randint(0, centers.shape[0], (num_samples,), device=device)
        chosen_centers = centers[component_ids]
        noise = torch.randn(num_samples, 2, device=device) * self.config.std
        return chosen_centers + noise

    def default_plot_limits(self) -> tuple[float, float, float, float]:
        return (-3.0, 3.0, -3.0, 3.0)


def build_distribution(name: str, **kwargs) -> ToyDistribution:
    registry: Dict[str, type[ToyDistribution]] = {
        TwoMoons.name: TwoMoons,
        EightGaussians.name: EightGaussians,
    }
    if name not in registry:
        raise ValueError(f"Unknown toy dataset: {name}")
    return registry[name](**kwargs)

