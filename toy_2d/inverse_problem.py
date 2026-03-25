from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LinearObservationConfig:
    matrix: list[list[float]]
    sigma_noise: float = 0.05


class LinearObservationModel:
    def __init__(self, matrix: torch.Tensor, sigma_noise: float = 0.05, device: torch.device | str = "cpu"):
        matrix = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        if matrix.ndim != 2 or matrix.shape[1] != 2:
            raise ValueError("Observation matrix must have shape [m, 2].")
        self.A = matrix
        self.sigma_noise = sigma_noise
        self.device = torch.device(device)

    @property
    def measurement_dim(self) -> int:
        return self.A.shape[0]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.t()

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(x.shape[0], self.measurement_dim, device=x.device) * self.sigma_noise
        return self.apply(x) + noise

    def squared_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        residual = self.apply(x) - y
        return (residual ** 2).sum(dim=-1)

    def gaussian_nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.squared_error(x, y) / max(self.sigma_noise ** 2, 1e-8)

    def state_mask(self) -> torch.Tensor | None:
        # FIG+ only makes sense when the observation acts like a coordinate mask.
        A_cpu = self.A.detach().cpu()
        if torch.any((A_cpu != 0.0) & (A_cpu != 1.0)):
            return None

        mask = torch.zeros(self.A.shape[1], dtype=self.A.dtype)
        for row in A_cpu:
            nonzero = torch.nonzero(row, as_tuple=False).reshape(-1)
            if nonzero.numel() != 1:
                return None
            idx = nonzero.item()
            if abs(row[idx].item() - 1.0) > 1e-8:
                return None
            mask[idx] = 1.0
        return mask.to(self.device)

    def supports_mask_mixing(self) -> bool:
        return self.state_mask() is not None

    def project_observed_components(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.state_mask()
        if mask is None:
            raise ValueError("State-mask projection is only defined for mask-like operators.")
        return x * mask.unsqueeze(0)

    def project_hidden_components(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.state_mask()
        if mask is None:
            raise ValueError("State-mask projection is only defined for mask-like operators.")
        return x * (1.0 - mask.unsqueeze(0))

    def measurement_interpolant(
        self,
        y: torch.Tensor,
        alpha_bar_prev: torch.Tensor,
        state_noise: torch.Tensor,
        w: float,
    ) -> torch.Tensor:
        measurement_noise = self.apply(state_noise)
        return alpha_bar_prev.sqrt() * y + w * (1.0 - alpha_bar_prev).sqrt() * measurement_noise

    def project_line(self, y: torch.Tensor, num_points: int = 200, x_range: tuple[float, float] = (-3.0, 3.0)) -> torch.Tensor:
        if self.measurement_dim != 1:
            raise ValueError("plotting helper only supports a 1D observation.")
        a0, a1 = self.A[0, 0].item(), self.A[0, 1].item()
        target = y.reshape(-1)[0].item()
        xs = torch.linspace(x_range[0], x_range[1], num_points)
        if abs(a1) > 1e-8:
            ys = (target - a0 * xs) / a1
        elif abs(a0) > 1e-8:
            xs = torch.full_like(xs, target / a0)
            ys = torch.linspace(x_range[0], x_range[1], num_points)
        else:
            raise ValueError("Degenerate observation matrix.")
        return torch.stack([xs, ys], dim=-1)


class ConditionalReferenceSampler:
    def __init__(
        self,
        distribution,
        observation_model: LinearObservationModel,
        pool_size: int = 50_000,
        device: torch.device | str = "cpu",
    ):
        self.distribution = distribution
        self.observation_model = observation_model
        self.device = torch.device(device)
        self.pool = distribution.sample(pool_size, device=self.device)
        self.pool_measurements = observation_model.apply(self.pool)

    def sample(self, y: torch.Tensor, num_samples: int) -> torch.Tensor:
        y = y.to(self.device).reshape(1, -1)
        squared_error = ((self.pool_measurements - y) ** 2).sum(dim=-1)
        log_weights = -squared_error / (2.0 * max(self.observation_model.sigma_noise ** 2, 1e-8))
        weights = torch.softmax(log_weights - log_weights.max(), dim=0)
        indices = torch.multinomial(weights, num_samples=num_samples, replacement=True)
        return self.pool[indices]
