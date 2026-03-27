from __future__ import annotations

from dataclasses import dataclass

import torch


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.linspace(0, num_steps, num_steps + 1)
    f = torch.cos(((steps / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-5, 0.999)


def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


@dataclass
class DiffusionScheduleConfig:
    num_steps: int = 256
    schedule_type: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class DiffusionSchedule:
    def __init__(
        self,
        num_steps: int = 256,
        schedule_type: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
    ):
        self.config = DiffusionScheduleConfig(
            num_steps=num_steps,
            schedule_type=schedule_type,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.device = torch.device(device)
        if schedule_type == "cosine":
            betas = cosine_beta_schedule(num_steps)
        elif schedule_type == "linear":
            betas = linear_beta_schedule(num_steps, beta_start=beta_start, beta_end=beta_end)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        self.betas = torch.cat([torch.zeros(1), betas], dim=0).to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars).to(self.device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars).to(self.device)

    @property
    def num_steps(self) -> int:
        return self.config.num_steps

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        return DiffusionSchedule(
            num_steps=self.config.num_steps,
            schedule_type=self.config.schedule_type,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            device=device,
        )

    def sample_timesteps(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(1, self.num_steps + 1, (batch_size,), device=device)

    def normalize_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        return t.float() / float(self.num_steps)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        return (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha_bar.clamp_min(1e-8)

    def ddpm_step(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        x0_hat = self.predict_x0_from_eps(x_t, t, eps_pred)
        t_prev = (t - 1).clamp_min(0)

        alpha_t = self.alphas[t].unsqueeze(-1)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
        alpha_bar_prev = self.alpha_bars[t_prev].unsqueeze(-1)
        beta_t = self.betas[t].unsqueeze(-1)

        denom = (1.0 - alpha_bar_t).clamp_min(1e-8)
        coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / denom
        coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / denom
        mean = coef_x0 * x0_hat + coef_xt * x_t

        posterior_var = beta_t * (1.0 - alpha_bar_prev) / denom
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 1).float().unsqueeze(-1)
        return mean + nonzero_mask * torch.sqrt(posterior_var.clamp_min(1e-8)) * noise

    def ddim_step(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        x0_hat = self.predict_x0_from_eps(x_t, t, eps_pred)
        t_prev = (t - 1).clamp_min(0)
        sqrt_alpha_bar_prev = self.sqrt_alpha_bars[t_prev].unsqueeze(-1)
        sqrt_one_minus_prev = self.sqrt_one_minus_alpha_bars[t_prev].unsqueeze(-1)
        return sqrt_alpha_bar_prev * x0_hat + sqrt_one_minus_prev * eps_pred
