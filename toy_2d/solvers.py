from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DPSConfig:
    zeta: float = 0.05
    grad_clip_norm: float = 10.0


@dataclass
class FIGConfig:
    correction_steps: int = 5
    w: float = 1.0
    lr: float = 0.5
    grad_clip_norm: float = 10.0
    use_snr_weighting: bool = True
    use_fig_plus: bool = False
    mix_coef: float = 0.95


class UnconditionalDDIMSampler:
    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device | str,
        initial_x: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        device = torch.device(device)
        x = initial_x.clone().to(device) if initial_x is not None else torch.randn(num_samples, 2, device=device)
        trajectory = [x.detach().cpu()] if return_trajectory else None
        for step in range(self.schedule.num_steps, 0, -1):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            eps_pred = self.model(x, self.schedule.normalize_timesteps(t))
            x = self.schedule.ddim_step(x, t, eps_pred)
            if return_trajectory:
                trajectory.append(x.detach().cpu())
        if return_trajectory:
            return x, trajectory
        return x


class DPSSolver:
    def __init__(self, model, schedule, observation_model, zeta: float = 0.05, grad_clip_norm: float = 10.0):
        self.model = model
        self.schedule = schedule
        self.observation_model = observation_model
        self.config = DPSConfig(zeta=zeta, grad_clip_norm=grad_clip_norm)

    def _clip_grad(self, grad: torch.Tensor) -> torch.Tensor:
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (self.config.grad_clip_norm / grad_norm).clamp(max=1.0)
        return grad * scale

    def sample(
        self,
        y: torch.Tensor,
        num_samples: int,
        device: torch.device | str,
        initial_x: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        device = torch.device(device)
        y = y.to(device).reshape(1, -1).expand(num_samples, -1)
        x = initial_x.clone().to(device) if initial_x is not None else torch.randn(num_samples, 2, device=device)
        trajectory = [x.detach().cpu()] if return_trajectory else None

        for step in range(self.schedule.num_steps, 0, -1):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)

            x = x.detach().requires_grad_(True)
            eps_pred = self.model(x, self.schedule.normalize_timesteps(t))
            x0_hat = self.schedule.predict_x0_from_eps(x, t, eps_pred)
            loss = self.observation_model.squared_error(x0_hat, y).sum()
            grad = torch.autograd.grad(loss, x)[0]
            grad = self._clip_grad(grad)
            x = (x - self.config.zeta * grad).detach()

            with torch.no_grad():
                eps_pred = self.model(x, self.schedule.normalize_timesteps(t))
                x = self.schedule.ddim_step(x, t, eps_pred)
            if return_trajectory:
                trajectory.append(x.detach().cpu())
        if return_trajectory:
            return x, trajectory
        return x


class FIGDiffusionSolver:
    def __init__(
        self,
        model,
        schedule,
        observation_model,
        correction_steps: int = 5,
        w: float = 1.0,
        lr: float = 0.5,
        grad_clip_norm: float = 10.0,
        use_snr_weighting: bool = True,
        use_fig_plus: bool = False,
        mix_coef: float = 0.95,
    ):
        self.model = model
        self.schedule = schedule
        self.observation_model = observation_model
        self.config = FIGConfig(
            correction_steps=correction_steps,
            w=w,
            lr=lr,
            grad_clip_norm=grad_clip_norm,
            use_snr_weighting=use_snr_weighting,
            use_fig_plus=use_fig_plus,
            mix_coef=mix_coef,
        )
        spectral_norm = torch.linalg.matrix_norm(self.observation_model.A.detach().cpu(), ord=2).item()
        self.spectral_norm_sq = max(spectral_norm ** 2, 1e-8)

    def _clip_grad(self, grad: torch.Tensor) -> torch.Tensor:
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (self.config.grad_clip_norm / grad_norm).clamp(max=1.0)
        return grad * scale

    def _guidance_step_size(self, alpha_bar_prev: torch.Tensor) -> torch.Tensor:
        delta_t = 1.0 / float(self.schedule.num_steps)
        if self.config.use_snr_weighting:
            snr_recip = (1.0 - alpha_bar_prev).clamp_min(0.0) / alpha_bar_prev.clamp_min(1e-8)
            return self.config.lr * delta_t * snr_recip / self.spectral_norm_sq
        return self.config.lr * (self.observation_model.sigma_noise ** 2) / self.spectral_norm_sq

    def sample(
        self,
        y: torch.Tensor,
        num_samples: int,
        device: torch.device | str,
        initial_x: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        device = torch.device(device)
        y = y.to(device).reshape(1, -1).expand(num_samples, -1)
        x = initial_x.clone().to(device) if initial_x is not None else torch.randn(num_samples, 2, device=device)
        trajectory = [x.detach().cpu()] if return_trajectory else None

        for step in range(self.schedule.num_steps, 0, -1):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            with torch.no_grad():
                eps_pred = self.model(x, self.schedule.normalize_timesteps(t))
                x0_hat = self.schedule.predict_x0_from_eps(x, t, eps_pred)
                x = self.schedule.ddim_step(x, t, eps_pred)

            alpha_bar_prev = self.schedule.alpha_bars[step - 1].to(device)
            state_noise = torch.randn_like(x)
            y_interp = self.observation_model.measurement_interpolant(
                y=y,
                alpha_bar_prev=alpha_bar_prev,
                state_noise=state_noise,
                w=self.config.w,
            )
            step_size = self._guidance_step_size(alpha_bar_prev)

            for _ in range(self.config.correction_steps):
                x = x.detach().requires_grad_(True)
                loss = self.observation_model.gaussian_nll(x, y_interp).sum()
                grad = torch.autograd.grad(loss, x)[0]
                grad = self._clip_grad(grad)
                x = (x - step_size * grad).detach()

            if self.config.use_fig_plus and self.observation_model.supports_mask_mixing():
                with torch.no_grad():
                    sqrt_alpha_bar_prev = self.schedule.sqrt_alpha_bars[step - 1].to(device)
                    sqrt_one_minus_prev = self.schedule.sqrt_one_minus_alpha_bars[step - 1].to(device)
                    tweedie_state = sqrt_alpha_bar_prev * x0_hat + sqrt_one_minus_prev * state_noise
                    x = (
                        self.observation_model.project_observed_components(x)
                        + (1.0 - self.config.mix_coef) * self.observation_model.project_hidden_components(x)
                        + self.config.mix_coef * self.observation_model.project_hidden_components(tweedie_state)
                    )
            if return_trajectory:
                trajectory.append(x.detach().cpu())
        if return_trajectory:
            return x, trajectory
        return x
