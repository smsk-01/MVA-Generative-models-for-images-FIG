from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F


class DiffusionTrainer:
    def __init__(
        self,
        model,
        schedule,
        distribution,
        device: torch.device | str = "cpu",
        lr: float = 1e-3,
    ):
        self.model = model
        self.schedule = schedule
        self.distribution = distribution
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self,
        num_steps: int,
        batch_size: int,
        log_every: int = 200,
    ) -> list[float]:
        self.model.train()
        losses = []
        for step in range(1, num_steps + 1):
            x0 = self.distribution.sample(batch_size, device=self.device)
            noise = torch.randn_like(x0)
            t = self.schedule.sample_timesteps(batch_size, device=self.device)
            x_t = self.schedule.q_sample(x0, t, noise)
            eps_pred = self.model(x_t, self.schedule.normalize_timesteps(t))
            loss = F.mse_loss(eps_pred, noise)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            if step % log_every == 0:
                mean_loss = sum(losses[-log_every:]) / min(log_every, len(losses))
                print(f"[train] step={step:06d} loss={mean_loss:.6f}")
        return losses

    def save_checkpoint(
        self,
        output_path: str | Path,
        losses: list[float],
        extra_config: dict | None = None,
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "schedule_config": asdict(self.schedule.config),
            "losses": losses,
            "extra_config": extra_config or {},
        }
        torch.save(payload, output_path)

