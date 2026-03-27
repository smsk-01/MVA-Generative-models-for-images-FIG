from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for sinusoidal embeddings.")
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10_000.0) * torch.arange(half_dim, device=t.device) / max(half_dim - 1, 1)
        freqs = torch.exp(exponent)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualTimeBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.time_to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_to_scale_shift(time_features).chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1.0 + scale) + shift
        h = self.net(h)
        return x + h


class EpsilonMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([ResidualTimeBlock(hidden_dim) for _ in range(num_layers)])
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, t_normalized: torch.Tensor) -> torch.Tensor:
        time_features = self.time_embedding(t_normalized)
        time_features = self.time_mlp(time_features)

        hidden = self.input_proj(x)
        for block in self.blocks:
            hidden = block(hidden, time_features)
        return self.output_head(hidden)
