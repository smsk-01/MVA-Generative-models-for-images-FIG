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
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = []
        feature_dim = input_dim + hidden_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(feature_dim, hidden_dim))
            layers.append(nn.SiLU())
            feature_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t_normalized: torch.Tensor) -> torch.Tensor:
        time_features = self.time_embedding(t_normalized)
        time_features = self.time_mlp(time_features)
        model_input = torch.cat([x, time_features], dim=-1)
        return self.net(model_input)

