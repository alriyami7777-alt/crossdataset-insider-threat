"""Two-layer MLP user-day scoring head (128-64-1).

Shared by the feature-only control (A1) and the day-supervised TGN readout
(A2/A3/B/C). Kept separate from ``temporal_gnn.TemporalHeteroGNN`` so the
existing edge-BCE API stays untouched.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UserDayMLP(nn.Module):
    """r -> logit with ReLU + dropout. ``in_dim`` is |x_ud|, mem_dim, or both."""

    def __init__(self, in_dim: int, hidden=(128, 64), dropout: float = 0.2):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r).squeeze(-1)
