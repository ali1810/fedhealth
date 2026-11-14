"""Model definition and helpers to move weights in and out as flat vectors."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def get_flat(model: nn.Module) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().numpy().astype(np.float64)


def set_flat(model: nn.Module, flat: np.ndarray) -> None:
    flat_t = torch.as_tensor(flat, dtype=torch.float32)
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat_t[i:i + n].reshape(p.shape))
            i += n
    assert i == flat_t.numel(), "parameter vector has the wrong length"


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy()
