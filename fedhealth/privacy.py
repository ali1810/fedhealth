"""Client-level differential privacy for federated averaging (DP-FedAvg style).

Each client clips its model update to L2 norm <= C. The server adds Gaussian noise with
standard deviation z * C to the *sum* of clipped updates. Adding or removing one hospital
changes that sum by at most C, so each round is a Gaussian mechanism with noise multiplier z.

Accounting: Renyi DP of the Gaussian mechanism is alpha / (2 z^2) per round; over T rounds
with every client participating (no subsampling amplification - conservative) this composes
to T * alpha / (2 z^2). Converted to (epsilon, delta) with the standard bound
    epsilon = min_alpha [ T*alpha/(2 z^2) + log(1/delta)/(alpha - 1) ].

Note this protects *whole hospitals* (client-level DP), not individual patients inside a
hospital; with only a handful of clients, meaningful epsilon needs a lot of noise, which the
benchmark makes visible.
"""

from __future__ import annotations

import numpy as np


def clip(update: np.ndarray, max_norm: float) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(update))
    if norm > max_norm:
        return update * (max_norm / norm), True
    return update, False


def gaussian_noise(shape, noise_multiplier: float, max_norm: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, noise_multiplier * max_norm, size=shape)


def epsilon(noise_multiplier: float, rounds: int, delta: float = 1e-5) -> float:
    if noise_multiplier <= 0:
        return float("inf")
    alphas = np.concatenate([np.linspace(1.01, 10, 200), np.linspace(10, 512, 400)])
    rdp = rounds * alphas / (2 * noise_multiplier ** 2)
    eps = rdp + np.log(1 / delta) / (alphas - 1)
    return float(eps.min())
