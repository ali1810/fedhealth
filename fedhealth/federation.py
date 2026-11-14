"""Federated training loop: hospitals train locally, the server only sees (masked) updates."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn

from . import privacy, secure_agg
from .data import ClientData
from .model import MLP, get_flat, n_params, predict_proba, set_flat


@dataclass
class FLConfig:
    rounds: int = 30
    local_epochs: int = 2
    batch_size: int = 16
    lr: float = 0.05
    strategy: str = "fedavg"        # "fedavg" or "fedprox"
    prox_mu: float = 0.1
    secure_aggregation: bool = True
    dp_noise_multiplier: float = 0.0  # 0 disables differential privacy
    dp_clip_norm: float = 1.0
    dp_delta: float = 1e-5
    hidden: int = 32
    seed: int = 0


@dataclass
class History:
    round: list[int] = field(default_factory=list)
    global_acc: list[float] = field(default_factory=list)
    global_auc: list[float] = field(default_factory=list)
    local_acc: list[float] = field(default_factory=list)
    bytes_sent: list[int] = field(default_factory=list)
    clipped_fraction: list[float] = field(default_factory=list)
    seconds: float = 0.0
    epsilon: float = float("inf")


class Hospital:
    """A federation participant. Raw data stays inside this object."""

    def __init__(self, data: ClientData, n_features: int, cfg: FLConfig, client_id: int):
        self.data = data
        self.id = client_id
        self.cfg = cfg
        self.model = MLP(n_features, cfg.hidden)
        self.mean = None
        self.std = None
        self.gen = torch.Generator().manual_seed(cfg.seed * 1000 + client_id)

    # -- federated standardisation: only sums leave the hospital
    def feature_stats(self) -> np.ndarray:
        X = self.data.X_train.astype(np.float64)
        return np.concatenate([[len(X)], X.sum(0), (X ** 2).sum(0)])

    def set_normalisation(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean, self.std = mean, std

    def _norm(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def local_update(self, global_w: np.ndarray) -> np.ndarray:
        set_flat(self.model, global_w)
        X = torch.as_tensor(self._norm(self.data.X_train))
        y = torch.as_tensor(self.data.y_train, dtype=torch.float32)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        anchor = [p.detach().clone() for p in self.model.parameters()]
        self.model.train()
        for _ in range(self.cfg.local_epochs):
            perm = torch.randperm(len(y), generator=self.gen)
            for i in range(0, len(y), self.cfg.batch_size):
                b = perm[i:i + self.cfg.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model(X[b]), y[b])
                if self.cfg.strategy == "fedprox":
                    loss = loss + 0.5 * self.cfg.prox_mu * sum(
                        ((p - a) ** 2).sum() for p, a in zip(self.model.parameters(), anchor))
                loss.backward()
                opt.step()
        return get_flat(self.model) - global_w

    def evaluate(self, w: np.ndarray) -> float:
        set_flat(self.model, w)
        p = predict_proba(self.model, self._norm(self.data.X_test))
        return accuracy_score(self.data.y_test, p > 0.5)


def _sum(vectors: list[np.ndarray], secure: bool, round_id: int) -> np.ndarray:
    return secure_agg.secure_sum(vectors, round_id) if secure else np.sum(vectors, axis=0)


def federated_normalisation(hospitals: list[Hospital], secure: bool) -> tuple[np.ndarray, np.ndarray]:
    total = _sum([h.feature_stats() for h in hospitals], secure, round_id=-1)
    n, d = total[0], (len(total) - 1) // 2
    mean = total[1:1 + d] / n
    var = np.maximum(total[1 + d:] / n - mean ** 2, 1e-12)
    std = np.sqrt(var)
    for h in hospitals:
        h.set_normalisation(mean, std)
    return mean, std


def _score(model: nn.Module, w, X, y) -> tuple[float, float]:
    set_flat(model, w)
    p = predict_proba(model, X)
    return accuracy_score(y, p > 0.5), roc_auc_score(y, p)


def run_federated(clients: list[ClientData], X_test: np.ndarray, y_test: np.ndarray,
                  cfg: FLConfig) -> History:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    d = X_test.shape[1]
    hospitals = [Hospital(c, d, cfg, i) for i, c in enumerate(clients)]
    mean, std = federated_normalisation(hospitals, cfg.secure_aggregation)
    Xg = ((X_test - mean) / std).astype(np.float32)

    server_model = MLP(d, cfg.hidden)
    w = get_flat(server_model)
    P, K = n_params(server_model), len(hospitals)
    use_dp = cfg.dp_noise_multiplier > 0
    sizes = np.array([h.data.n_train for h in hospitals], dtype=np.float64)

    hist = History()
    t0 = time.perf_counter()
    for r in range(1, cfg.rounds + 1):
        updates = [h.local_update(w) for h in hospitals]
        if use_dp:
            clipped = [privacy.clip(u, cfg.dp_clip_norm) for u in updates]
            hist.clipped_fraction.append(float(np.mean([c for _, c in clipped])))
            total = _sum([u for u, _ in clipped], cfg.secure_aggregation, r)
            total += privacy.gaussian_noise(total.shape, cfg.dp_noise_multiplier, cfg.dp_clip_norm, rng)
            w = w + total / K  # uniform weights: sample counts would themselves leak information
        else:
            # weighted FedAvg: send n_k * update and n_k, so even weights stay inside the secure sum
            payload = [np.concatenate([[n], n * u]) for n, u in zip(sizes, updates)]
            total = _sum(payload, cfg.secure_aggregation, r)
            w = w + total[1:] / total[0]

        acc, auc = _score(server_model, w, Xg, y_test)
        hist.round.append(r)
        hist.global_acc.append(acc)
        hist.global_auc.append(auc)
        hist.local_acc.append(float(np.mean([h.evaluate(w) for h in hospitals])))
        # download float32 weights; upload float32 (plain) or 8-byte masked integers (SecAgg),
        # plus each client sending its 32-byte public key and receiving all K keys
        up = secure_agg.BYTES_PER_VALUE if cfg.secure_aggregation else 4
        sent = K * P * 4 + K * P * up + (K * 32 + K * K * 32 if cfg.secure_aggregation else 0)
        hist.bytes_sent.append(sent)
    hist.seconds = time.perf_counter() - t0
    if use_dp:
        hist.epsilon = privacy.epsilon(cfg.dp_noise_multiplier, cfg.rounds, cfg.dp_delta)
    return hist


# ----------------------------------------------------------------- baselines

def _train_plain(X, y, d, cfg: FLConfig, epochs: int) -> MLP:
    torch.manual_seed(cfg.seed)
    model = MLP(d, cfg.hidden)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(cfg.seed)
    for _ in range(epochs):
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(yt), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss_fn(model(Xt[b]), yt[b]).backward()
            opt.step()
    return model


def run_centralised(clients, X_test, y_test, cfg: FLConfig) -> tuple[float, float]:
    """Upper bound: what you would get if all hospitals pooled their raw data."""
    X = np.concatenate([c.X_train for c in clients])
    y = np.concatenate([c.y_train for c in clients])
    mean, std = X.mean(0), X.std(0) + 1e-6
    model = _train_plain(((X - mean) / std).astype(np.float32), y, X.shape[1], cfg,
                         epochs=cfg.rounds * cfg.local_epochs)
    return _score(model, get_flat(model), ((X_test - mean) / std).astype(np.float32), y_test)


def run_local_only(clients, X_test, y_test, cfg: FLConfig) -> tuple[float, float]:
    """Lower bound: every hospital trains alone on its own data. Returns mean global-test scores."""
    accs, aucs = [], []
    for c in clients:
        mean, std = c.X_train.mean(0), c.X_train.std(0) + 1e-6
        model = _train_plain(((c.X_train - mean) / std).astype(np.float32), c.y_train,
                             c.X_train.shape[1], cfg, epochs=cfg.rounds * cfg.local_epochs)
        a, u = _score(model, get_flat(model), ((X_test - mean) / std).astype(np.float32), y_test)
        accs.append(a)
        aucs.append(u)
    return float(np.mean(accs)), float(np.mean(aucs))
