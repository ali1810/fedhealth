"""Data loading, non-IID partitioning across simulated hospitals, and data-quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass
class ClientData:
    """The private data held by one hospital. Never leaves the client object."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.y_train)


@dataclass
class QualityReport:
    name: str
    n_samples: int
    n_features: int
    missing_fraction: float
    positive_rate: float
    constant_features: int
    outlier_fraction: float  # share of values beyond 4 robust z-scores
    warnings: list[str] = field(default_factory=list)


def load_dataset(seed: int = 0, test_size: float = 0.2):
    """Wisconsin Diagnostic Breast Cancer data (569 patients, 30 features, ships with scikit-learn).

    Returns a pooled train split (to be partitioned across hospitals) and a held-out global
    test set that stands in for an external validation cohort.
    """
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, rng: np.random.Generator,
                        min_size: int = 10) -> list[np.ndarray]:
    """Split sample indices across clients with label skew drawn from Dirichlet(alpha).

    Small alpha -> very different class mixes per hospital (strongly non-IID).
    Large alpha -> close to IID. Resamples until every client has at least `min_size` samples.
    """
    classes = np.unique(y)
    for _ in range(1000):
        buckets: list[list[int]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.where(y == c)[0])
            props = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                buckets[k].extend(part.tolist())
        if min(len(b) for b in buckets) >= min_size:
            return [np.array(sorted(b)) for b in buckets]
    raise RuntimeError("Could not find a partition meeting min_size; increase alpha or reduce clients.")


def make_clients(X: np.ndarray, y: np.ndarray, n_clients: int = 5, alpha: float = 0.5,
                 seed: int = 0, local_test_size: float = 0.25) -> list[ClientData]:
    rng = np.random.default_rng(seed)
    parts = dirichlet_partition(y, n_clients, alpha, rng)
    clients = []
    for k, idx in enumerate(parts):
        Xk, yk = X[idx], y[idx]
        strat = yk if min(np.bincount(yk, minlength=2)) >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(Xk, yk, test_size=local_test_size,
                                              stratify=strat, random_state=seed + k)
        clients.append(ClientData(f"hospital_{k + 1}", Xtr, ytr, Xte, yte))
    return clients


def quality_report(c: ClientData) -> QualityReport:
    """Checks a hospital can run locally before joining a federation.

    Only summary numbers are produced here, so the report itself can be shared.
    """
    X = c.X_train
    missing = float(np.isnan(X).mean())
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    mad[mad == 0] = 1.0
    robust_z = np.abs(X - med) / (1.4826 * mad)
    outliers = float(np.nanmean(robust_z > 4))
    constant = int((np.nanstd(X, axis=0) == 0).sum())
    pos = float(c.y_train.mean())
    warnings = []
    if c.n_train < 50:
        warnings.append("small sample size")
    if pos < 0.1 or pos > 0.9:
        warnings.append("strong class imbalance")
    if missing > 0:
        warnings.append("missing values present")
    if constant:
        warnings.append(f"{constant} constant feature(s)")
    if outliers > 0.02:
        warnings.append("many extreme values")
    return QualityReport(c.name, c.n_train, X.shape[1], missing, pos, constant, outliers, warnings)
