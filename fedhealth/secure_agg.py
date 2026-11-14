"""Secure aggregation with pairwise masking (after Bonawitz et al., CCS 2017, simplified).

Every pair of clients agrees on a shared secret with X25519 Diffie-Hellman and expands it
with HKDF into a seed. Client i adds the mask PRG(seed_ij) for every j > i and subtracts it
for every j < i. Each masked vector on its own looks like uniform noise to the server, but
in the sum all masks cancel, so the server learns only the aggregate.

Arithmetic is done in fixed point modulo 2**62 (stored in int64, so the sum of two residues
never overflows) and the cancellation is exact.

Simplifications versus the full protocol (documented on purpose):
  * no dropout recovery (Shamir secret sharing of seeds) - all clients must report;
  * public keys are exchanged through the server without authentication, so this protects
    against an honest-but-curious server, not an active man-in-the-middle;
  * values must stay inside the fixed-point range (checked).
"""

from __future__ import annotations

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MOD_BITS = 62
MOD = 1 << MOD_BITS
SCALE = 1 << 20          # fixed-point precision: about 1e-6
MAX_ABS = (MOD // 2) / SCALE  # about 2.2e12: largest |sum| we can represent
BYTES_PER_VALUE = 8


def encode(x: np.ndarray) -> np.ndarray:
    if np.any(np.abs(x) >= MAX_ABS):
        raise ValueError("value outside fixed-point range; clip or rescale before aggregation")
    return np.round(x * SCALE).astype(np.int64) % MOD


def decode(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.int64) % MOD
    v = np.where(v >= MOD // 2, v - MOD, v)
    return v.astype(np.float64) / SCALE


class SecAggClient:
    def __init__(self, client_id: int):
        self.id = client_id
        self._sk = X25519PrivateKey.generate()
        self._seeds: dict[int, bytes] = {}

    def public_key(self) -> bytes:
        return self._sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def agree(self, public_keys: dict[int, bytes], round_id: int) -> None:
        """Derive one seed per peer. Including round_id gives fresh masks every round."""
        self._seeds = {}
        for j, pk in public_keys.items():
            if j == self.id:
                continue
            shared = self._sk.exchange(X25519PublicKey.from_public_bytes(pk))
            lo, hi = sorted((self.id, j))
            info = f"fedhealth-secagg|{lo}|{hi}|round{round_id}".encode()
            self._seeds[j] = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(shared)

    def mask(self, x: np.ndarray) -> np.ndarray:
        y = encode(x)
        for j, seed in self._seeds.items():
            prg = np.random.Generator(np.random.PCG64(int.from_bytes(seed, "big")))
            m = prg.integers(0, MOD, size=y.shape, dtype=np.int64)
            y = (y + m) % MOD if self.id < j else (y - m) % MOD
        return y


def aggregate(masked: list[np.ndarray]) -> np.ndarray:
    """Server side: add the masked vectors; the pairwise masks cancel."""
    total = np.zeros_like(masked[0])
    for m in masked:
        total = (total + m) % MOD
    return decode(total)


def secure_sum(vectors: list[np.ndarray], round_id: int = 0) -> np.ndarray:
    """Run one full round of the protocol in-process and return the sum of `vectors`."""
    clients = [SecAggClient(i) for i in range(len(vectors))]
    pks = {c.id: c.public_key() for c in clients}  # relayed by the server
    for c in clients:
        c.agree(pks, round_id)
    return aggregate([c.mask(v) for c, v in zip(clients, vectors)])
