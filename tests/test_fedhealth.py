import numpy as np
import pytest

from fedhealth import privacy, secure_agg
from fedhealth.data import load_dataset, make_clients, quality_report
from fedhealth.federation import FLConfig, run_federated


def test_secure_sum_matches_plain_sum():
    rng = np.random.default_rng(1)
    vecs = [rng.normal(size=50) * 10 for _ in range(5)]
    assert np.allclose(secure_agg.secure_sum(vecs, round_id=3), np.sum(vecs, axis=0), atol=1e-5)


def test_secure_sum_handles_large_values():
    vecs = [np.array([5e8, -3e8, 1.0]) for _ in range(4)]
    assert np.allclose(secure_agg.secure_sum(vecs), [2e9, -1.2e9, 4.0], atol=1e-3)


def test_single_masked_vector_hides_input():
    clients = [secure_agg.SecAggClient(i) for i in range(3)]
    pks = {c.id: c.public_key() for c in clients}
    for c in clients:
        c.agree(pks, round_id=0)
    x = np.zeros(1000)
    masked = clients[0].mask(x)
    decoded = secure_agg.decode(masked)
    # an all-zero input should look like large random numbers to the server
    assert np.abs(decoded).mean() > 1e6


def test_masks_change_between_rounds():
    a, b = secure_agg.SecAggClient(0), secure_agg.SecAggClient(1)
    pks = {0: a.public_key(), 1: b.public_key()}
    a.agree(pks, 1)
    m1 = a.mask(np.zeros(10))
    a.agree(pks, 2)
    m2 = a.mask(np.zeros(10))
    assert not np.array_equal(m1, m2)


def test_out_of_range_rejected():
    with pytest.raises(ValueError):
        secure_agg.encode(np.array([1e13]))


def test_clip_bounds_norm():
    u, was = privacy.clip(np.ones(100), 1.0)
    assert was and np.isclose(np.linalg.norm(u), 1.0)
    u2, was2 = privacy.clip(np.full(4, 0.1), 1.0)
    assert not was2 and np.allclose(u2, 0.1)


def test_epsilon_monotonic():
    assert privacy.epsilon(2.0, 10) < privacy.epsilon(1.0, 10)
    assert privacy.epsilon(1.0, 10) < privacy.epsilon(1.0, 50)
    assert privacy.epsilon(0.0, 10) == float("inf")


def test_partition_covers_all_samples_once():
    Xtr, Xte, ytr, yte = load_dataset()
    clients = make_clients(Xtr, ytr, n_clients=5, alpha=0.5)
    total = sum(len(c.y_train) + len(c.y_test) for c in clients)
    assert total == len(ytr)


def test_quality_report_flags_problems():
    Xtr, _, ytr, _ = load_dataset()
    c = make_clients(Xtr, ytr, n_clients=3, alpha=100)[0]
    c.X_train = c.X_train.copy()
    c.X_train[:, 0] = 1.0
    rep = quality_report(c)
    assert rep.constant_features == 1
    assert any("constant" in w for w in rep.warnings)


@pytest.mark.parametrize("secure", [False, True])
def test_federated_training_learns(secure):
    Xtr, Xte, ytr, yte = load_dataset()
    clients = make_clients(Xtr, ytr, n_clients=4, alpha=1.0)
    hist = run_federated(clients, Xte, yte, FLConfig(rounds=8, secure_aggregation=secure))
    assert hist.global_acc[-1] > 0.85


def test_secure_and_plain_training_agree():
    Xtr, Xte, ytr, yte = load_dataset()
    clients = make_clients(Xtr, ytr, n_clients=4, alpha=1.0)
    h1 = run_federated(clients, Xte, yte, FLConfig(rounds=3, secure_aggregation=False))
    h2 = run_federated(clients, Xte, yte, FLConfig(rounds=3, secure_aggregation=True))
    assert np.allclose(h1.global_auc, h2.global_auc, atol=1e-3)
