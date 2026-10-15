import numpy as np
import torch

from sentinelfed.detector import (REJECT, TRUST, VERIFY, EmpiricalScaler, Mahalanobis,
                                  RiskModel, decide)
from sentinelfed.modules import ego_batch
from sentinelfed.train import collate

from test_phase4 import build_toy  # noqa: E402


def test_decision_policy_matches_the_guide():
    p_sup = np.array([0.1, 0.9, 0.9, 0.1])
    d_norm = np.array([0.1, 0.995, 0.1, 0.995])
    labelled = decide(p_sup, d_norm, tau_sup=0.5, tau_unsup=0.99)
    assert labelled.tolist() == [TRUST, REJECT, VERIFY, VERIFY]
    unlabelled = decide(p_sup, d_norm, 0.5, 0.99, labels_available=False)
    assert unlabelled.tolist() == [TRUST, REJECT, TRUST, REJECT]


def test_mahalanobis_matches_definition():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(500, 3)) @ np.diag([1.0, 2.0, 0.5])
    m = Mahalanobis.fit(z, ridge=1e-3)
    q = rng.normal(size=(5, 3))
    cov = np.cov(z, rowvar=False) + 1e-3 * np.eye(3)
    d = q - z.mean(0)
    expected = np.sqrt(np.sum(d @ np.linalg.inv(cov) * d, axis=1))
    assert np.allclose(m(q), expected)


def test_empirical_scaler_is_a_percentile():
    s = EmpiricalScaler(np.arange(100.0))
    assert s(np.array([-1.0, 49.0, 1000.0])).tolist() == [0.0, 0.5, 1.0]


def test_risk_model_is_monotone_probability():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.6 + rng.uniform(0, 0.4, 400), 0, 1)
    d = rng.uniform(0, 1, 400)
    r = RiskModel.fit(p, d, y)
    out = r(np.linspace(0, 1, 11), np.full(11, 0.5))
    assert ((out >= 0) & (out <= 1)).all()
    assert np.all(np.diff(out) >= 0) or np.all(np.diff(out) <= 0)


def test_fast_collate_equals_pyg_batch():
    built, _ = build_toy()
    ego = built["ego"]
    ids = np.array([3, 0, 7, 7, 1])
    fast, ref = collate(ego, ids), ego_batch(ego, ids)
    assert torch.equal(fast.x, ref.x)
    assert torch.equal(fast.edge_index, ref.edge_index)
    assert torch.equal(fast.edge_attr, ref.edge_attr)
    assert torch.equal(fast.batch, ref.batch)
    assert torch.equal(fast.ptr, ref.ptr)
