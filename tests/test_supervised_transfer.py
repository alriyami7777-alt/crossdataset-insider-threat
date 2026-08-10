"""Honest diagonal + lift helpers (fast, synthetic)."""
import numpy as np

from src.data import synth
from src.train.supervised_transfer import (
    cross_dataset_transfer, generalization_gap, _with_lift,
)
from src.train.evaluate import compute_metrics


def test_lift_and_base_rate():
    y = np.array([0, 0, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9])
    m = _with_lift(compute_metrics(y, s), y)
    assert abs(m["base_rate"] - 0.25) < 1e-9
    assert abs(m["lift"] - m["pr_auc"] / 0.25) < 1e-9


def test_honest_diagonal_differs_from_full_leak():
    a = synth.generate("synthA", n_users=25, days=8, seed=1)
    b = synth.generate("synthB", n_users=25, days=8, domain_shift=0.5, seed=2)
    dfs = {"A": a, "B": b}
    pr_t, _, det_t, _ = cross_dataset_transfer(
        dfs, "random_forest", deviation=True, diagonal_protocol="temporal", seed=0
    )
    pr_u, _, det_u, _ = cross_dataset_transfer(
        dfs, "random_forest", deviation=True, diagonal_protocol="user_disjoint", seed=0
    )
    assert pr_t.shape == (2, 2)
    assert det_t[("A", "A")]["diagonal_protocol"] == "temporal"
    assert det_t[("A", "B")]["diagonal_protocol"] == "full_source"
    assert det_u[("A", "A")]["diagonal_protocol"] == "user_disjoint"
    # Off-diagonal should be finite; gap defined
    assert np.isfinite(generalization_gap(pr_t))
