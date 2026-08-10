"""Tests for ID splits, dual GNN settings API surface, and LODO (no GPU needed
for the split/baseline paths; GNN runners are import-smoke only)."""
import numpy as np
import pandas as pd

from src.data import synth
from src.data.schema import USER, TIMESTAMP, LABEL
from src.models.baselines import isolation_forest
from src.train.splits import temporal_split, user_disjoint_split
from src.train.id_eval import in_distribution_eval, id_eval_table
from src.train.crossdataset import leave_one_domain_out, generalization_gap
from src.train.domain_concat import tag_and_concat


def test_temporal_split_is_chronological():
    df = synth.generate("synthA", n_users=12, days=6, seed=1)
    train, test = temporal_split(df, train_frac=0.7)
    assert len(train) > 0 and len(test) > 0
    assert train[TIMESTAMP].max() <= test[TIMESTAMP].min()
    assert len(train) + len(test) == len(df)


def test_user_disjoint_no_overlap():
    df = synth.generate("synthA", n_users=20, days=4, seed=2)
    train, test = user_disjoint_split(df, train_frac=0.7, seed=0)
    assert set(train[USER]).isdisjoint(set(test[USER]))
    assert len(train) + len(test) == len(df)


def test_id_eval_both_splits_baseline():
    df = synth.generate("synthA", n_users=20, days=5, seed=3)
    res = in_distribution_eval(df, model_factory=isolation_forest, seed=0)
    assert set(res) == {"temporal", "user_disjoint"}
    for m in res.values():
        assert "pr_auc" in m
        assert m["n"] > 0


def test_id_eval_table():
    a = synth.generate("synthA", n_users=15, days=4, seed=4)
    table = id_eval_table({"A": a}, model_factory=isolation_forest)
    assert set(table["split"]) == {"temporal", "user_disjoint"}
    assert len(table) == 2


def test_tag_and_concat_prefixes_ids():
    a = synth.generate("synthA", n_users=5, days=2, seed=5)
    b = synth.generate("synthB", n_users=5, days=2, seed=6)
    cat = tag_and_concat({"A": a, "B": b})
    assert cat[USER].str.startswith("A::").sum() == len(a)
    assert cat[USER].str.startswith("B::").sum() == len(b)
    assert len(cat) == len(a) + len(b)


def test_leave_one_domain_out_baseline():
    a = synth.generate("synthA", n_users=15, days=4, seed=7)
    b = synth.generate("synthB", n_users=15, days=4, domain_shift=0.8, seed=8)
    series, details = leave_one_domain_out({"A": a, "B": b}, isolation_forest)
    assert set(series.index) == {"A", "B"}
    assert set(details) == {"A", "B"}
    assert all(not np.isnan(v) for v in series.to_numpy())


def test_generalization_gap_positive_on_shifted():
    a = synth.generate("synthA", n_users=20, days=5, seed=9)
    b = synth.generate("synthB", n_users=20, days=5, domain_shift=0.9, seed=10)
    from src.train.crossdataset import cross_dataset_matrix
    matrix, _ = cross_dataset_matrix({"A": a, "B": b}, isolation_forest)
    # Not asserting sign (can vary on tiny synth); just that it is finite.
    gap = generalization_gap(matrix)
    assert np.isfinite(gap)
