"""Fast unit tests for the data + eval layer (no torch)."""
import numpy as np

from src.data import synth
from src.data.schema import validate, CANONICAL_COLUMNS
from src.data.features import user_day_features, X_y, FEATURE_COLUMNS
from src.data.graph_builder import summary_stats
from src.train.evaluate import compute_metrics, detection_rate_at_fpr, precision_at_k
from src.train.crossdataset import cross_dataset_matrix, generalization_gap
from src.models.baselines import isolation_forest


def test_synth_schema():
    df = synth.generate("synthA", n_users=10, days=3, seed=1)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert validate(df)
    assert df["label"].isin([0, 1]).all()


def test_features_fixed_width():
    from src.data.features import BASE_FEATURE_COLUMNS
    df = synth.generate("synthA", n_users=10, days=3, seed=1)
    feat = user_day_features(df)
    X, y = X_y(feat)
    assert X.shape[1] == len(BASE_FEATURE_COLUMNS)
    feat_d = user_day_features(df, deviation=True)
    Xd, _ = X_y(feat_d)
    assert Xd.shape[1] == len(FEATURE_COLUMNS)
    assert set(np.unique(y)).issubset({0, 1})


def test_causal_deviation_spike():
    """Synthetic user: flat activity then a spike on the last day -> large +dev."""
    import pandas as pd
    from src.data.schema import (
        TIMESTAMP, USER, SRC_HOST, DST_HOST, ACTION, OBJECT,
        LABEL, INSIDER_TYPE, DATASET, CANONICAL_COLUMNS,
    )
    rows = []
    # 5 quiet days: 2 http events/day; day 6: 40 http events
    for day_i, n_http in enumerate([2, 2, 2, 2, 2, 40]):
        for k in range(n_http):
            rows.append({
                TIMESTAMP: pd.Timestamp("2020-01-01") + pd.Timedelta(days=day_i, minutes=k),
                USER: "U_spike",
                SRC_HOST: "H1", DST_HOST: "H1",
                ACTION: "http", OBJECT: "x",
                LABEL: 0, INSIDER_TYPE: "benign", DATASET: "synth",
            })
    df = pd.DataFrame(rows)[CANONICAL_COLUMNS]
    feat = user_day_features(df, deviation=True).sort_values("day")
    assert feat.iloc[0]["dev_cnt_http"] == 0.0  # cold start
    assert feat.iloc[-1]["dev_cnt_http"] > 3.0  # clear positive spike


def test_graph_stats():
    df = synth.generate("synthB", n_users=8, days=3, seed=2)
    s = summary_stats(df)
    assert s["n_users"] == 8
    assert s["n_edges"] > 0


def test_metrics_ranges():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    m = compute_metrics(y, s)
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert precision_at_k(y, s, k=2) == 1.0
    assert 0.0 <= detection_rate_at_fpr(y, s, 0.5) <= 1.0


def test_cross_dataset_matrix():
    a = synth.generate("synthA", n_users=15, days=4, seed=3)
    b = synth.generate("synthB", n_users=15, days=4, domain_shift=0.8, seed=4)
    matrix, details = cross_dataset_matrix({"A": a, "B": b}, isolation_forest)
    assert matrix.shape == (2, 2)
    assert ("A", "B") in details
    _ = generalization_gap(matrix)   # should not raise
