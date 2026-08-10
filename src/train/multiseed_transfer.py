"""
Multi-seed honest cross-dataset transfer confirmation.

Protocol (IDENTICAL for GNN and RF / baselines)
----------------------------------------------
* Features: ``user_day_features(..., deviation=True)`` (counts + causal deviation).
* Scaler: StandardScaler on BASE count columns, fit on TRAIN rows only
  (GNN / neural / distance models). RF is scale-invariant and uses the same
  raw counts+deviation matrix (matches prior STEP D claim numbers).
* Diagonal: held-out temporal OR user-disjoint split (never train=test rows).
* Off-diagonal: fit FULL source, score FULL target (shared across protocols).
* Metrics: PR-AUC and lift = PR-AUC / target_base_rate.
* Uncertainty: mean and 95% bootstrap CI across seeds (data split + model init).

Claim holds for an off-diagonal cell iff mean(GNN − RF) > 0 and the bootstrap
CI excludes 0.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.features import user_day_features, X_y
from ..data.schema import LABEL
from ..models.baselines import isolation_forest, ocsvm
from ..utils.seed import set_seed
from .day_supervised_gnn import DaySupervisedGNN, _with_lift
from .evaluate import compute_metrics
from .splits import temporal_split, user_disjoint_split
from .supervised_transfer import (
    SupervisedAdapter,
    UnsupervisedAdapter,
    _base_rate,
    _split_user_day_frame,
    rf_factory,
)

log = logging.getLogger(__name__)

PROTOCOLS = ("temporal", "user_disjoint")
OCSVM_MAX_TRAIN = 20_000


def seed_bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap CI of the mean across seed replicates."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = rng.choice(vals, size=vals.size, replace=True).mean()
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1.0 - alpha / 2))
    return mean, lo, hi


def paired_delta_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap CI for mean(a − b) with paired seeds."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    aa, bb = aa[mask], bb[mask]
    if aa.size == 0:
        return float("nan"), float("nan"), float("nan")
    delta = aa - bb
    return seed_bootstrap_ci(delta, n_boot=n_boot, alpha=alpha, seed=seed)


def gnn_kw_for_source(source: str, seed: int, use_dann: bool = False) -> dict:
    """Competitive day-supervised settings from STEP D / ablation ladder.

    CERT-scale sources (``cert``, ``cert42``, ``cert52``, ``cert62``, ...) use
    the faster CERT schedule; small domains (e.g. SPEDIA) keep longer patience.
    """
    is_cert = source == "cert" or source.startswith("cert")
    return dict(
        variant="BC",
        loss="bce",
        epochs=60 if is_cert else 80,
        ssl_epochs=1 if is_cert else 2,
        patience=8 if is_cert else 12,
        min_epochs=12 if is_cert else 20,
        val_every=4 if is_cert else 2,
        batch_edges=16384 if is_cert else 8192,
        use_dann=use_dann,
        seed=seed,
    )


def log_protocol_identity(names: List[str], seeds: Sequence[int]):
    log.info("=" * 72)
    log.info("PROTOCOL IDENTITY CONFIRMATION")
    log.info("  domains=%s", names)
    log.info("  features=user_day_features(deviation=True)  # counts + causal dev")
    log.info("  scaler=StandardScaler(base counts) fit on TRAIN only (neural/GNN)")
    log.info("  RF uses same raw counts+deviation (tree scale-invariant)")
    log.info("  diagonal protocols=%s (honest held-out; never full=full)", PROTOCOLS)
    log.info("  off-diagonal=fit FULL source / score FULL target (shared)")
    log.info("  seeds=%s (data split + model init)", list(seeds))
    log.info("  metric=PR-AUC + lift=PR-AUC/target_base_rate")
    log.info("=" * 72)


# ---------------------------------------------------------------------------
# Single-cell runners
# ---------------------------------------------------------------------------
def run_rf_cell(
    train_f: pd.DataFrame, test_f: pd.DataFrame, seed: int,
) -> dict:
    set_seed(seed)
    Xtr, ytr = X_y(train_f)
    Xte, yte = X_y(test_f)
    if len(np.unique(ytr)) < 2:
        # Degenerate split (common on tiny synth); constant score.
        scores = np.zeros(len(yte), dtype=float)
        return _with_lift(compute_metrics(yte, scores), yte)
    adapter = SupervisedAdapter(rf_factory(seed))
    adapter.fit(Xtr, ytr)
    proba = adapter.model.predict_proba(Xte)
    # Handle sklearn single-class edge case after fit
    if proba.shape[1] == 1:
        cls = int(adapter.model.classes_[0])
        scores = np.full(len(yte), float(cls), dtype=float)
    else:
        scores = proba[:, 1]
    return _with_lift(compute_metrics(yte, scores), yte)


def run_if_cell(train_f: pd.DataFrame, test_f: pd.DataFrame, seed: int) -> dict:
    set_seed(seed)
    Xtr, ytr = X_y(train_f)
    Xte, yte = X_y(test_f)
    # fit on (mostly) negatives
    Xfit = Xtr[ytr == 0] if (ytr == 0).any() else Xtr
    det = isolation_forest(seed=seed)
    det.fit(Xfit)
    return _with_lift(compute_metrics(yte, det.anomaly_scores(Xte)), yte)


def run_ocsvm_cell(
    train_f: pd.DataFrame, test_f: pd.DataFrame, seed: int,
    max_train: int = OCSVM_MAX_TRAIN,
) -> dict:
    set_seed(seed)
    Xtr, ytr = X_y(train_f)
    Xte, yte = X_y(test_f)
    Xfit = Xtr[ytr == 0] if (ytr == 0).any() else Xtr
    if len(Xfit) > max_train:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xfit), size=max_train, replace=False)
        log.info(
            "OCSVM subsample train %d -> %d (seed=%d)",
            len(Xfit), max_train, seed,
        )
        Xfit = Xfit[idx]
    det = ocsvm()
    det.fit(Xfit)
    return _with_lift(compute_metrics(yte, det.anomaly_scores(Xte)), yte)


def run_lstm_ae_cell(train_f: pd.DataFrame, test_f: pd.DataFrame, seed: int) -> dict:
    from ..models.lstm_ae import LSTMAEDetector
    set_seed(seed)
    det = LSTMAEDetector(epochs=15, seed=seed)
    det.fit(train_f)
    scores = det.anomaly_scores(test_f)
    yte = test_f[LABEL].to_numpy()
    return _with_lift(compute_metrics(yte, scores), yte)


def run_transformer_cell(train_f: pd.DataFrame, test_f: pd.DataFrame, seed: int) -> dict:
    from ..models.transformer_enc import TransformerFeatureDetector
    set_seed(seed)
    det = TransformerFeatureDetector(epochs=20, seed=seed)
    det.fit(train_f)
    scores = det.anomaly_scores(test_f)
    yte = test_f[LABEL].to_numpy()
    return _with_lift(compute_metrics(yte, scores), yte)


def run_static_gcn_cell(train_f: pd.DataFrame, test_f: pd.DataFrame, seed: int) -> dict:
    from ..models.static_gnn import StaticGCNDetector
    set_seed(seed)
    det = StaticGCNDetector(epochs=35, seed=seed, max_train_nodes=8000)
    det.fit(train_f)
    scores = det.anomaly_scores(test_f)
    yte = test_f[LABEL].to_numpy()
    return _with_lift(compute_metrics(yte, scores), yte)


def run_athitd_cell(train_f: pd.DataFrame, test_f: pd.DataFrame, seed: int) -> dict:
    from ..models.static_gnn import ATHITDDetector
    set_seed(seed)
    det = ATHITDDetector(epochs=25, seed=seed)
    det.fit(train_f)
    scores = det.anomaly_scores(test_f)
    yte = test_f[LABEL].to_numpy()
    return _with_lift(compute_metrics(yte, scores), yte)


def run_day_gnn_cell(
    dfs: dict,
    full_feats: dict,
    source: str,
    target: str,
    seed: int,
    *,
    protocol: Optional[str] = None,
    use_dann: bool = False,
    train_frac: float = 0.7,
) -> dict:
    """Day-supervised GNN zero-shot (or DANN-UDA when use_dann and off-diag)."""
    set_seed(seed)
    adapt = bool(use_dann and source != target)
    det = DaySupervisedGNN(**gnn_kw_for_source(source, seed, use_dann=adapt))
    if source == target:
        assert protocol is not None
        if protocol == "temporal":
            tr, te = temporal_split(dfs[source], train_frac=train_frac)
        else:
            tr, te = user_disjoint_split(
                dfs[source], train_frac=train_frac, seed=seed,
            )
        det.set_full_features(full_feats[source])
        det.fit(tr)
        agg = det.score_user_day(te)
    else:
        det.set_full_features(full_feats[source])
        det.fit(dfs[source], df_tgt=(dfs[target] if adapt else None))
        det.set_full_features(full_feats[target])
        agg = det.score_user_day(dfs[target])
    y = agg["label"].to_numpy()
    m = _with_lift(compute_metrics(y, agg["score"].to_numpy()), y)
    m["best_epoch"] = det.best_epoch
    m["best_val_pr"] = det.best_val_pr
    return m


FEATURE_MODEL_RUNNERS: Dict[str, Callable] = {
    "random_forest": run_rf_cell,
    "isolation_forest": run_if_cell,
    "ocsvm": run_ocsvm_cell,
    "lstm_ae": run_lstm_ae_cell,
    "transformer": run_transformer_cell,
    "static_gcn": run_static_gcn_cell,
    "athitd": run_athitd_cell,
}


def _train_test_frames(
    dfs: dict,
    full_feats: dict,
    source: str,
    target: str,
    *,
    protocol: Optional[str],
    train_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (train_feat, test_feat, diag_tag) under the honest protocol."""
    if source == target:
        assert protocol is not None
        # Feature-frame split for tabular/neural baselines (matches RF honest diag)
        train_f, test_f = _split_user_day_frame(
            full_feats[source], protocol, train_frac=train_frac, seed=seed,
        )
        return train_f, test_f, protocol
    return full_feats[source], full_feats[target], "full_source"


def _cell_key(model: str, seed: int, source: str, target: str, diag: str):
    return (str(model), int(seed), str(source), str(target), str(diag))


def run_feature_model_matrix_seed(
    model_name: str,
    dfs: dict,
    full_feats: dict,
    seed: int,
    *,
    train_frac: float = 0.7,
    skip_cells: Optional[set] = None,
) -> List[dict]:
    """One seed: off-diag once + both diagonal protocols for a feature model."""
    runner = FEATURE_MODEL_RUNNERS[model_name]
    names = list(dfs.keys())
    rows: List[dict] = []
    skip_cells = skip_cells or set()
    # Off-diagonal (protocol-independent)
    for s in names:
        for t in names:
            if s == t:
                continue
            if _cell_key(model_name, seed, s, t, "full_source") in skip_cells:
                log.info("skip %s seed=%d %s->%s (resume)", model_name, seed, s, t)
                continue
            tr, te, tag = _train_test_frames(
                dfs, full_feats, s, t, protocol=None, train_frac=train_frac, seed=seed,
            )
            m = runner(tr, te, seed)
            m.update({
                "source": s, "target": t, "model": model_name, "seed": seed,
                "diagonal_protocol": tag, "cell_kind": "off_diagonal",
            })
            rows.append(m)
            log.info(
                "%s seed=%d %s->%s: pr_auc=%.4f lift=%.2f",
                model_name, seed, s, t, m["pr_auc"], m["lift"],
            )
    # Diagonals
    for protocol in PROTOCOLS:
        for s in names:
            if _cell_key(model_name, seed, s, s, protocol) in skip_cells:
                log.info(
                    "skip %s seed=%d %s->%s [%s] (resume)",
                    model_name, seed, s, s, protocol,
                )
                continue
            tr, te, tag = _train_test_frames(
                dfs, full_feats, s, s,
                protocol=protocol, train_frac=train_frac, seed=seed,
            )
            m = runner(tr, te, seed)
            m.update({
                "source": s, "target": s, "model": model_name, "seed": seed,
                "diagonal_protocol": tag, "cell_kind": "diagonal",
            })
            rows.append(m)
            log.info(
                "%s seed=%d %s->%s [%s]: pr_auc=%.4f lift=%.2f",
                model_name, seed, s, s, protocol, m["pr_auc"], m["lift"],
            )
    return rows


def run_gnn_matrix_seed(
    dfs: dict,
    full_feats: dict,
    seed: int,
    *,
    use_dann: bool = False,
    train_frac: float = 0.7,
    skip_cells: Optional[set] = None,
    on_cell=None,
) -> List[dict]:
    """One seed for day-supervised GNN (zero-shot or DANN). Off-diag once.

    ``on_cell``: optional callback(row_dict) invoked after each completed cell
    so callers can checkpoint mid-seed (survives connection drops).
    """
    names = list(dfs.keys())
    setting = "dann_uda" if use_dann else "zero_shot"
    model_name = f"day_gnn_{setting}"
    rows: List[dict] = []
    skip_cells = skip_cells or set()

    def _emit(m: dict):
        rows.append(m)
        if on_cell is not None:
            on_cell(m)

    for s in names:
        for t in names:
            if s == t:
                continue
            if _cell_key(model_name, seed, s, t, "full_source") in skip_cells:
                log.info("skip %s seed=%d %s->%s (resume)", model_name, seed, s, t)
                continue
            m = run_day_gnn_cell(
                dfs, full_feats, s, t, seed, use_dann=use_dann, train_frac=train_frac,
            )
            m.update({
                "source": s, "target": t, "model": model_name, "seed": seed,
                "diagonal_protocol": "full_source", "cell_kind": "off_diagonal",
                "setting": setting,
            })
            _emit(m)
            log.info(
                "%s seed=%d %s->%s: pr_auc=%.4f lift=%.2f",
                model_name, seed, s, t, m["pr_auc"], m["lift"],
            )
    for protocol in PROTOCOLS:
        for s in names:
            if _cell_key(model_name, seed, s, s, protocol) in skip_cells:
                log.info(
                    "skip %s seed=%d %s->%s [%s] (resume)",
                    model_name, seed, s, s, protocol,
                )
                continue
            # DANN not meaningful on diagonal; still run supervised zero-shot head
            m = run_day_gnn_cell(
                dfs, full_feats, s, s, seed,
                protocol=protocol, use_dann=False, train_frac=train_frac,
            )
            m.update({
                "source": s, "target": s, "model": model_name, "seed": seed,
                "diagonal_protocol": protocol, "cell_kind": "diagonal",
                "setting": setting,
            })
            _emit(m)
            log.info(
                "%s seed=%d %s->%s [%s]: pr_auc=%.4f lift=%.2f",
                model_name, seed, s, s, protocol, m["pr_auc"], m["lift"],
            )
    return rows


def aggregate_seed_rows(
    details: pd.DataFrame,
    *,
    n_boot: int = 5000,
) -> pd.DataFrame:
    """Aggregate per-seed detail rows → mean PR-AUC/lift with bootstrap CIs."""
    keys = ["model", "source", "target", "diagonal_protocol", "cell_kind"]
    rows = []
    for key, g in details.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        pr_vals = g["pr_auc"].to_numpy(dtype=float)
        lift_vals = g["lift"].to_numpy(dtype=float)
        br_vals = g["base_rate"].to_numpy(dtype=float)
        pr_m, pr_lo, pr_hi = seed_bootstrap_ci(pr_vals, n_boot=n_boot, seed=0)
        lf_m, lf_lo, lf_hi = seed_bootstrap_ci(lift_vals, n_boot=n_boot, seed=1)
        rec.update({
            "n_seeds": int(np.isfinite(pr_vals).sum()),
            "pr_auc_mean": pr_m,
            "pr_auc_lo": pr_lo,
            "pr_auc_hi": pr_hi,
            "lift_mean": lf_m,
            "lift_lo": lf_lo,
            "lift_hi": lf_hi,
            "base_rate": float(np.nanmean(br_vals)),
            "pr_auc_seeds": ";".join(f"{v:.6f}" for v in pr_vals),
            "lift_seeds": ";".join(f"{v:.6f}" for v in lift_vals),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def gnn_minus_rf_table(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    *,
    gnn_model: str = "day_gnn_zero_shot",
    rf_model: str = "random_forest",
    n_boot: int = 5000,
) -> pd.DataFrame:
    """Per-cell paired GNN−RF delta with bootstrap CI across seeds."""
    rows = []
    keys = ["source", "target", "diagonal_protocol", "cell_kind"]
    gnn_d = details[details["model"] == gnn_model]
    rf_d = details[details["model"] == rf_model]
    for key, g_g in gnn_d.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        g_r = rf_d[
            (rf_d["source"] == rec["source"])
            & (rf_d["target"] == rec["target"])
            & (rf_d["diagonal_protocol"] == rec["diagonal_protocol"])
            & (rf_d["cell_kind"] == rec["cell_kind"])
        ]
        # align by seed
        merged = g_g[["seed", "pr_auc", "lift"]].merge(
            g_r[["seed", "pr_auc", "lift"]],
            on="seed",
            suffixes=("_gnn", "_rf"),
        )
        if merged.empty:
            continue
        d_pr, lo, hi = paired_delta_ci(
            merged["pr_auc_gnn"], merged["pr_auc_rf"], n_boot=n_boot, seed=2,
        )
        d_lf, llo, lhi = paired_delta_ci(
            merged["lift_gnn"], merged["lift_rf"], n_boot=n_boot, seed=3,
        )
        holds = bool(d_pr > 0 and lo > 0) if rec["cell_kind"] == "off_diagonal" else None
        rec.update({
            "n_seeds": len(merged),
            "gnn_pr_mean": float(merged["pr_auc_gnn"].mean()),
            "rf_pr_mean": float(merged["pr_auc_rf"].mean()),
            "delta_pr_mean": d_pr,
            "delta_pr_lo": lo,
            "delta_pr_hi": hi,
            "delta_lift_mean": d_lf,
            "delta_lift_lo": llo,
            "delta_lift_hi": lhi,
            "claim_holds": holds,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def target_base_rates(full_feats: dict) -> pd.DataFrame:
    rows = []
    for n, feat in full_feats.items():
        y = feat[LABEL].to_numpy()
        rows.append({
            "target": n,
            "base_rate": _base_rate(y),
            "n": int(len(y)),
            "n_pos": int(y.sum()),
        })
    return pd.DataFrame(rows)


def run_multiseed_confirm(
    dfs: dict,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    dann_seeds: Sequence[int] = (0,),
    feature_models: Optional[List[str]] = None,
    run_gnn: bool = True,
    run_dann: bool = True,
    n_boot: int = 5000,
    parallel_feature_seeds: int = 1,
    checkpoint_path: Optional[str] = None,
    resume_details_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Full multi-seed confirmation suite. Returns dict of DataFrames.

    ``checkpoint_path``: if set, rewrite a details CSV after each seed finishes
    (survives connection drops on long multi-domain runs).
    ``resume_details_path``: if set, load prior detail rows and skip (model, seed)
    pairs already present — useful after a mid-GNN crash.
    """
    names = list(dfs.keys())
    log_protocol_identity(names, seeds)
    full_feats = {n: user_day_features(dfs[n], deviation=True) for n in names}
    tstats = target_base_rates(full_feats)
    for _, r in tstats.iterrows():
        log.info(
            "target %s: base_rate=%.6f n=%d n_pos=%d",
            r["target"], r["base_rate"], r["n"], r["n_pos"],
        )

    feature_models = feature_models or [
        "random_forest", "isolation_forest", "ocsvm",
        "lstm_ae", "transformer", "static_gcn", "athitd",
    ]
    detail_rows: List[dict] = []
    skip_cells: set = set()
    if resume_details_path:
        prev = pd.read_csv(resume_details_path)
        detail_rows.extend(prev.to_dict(orient="records"))
        skip_cells = {
            _cell_key(
                r["model"], r["seed"], r["source"], r["target"], r["diagonal_protocol"],
            )
            for r in detail_rows
        }
        log.info(
            "Resumed %d detail rows from %s (%d cells done)",
            len(detail_rows), resume_details_path, len(skip_cells),
        )

    def _checkpoint():
        if not checkpoint_path:
            return
        pd.DataFrame(detail_rows).to_csv(checkpoint_path, index=False)
        log.info("Checkpointed %d detail rows -> %s", len(detail_rows), checkpoint_path)

    def _one_feat(model_name: str, seed: int):
        return run_feature_model_matrix_seed(
            model_name, dfs, full_feats, seed, skip_cells=skip_cells,
        )

    for model_name in feature_models:
        log.info("==== feature model %s | seeds=%s ====", model_name, list(seeds))
        if parallel_feature_seeds > 1 and model_name in (
            "random_forest", "isolation_forest", "ocsvm",
        ):
            with ThreadPoolExecutor(max_workers=parallel_feature_seeds) as ex:
                futs = {ex.submit(_one_feat, model_name, s): s for s in seeds}
                for fut in as_completed(futs):
                    new_rows = fut.result()
                    detail_rows.extend(new_rows)
                    for r in new_rows:
                        skip_cells.add(_cell_key(
                            r["model"], r["seed"], r["source"], r["target"],
                            r["diagonal_protocol"],
                        ))
                    _checkpoint()
        else:
            for seed in seeds:
                new_rows = _one_feat(model_name, seed)
                detail_rows.extend(new_rows)
                for r in new_rows:
                    skip_cells.add(_cell_key(
                        r["model"], r["seed"], r["source"], r["target"],
                        r["diagonal_protocol"],
                    ))
                _checkpoint()

    def _on_cell(row: dict):
        detail_rows.append(row)
        skip_cells.add(_cell_key(
            row["model"], row["seed"], row["source"], row["target"],
            row["diagonal_protocol"],
        ))
        _checkpoint()

    if run_gnn:
        log.info("==== day_gnn_zero_shot | seeds=%s ====", list(seeds))
        for seed in seeds:
            # Per-cell checkpoint via on_cell (do not also extend detail_rows here)
            run_gnn_matrix_seed(
                dfs, full_feats, seed, use_dann=False, skip_cells=skip_cells,
                on_cell=_on_cell,
            )

    if run_dann:
        log.info("==== day_gnn_dann_uda | seeds=%s ====", list(dann_seeds))
        for seed in dann_seeds:
            run_gnn_matrix_seed(
                dfs, full_feats, seed, use_dann=True, skip_cells=skip_cells,
                on_cell=_on_cell,
            )

    details = pd.DataFrame(detail_rows)
    summary = aggregate_seed_rows(details, n_boot=n_boot)
    deltas = gnn_minus_rf_table(summary, details, n_boot=n_boot)

    # Verdict table for off-diagonals under each "primary" reporting protocol.
    # Off-diag cells are protocol-independent; we attach both protocol labels
    # for readability when printing claim status beside ID diagonals.
    verdict_rows = []
    off = deltas[deltas["cell_kind"] == "off_diagonal"]
    for protocol in PROTOCOLS:
        for _, r in off.iterrows():
            verdict_rows.append({
                "report_protocol": protocol,
                "source": r["source"],
                "target": r["target"],
                "gnn_pr_mean": r["gnn_pr_mean"],
                "rf_pr_mean": r["rf_pr_mean"],
                "delta_pr_mean": r["delta_pr_mean"],
                "delta_pr_lo": r["delta_pr_lo"],
                "delta_pr_hi": r["delta_pr_hi"],
                "claim_holds": r["claim_holds"],
                "n_seeds": r["n_seeds"],
            })
    verdict = pd.DataFrame(verdict_rows)

    return {
        "details": details,
        "summary": summary,
        "deltas": deltas,
        "verdict": verdict,
        "target_stats": tstats,
    }


def format_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "nan"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def print_confirmation_report(results: Dict[str, pd.DataFrame]):
    tstats = results["target_stats"]
    summary = results["summary"]
    deltas = results["deltas"]
    print("\n======== TARGET BASE RATES ========")
    print(tstats.to_string(index=False))

    print("\n======== MULTI-SEED SUMMARY (PR-AUC mean [95% CI], lift) ========")
    for protocol in list(PROTOCOLS) + ["full_source"]:
        sub = summary[summary["diagonal_protocol"] == protocol]
        if sub.empty:
            continue
        print(f"\n--- protocol/tag = {protocol} ---")
        for model, g in sub.groupby("model"):
            print(f"\n[{model}]")
            for _, r in g.sort_values(["source", "target"]).iterrows():
                print(
                    f"  {r['source']}->{r['target']}: "
                    f"PR={format_ci(r['pr_auc_mean'], r['pr_auc_lo'], r['pr_auc_hi'])}  "
                    f"lift={format_ci(r['lift_mean'], r['lift_lo'], r['lift_hi'], 2)}  "
                    f"p={r['base_rate']:.5f}  n_seeds={r['n_seeds']}"
                )

    print("\n======== GNN − RF DELTAS (paired seeds) ========")
    for _, r in deltas.sort_values(["cell_kind", "source", "target"]).iterrows():
        hold = r["claim_holds"]
        hold_s = (
            "HOLDS" if hold is True else ("FAILS" if hold is False else "n/a (diag)")
        )
        print(
            f"  {r['source']}->{r['target']} [{r['diagonal_protocol']}]: "
            f"ΔPR={format_ci(r['delta_pr_mean'], r['delta_pr_lo'], r['delta_pr_hi'])}  "
            f"GNN={r['gnn_pr_mean']:.3f} RF={r['rf_pr_mean']:.3f}  "
            f"claim={hold_s}"
        )

    print("\n======== OFF-DIAGONAL CLAIM VERDICT ========")
    off = deltas[deltas["cell_kind"] == "off_diagonal"]
    for _, r in off.iterrows():
        status = "HOLDS" if r["claim_holds"] else "FAILS"
        print(
            f"  {r['source']}->{r['target']}: {status}  "
            f"(ΔPR={r['delta_pr_mean']:.3f} CI=[{r['delta_pr_lo']:.3f}, {r['delta_pr_hi']:.3f}])"
        )
    # Highlight spedia->cert lift
    sp = summary[
        (summary["source"] == "spedia")
        & (summary["target"] == "cert")
        & (summary["cell_kind"] == "off_diagonal")
    ]
    if not sp.empty:
        print("\n======== HIGHLIGHT: spedia -> cert LIFT ========")
        for _, r in sp.iterrows():
            print(
                f"  [{r['model']}] lift={format_ci(r['lift_mean'], r['lift_lo'], r['lift_hi'], 2)}  "
                f"PR={format_ci(r['pr_auc_mean'], r['pr_auc_lo'], r['pr_auc_hi'])}  "
                f"base_rate={r['base_rate']:.6f}"
            )
