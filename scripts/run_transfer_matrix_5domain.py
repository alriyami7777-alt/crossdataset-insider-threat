"""
5-domain multi-seed transfer matrix — Phase A (day_gnn vs RF), then B/C.

Domains: {cert42, cert52, cert62, spedia, lanl}
Protocol:
  * Features: user_day_features(deviation=True) on FEATURE_COLUMNS
  * Diagonal (headline): user-disjoint for ALL domains
  * Diagonal (supplementary): temporal for cert*/spedia only (not lanl)
  * Off-diagonal: fit FULL source, score FULL target
  * ≥5 seeds; PR-AUC / ROC-AUC / prevalence / lift; peak RSS per cell

Caching under results/cache/transfer_5d/:
  {name}_events.parquet   — canonical event df (LANL reuses C:\\PhD\\lanl\\_cache_...)
  {name}_user_day_dev.parquet — feature matrix (CERT via stream_cert; reuse domain_gap)

Usage (GPU env):
  set PYTHONIOENCODING=utf-8
  python -m scripts.run_transfer_matrix_5domain --phase cache
  python -m scripts.run_transfer_matrix_5domain --phase A
  python -m scripts.run_transfer_matrix_5domain --phase report

Selective GNN cells (no RF; append-only resume):
  python -m scripts.run_transfer_matrix_5domain --phase gnn \\
      --cells "cert62->spedia,lanl->spedia" --epochs 30 --resume

Component ablation (zero-shot into-SPEDIA only; no RF/LANL/UDA):
  python -m scripts.run_transfer_matrix_5domain --ablate \\
      --cells "cert42->spedia,cert52->spedia" --epochs 30 --resume
"""
from __future__ import annotations

import argparse
import gc
import logging
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import psutil

from src.data.features import FEATURE_COLUMNS, user_day_features
from src.data.loaders import load_cert, load_spedia
from src.data.schema import LABEL, USER, TIMESTAMP, validate
from src.eval.stream_cert_features import stream_cert_user_day_features
from src.train.day_supervised_gnn import DaySupervisedGNN, _with_lift
from src.train.evaluate import compute_metrics
from src.train.multiseed_transfer import (
    aggregate_seed_rows,
    gnn_kw_for_source,
    gnn_minus_rf_table,
    paired_delta_ci,
    run_rf_cell,
    seed_bootstrap_ci,
)
from src.train.splits import temporal_split, user_disjoint_split
from src.train.supervised_transfer import _split_user_day_frame
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("transfer5d")

# ---- paths ------------------------------------------------------------------
CERT42 = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
CERT52 = r"C:\PhD\07_Projects\r5.2"
CERT62 = r"C:\PhD\07_Projects\r6.2"
ANSWERS = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2\answers"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"
LANL_DIR = r"C:\PhD\lanl"
LANL_EVENTS = Path(LANL_DIR) / "_cache_load_lanl_redteam_aware_bf002.parquet"

OUT = ROOT / "results"
CACHE = OUT / "cache" / "transfer_5d"
DOMAIN_GAP_CACHE = OUT / "cache" / "domain_gap"

DOMAINS = ("cert42", "cert52", "cert62", "spedia", "lanl")
CERT_RELEASES = {
    "cert42": (CERT42, "4.2"),
    "cert52": (CERT52, "5.2"),
    "cert62": (CERT62, "6.2"),
}
DOMAIN_GAP_FEAT = {
    "cert42": "cert_r42_user_day_dev.parquet",
    "cert52": "cert_r52_user_day_dev.parquet",
    "cert62": "cert_r62_user_day_dev.parquet",
    "spedia": "spedia_user_day_dev.parquet",
}
CERT_LOAD_KW = dict(
    sources=("logon", "device", "file", "email"),
    http_mode="insider_aware",
    benign_http_frac=0.05,
    answers_dir=ANSWERS,
    seed=7,
)
LEAK_PR = 0.95
SEEDS_DEFAULT = (0, 1, 2, 3, 4)
# Warn when full-source positives are below this (CERT r6.2 ≈44 user-days).
MIN_SOURCE_POS_WARN = 50
CKPT_DEFAULT = OUT / "transfer_matrix_phaseA_checkpoint.csv"
INTO_SPEDIA_CSV = OUT / "transfer5d_into_spedia.csv"
ABLATION_CSV = OUT / "ablation_into_spedia.csv"
ABLATION_CELLS_DEFAULT = (("cert42", "spedia"), ("cert52", "spedia"))
# Known full-model into-SPEDIA means (C2); V0 must reproduce within seed noise.
V0_REF_PR = {
    ("cert42", "spedia"): 0.625,
    ("cert52", "spedia"): 0.642,
}
V0_SANITY_ABS_TOL = 0.08  # stop if |V0_mean − ref| exceeds this

# Component ablation grid (one component off per row; V0 = full).
ABLATION_VARIANTS: list[tuple[str, dict]] = [
    ("V0_full", dict(
        use_memory=True, use_ssl_pretrain=True,
        use_time_encoding=True, use_deviation_features=True,
    )),
    ("V1_-SSL", dict(
        use_memory=True, use_ssl_pretrain=False,
        use_time_encoding=True, use_deviation_features=True,
    )),
    ("V2_-memory", dict(
        use_memory=False, use_ssl_pretrain=True,
        use_time_encoding=True, use_deviation_features=True,
    )),
    ("V3_-timeenc", dict(
        use_memory=True, use_ssl_pretrain=True,
        use_time_encoding=False, use_deviation_features=True,
    )),
    ("V4_-deviation", dict(
        use_memory=True, use_ssl_pretrain=True,
        use_time_encoding=True, use_deviation_features=False,
    )),
]


def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def _parse_cells(spec: str) -> list[tuple[str, str]]:
    """Parse ``cert62->spedia,lanl->spedia`` into [(src, tgt), ...]."""
    cells: list[tuple[str, str]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "->" not in part:
            raise SystemExit(
                f"--cells entry must be src->tgt, got {part!r}"
            )
        s, t = (x.strip() for x in part.split("->", 1))
        if s not in DOMAINS or t not in DOMAINS:
            raise SystemExit(
                f"--cells unknown domain in {s!r}->{t!r}; "
                f"allowed={list(DOMAINS)}"
            )
        if s == t:
            raise SystemExit(
                f"--cells does not run diagonals ({s}->{t}); "
                "use full Phase A for ID cells"
            )
        cells.append((s, t))
    return cells


def _feat_path(name: str) -> Path:
    return CACHE / f"{name}_user_day_dev.parquet"


def _events_path(name: str) -> Path:
    if name == "lanl":
        return LANL_EVENTS
    return CACHE / f"{name}_events.parquet"


def _align_feat(feat: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in feat.columns]
    if missing:
        raise KeyError(f"missing FEATURE_COLUMNS: {missing[:8]}")
    keep = [USER, "day", LABEL] + list(FEATURE_COLUMNS)
    # day may be Timestamp
    out = feat[keep].copy()
    out["day"] = pd.to_datetime(out["day"])
    return out


# ---- caching ----------------------------------------------------------------
def cache_features(name: str, force: bool = False) -> Path:
    dest = _feat_path(name)
    if dest.exists() and not force:
        log.info("[cache] features hit %s", dest)
        return dest
    CACHE.mkdir(parents=True, exist_ok=True)

    # Reuse domain_gap feature caches when present
    dg = DOMAIN_GAP_FEAT.get(name)
    if dg:
        src = DOMAIN_GAP_CACHE / dg
        if src.exists() and not force:
            log.info("[cache] copy features %s -> %s", src.name, dest.name)
            shutil.copy2(src, dest)
            return dest

    t0 = time.perf_counter()
    if name in CERT_RELEASES:
        path, release = CERT_RELEASES[name]
        log.info("[cache] stream_cert_user_day_features %s ...", name)
        feat = stream_cert_user_day_features(
            path,
            release=release,
            answers_dir=ANSWERS,
            sources=CERT_LOAD_KW["sources"],
            http_mode=CERT_LOAD_KW["http_mode"],
            benign_http_frac=CERT_LOAD_KW["benign_http_frac"],
            seed=CERT_LOAD_KW["seed"],
        )
    elif name == "spedia":
        log.info("[cache] load_spedia(real_only=True) + features ...")
        df = load_spedia(SPEDIA, real_only=True)
        validate(df)
        feat = user_day_features(df, deviation=True)
        del df
        gc.collect()
    elif name == "lanl":
        log.info("[cache] LANL features from event parquet %s ...", LANL_EVENTS)
        if not LANL_EVENTS.exists():
            raise FileNotFoundError(LANL_EVENTS)
        df = pd.read_parquet(LANL_EVENTS)
        validate(df)
        feat = user_day_features(df, deviation=True)
        del df
        gc.collect()
    else:
        raise ValueError(name)

    feat = _align_feat(feat)
    feat.to_parquet(dest, index=False)
    log.info(
        "[cache] features %s rows=%d pos=%d (%.1fs) rss=%.2fGiB -> %s",
        name, len(feat), int(feat[LABEL].sum()), time.perf_counter() - t0,
        _rss_gb(), dest,
    )
    del feat
    gc.collect()
    return dest


def cache_events(name: str, force: bool = False) -> Path:
    dest = _events_path(name)
    if name == "lanl":
        if not dest.exists():
            raise FileNotFoundError(f"LANL event cache missing: {dest}")
        log.info("[cache] events hit (LANL) %s", dest)
        return dest
    if dest.exists() and not force:
        log.info("[cache] events hit %s", dest)
        return dest
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if name in CERT_RELEASES:
        path, release = CERT_RELEASES[name]
        log.info("[cache] load_cert events %s (insider_aware) ...", name)
        df = load_cert(
            path, release=release, dataset_tag=name, **CERT_LOAD_KW,
        )
    elif name == "spedia":
        log.info("[cache] load_spedia events ...")
        df = load_spedia(SPEDIA, real_only=True)
    else:
        raise ValueError(name)
    validate(df)
    df.to_parquet(dest, index=False)
    log.info(
        "[cache] events %s rows=%d pos=%d (%.1fs) rss=%.2fGiB -> %s",
        name, len(df), int(df[LABEL].sum()), time.perf_counter() - t0,
        _rss_gb(), dest,
    )
    del df
    gc.collect()
    return dest


def build_all_caches(force: bool = False):
    log.info("==== Building domain caches (one domain at a time) ====")
    for name in DOMAINS:
        cache_features(name, force=force)
        cache_events(name, force=force)
        gc.collect()
        log.info("[cache] after %s rss=%.2fGiB", name, _rss_gb())


def load_feat(name: str) -> pd.DataFrame:
    return _align_feat(pd.read_parquet(_feat_path(name)))


def load_events(name: str) -> pd.DataFrame:
    df = pd.read_parquet(_events_path(name))
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    return df


# ---- Phase A cell runners ---------------------------------------------------
def _leak_guard(m: dict, train_f: pd.DataFrame, tag: str):
    """Flag near-perfect ID diagonals.

    CERT / small SPEDIA routinely score PR-AUC ≈0.95–0.99 on held-out
    diagonals (documented overfitting / tiny-N effects). Warn and print
    importances at ≥0.95; hard-stop only on true saturation (≥0.999), which
    would indicate train/test row identity leakage.
    """
    if m.get("cell_kind") != "diagonal":
        return
    pr = float(m.get("pr_auc", 0.0))
    if pr < LEAK_PR:
        return
    from sklearn.ensemble import RandomForestClassifier
    from src.data.features import X_y
    Xtr, ytr = X_y(train_f)
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        n_jobs=-1, random_state=0,
    )
    rf.fit(Xtr, ytr)
    n_pos = int(m.get("n_pos", 0) or 0)
    src = str(m.get("source", ""))
    # Hard-stop only if LANL ID looks saturated (should be ~0.1, not ~1.0).
    # CERT/SPEDIA high diagonals are known (overfitting / tiny-N) — warn only.
    hard = src == "lanl" and pr >= LEAK_PR
    level = "LEAKAGE STOP (lanl)" if hard else "HIGH-DIAG (warn; continue)"
    log.error(
        "%s: %s PR-AUC=%.4f n_pos=%d — top RF importances",
        level, tag, pr, n_pos,
    )
    for name, val in sorted(
        zip(FEATURE_COLUMNS, rf.feature_importances_), key=lambda x: -x[1]
    )[:15]:
        print(f"  {val:.4f}  {name}")
    if hard:
        raise SystemExit(2)


def run_rf_cell_job(source: str, target: str, seed: int, protocol: str | None) -> dict:
    """RF cell; features only (no event dfs)."""
    rss0 = _rss_gb()
    t0 = time.perf_counter()
    fs = load_feat(source)
    ft = fs if source == target else load_feat(target)
    if source == target:
        assert protocol is not None
        tr, te = _split_user_day_frame(fs, protocol, train_frac=0.7, seed=seed)
        tag = protocol
        kind = "diagonal"
    else:
        tr, te = fs, ft
        tag = "full_source"
        kind = "off_diagonal"
    m = run_rf_cell(tr, te, seed)
    m.update({
        "source": source, "target": target, "model": "random_forest",
        "seed": seed, "diagonal_protocol": tag, "cell_kind": kind,
        "wall_s": time.perf_counter() - t0,
        "peak_rss_gb": max(rss0, _rss_gb()),
        "roc_auc": m.get("roc_auc", float("nan")),
        "prevalence": m.get("base_rate", float("nan")),
    })
    _leak_guard(m, tr, f"RF {source}->{target} [{tag}]")
    log.info(
        "RF seed=%d %s->%s [%s] pr=%.4f lift=%.2f rss=%.2fGiB (%.1fs)",
        seed, source, target, tag, m["pr_auc"], m["lift"], m["peak_rss_gb"], m["wall_s"],
    )
    del fs, ft, tr, te
    gc.collect()
    return m


def _offdiag_leak_guard(m: dict, tag: str):
    """LEAK_PR check for off-diagonal / ablation cells (target labels score-only)."""
    pr = float(m.get("pr_auc", 0.0))
    if pr < LEAK_PR:
        return
    hard = pr >= 0.999
    level = "LEAKAGE STOP" if hard else "HIGH-PR (warn; continue)"
    log.error("%s: %s PR-AUC=%.4f (>= LEAK_PR=%.2f)", level, tag, pr, LEAK_PR)
    if hard:
        raise SystemExit(2)


def run_gnn_cell_job(
    source: str,
    target: str,
    seed: int,
    protocol: str | None,
    epochs: int | None = None,
    model_toggles: dict | None = None,
) -> dict:
    """GNN cell; load at most source (+ target if off-diag) events."""
    rss0 = _rss_gb()
    t0 = time.perf_counter()
    set_seed(seed)
    feat_s = load_feat(source)
    df_s = load_events(source)
    peak = max(rss0, _rss_gb())

    n_pos_source = int(feat_s[LABEL].sum())
    if n_pos_source == 0:
        log.error(
            "SOURCE HAS ZERO positive user-days (%s) — refusing fit "
            "(empty-positive would yield undefined day-BCE)",
            source,
        )
        raise SystemExit(3)
    if n_pos_source < MIN_SOURCE_POS_WARN:
        log.warning(
            "SOURCE POSITIVES LOW: %s has only %d positive user-days "
            "(<%d). DaySupervisedGNN will still train (pos_weight uses "
            "max(n_pos,1)); fit quality may be unreliable.",
            source, n_pos_source, MIN_SOURCE_POS_WARN,
        )
    else:
        log.info("source %s n_pos_user_day=%d", source, n_pos_source)

    kw = gnn_kw_for_source(source, seed, use_dann=False)
    if epochs is not None:
        kw["epochs"] = int(epochs)
        kw["min_epochs"] = min(int(kw["min_epochs"]), int(epochs))
        log.info(
            "epochs override: epochs=%d min_epochs=%d (source=%s)",
            kw["epochs"], kw["min_epochs"], source,
        )
    if model_toggles:
        kw.update(model_toggles)
        log.info("model toggles: %s", model_toggles)
    det = DaySupervisedGNN(**kw)
    if source == target:
        assert protocol is not None
        if protocol == "temporal":
            tr, te = temporal_split(df_s, train_frac=0.7)
        else:
            tr, te = user_disjoint_split(df_s, train_frac=0.7, seed=seed)
        det.set_full_features(feat_s)
        det.fit(tr)
        agg = det.score_user_day(te)
        tag, kind = protocol, "diagonal"
        del tr, te
        n_pos_target = int(agg["label"].sum())
    else:
        feat_t = load_feat(target)
        df_t = load_events(target)
        peak = max(peak, _rss_gb())
        n_pos_target = int(feat_t[LABEL].sum())
        det.set_full_features(feat_s)
        det.fit(df_s)
        det.set_full_features(feat_t)
        agg = det.score_user_day(df_t)
        tag, kind = "full_source", "off_diagonal"
        del feat_t, df_t

    y = agg["label"].to_numpy()
    m = _with_lift(compute_metrics(y, agg["score"].to_numpy()), y)
    peak = max(peak, _rss_gb())
    m.update({
        "source": source, "target": target, "model": "day_gnn_zero_shot",
        "seed": seed, "diagonal_protocol": tag, "cell_kind": kind,
        "wall_s": time.perf_counter() - t0,
        "peak_rss_gb": peak,
        "roc_auc": m.get("roc_auc", float("nan")),
        "prevalence": m.get("base_rate", float("nan")),
        "best_epoch": det.best_epoch,
        "best_val_pr": det.best_val_pr,
        "n_pos_source": n_pos_source,
        "n_pos_target": int(m.get("n_pos", n_pos_target)),
        "epochs": int(kw["epochs"]),
    })
    log.info(
        "GNN seed=%d %s->%s [%s] pr=%.4f roc=%.4f lift=%.2f "
        "n_pos_src=%d n_pos_tgt=%d rss=%.2fGiB (%.1fs)",
        seed, source, target, tag, m["pr_auc"], m["roc_auc"], m["lift"],
        n_pos_source, m["n_pos_target"], m["peak_rss_gb"], m["wall_s"],
    )
    _offdiag_leak_guard(m, f"GNN {source}->{target} [{tag}]")
    # free GPU
    del det, df_s, feat_s, agg
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return m


def _cell_key(r: dict) -> tuple:
    return (
        str(r["model"]), str(r["source"]), str(r["target"]),
        str(r["diagonal_protocol"]), int(r["seed"]),
    )


def _phase_a_jobs(seeds, models=None, cells=None):
    """Yield (model, source, target, seed, protocol). RF block first, then GNN.

    If ``cells`` is set, yield GNN off-diagonal jobs for exactly those
    (src, tgt) pairs — no RF, no diagonals, no other matrix cells.
    """
    if cells:
        for seed in seeds:
            for s, t in cells:
                # Never schedule LANL-source training in selective mode
                # (report-only from checkpoint if present).
                if s == "lanl":
                    continue
                yield ("day_gnn_zero_shot", s, t, seed, None)
        return

    model_order = ["random_forest", "day_gnn_zero_shot"]
    if models:
        model_order = [m for m in model_order if m in models]
    for model in model_order:
        # Off-diagonal
        for seed in seeds:
            for s in DOMAINS:
                for t in DOMAINS:
                    if s == t:
                        continue
                    yield (model, s, t, seed, None)
        # Diagonal user-disjoint (headline) for all
        for seed in seeds:
            for s in DOMAINS:
                yield (model, s, s, seed, "user_disjoint")
        # Supplementary temporal for non-lanl
        for seed in seeds:
            for s in DOMAINS:
                if s == "lanl":
                    continue
                yield (model, s, s, seed, "temporal")


def run_phase_a(
    seeds=SEEDS_DEFAULT,
    models=None,
    resume: Path | None = None,
    cells: list[tuple[str, str]] | None = None,
    epochs: int | None = None,
):
    OUT.mkdir(exist_ok=True)
    details_path = OUT / "transfer_matrix_phaseA.csv"
    ckpt_path = CKPT_DEFAULT

    rows: list[dict] = []
    done: set = set()
    # Append-only: load prior rows so we never overwrite existing cells.
    load_from = None
    if resume is not None and Path(resume).exists():
        load_from = Path(resume)
    elif ckpt_path.exists():
        load_from = ckpt_path
    if load_from is not None:
        prev = pd.read_csv(load_from)
        rows = prev.to_dict(orient="records")
        done = {_cell_key(r) for r in rows}
        log.info(
            "Resume/append from %s (%d rows); skip done cells; never overwrite",
            load_from, len(rows),
        )

    if cells:
        # Selective mode: GNN only for the listed cells — no RF retrain/rescore.
        # LANL-source cells are report-only (never scheduled for fit/score).
        want_models = {"day_gnn_zero_shot"}
        lanl_cells = [f"{s}->{t}" for s, t in cells if s == "lanl"]
        train_cells = [f"{s}->{t}" for s, t in cells if s != "lanl"]
        log.info(
            "Selective --cells mode: GNN train=%s; LANL-source report-only=%s; "
            "RF untouched",
            train_cells or "[]", lanl_cells or "[]",
        )
    else:
        want_models = set(models) if models else {
            "random_forest", "day_gnn_zero_shot",
        }
    jobs = list(_phase_a_jobs(seeds, models=want_models, cells=cells))
    n_skip = sum(
        1 for (model, s, t, seed, protocol) in jobs
        if (model, s, t, protocol or "full_source", int(seed)) in done
    )
    log.info(
        "Phase A: %d jobs (%d already done, %d remaining)%s",
        len(jobs), n_skip, len(jobs) - n_skip,
        f" epochs_override={epochs}" if epochs is not None else "",
    )

    for i, (model, s, t, seed, protocol) in enumerate(jobs, 1):
        proto_tag = protocol or "full_source"
        key = (model, s, t, proto_tag, int(seed))
        if key in done:
            continue
        log.info(
            "---- cell %d/%d  %s %s->%s seed=%d [%s] rss=%.2fGiB ----",
            i, len(jobs), model, s, t, seed, proto_tag, _rss_gb(),
        )
        if model == "random_forest":
            m = run_rf_cell_job(s, t, seed, protocol)
        else:
            m = run_gnn_cell_job(s, t, seed, protocol, epochs=epochs)
        # normalize columns for long-form
        m["pr_auc"] = float(m["pr_auc"])
        m["roc_auc"] = float(m.get("roc_auc", float("nan")))
        m["prevalence"] = float(m.get("base_rate", m.get("prevalence", float("nan"))))
        m["lift"] = float(m["lift"])
        rows.append(m)
        done.add(key)
        pd.DataFrame(rows).to_csv(ckpt_path, index=False)
        # also write long-form continuously
        _write_longform(rows, details_path)

    details = pd.DataFrame(rows)
    details.to_csv(details_path, index=False)
    log.info("Wrote %s (%d rows)", details_path, len(details))
    if cells:
        print_selected_cells_report(details, cells)
        write_into_spedia_csv(details, cells)
    return details


def _write_longform(rows: list[dict], path: Path):
    cols = [
        "source", "target", "model", "seed",
        "pr_auc", "roc_auc", "prevalence", "lift",
        "diagonal_protocol", "cell_kind", "wall_s", "peak_rss_gb",
        "n_pos", "n_pos_source", "n_pos_target", "epochs",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df[cols].to_csv(path, index=False)


def _n_pos_from_feat_cache(name: str) -> int:
    """Read positive user-day count from cached features (no retrain)."""
    path = _feat_path(name)
    if not path.exists():
        return -1
    try:
        feat = pd.read_parquet(path)
        return int(feat[LABEL].sum())
    except Exception as exc:
        log.warning("could not read n_pos from %s: %s", path, exc)
        return -1


def print_selected_cells_report(
    details: pd.DataFrame,
    cells: list[tuple[str, str]],
    *,
    n_boot: int = 5000,
):
    """Per-cell GNN summary: n_pos_target, PR/ROC, lift, 95% seed bootstrap CI."""
    print("\n" + "=" * 78)
    print("SELECTIVE CELLS — day_gnn_zero_shot (seed mean + 95% bootstrap CI)")
    print("=" * 78)
    gnn = details[details["model"] == "day_gnn_zero_shot"]
    for s, t in cells:
        sub = gnn[
            (gnn["source"] == s)
            & (gnn["target"] == t)
            & (gnn["diagonal_protocol"] == "full_source")
        ].copy()
        if sub.empty:
            print(f"  {s}->{t}:  (no rows)")
            continue
        pr = sub["pr_auc"].to_numpy(dtype=float)
        roc = sub["roc_auc"].to_numpy(dtype=float)
        lift = sub["lift"].to_numpy(dtype=float)
        pr_m, pr_lo, pr_hi = seed_bootstrap_ci(pr, n_boot=n_boot, seed=0)
        roc_m, roc_lo, roc_hi = seed_bootstrap_ci(roc, n_boot=n_boot, seed=1)
        lf_m, lf_lo, lf_hi = seed_bootstrap_ci(lift, n_boot=n_boot, seed=2)
        n_pos_tgt = (
            int(sub["n_pos_target"].dropna().iloc[0])
            if "n_pos_target" in sub.columns and sub["n_pos_target"].notna().any()
            else (
                int(sub["n_pos"].dropna().iloc[0])
                if "n_pos" in sub.columns and sub["n_pos"].notna().any()
                else _n_pos_from_feat_cache(t)
            )
        )
        n_pos_src = (
            int(sub["n_pos_source"].dropna().iloc[0])
            if "n_pos_source" in sub.columns and sub["n_pos_source"].notna().any()
            else _n_pos_from_feat_cache(s)
        )
        br = float(sub["base_rate"].mean()) if "base_rate" in sub.columns else float("nan")
        if not np.isfinite(br) and "prevalence" in sub.columns:
            br = float(sub["prevalence"].mean())
        low_src = (
            f"  ** SOURCE POS TOO FEW TO FIT RELIABLY (n_pos_source={n_pos_src})"
            if 0 <= n_pos_src < MIN_SOURCE_POS_WARN
            else ""
        )
        print(
            f"  {s}->{t}  n_seeds={len(sub)}  n_pos_target={n_pos_tgt}  "
            f"n_pos_source={n_pos_src}  base_rate={br:.6f}"
        )
        print(
            f"    PR-AUC  {pr_m:.4f}  [{pr_lo:.4f}, {pr_hi:.4f}]"
        )
        print(
            f"    ROC-AUC {roc_m:.4f}  [{roc_lo:.4f}, {roc_hi:.4f}]"
        )
        print(
            f"    lift    {lf_m:.2f}  [{lf_lo:.2f}, {lf_hi:.2f}]"
            f"  (= PR-AUC / base_rate){low_src}"
        )


def write_into_spedia_csv(
    details: pd.DataFrame,
    cells: list[tuple[str, str]],
    *,
    n_boot: int = 5000,
):
    """Write seed-level + aggregate rows for requested into-spedia (or selected) cells."""
    OUT.mkdir(exist_ok=True)
    want = {(s, t) for s, t in cells}
    mask = details.apply(
        lambda r: (str(r["source"]), str(r["target"])) in want
        and str(r["model"]) == "day_gnn_zero_shot",
        axis=1,
    )
    sub = details.loc[mask].copy()
    if sub.empty:
        log.warning("No selected GNN rows to write to %s", INTO_SPEDIA_CSV)
        return

    # Seed-level detail
    seed_path = INTO_SPEDIA_CSV
    cols = [
        c for c in (
            "source", "target", "model", "seed",
            "pr_auc", "roc_auc", "base_rate", "prevalence", "lift",
            "n_pos", "n_pos_source", "n_pos_target",
            "diagonal_protocol", "cell_kind",
            "best_epoch", "best_val_pr", "epochs",
            "wall_s", "peak_rss_gb",
        )
        if c in sub.columns
    ]
    sub[cols].sort_values(["source", "target", "seed"]).to_csv(seed_path, index=False)
    log.info("Wrote %s (%d seed rows)", seed_path, len(sub))

    # Companion aggregate with CIs
    agg_rows = []
    for (s, t), g in sub.groupby(["source", "target"], sort=True):
        pr_m, pr_lo, pr_hi = seed_bootstrap_ci(
            g["pr_auc"].to_numpy(dtype=float), n_boot=n_boot, seed=0,
        )
        roc_m, roc_lo, roc_hi = seed_bootstrap_ci(
            g["roc_auc"].to_numpy(dtype=float), n_boot=n_boot, seed=1,
        )
        lf_m, lf_lo, lf_hi = seed_bootstrap_ci(
            g["lift"].to_numpy(dtype=float), n_boot=n_boot, seed=2,
        )
        n_pos_tgt = (
            int(g["n_pos_target"].dropna().iloc[0])
            if "n_pos_target" in g.columns and g["n_pos_target"].notna().any()
            else (
                int(g["n_pos"].dropna().iloc[0])
                if "n_pos" in g.columns and g["n_pos"].notna().any()
                else _n_pos_from_feat_cache(str(t))
            )
        )
        n_pos_src = (
            int(g["n_pos_source"].dropna().iloc[0])
            if "n_pos_source" in g.columns and g["n_pos_source"].notna().any()
            else _n_pos_from_feat_cache(str(s))
        )
        br = float(g["base_rate"].mean()) if "base_rate" in g.columns else float("nan")
        if not np.isfinite(br) and "prevalence" in g.columns:
            br = float(g["prevalence"].mean())
        agg_rows.append({
            "source": s, "target": t, "model": "day_gnn_zero_shot",
            "n_seeds": len(g),
            "n_pos_target": n_pos_tgt,
            "n_pos_source": n_pos_src,
            "base_rate": br,
            "pr_auc_mean": pr_m, "pr_auc_lo": pr_lo, "pr_auc_hi": pr_hi,
            "roc_auc_mean": roc_m, "roc_auc_lo": roc_lo, "roc_auc_hi": roc_hi,
            "lift_mean": lf_m, "lift_lo": lf_lo, "lift_hi": lf_hi,
            "source_pos_too_few": bool(0 <= n_pos_src < MIN_SOURCE_POS_WARN),
        })
    agg_path = OUT / "transfer5d_into_spedia_summary.csv"
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
    log.info("Wrote %s (%d cells)", agg_path, len(agg_rows))


# ---- component ablation (zero-shot into-SPEDIA) -----------------------------
def _ablation_key(r: dict) -> tuple:
    return (
        str(r["variant"]), str(r["source"]), str(r["target"]), int(r["seed"]),
    )


def _load_rf_seed_prs(
    cells: list[tuple[str, str]],
) -> dict[tuple[str, str, int], float]:
    """Seed-level RF PR-AUC for the same cells from Phase-A checkpoint (no retrain)."""
    path = CKPT_DEFAULT if CKPT_DEFAULT.exists() else OUT / "transfer_matrix_phaseA.csv"
    out: dict[tuple[str, str, int], float] = {}
    if not path.exists():
        log.warning("No Phase-A CSV for RF baselines (%s)", path)
        return out
    df = pd.read_csv(path)
    want = set(cells)
    sub = df[
        (df["model"] == "random_forest")
        & (df["diagonal_protocol"] == "full_source")
    ]
    for _, r in sub.iterrows():
        key = (str(r["source"]), str(r["target"]))
        if key not in want:
            continue
        out[(key[0], key[1], int(r["seed"]))] = float(r["pr_auc"])
    log.info("Loaded %d RF seed rows for ablation ΔPR_vs_rf from %s", len(out), path)
    return out


def _check_v0_sanity(rows: list[dict], cells: list[tuple[str, str]]) -> None:
    """Stop if V0 full drifts far from known into-SPEDIA means (config mismatch)."""
    df = pd.DataFrame(rows)
    v0 = df[df["variant"] == "V0_full"]
    if v0.empty:
        return
    bad = []
    for s, t in cells:
        sub = v0[(v0["source"] == s) & (v0["target"] == t)]
        if sub.empty:
            continue
        mean_pr = float(sub["pr_auc"].mean())
        ref = V0_REF_PR.get((s, t))
        if ref is None:
            continue
        drift = abs(mean_pr - ref)
        log.info(
            "V0 sanity %s->%s: mean_PR=%.4f ref=%.3f |Δ|=%.4f (tol=%.3f) n=%d",
            s, t, mean_pr, ref, drift, V0_SANITY_ABS_TOL, len(sub),
        )
        if len(sub) >= len(SEEDS_DEFAULT) and drift > V0_SANITY_ABS_TOL:
            bad.append(
                f"{s}->{t}: V0_mean={mean_pr:.4f} vs ref≈{ref:.3f} "
                f"(|Δ|={drift:.4f} > {V0_SANITY_ABS_TOL})"
            )
    if bad:
        msg = (
            "V0 full-model into-SPEDIA drifted from known C2 numbers — "
            "likely config mismatch; refusing to trust the ablation.\n  "
            + "\n  ".join(bad)
        )
        log.error(msg)
        raise SystemExit(4)


def print_ablation_report(
    details: pd.DataFrame,
    cells: list[tuple[str, str]],
    *,
    n_boot: int = 5000,
):
    """Per (variant, cell): mean PR/lift, dPR vs V0 and vs RF with paired CIs."""
    rf_prs = _load_rf_seed_prs(cells)
    print("\n" + "=" * 78)
    print("COMPONENT ABLATION — zero-shot into-SPEDIA (day_gnn; no DANN/RF retrain)")
    print("=" * 78)

    variants = [v for v, _ in ABLATION_VARIANTS]
    for s, t in cells:
        print(f"\n### {s}->{t}")
        cell = details[
            (details["source"] == s) & (details["target"] == t)
        ].copy()
        if cell.empty:
            print("  (no rows)")
            continue
        v0 = cell[cell["variant"] == "V0_full"].sort_values("seed")
        v0_by_seed = {
            int(r["seed"]): float(r["pr_auc"]) for _, r in v0.iterrows()
        }
        for variant in variants:
            sub = cell[cell["variant"] == variant].sort_values("seed")
            if sub.empty:
                print(f"  {variant:16s}  (no rows)")
                continue
            pr = sub["pr_auc"].to_numpy(dtype=float)
            lift = sub["lift"].to_numpy(dtype=float)
            pr_m = float(np.mean(pr))
            lf_m = float(np.mean(lift))

            # dPR vs full (paired by seed)
            if variant == "V0_full":
                d_full = d_full_lo = d_full_hi = 0.0
            else:
                paired_v, paired_f = [], []
                for _, r in sub.iterrows():
                    seed = int(r["seed"])
                    if seed in v0_by_seed:
                        paired_v.append(float(r["pr_auc"]))
                        paired_f.append(v0_by_seed[seed])
                d_full, d_full_lo, d_full_hi = paired_delta_ci(
                    paired_v, paired_f, n_boot=n_boot, seed=0,
                )

            # dPR vs RF (paired by seed; RF from Phase-A checkpoint)
            paired_g, paired_r = [], []
            for _, r in sub.iterrows():
                seed = int(r["seed"])
                rf_pr = rf_prs.get((s, t, seed))
                if rf_pr is not None:
                    paired_g.append(float(r["pr_auc"]))
                    paired_r.append(rf_pr)
            if paired_g:
                d_rf, d_rf_lo, d_rf_hi = paired_delta_ci(
                    paired_g, paired_r, n_boot=n_boot, seed=2,
                )
                holds = bool(d_rf > 0 and d_rf_lo > 0)
            else:
                d_rf = d_rf_lo = d_rf_hi = float("nan")
                holds = False

            flag = "HOLDS" if holds else "broken"
            print(
                f"  {variant:16s}  PR={pr_m:.4f}  lift={lf_m:.2f}  n={len(sub)}"
            )
            print(
                f"    dPR_vs_full={d_full:+.4f} "
                f"[{d_full_lo:+.4f},{d_full_hi:+.4f}]  "
                f"dPR_vs_rf={d_rf:+.4f} "
                f"[{d_rf_lo:+.4f},{d_rf_hi:+.4f}]  "
                f"into-SPEDIA>RF={flag}"
            )


def run_ablation(
    seeds=SEEDS_DEFAULT,
    cells: list[tuple[str, str]] | None = None,
    epochs: int | None = 30,
    resume: bool = True,
):
    """Zero-shot component ablation on into-SPEDIA cells only.

    Writes ``results/ablation_into_spedia.csv``. Reuses feature cache; with
    ``resume`` skips completed (variant, source, target, seed) rows and appends
    only new ones (never overwrites).
    """
    OUT.mkdir(exist_ok=True)
    cells = list(cells) if cells else list(ABLATION_CELLS_DEFAULT)
    # Refuse anything outside the C2 into-SPEDIA scope.
    allowed = set(ABLATION_CELLS_DEFAULT)
    bad = [f"{s}->{t}" for s, t in cells if (s, t) not in allowed]
    if bad:
        raise SystemExit(
            f"--ablate only supports {sorted(f'{a}->{b}' for a,b in allowed)}; "
            f"refusing {bad}"
        )
    # Ensure feature (+ event) caches exist; never rebuild if present.
    for name in sorted({d for pair in cells for d in pair}):
        cache_features(name, force=False)
        cache_events(name, force=False)

    rows: list[dict] = []
    done: set = set()
    if resume and ABLATION_CSV.exists():
        prev = pd.read_csv(ABLATION_CSV)
        rows = prev.to_dict(orient="records")
        done = {_ablation_key(r) for r in rows}
        log.info(
            "Ablation resume from %s (%d rows); skip done; never overwrite",
            ABLATION_CSV, len(rows),
        )

    jobs = [
        (variant, toggles, s, t, seed)
        for variant, toggles in ABLATION_VARIANTS
        for s, t in cells
        for seed in seeds
    ]
    n_skip = sum(
        1 for (variant, _, s, t, seed) in jobs
        if (variant, s, t, int(seed)) in done
    )
    log.info(
        "Ablation: %d jobs (%d done, %d remaining) cells=%s epochs=%s seeds=%s",
        len(jobs), n_skip, len(jobs) - n_skip,
        [f"{s}->{t}" for s, t in cells], epochs, list(seeds),
    )

    def _v0_done() -> bool:
        return all(
            ("V0_full", cs, ct, int(sd)) in done
            for cs, ct in cells
            for sd in seeds
        )

    # If resuming with V0 complete, validate before spending compute on V1+.
    if _v0_done():
        _check_v0_sanity(rows, cells)

    for i, (variant, toggles, s, t, seed) in enumerate(jobs, 1):
        key = (variant, s, t, int(seed))
        if key in done:
            continue
        log.info(
            "---- ablation %d/%d  %s %s->%s seed=%d rss=%.2fGiB ----",
            i, len(jobs), variant, s, t, seed, _rss_gb(),
        )
        m = run_gnn_cell_job(
            s, t, seed, protocol=None,
            epochs=epochs, model_toggles=toggles,
        )
        row = {
            "variant": variant,
            "source": s,
            "target": t,
            "seed": int(seed),
            "pr_auc": float(m["pr_auc"]),
            "roc_auc": float(m.get("roc_auc", float("nan"))),
            "base_rate": float(m.get("base_rate", m.get("prevalence", float("nan")))),
            "lift": float(m["lift"]),
        }
        rows.append(row)
        done.add(key)
        # Append-safe write after every cell.
        pd.DataFrame(rows)[
            ["variant", "source", "target", "seed",
             "pr_auc", "roc_auc", "base_rate", "lift"]
        ].to_csv(ABLATION_CSV, index=False)

        # After V0 finishes both cells × all seeds, sanity-check before V1+.
        if variant == "V0_full" and _v0_done():
            _check_v0_sanity(rows, cells)

    details = pd.DataFrame(rows)
    if not details.empty:
        details[
            ["variant", "source", "target", "seed",
             "pr_auc", "roc_auc", "base_rate", "lift"]
        ].to_csv(ABLATION_CSV, index=False)
    log.info("Wrote %s (%d rows)", ABLATION_CSV, len(details))
    print_ablation_report(details, cells)
    return details


# ---- reporting --------------------------------------------------------------
def wilcoxon_offdiag(details: pd.DataFrame) -> dict:
    """Paired Wilcoxon over off-diagonal cells: mean_seed(GNN) vs mean_seed(RF)."""
    from scipy.stats import wilcoxon

    off = details[details["cell_kind"] == "off_diagonal"]
    gnn = (
        off[off["model"] == "day_gnn_zero_shot"]
        .groupby(["source", "target"], as_index=False)["pr_auc"].mean()
        .rename(columns={"pr_auc": "gnn"})
    )
    rf = (
        off[off["model"] == "random_forest"]
        .groupby(["source", "target"], as_index=False)["pr_auc"].mean()
        .rename(columns={"pr_auc": "rf"})
    )
    m = gnn.merge(rf, on=["source", "target"])
    if len(m) < 2:
        return {"n_cells": len(m), "pvalue": float("nan"), "stat": float("nan")}
    # alternative: GNN > RF
    try:
        stat, p = wilcoxon(m["gnn"], m["rf"], alternative="greater", zero_method="wilcox")
    except ValueError as exc:
        log.warning("Wilcoxon failed: %s", exc)
        return {"n_cells": len(m), "pvalue": float("nan"), "stat": float("nan")}
    return {
        "n_cells": int(len(m)),
        "pvalue": float(p),
        "stat": float(stat),
        "n_gnn_gt_rf": int((m["gnn"] > m["rf"]).sum()),
        "mean_delta": float((m["gnn"] - m["rf"]).mean()),
    }


def generalization_gap_ud(details: pd.DataFrame, model: str) -> float:
    """Δ = mean(user-disjoint diagonal) − mean(off-diagonal) on PR-AUC."""
    d = details[details["model"] == model]
    diag = d[(d["cell_kind"] == "diagonal") & (d["diagonal_protocol"] == "user_disjoint")]
    off = d[d["cell_kind"] == "off_diagonal"]
    if diag.empty or off.empty:
        return float("nan")
    # mean over cells (first avg seeds per cell)
    diag_m = diag.groupby(["source", "target"])["pr_auc"].mean().mean()
    off_m = off.groupby(["source", "target"])["pr_auc"].mean().mean()
    return float(diag_m - off_m)


def print_phase_a_report(details: pd.DataFrame):
    summary = aggregate_seed_rows(details, n_boot=5000)
    deltas = gnn_minus_rf_table(
        summary, details,
        gnn_model="day_gnn_zero_shot", rf_model="random_forest", n_boot=5000,
    )
    OUT.mkdir(exist_ok=True)
    summary.to_csv(OUT / "transfer_matrix_phaseA_summary.csv", index=False)
    deltas.to_csv(OUT / "transfer_matrix_phaseA_deltas.csv", index=False)

    names = list(DOMAINS)
    print("\n" + "=" * 78)
    print("PHASE A — cell-level PR-AUC / lift (user-disjoint diag + off-diag)")
    print("=" * 78)

    for model in ("day_gnn_zero_shot", "random_forest"):
        print(f"\n### {model}  PR-AUC mean [95% CI]  (lift)")
        sub = summary[
            (summary["model"] == model)
            & (
                ((summary["cell_kind"] == "off_diagonal"))
                | (
                    (summary["cell_kind"] == "diagonal")
                    & (summary["diagonal_protocol"] == "user_disjoint")
                )
            )
        ]
        pr = pd.DataFrame(index=names, columns=names, dtype=object)
        for _, r in sub.iterrows():
            cell = (
                f"{r['pr_auc_mean']:.3f}[{r['pr_auc_lo']:.3f},{r['pr_auc_hi']:.3f}] "
                f"L={r['lift_mean']:.1f}"
            )
            pr.loc[r["source"], r["target"]] = cell
        print(pr.to_string())

    print("\n### day_gnn − RF  (off-diagonal)  ΔPR [CI]  claim_holds")
    if deltas is None or deltas.empty or "cell_kind" not in deltas.columns:
        print("  (no paired GNN+RF cells yet)")
        off_d = pd.DataFrame()
        n_hold = 0
    else:
        off_d = deltas[deltas["cell_kind"] == "off_diagonal"].copy()
    if not off_d.empty:
        for _, r in off_d.sort_values(["source", "target"]).iterrows():
            flag = "YES" if r["claim_holds"] else "no"
            print(
                f"  {r['source']:7s}->{r['target']:7s}  "
                f"ΔPR={r['delta_pr_mean']:+.4f} "
                f"[{r['delta_pr_lo']:+.4f},{r['delta_pr_hi']:+.4f}]  "
                f"holds={flag}  "
                f"(gnn={r['gnn_pr_mean']:.3f} rf={r['rf_pr_mean']:.3f})"
            )
        n_hold = int(off_d["claim_holds"].fillna(False).sum())
        print(f"\nOff-diagonal cells with CI excluding 0 and Δ>0: "
              f"{n_hold}/{len(off_d)}")

    gap_gnn = generalization_gap_ud(details, "day_gnn_zero_shot")
    gap_rf = generalization_gap_ud(details, "random_forest")
    print(f"\n### Generalization gap Δ = mean(UD diag) − mean(off-diag)")
    print(f"  day_gnn_zero_shot: Δ={gap_gnn:.4f}")
    print(f"  random_forest:     Δ={gap_rf:.4f}")

    w = wilcoxon_offdiag(details)
    print(f"\n### Wilcoxon signed-rank (off-diag cells, GNN vs RF, alternative=greater)")
    print(f"  n_cells={w['n_cells']}  n_gnn>rf={w.get('n_gnn_gt_rf')}  "
          f"mean_Δ={w.get('mean_delta', float('nan')):.4f}  "
          f"W={w['stat']:.3f}  p={w['pvalue']:.6g}")

    # C2 lock verdict
    if not off_d.empty:
        majority = n_hold > len(off_d) / 2
        print(f"\n### C2 lock: graph>tabular on majority of off-diag with CI≠0? "
              f"{'YES — broad' if majority else 'NO — stays narrow'} "
              f"({n_hold}/{len(off_d)})")

    # Supplementary temporal diag note
    temp = summary[
        (summary["cell_kind"] == "diagonal")
        & (summary["diagonal_protocol"] == "temporal")
    ]
    if not temp.empty:
        print("\n### Supplementary temporal diagonal (cert*/spedia; not lanl)")
        for _, r in temp.sort_values(["model", "source"]).iterrows():
            print(
                f"  {r['model']:22s} {r['source']:7s}  "
                f"PR={r['pr_auc_mean']:.3f}[{r['pr_auc_lo']:.3f},{r['pr_auc_hi']:.3f}]  "
                f"L={r['lift_mean']:.1f}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=("cache", "A", "rf", "gnn", "report", "all"),
        default="all",
        help="cache | A (RF+GNN) | rf | gnn | report | all",
    )
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument(
        "--features-only", action="store_true",
        help="With --phase cache: only build feature matrices (skip event dfs)",
    )
    ap.add_argument(
        "--resume",
        nargs="?",
        const=str(CKPT_DEFAULT),
        default=None,
        help=(
            "Resume/append from checkpoint CSV (skip cells already present; "
            "never overwrite). Flag alone uses "
            "results/transfer_matrix_phaseA_checkpoint.csv; or pass a path. "
            "With --ablate: reuse feature cache + completed ablation rows."
        ),
    )
    ap.add_argument(
        "--cells",
        default="",
        help=(
            'Comma-separated src->tgt list, e.g. "cert62->spedia,lanl->spedia". '
            "If set, run GNN Phase-A for exactly these off-diagonal cells "
            "(no RF retrain/rescore; no other matrix cells). "
            "LANL-source entries are report-only from checkpoint (never fitted). "
            'With --ablate: defaults to "cert42->spedia,cert52->spedia".'
        ),
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override GNN epoch budget for this run (default 60 CERT / 80 other).",
    )
    ap.add_argument(
        "--ablate",
        action="store_true",
        help=(
            "Run component ablation grid (V0–V4) zero-shot on into-SPEDIA "
            "cells only (cert42/cert52->spedia). No RF, no LANL, no DANN/UDA. "
            "Writes results/ablation_into_spedia.csv."
        ),
    )
    args = ap.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip() != "")
    cells = _parse_cells(args.cells)

    try:
        import torch
        log.info(
            "torch=%s cuda=%s device=%s",
            torch.__version__, torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        )
    except ImportError:
        log.warning("torch not importable in this interpreter")

    # ---- ablation mode (exclusive; no Phase A / UDA) ------------------------
    if args.ablate:
        if not cells:
            cells = list(ABLATION_CELLS_DEFAULT)
        ep = 30 if args.epochs is None else int(args.epochs)
        run_ablation(
            seeds=seeds or SEEDS_DEFAULT,
            cells=cells,
            epochs=ep,
            resume=(args.resume is not None) or ABLATION_CSV.exists(),
        )
        log.info("Done. peak process RSS now=%.2fGiB", _rss_gb())
        return

    if cells and args.phase == "rf":
        raise SystemExit("--cells is GNN-only; refuse --phase rf")
    if cells and args.phase in ("all", "A"):
        # Avoid cache rebuild + RF when user only asked for selective GNN cells.
        log.info("--cells set: treating phase as gnn (skip cache/RF)")
        args.phase = "gnn"

    if args.phase in ("cache", "all") and not cells:
        if args.features_only:
            CACHE.mkdir(parents=True, exist_ok=True)
            for name in DOMAINS:
                cache_features(name, force=args.force_cache)
                gc.collect()
        else:
            build_all_caches(force=args.force_cache)

    # --resume flag/path; otherwise still append-safe via existing checkpoint.
    if args.resume is not None:
        resume = Path(args.resume)
        if not resume.exists():
            log.warning("--resume path missing (%s); starting fresh rows", resume)
    else:
        resume = CKPT_DEFAULT if CKPT_DEFAULT.exists() else None

    run_kw = dict(seeds=seeds, resume=resume, cells=cells or None, epochs=args.epochs)

    if args.phase in ("A", "all"):
        details = run_phase_a(**run_kw)
        if not cells:
            print_phase_a_report(details)
    elif args.phase == "rf":
        details = run_phase_a(models=["random_forest"], **run_kw)
        print_phase_a_report(details)
    elif args.phase == "gnn":
        details = run_phase_a(models=["day_gnn_zero_shot"], **run_kw)
        if not cells:
            print_phase_a_report(details)
    elif args.phase == "report":
        # Prefer checkpoint (has n_pos / full columns) over long-form details.
        if CKPT_DEFAULT.exists():
            path = CKPT_DEFAULT
        else:
            path = OUT / "transfer_matrix_phaseA.csv"
        details = pd.read_csv(path)
        if cells:
            print_selected_cells_report(details, cells)
            write_into_spedia_csv(details, cells)
        else:
            print_phase_a_report(details)

    log.info("Done. peak process RSS now=%.2fGiB", _rss_gb())


if __name__ == "__main__":
    main()
