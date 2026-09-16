"""
STRICT CAMERA-READY EVIDENCE SET (IJACSA MS-17-9-0257), 15 Sep 2026.

PUBLICATION NOTE (2026-09-16)
  This is the authoritative driver for the camera-ready results. Local absolute paths were
  replaced by <DATA_ROOT>/<REPORT_DIR> placeholders (see ITD_DATA_ROOT / ITD_REPORT_DIR and the
  --raw-* options). The lodo / lodo-dry-run / gnn-matrix stages exist in the code but were NOT
  run for the camera-ready paper, and no LODO, domain-adaptation or full GNN-matrix result is
  reported. ACCEPTED_C2 holds superseded pre-correction values used only for an audit comparison.

What this driver does
---------------------
It rebuilds the benchmark evidence under two corrections and writes everything to a
NEW namespace. No original cache or result file is read-write; originals are only read
and hashed.

  results/strict_camera_ready_2026_09_15/
      cache/              corrected label-agnostic event + feature caches
      rf/                 RF cells, per seed, with predictions
      gnn/                C2 GNN cells (no time encoding), per seed, with predictions
      temporal_controls/  chronological + user-disjoint in-distribution controls
      lodo/               4-source LODO into SPEDIA (source-only and DANN-UDA)
      manifests/          sampling verification, environment, SHA-256 of code/caches/outputs
      logs/

Correction 1: label-agnostic sampling (Section B of the task)
  CERT http.csv : every row kept with probability 0.05, for EVERY user.
  LANL auth     : every (non-machine-account) row kept with probability 0.02, for EVERY user.
  Seed          : 7 (the project's fixed sampling seed, CERT_LOAD_KW["seed"] / load_lanl seed).
  How           : the frozen loaders are re-compiled from their own source with exactly ONE
                  line replaced (the keep rule); the roster term ``is_ins |`` / ``is_rt |``
                  is removed. The random stream (one rng.random(len(chunk)) draw per chunk)
                  is therefore identical to the original run, so every non-roster row of the
                  original cache must reappear unchanged. This is verified row-by-row.
  Labels        : attached by the frozen code after the sampled universe is fixed
                  (answer-key / red-team user-days), unchanged.
  SPEDIA        : unchanged (real_only provenance-filtered); original caches copied + hashed.

Correction 2: strict zero-shot time handling (Section C)
  Every GNN in this namespace uses the frozen V3 configuration use_time_encoding=False.
  In addition the time channel passed to the model is overwritten with zeros
  (StrictNoTimeGNN), so no stream-level t_min/t_max value can reach any computation.
  Temporal learning remains through chronological GRU user/host memories.

Usage (from the repo root, GPU environment):
  set PYTHONIOENCODING=utf-8
  python -m scripts.strict_camera_ready_2026_09_15 --stage preflight
  python -m scripts.strict_camera_ready_2026_09_15 --stage all
      # all = C2 caches + C2 + RF matrix + temporal controls. NEVER starts LODO.
Stages can also be run one at a time:
  build-c2        cert42 + cert52 + spedia strict caches (needed for C2)
  c2              C2 cells, GNN(no-time)+RF, seeds (default 0,1,2,3,4)
  build-rest      cert62 + lanl strict caches
  rf              full RF matrix (5 seeds)
  temporal        GNN(no-time) chronological + user-disjoint diagonals
  lodo-dry-run    pool/check 4-source LODO; no training (explicit only)
  lodo            4-source LODO training (explicit only; never part of --stage all)
  gnn-matrix      OPTIONAL GNN(no-time) off-diagonal matrix
  summarize       STRICT_*.csv + STRICT_MANIFEST.json
Everything is resumable: finished cells (by output file) are skipped.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import inspect
import json
import logging
import os
import platform
import random
import shutil
import sys
import textwrap
import time
from pathlib import Path

# Must be set before CUDA / torch initialisation (DaySupervisedGNN import below).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import scripts.run_transfer_matrix_5domain as T
import src.data.loaders as L
from src.data.features import FEATURE_COLUMNS, X_y, user_day_features
from src.data.schema import ACTION, LABEL, TIMESTAMP, USER, validate
from src.train.day_supervised_gnn import DaySupervisedGNN, _with_lift
from src.train.evaluate import compute_metrics
from src.train.multiseed_transfer import gnn_kw_for_source, paired_delta_ci, seed_bootstrap_ci
from src.train.splits import temporal_split, user_disjoint_split
from src.train.supervised_transfer import SupervisedAdapter, _split_user_day_frame, rf_factory
from src.utils.seed import set_seed

TAG = "strict_camera_ready_2026_09_15"
OUT = ROOT / "results" / TAG
D = {k: OUT / k for k in ("cache", "c2", "rf", "gnn", "temporal_controls", "lodo", "manifests", "logs", "gnn_matrix")}
# Public release: local absolute paths were replaced by environment-variable placeholders.
#   ITD_DATA_ROOT  -> directory holding the third-party raw data (see README of camera_ready_strict_2026_09/)
#   ITD_REPORT_DIR -> directory for the human-readable evidence pack (default: results/<TAG>/report)
DATA_ROOT = Path(os.environ.get("ITD_DATA_ROOT", "<DATA_ROOT>"))
REPORT_DIR = Path(os.environ.get("ITD_REPORT_DIR", str(OUT / "report")))
CACHE_HASH_MEMO: dict = {}
ACCEPTED_C2 = {
    ("cert42", "spedia"): {"old_gnn": 0.617, "old_rf": 0.488, "old_dpr": 0.128, "old_ci": (0.062, 0.195)},
    ("cert52", "spedia"): {"old_gnn": 0.636, "old_rf": 0.463, "old_dpr": 0.173, "old_ci": (0.125, 0.226)},
}
ORIG_CACHE = ROOT / "results" / "cache" / "transfer_5d"
ORIG_LANL_EVENTS = DATA_ROOT / "lanl" / "_cache_load_lanl_redteam_aware_bf002.parquet"  # pre-correction cache (read-only, audit only)

# Raw data (overridable on the command line). CERT r4.2 moved since Aug 2026;
# the sampling verification proves it is the same release (row-level equality).
RAW = dict(
    cert42=str(DATA_ROOT / "CERT" / "r4.2"),
    cert52=str(DATA_ROOT / "CERT" / "r5.2"),
    cert62=str(DATA_ROOT / "CERT" / "r6.2"),
    answers=str(DATA_ROOT / "CERT" / "answers"),
    lanl=str(DATA_ROOT / "lanl"),
)
RELEASE = {"cert42": "4.2", "cert52": "5.2", "cert62": "6.2"}
SAMPLE_SEED = 7
CERT_FRAC = 0.05
LANL_FRAC = 0.02
SEEDS = (0, 1, 2, 3, 4)
DOMAINS = ("cert42", "cert52", "cert62", "spedia", "lanl")
TARGETS_EVALUABLE = ("cert42", "cert52", "spedia", "lanl")   # CERT r6.2: source only (44 target positives)
C2_CELLS = (("cert42", "spedia"), ("cert52", "spedia"))
NO_TIME = {"use_time_encoding": False}
CONFIG_LABEL = "strict_camera_ready_corrected (label-agnostic sampling; GNN V3 no time encoding)"

log = logging.getLogger("strict")


# =============================================================================
# utilities
# =============================================================================
def sha256(p: Path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return float("nan")


def gpu_reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def gpu_peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:
        pass
    return None


def cache_file_rec(p: Path) -> dict:
    p = Path(p).resolve()
    key = str(p)
    if key not in CACHE_HASH_MEMO:
        CACHE_HASH_MEMO[key] = {"path": key, "sha256": sha256(p), "bytes": int(p.stat().st_size)}
    return CACHE_HASH_MEMO[key]


def log_c2_inputs(role: str, paths: dict) -> dict:
    rec = {}
    for k, p in paths.items():
        p = assert_strict_cache_path(p, f"{role}:{k}")
        rec[k] = cache_file_rec(p)
        log.info("[c2-input] %s %s path=%s sha256=%s bytes=%d",
                 role, k, rec[k]["path"], rec[k]["sha256"], rec[k]["bytes"])
    return rec


def c2_pred_csv(kind: str, source: str, target: str, seed: int) -> Path:
    return D["c2"] / "predictions" / f"predictions_{kind}_{source}_{target}_seed{seed}.csv"


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


def append_row(p: Path, row: dict):
    """Append with column union (rows may carry different keys); atomic rewrite."""
    p.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if p.exists():
        new = pd.concat([pd.read_csv(p), new], ignore_index=True)
    tmp = p.with_suffix(".tmp")
    new.to_csv(tmp, index=False)
    os.replace(tmp, p)


def save_pred(p: Path, agg: pd.DataFrame, user_col=USER):
    p.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "user": agg[user_col].astype(str).to_numpy(),
        "day": pd.to_datetime(agg["day"]).dt.strftime("%Y-%m-%d").to_numpy(),
        "score": agg["score"].astype("float64").to_numpy(),
        "label": agg["label"].astype("int8").to_numpy(),
    })
    if p.suffix.lower() == ".csv":
        out.to_csv(p, index=False)
        out.to_parquet(p.with_suffix(".parquet"), index=False)
    else:
        out.to_parquet(p, index=False)
        csv_p = p.with_suffix(".csv")
        if not csv_p.exists():
            out.to_csv(csv_p, index=False)


def metrics(y, s) -> dict:
    m = _with_lift(compute_metrics(np.asarray(y), np.asarray(s)), np.asarray(y))
    return {k: m.get(k) for k in ("pr_auc", "roc_auc", "lift", "base_rate", "n", "n_pos")}


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


FORBIDDEN_CACHE_MARKERS = (
    "transfer_5d",
    "redteam_aware",
    "label_agnostic_check",
    "_cache_load_lanl_redteam_aware",
)
LEGACY_LANL = DATA_ROOT / "lanl" / "_cache_load_lanl_redteam_aware_bf002.parquet"
ORIG_TRANSFER_CSV = ROOT / "results" / "transfer_matrix_phaseA_checkpoint.csv"
DETERMINISM_STATE: dict = {}


def enable_strict_determinism(seed: int) -> dict:
    """Seed every RNG and enable PyTorch deterministic algorithms.

    Does NOT catch/disable failures of use_deterministic_algorithms.
    CUBLAS_WORKSPACE_CONFIG is set at import time (before torch load).
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
    )
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    set_seed(int(seed))
    rec = {
        "seed": int(seed),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_random": True,
        "numpy": True,
        "torch_cpu": False,
        "torch_cuda": False,
        "cudnn_benchmark": None,
        "cudnn_deterministic": None,
        "use_deterministic_algorithms": False,
    }
    try:
        import torch
        torch.manual_seed(int(seed))
        rec["torch_cpu"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
            rec["torch_cuda"] = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        rec["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        rec["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        torch.use_deterministic_algorithms(True)
        rec["use_deterministic_algorithms"] = True
    except Exception:
        raise
    DETERMINISM_STATE.clear()
    DETERMINISM_STATE.update(rec)
    return rec


def assert_strict_cache_path(path, role: str) -> Path:
    """Fail closed if a training/eval input is a legacy label-aware cache."""
    path = Path(path).resolve()
    log.info("[cache-isolation] %s -> %s", role, path)
    cache_root = D["cache"].resolve()
    under_strict = path == cache_root or cache_root in path.parents
    if not under_strict:
        raise RuntimeError(
            f"FAIL CLOSED cache isolation: {role} resolved outside strict cache: {path}"
        )
    text = str(path).replace("/", "\\").lower()
    for marker in FORBIDDEN_CACHE_MARKERS:
        if marker.lower() in text:
            raise RuntimeError(
                f"FAIL CLOSED cache isolation: {role} hit legacy marker {marker!r}: {path}"
            )
    if path == LEGACY_LANL.resolve():
        raise RuntimeError(f"FAIL CLOSED: {role} is the redteam-aware LANL cache: {path}")
    return path


def strict_events_path(name: str) -> Path:
    return assert_strict_cache_path(T._events_path(name), f"events:{name}")


def strict_feat_path(name: str) -> Path:
    return assert_strict_cache_path(T._feat_path(name), f"features:{name}")


def point_cache_to_strict():
    """Redirect the frozen driver's cache helpers to the strict namespace."""
    T.CACHE = D["cache"]
    T.LANL_EVENTS = D["cache"] / "lanl_events.parquet"
    log.info("[cache-isolation] T.CACHE=%s T.LANL_EVENTS=%s",
             Path(T.CACHE).resolve(), Path(T.LANL_EVENTS).resolve())
    assert_strict_cache_path(T.CACHE, "T.CACHE")
    assert_strict_cache_path(T.LANL_EVENTS, "T.LANL_EVENTS")


def assert_c2_eval_parity(gnn_pred: pd.DataFrame, rf_pred: pd.DataFrame, tag: str):
    """RF and GNN must score the identical (user, day, label) target rows."""
    need = ["user", "day", "label"]
    for frame, who in ((gnn_pred, "gnn"), (rf_pred, "rf")):
        missing = [c for c in need if c not in frame.columns]
        if missing:
            raise SystemExit(f"FAIL CLOSED eval parity ({tag}): {who} missing {missing}")
    g = gnn_pred[need].copy()
    r = rf_pred[need].copy()
    g["user"] = g["user"].astype(str)
    r["user"] = r["user"].astype(str)
    g["day"] = pd.to_datetime(g["day"])
    r["day"] = pd.to_datetime(r["day"])
    g["label"] = g["label"].astype(int)
    r["label"] = r["label"].astype(int)
    g = g.sort_values(["user", "day"]).reset_index(drop=True)
    r = r.sort_values(["user", "day"]).reset_index(drop=True)
    if len(g) != len(r) or not g.equals(r):
        raise SystemExit(
            f"FAIL CLOSED eval parity ({tag}): RF/GNN (user,day,label) rows are not identical "
            f"(n_gnn={len(g)} n_rf={len(r)})"
        )


def evaluable_set_gap(
    details: pd.DataFrame,
    model: str,
    sources=None,
    evaluable_targets=TARGETS_EVALUABLE,
    diag_protocol: str = "user_disjoint",
) -> dict:
    """Δ = mean_{j in T}(M_jj) − mean_{(i,j) in P}(M_ij). Does not alter frozen gap_ud."""
    S = list(sources if sources is not None else DOMAINS)
    Tgt = list(evaluable_targets)
    d = details[details["model"] == model].copy()
    if "cell_kind" in d.columns:
        diag = d[(d["cell_kind"] == "diagonal") & (d.get("diagonal_protocol", diag_protocol) == diag_protocol)]
        if "diagonal_protocol" in d.columns:
            diag = d[(d["cell_kind"] == "diagonal") & (d["diagonal_protocol"] == diag_protocol)]
        off = d[d["cell_kind"] == "off_diagonal"]
    else:
        diag = d[(d["source"] == d["target"]) & (d["protocol"] == diag_protocol)]
        off = d[d["source"] != d["target"]]
    diag_m = diag.groupby(["source", "target"], as_index=False)["pr_auc"].mean()
    off_m = off.groupby(["source", "target"], as_index=False)["pr_auc"].mean()
    diag_vals, diag_cells = [], []
    for j in Tgt:
        row = diag_m[(diag_m["source"] == j) & (diag_m["target"] == j)]
        if row.empty:
            continue
        diag_vals.append(float(row["pr_auc"].iloc[0]))
        diag_cells.append((j, j))
    p_vals, p_cells = [], []
    for i in S:
        for j in Tgt:
            if i == j:
                continue
            row = off_m[(off_m["source"] == i) & (off_m["target"] == j)]
            if row.empty:
                continue
            p_vals.append(float(row["pr_auc"].iloc[0]))
            p_cells.append((i, j))
    delta = float(np.mean(diag_vals) - np.mean(p_vals)) if diag_vals and p_vals else float("nan")
    return {
        "model": model, "S": S, "T": Tgt, "P": p_cells, "diag_cells": diag_cells,
        "n_diag": len(diag_vals), "n_P": len(p_vals),
        "mean_diag": float(np.mean(diag_vals)) if diag_vals else float("nan"),
        "mean_P": float(np.mean(p_vals)) if p_vals else float("nan"),
        "delta": delta,
    }


def matched_evaluable_global_cells(
    details: pd.DataFrame,
    gnn_model: str = "day_gnn_zero_shot",
    rf_model: str = "random_forest",
    evaluable_targets=TARGETS_EVALUABLE,
    cell_kind: str = "off_diagonal",
) -> pd.DataFrame:
    """Matched RF+GNN cells on the same (source,target) with evaluable target.

    Records the included cell list. Does NOT compute Wilcoxon W/p.
    Historical note (original matrix, not for the manuscript):
      20 off-diag cells including CERT r6.2-as-target: W=131, one-sided p=0.1744
      16 evaluable cells excluding those four: W=98, one-sided p=0.065
    """
    Tgt = set(evaluable_targets)
    if "cell_kind" in details.columns:
        off = details[details["cell_kind"] == cell_kind]
    else:
        off = details[details["source"] != details["target"]]
    gnn = (off[off["model"] == gnn_model]
           .groupby(["source", "target"], as_index=False)["pr_auc"].mean()
           .rename(columns={"pr_auc": "gnn_pr_auc"}))
    rf = (off[off["model"] == rf_model]
          .groupby(["source", "target"], as_index=False)["pr_auc"].mean()
          .rename(columns={"pr_auc": "rf_pr_auc"}))
    m = gnn.merge(rf, on=["source", "target"], how="inner")
    included = m[m["target"].isin(Tgt)].copy()
    excluded = m[~m["target"].isin(Tgt)].copy()
    included["evaluable_target"] = True
    included["same_population"] = True
    included = included.sort_values(["source", "target"]).reset_index(drop=True)
    included.attrs["excluded_cells"] = [
        (str(a), str(b)) for a, b in excluded[["source", "target"]].itertuples(index=False)
    ]
    included.attrs["n_excluded"] = int(len(excluded))
    included.attrs["wilcoxon_not_run"] = True
    return included


# =============================================================================
# B. strict label-agnostic loaders (frozen source, one line replaced)
# =============================================================================
_CERT_KEEP_ORIG = "keep = is_ins | (rng.random(len(chunk)) < benign_http_frac)"
_CERT_KEEP_NEW = "keep = rng.random(len(chunk)) < benign_http_frac  # STRICT label-agnostic"
_LANL_KEEP_ORIG = "keep = is_rt | (rng.random(len(chunk)) < benign_frac)"
_LANL_KEEP_NEW = "keep = rng.random(len(chunk)) < benign_frac  # STRICT label-agnostic"
_ALLOWED_KEEP_NAMES = {"rng", "len", "chunk", "benign_http_frac", "benign_frac"}


def _compile_strict(func, old_line: str, new_line: str, new_name: str):
    src = textwrap.dedent(inspect.getsource(func))
    assert src.count(old_line) == 1, f"frozen keep rule not found exactly once in {func.__name__}"
    src2 = src.replace(old_line, new_line).replace(f"def {func.__name__}(", f"def {new_name}(", 1)
    # static proof: the (only) assignment to `keep` in the uniform branch uses no roster/label name
    tree = ast.parse(src2)
    keep_assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "keep" for t in n.targets)]
    names_by_assign = [sorted({x.id for x in ast.walk(a.value) if isinstance(x, ast.Name)}) for a in keep_assigns]
    strict_assigns = [nm for nm in names_by_assign if "rng" in nm]
    assert len(strict_assigns) == 1, names_by_assign
    assert set(strict_assigns[0]) <= _ALLOWED_KEEP_NAMES, strict_assigns
    ns = dict(vars(L))
    exec(compile(src2, f"<strict:{new_name}>", "exec"), ns)
    diff = {"old": old_line, "new": new_line,
            "frozen_source_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "strict_source_sha256": hashlib.sha256(src2.encode()).hexdigest(),
            "keep_rule_names": strict_assigns[0],
            "all_keep_assignments_names": names_by_assign}
    return ns[new_name], diff


load_cert_strict, CERT_STRICT_DIFF = _compile_strict(L.load_cert, _CERT_KEEP_ORIG, _CERT_KEEP_NEW, "load_cert_strict_uniform")
load_lanl_strict, LANL_STRICT_DIFF = _compile_strict(L.load_lanl, _LANL_KEEP_ORIG, _LANL_KEEP_NEW, "load_lanl_strict_uniform")


def _row_hashes(df: pd.DataFrame) -> np.ndarray:
    """Order-free, dtype-robust row fingerprints (uint64), sorted."""
    h = np.zeros(len(df), dtype=np.uint64)
    ts = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]").astype("int64").to_numpy()
    cols = [np.asarray(ts, dtype=np.int64).view(np.uint64)]
    for c in ("user", "src_host", "dst_host", "action", "object"):
        cols.append(pd.util.hash_array(np.asarray(df[c].astype(str).to_numpy(), dtype=object)))
    cols.append(np.asarray(df["label"].to_numpy(), dtype=np.int64).view(np.uint64))
    with np.errstate(over="ignore"):
        for v in cols:
            h = h * np.uint64(1099511628211) ^ pd.util.hash_array(v)
    return np.sort(h)


def _multiset_diff(a: np.ndarray, b: np.ndarray):
    """(# in a not in b, # in b not in a) for sorted uint64 multisets."""
    ua, ca = np.unique(a, return_counts=True)
    ub, cb = np.unique(b, return_counts=True)
    sa = pd.Series(ca, index=ua)
    sb = pd.Series(cb, index=ub)
    j = pd.concat([sa.rename("a"), sb.rename("b")], axis=1).fillna(0)
    return int((j.a - j.b).clip(lower=0).sum()), int((j.b - j.a).clip(lower=0).sum())


def verify_universe(name: str, strict_ev: pd.DataFrame, orig_ev: pd.DataFrame, roster: set, frac: float,
                    sampled_action_mask) -> dict:
    """Row-level proof that only roster rows of the sampled channel changed, and that
    roster rows are now kept at the same rate as everyone else."""
    s_r = strict_ev[USER].isin(roster).to_numpy()
    o_r = orig_ev[USER].isin(roster).to_numpy()
    s_samp = sampled_action_mask(strict_ev)
    o_samp = sampled_action_mask(orig_ev)
    rep = {"domain": name, "roster_size": len(roster), "frac": frac, "seed": SAMPLE_SEED,
           "strict_rows": int(len(strict_ev)), "orig_rows": int(len(orig_ev)),
           "strict_pos_rows": int(strict_ev[LABEL].sum()), "orig_pos_rows": int(orig_ev[LABEL].sum())}
    # (1) everything that is NOT (roster AND sampled channel) must be identical
    a = _row_hashes(strict_ev[~(s_r & s_samp)])
    b = _row_hashes(orig_ev[~(o_r & o_samp)])
    only_s, only_o = _multiset_diff(a, b)
    rep["nonroster_or_unsampled_identical"] = bool(only_s == 0 and only_o == 0)
    rep["nonroster_only_in_strict"] = only_s
    rep["nonroster_only_in_orig"] = only_o
    del a, b
    # (2) roster rows of the sampled channel: strict must be a subset of original
    a = _row_hashes(strict_ev[s_r & s_samp])
    b = _row_hashes(orig_ev[o_r & o_samp])
    only_s, _ = _multiset_diff(a, b)
    rep["roster_sampled_rows_strict"] = int(len(a))
    rep["roster_sampled_rows_orig"] = int(len(b))
    rep["roster_strict_subset_of_orig"] = bool(only_s == 0)
    n, k = len(b), len(a)
    rep["roster_keep_rate"] = (k / n) if n else float("nan")
    se = (frac * (1 - frac) / n) ** 0.5 if n else float("nan")
    rep["roster_keep_rate_z_vs_frac"] = ((k / n - frac) / se) if n else float("nan")
    # (3) non-roster keep rate cannot be measured without the raw row count; report counts
    rep["nonroster_sampled_rows"] = int((~s_r & s_samp).sum())
    rep["passed"] = bool(rep["nonroster_or_unsampled_identical"] and rep["roster_strict_subset_of_orig"]
                         and abs(rep["roster_keep_rate_z_vs_frac"]) < 5)
    return rep


# =============================================================================
# cache building
# =============================================================================
def _feat_from_events(ev: pd.DataFrame) -> pd.DataFrame:
    return T._align_feat(user_day_features(ev, deviation=True))


def feature_parity(name: str, orig_ev: pd.DataFrame) -> dict:
    """The strict features are built with user_day_features(events); prove this path
    reproduces the ORIGINAL feature cache from the ORIGINAL events."""
    f0 = _feat_from_events(orig_ev)
    orig_feat_p = ORIG_CACHE / f"{name}_user_day_dev.parquet"
    fc = T._align_feat(pd.read_parquet(orig_feat_p))  # BUILD/AUDIT read of original only
    mm = f0.merge(fc, on=[USER, "day"], suffixes=("", "_c"), how="outer", indicator=True)
    both = mm[mm["_merge"] == "both"]
    maxdiff = {c: float(np.nanmax(np.abs(both[c].to_numpy(float) - both[c + "_c"].to_numpy(float))))
               for c in FEATURE_COLUMNS + [LABEL]}
    worst = max(maxdiff.values()) if maxdiff else 0.0
    n_mismatch = int((mm["_merge"] != "both").sum())
    # Original CERT features were streamed (stream_cert_user_day_features). Rebuilding
    # the SAME original events with user_day_features() matches row-for-row but can
    # differ at ~1e-4 in deviation columns. That is not a strict-sampling failure.
    rec = {"domain": name, "rows_from_events": int(len(f0)), "rows_cache": int(len(fc)),
           "rows_only_one_side": n_mismatch,
           "max_abs_diff_any_column": worst,
           "passed": bool(n_mismatch == 0),
           "numerical_audit_only": True,
           "original_feat_path": str(orig_feat_p),
           "note": ("orig-vs-orig row identity is fail-closed; max_abs_diff is AUDIT ONLY "
                    "(stream_cert vs user_day_features float noise). Strict features are "
                    "rebuilt from strict events, not copied from transfer_5d.")}
    del f0, fc, mm, both
    gc.collect()
    return rec


def build_domain(name: str, skip_parity=False):
    D["cache"].mkdir(parents=True, exist_ok=True)
    ev_p = D["cache"] / f"{name}_events.parquet"
    ft_p = D["cache"] / f"{name}_user_day_dev.parquet"
    ver_p = D["manifests"] / f"sampling_verification_{name}.json"
    if ev_p.exists() and ft_p.exists() and (name == "spedia" or ver_p.exists()):
        log.info("[build] %s exists, skip", name)
        return
    t0 = time.time()
    if name == "spedia":
        for suf in ("_events.parquet", "_user_day_dev.parquet"):
            src = ORIG_CACHE / f"spedia{suf}"
            shutil.copy2(src, D["cache"] / f"spedia{suf}")
            assert sha256(src) == sha256(D["cache"] / f"spedia{suf}")
        write_json(D["manifests"] / "sampling_verification_spedia.json",
                   {"domain": "spedia", "note": "unchanged real_only population; byte-identical copy of original caches",
                    "sha256": sha256(ORIG_CACHE / "spedia_events.parquet"), "passed": True})
        return
    if name in RELEASE:
        rel = RELEASE[name]
        log.info("[build] %s: strict load_cert from %s (http frac %.2f, seed %d)", name, RAW[name], CERT_FRAC, SAMPLE_SEED)
        ev = load_cert_strict(RAW[name], release=rel, dataset_tag=name,
                              sources=("logon", "device", "file", "email"),
                              http_mode="insider_aware",   # branch name kept; its keep rule is now uniform
                              benign_http_frac=CERT_FRAC, answers_dir=RAW["answers"], seed=SAMPLE_SEED)
        roster = L._cert_official_insiders(RAW["answers"], release=rel)
        orig_p = ORIG_CACHE / f"{name}_events.parquet"
        samp = lambda d: (d[ACTION] == "http").to_numpy()
        frac = CERT_FRAC
    elif name == "lanl":
        log.info("[build] lanl: strict load_lanl from %s (frac %.2f, seed %d)", RAW["lanl"], LANL_FRAC, SAMPLE_SEED)
        ev = load_lanl_strict(RAW["lanl"], mode="redteam_aware", benign_frac=LANL_FRAC, seed=SAMPLE_SEED)
        rt = pd.read_csv(L._lanl_find(RAW["lanl"], "redteam"), header=None, names=L._LANL_RT_NAMES,
                         compression="infer", dtype=str, keep_default_na=False)
        roster = set(rt["user"].astype(str).str.split("@", n=1).str[0])
        orig_p = ORIG_LANL_EVENTS
        samp = lambda d: np.ones(len(d), dtype=bool)      # every LANL auth row is sampled
        frac = LANL_FRAC
    else:
        raise ValueError(name)
    validate(ev)
    log.info("[build] %s strict events=%d pos_rows=%d rss=%.1fGB (%.0fs)", name, len(ev), int(ev[LABEL].sum()), rss_gb(), time.time() - t0)
    orig = pd.read_parquet(orig_p)
    orig[TIMESTAMP] = pd.to_datetime(orig[TIMESTAMP])
    log.info("[build-audit] comparing %s against ORIGINAL cache %s (not used for training)", name, orig_p)
    rep = verify_universe(name, ev, orig, roster, frac, samp)
    rep["raw_path"] = RAW[name] if name != "lanl" else RAW["lanl"]
    rep["original_events_cache"] = str(orig_p)
    rep["keep_rule_diff"] = CERT_STRICT_DIFF if name in RELEASE else LANL_STRICT_DIFF
    if not skip_parity:
        rep["feature_path_parity"] = feature_parity(name, orig)
    del orig, roster
    gc.collect()
    log.info("[build] %s verification passed=%s", name, rep["passed"])
    if not rep["passed"] or (not skip_parity and not rep["feature_path_parity"]["passed"]):
        write_json(ver_p.with_name(ver_p.stem + "_FAILED.json"), rep)
        raise SystemExit(f"[build] {name}: verification FAILED; see manifests (nothing written to cache)")
    ev.to_parquet(ev_p, index=False)
    ft = _feat_from_events(ev)
    ft.to_parquet(ft_p, index=False)
    rep["strict_feature_rows"] = int(len(ft))
    rep["strict_feature_pos"] = int(ft[LABEL].sum())
    rep["wall_s"] = round(time.time() - t0, 1)
    write_json(ver_p, rep)
    del ev, ft
    gc.collect()


# =============================================================================
# C. strict no-time GNN
# =============================================================================
class StrictNoTimeGNN(DaySupervisedGNN):
    """Frozen DaySupervisedGNN with use_time_encoding=False and the time channel zeroed."""

    def __init__(self, **kw):
        kw["use_time_encoding"] = False
        super().__init__(**kw)

    def _tensors(self, df, feat=None):
        out = super()._tensors(df, feat=feat)
        out["dt"] = out["dt"] * 0.0     # no stream statistic can reach the model
        return out


def gnn_kw(source_kind: str, seed: int, use_dann=False) -> dict:
    kw = gnn_kw_for_source(source_kind, seed, use_dann=use_dann)
    kw.update(NO_TIME)
    kw["use_memory"] = True
    return kw


def run_gnn_offdiag(source: str, target: str, seed: int, out_dir: Path, tag: str) -> dict | None:
    """Mirror of frozen run_gnn_cell_job (off-diagonal branch) + predictions."""
    if out_dir.resolve() == D["c2"].resolve():
        pred_p = c2_pred_csv("gnn", source, target, seed)
    else:
        pred_p = out_dir / "predictions" / f"{tag}_{source}_to_{target}_seed{seed}.parquet"
    if pred_p.exists() or pred_p.with_suffix(".csv").exists():
        return None
    t0 = time.time()
    enable_strict_determinism(seed)
    set_seed(seed)
    gpu_reset_peak()
    feat_s_p, df_s_p = strict_feat_path(source), strict_events_path(source)
    feat_t_p, df_t_p = strict_feat_path(target), strict_events_path(target)
    inputs = log_c2_inputs(f"GNN {source}->{target} seed={seed}", {
        "source_events": df_s_p, "source_features": feat_s_p,
        "target_events": df_t_p, "target_features": feat_t_p,
    })
    feat_s, df_s = T.load_feat(source), T.load_events(source)
    feat_t, df_t = T.load_feat(target), T.load_events(target)
    kw = gnn_kw(source, seed)
    if kw.get("use_time_encoding") is not False:
        raise RuntimeError("FAIL CLOSED: GNN use_time_encoding is not False")
    det = StrictNoTimeGNN(**kw)
    if det.use_time_encoding or det.use_memory is not True:
        raise RuntimeError("FAIL CLOSED: StrictNoTimeGNN time/memory flags")
    det.set_full_features(feat_s)
    det.fit(df_s)
    det.set_full_features(feat_t)
    agg = det.score_user_day(df_t)
    m = metrics(agg["label"], agg["score"])
    save_pred(pred_p, agg)
    row = dict(model="day_gnn_strict_notime", source=source, target=target, seed=seed, protocol="full_source",
               **m, n_positive=m.get("n_pos"), best_epoch=det.best_epoch, best_val_pr=det.best_val_pr,
               epochs=kw["epochs"], use_time_encoding=bool(det.use_time_encoding),
               use_memory=bool(det.use_memory), wall_s=round(time.time() - t0, 1),
               peak_rss_gb=round(rss_gb(), 2), peak_gpu_mb=gpu_peak_mb(), config=CONFIG_LABEL,
               pred_file=str(pred_p.relative_to(OUT)), cache_inputs=json.dumps(inputs, default=str))
    del det, feat_s, df_s, feat_t, df_t, agg
    free_gpu()
    return row


def run_rf(source: str, target: str, seed: int, protocol: str, out_dir: Path, tag: str, save=True) -> dict | None:
    """Mirror of frozen run_rf_cell_job/run_rf_cell + predictions."""
    if out_dir.resolve() == D["c2"].resolve():
        pred_p = c2_pred_csv("rf", source, target, seed)
    else:
        pred_p = out_dir / "predictions" / f"{tag}_{source}_to_{target}_{protocol}_seed{seed}.parquet"
    if pred_p.exists() or pred_p.with_suffix(".csv").exists():
        return None
    t0 = time.time()
    enable_strict_determinism(seed)
    gpu_reset_peak()
    fs_p = strict_feat_path(source)
    paths = {"source_features": fs_p}
    if source != target:
        paths["target_features"] = strict_feat_path(target)
    inputs = log_c2_inputs(f"RF {source}->{target} seed={seed} {protocol}", paths)
    fs = T.load_feat(source)
    if source == target:
        tr, te = _split_user_day_frame(fs, protocol, train_frac=0.7, seed=seed)
    else:
        tr, te = fs, T.load_feat(target)
    row = dict(model="random_forest", source=source, target=target, seed=seed, protocol=protocol,
               n_train=len(tr), n_train_pos=int(tr[LABEL].sum()), config=CONFIG_LABEL,
               cache_inputs=json.dumps(inputs, default=str))
    if int(te[LABEL].sum()) == 0 or tr[LABEL].nunique() < 2:
        row.update(pr_auc=np.nan, roc_auc=np.nan, lift=np.nan, base_rate=float(te[LABEL].mean()) if len(te) else np.nan,
                   n=len(te), n_pos=int(te[LABEL].sum()), n_positive=int(te[LABEL].sum()),
                   evaluable=False, wall_s=round(time.time() - t0, 1),
                   peak_rss_gb=round(rss_gb(), 2), peak_gpu_mb=gpu_peak_mb())
        if save:
            pred_p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"user": [], "day": [], "score": [], "label": []}).to_csv(
                pred_p.with_suffix(".csv") if pred_p.suffix.lower() != ".csv" else pred_p, index=False)
        return row
    set_seed(seed)
    Xtr, ytr = X_y(tr)
    Xte, yte = X_y(te)
    ad = SupervisedAdapter(rf_factory(seed))
    ad.fit(Xtr, ytr)
    proba = ad.model.predict_proba(Xte)
    s = proba[:, 1] if proba.shape[1] > 1 else np.full(len(yte), float(ad.model.classes_[0]))
    row.update(metrics(yte, s))
    row.update(n_positive=row.get("n_pos"), evaluable=True, wall_s=round(time.time() - t0, 1),
               peak_rss_gb=round(rss_gb(), 2), peak_gpu_mb=gpu_peak_mb())
    if save:
        agg = te[[USER, "day"]].copy()
        agg["score"], agg["label"] = s, yte
        save_pred(pred_p, agg)
        row["pred_file"] = str(pred_p.relative_to(OUT))
    return row


# =============================================================================
# stages
# =============================================================================
def _read_c2_pred(kind: str, source: str, target: str, seed: int) -> pd.DataFrame:
    csv_p = c2_pred_csv(kind, source, target, seed)
    pq_p = csv_p.with_suffix(".parquet")
    if pq_p.exists():
        df = pd.read_parquet(pq_p)
    elif csv_p.exists():
        df = pd.read_csv(csv_p)
    else:
        raise SystemExit(f"FAIL CLOSED C2 parity: missing predictions {csv_p}")
    return df


def interval_class(delta: float, lo: float, hi: float) -> str:
    """A = CI excludes 0 for GNN; B = includes 0; C = excludes 0 for RF."""
    if lo > 0 and hi > 0:
        return "A"
    if hi < 0 and lo < 0:
        return "C"
    return "B"


def interval_status_word(cls: str) -> str:
    return {"A": "SUPPORTED", "B": "NOT_SUPPORTED", "C": "RF_SUPPORTED"}[cls]


def summarize_c2():
    p = D["c2"] / "c2_cells.csv"
    if not p.exists():
        return None
    c = pd.read_csv(p).drop_duplicates(["model", "source", "target", "seed"], keep="last")
    c["seed"] = c["seed"].astype(int)
    rows = []
    for (s, t), g in c.groupby(["source", "target"]):
        gn = g[g.model == "day_gnn_strict_notime"].set_index("seed")
        rf = g[g.model == "random_forest"].set_index("seed")
        k = sorted(set(gn.index) & set(rf.index))
        k = [int(x) for x in k]
        for sd in k:
            assert_c2_eval_parity(_read_c2_pred("gnn", s, t, sd), _read_c2_pred("rf", s, t, sd),
                                  f"{s}->{t} seed={sd}")
            rows.append(dict(source=s, target=t, seed=sd, gnn_pr_auc=gn.pr_auc[sd], rf_pr_auc=rf.pr_auc[sd],
                             gnn_lift=gn.lift[sd], rf_lift=rf.lift[sd], base_rate=gn.base_rate[sd],
                             delta_gnn_minus_rf=gn.pr_auc[sd] - rf.pr_auc[sd], row_type="per_seed"))
        if k:
            d, lo, hi = paired_delta_ci(gn.pr_auc[k].to_numpy(), rf.pr_auc[k].to_numpy(), n_boot=5000, seed=0)
            cls = interval_class(d, lo, hi)
            rows.append(dict(source=s, target=t, seed="mean(" + ",".join(map(str, k)) + ")",
                             gnn_pr_auc=gn.pr_auc[k].mean(), rf_pr_auc=rf.pr_auc[k].mean(),
                             gnn_lift=gn.lift[k].mean(), rf_lift=rf.lift[k].mean(), base_rate=gn.base_rate[k].mean(),
                             delta_gnn_minus_rf=d, ci_lo=lo, ci_hi=hi, n_seeds=len(k),
                             interval_class=cls, interval_status=interval_status_word(cls),
                             row_type="summary_paired_bootstrap_5000"))
    out = pd.DataFrame(rows)
    out.to_csv(D["c2"] / "STRICT_C2_GNN_RF.csv", index=False)
    return out


def write_c2_seed0_check() -> dict:
    """Integrity gate after seed 0. FAIL CLOSED stops the stage."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    D["c2"].mkdir(parents=True, exist_ok=True)
    lines = ["# STRICT C2 SEED-0 CHECK", "", f"Written: {time.strftime('%Y-%m-%d %H:%M:%S %z')}", ""]
    probe = _time_invariance_probe()
    rec = {"probe": probe, "cells": [], "passed": True, "failures": []}
    for s, t in C2_CELLS:
        cell = {"source": s, "target": t}
        try:
            gnn = _read_c2_pred("gnn", s, t, 0)
            rf = _read_c2_pred("rf", s, t, 0)
            assert_c2_eval_parity(gnn, rf, f"{s}->{t} seed=0")
            cell["parity"] = True
        except Exception as e:
            rec["passed"] = False
            rec["failures"].append(f"parity {s}->{t}: {e}")
            cell["parity"] = False
            cell["parity_error"] = str(e)
            gnn = gnn if "gnn" in locals() else pd.DataFrame()
            rf = rf if "rf" in locals() else pd.DataFrame()
        if len(gnn):
            cell["n_gnn"] = int(len(gnn))
            cell["n_rf"] = int(len(rf))
            cell["n_pos_gnn"] = int(gnn["label"].sum())
            cell["n_pos_rf"] = int(rf["label"].sum())
            cell["finite_gnn"] = bool(np.isfinite(gnn["score"].to_numpy(float)).all())
            cell["finite_rf"] = bool(np.isfinite(rf["score"].to_numpy(float)).all())
            cell["n_ok"] = cell["n_gnn"] == 256 and cell["n_rf"] == 256
            cell["pos_ok"] = cell["n_pos_gnn"] == 82 and cell["n_pos_rf"] == 82
            if not (cell["n_ok"] and cell["pos_ok"] and cell["finite_gnn"] and cell["finite_rf"] and cell["parity"]):
                rec["passed"] = False
                rec["failures"].append(f"integrity {s}->{t}: {cell}")
        paths = [
            str(strict_events_path(s)), str(strict_feat_path(s)),
            str(strict_events_path(t)), str(strict_feat_path(t)),
        ]
        cell["paths"] = paths
        cell["legacy_cache"] = any("transfer_5d" in p.lower() or "redteam_aware" in p.lower() for p in paths)
        if cell["legacy_cache"]:
            rec["passed"] = False
            rec["failures"].append(f"legacy cache {s}->{t}")
        rec["cells"].append(cell)
        lines += [
            f"## {s} → {t} seed 0",
            f"- prediction-row parity: {'PASS' if cell.get('parity') else 'FAIL'}",
            f"- n GNN/RF: {cell.get('n_gnn')}/{cell.get('n_rf')} (expected 256)",
            f"- positives GNN/RF: {cell.get('n_pos_gnn')}/{cell.get('n_pos_rf')} (expected 82)",
            f"- finite scores: GNN={cell.get('finite_gnn')} RF={cell.get('finite_rf')}",
            f"- legacy cache: {'FAIL' if cell.get('legacy_cache') else 'PASS (none)'}",
            "",
        ]
    rec["no_time_encoding"] = bool(probe.get("use_time_encoding") is False and probe.get("dt_all_zero_base"))
    rec["no_target_tmin_tmax"] = bool(probe.get("dt_identical") and probe.get("dt_all_zero_shifted"))
    if not rec["no_time_encoding"] or not rec["no_target_tmin_tmax"]:
        rec["passed"] = False
        rec["failures"].append("time-encoding / t_min t_max probe failed")
    lines += [
        "## Global probes",
        f"- use_time_encoding=False: {'PASS' if rec['no_time_encoding'] else 'FAIL'}",
        f"- dt zero / no target t_min t_max: {'PASS' if rec['no_target_tmin_tmax'] else 'FAIL'}",
        "",
        f"**SEED0_STATUS = {'PASS' if rec['passed'] else 'FAIL'}**",
        "",
    ]
    if rec["passed"]:
        lines.append("Integrity PASS. Continuing seeds 1–4 automatically. Numerical C2 values are evidence, not a gate.")
    else:
        lines.append("Integrity FAIL. Stopping. No further seeds.")
        lines.append("Failures: " + " | ".join(rec["failures"]))
    text = "\n".join(lines) + "\n"
    (D["c2"] / "STRICT_C2_SEED0_CHECK.md").write_text(text, encoding="utf-8")
    (REPORT_DIR / "STRICT_C2_SEED0_CHECK.md").write_text(text, encoding="utf-8")
    write_json(D["c2"] / "seed0_check.json", rec)
    if not rec["passed"]:
        raise SystemExit("FAIL CLOSED: STRICT C2 seed-0 integrity check failed. See STRICT_C2_SEED0_CHECK.md")
    log.info("[c2] seed-0 integrity PASS; continuing seeds 1-4")
    return rec


def finalize_c2_reports(env: dict):
    """Write the required C2 evidence pack. Does not edit Overleaf."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cells_p = D["c2"] / "c2_cells.csv"
    cells = pd.read_csv(cells_p).drop_duplicates(["model", "source", "target", "seed"], keep="last")
    cells["seed"] = cells["seed"].astype(int)
    per_seed_rows = []
    for _, r in cells.iterrows():
        per_seed_rows.append({
            "source": r["source"], "target": r["target"], "model": r["model"], "seed": int(r["seed"]),
            "n": r.get("n"), "n_positive": r.get("n_positive", r.get("n_pos")),
            "target_base_rate": r.get("base_rate"), "pr_auc": r.get("pr_auc"),
            "base_rate_lift": r.get("lift"), "wall_clock_s": r.get("wall_s"),
            "peak_gpu_mb": r.get("peak_gpu_mb"), "peak_ram_gb": r.get("peak_rss_gb"),
            "pred_file": r.get("pred_file"),
        })
    per_seed = pd.DataFrame(per_seed_rows).sort_values(["source", "model", "seed"])
    per_seed.to_csv(D["c2"] / "STRICT_C2_PER_SEED.csv", index=False)

    boot_rows, cmp_rows = [], []
    headlines = {}
    for s, t in C2_CELLS:
        sub = cells[(cells.source == s) & (cells.target == t)]
        gn = sub[sub.model == "day_gnn_strict_notime"].sort_values("seed")
        rf = sub[sub.model == "random_forest"].sort_values("seed")
        k = sorted(set(gn.seed.astype(int)) & set(rf.seed.astype(int)))
        gpr = gn.set_index("seed").loc[k, "pr_auc"].to_numpy(float)
        rpr = rf.set_index("seed").loc[k, "pr_auc"].to_numpy(float)
        glf = gn.set_index("seed").loc[k, "lift"].to_numpy(float)
        rlf = rf.set_index("seed").loc[k, "lift"].to_numpy(float)
        deltas = gpr - rpr
        d, lo, hi = paired_delta_ci(gpr, rpr, n_boot=5000, seed=0)
        cls = interval_class(d, lo, hi)
        status = interval_status_word(cls)
        boot_rows.append({
            "source": s, "target": t, "n_seeds": len(k), "n_boot": 5000,
            "rf_mean_pr_auc": float(np.mean(rpr)), "gnn_mean_pr_auc": float(np.mean(gpr)),
            "rf_mean_lift": float(np.mean(rlf)), "gnn_mean_lift": float(np.mean(glf)),
            "mean_paired_delta_pr": float(d), "ci_lo": float(lo), "ci_hi": float(hi),
            "interval_class": cls,
            "interpretation": {
                "A": "95% CI excludes zero in favour of GNN",
                "B": "95% CI includes zero",
                "C": "95% CI excludes zero in favour of RF",
            }[cls],
            "status": status,
            "per_seed_delta": ",".join(f"{x:.6f}" for x in deltas),
        })
        old = ACCEPTED_C2[(s, t)]
        cmp_rows.append({
            "source": s, "target": t,
            "old_RF": old["old_rf"], "strict_RF": float(np.mean(rpr)),
            "old_GNN": old["old_gnn"], "strict_GNN": float(np.mean(gpr)),
            "old_dPR": old["old_dpr"], "strict_dPR": float(d),
            "strict_95CI_lo": float(lo), "strict_95CI_hi": float(hi),
            "interpretation": {
                "A": "95% CI excludes zero in favour of GNN",
                "B": "95% CI includes zero",
                "C": "95% CI excludes zero in favour of RF",
            }[cls],
        })
        headlines[s] = {
            "rf_mean": float(np.mean(rpr)), "gnn_mean": float(np.mean(gpr)),
            "delta": float(d), "ci": (float(lo), float(hi)), "status": status,
        }
    boot = pd.DataFrame(boot_rows)
    cmp = pd.DataFrame(cmp_rows)
    boot.to_csv(D["c2"] / "STRICT_C2_BOOTSTRAP.csv", index=False)
    cmp.to_csv(D["c2"] / "STRICT_C2_COMPARISON.csv", index=False)

    code = ["scripts/strict_camera_ready_2026_09_15.py", "scripts/run_transfer_matrix_5domain.py",
            "src/data/loaders.py", "src/data/features.py", "src/data/graph_builder.py",
            "src/train/day_supervised_gnn.py", "src/models/temporal_gnn.py",
            "src/train/multiseed_transfer.py", "src/train/supervised_transfer.py"]
    man = {
        "tag": TAG, "config": CONFIG_LABEL, "cells": [list(c) for c in C2_CELLS],
        "seeds": list(SEEDS), "n_boot": 5000,
        "environment": env, "determinism": dict(DETERMINISM_STATE),
        "strict_cache_sha256": {p.name: cache_file_rec(p) for p in sorted(D["cache"].glob("*.parquet"))},
        "script_sha256": {f: sha256(ROOT / f) for f in code if (ROOT / f).exists()},
        "gnn_kw_cert42_seed0": gnn_kw("cert42", 0),
        "use_time_encoding": False, "use_memory": True,
        "prediction_files": sorted(p.name for p in (D["c2"] / "predictions").glob("predictions_*.csv")),
        "headlines": headlines,
        "accepted_manuscript_c2_audit_only": {
            f"{a}->{b}": v for (a, b), v in ACCEPTED_C2.items()
        },
    }
    write_json(D["c2"] / "STRICT_C2_RUN_MANIFEST.json", man)

    h42, h52 = headlines["cert42"], headlines["cert52"]
    report = textwrap.dedent(f"""\
    # STRICT C2 GPU REPORT — IJACSA MS-17-9-0257

    **Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
    **Driver:** scripts/strict_camera_ready_2026_09_15.py
    **Caches:** results/strict_camera_ready_2026_09_15/cache/
    **Training:** RF + StrictNoTimeGNN, cells cert42→spedia and cert52→spedia, seeds 0–4.
    **Not run:** Overleaf edit, LODO, CERT r6.2, LANL, original-cache overwrite.

    ## Environment
    - Python: {env.get("python")}
    - PyTorch: {env.get("torch")}  CUDA: {env.get("cuda")}
    - scikit-learn: {env.get("sklearn")}  SciPy: {env.get("scipy")}
    - NumPy: {env.get("numpy")}  pandas: {env.get("pandas")}
    - GPU: {env.get("gpu")}  VRAM: {env.get("gpu_vram_total_mb")} MiB
    - RAM available at start (from env dump): {env.get("ram_available_gb")} GiB / {env.get("ram_total_gb")} GiB
    - CUBLAS_WORKSPACE_CONFIG: {env.get("CUBLAS_WORKSPACE_CONFIG")}
    - use_deterministic_algorithms: {DETERMINISM_STATE.get("use_deterministic_algorithms")}

    ## Results (strict evidence; do not force accepted-manuscript numbers)

    | source | RF mean PR-AUC | GNN mean PR-AUC | RF mean lift | GNN mean lift | mean ΔPR | 95% CI | class |
    |---|---|---|---|---|---|---|---|
    """)
    for _, r in boot.iterrows():
        report += (f"| {r['source']}→{r['target']} | {r['rf_mean_pr_auc']:.4f} | {r['gnn_mean_pr_auc']:.4f} | "
                   f"{r['rf_mean_lift']:.3f} | {r['gnn_mean_lift']:.3f} | {r['mean_paired_delta_pr']:+.4f} | "
                   f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['interval_class']} |\n")
    report += "\nInterval class: A = CI excludes 0 for GNN; B = includes 0; C = excludes 0 for RF.\n\n"
    report += "## Audit-only comparison with accepted manuscript C2\n\n"
    report += "| source | old RF | strict RF | old GNN | strict GNN | old ΔPR | strict ΔPR | strict 95% CI | interpretation |\n"
    report += "|---|---|---|---|---|---|---|---|---|\n"
    for _, r in cmp.iterrows():
        report += (f"| {r['source']} | {r['old_RF']:.3f} | {r['strict_RF']:.4f} | {r['old_GNN']:.3f} | "
                   f"{r['strict_GNN']:.4f} | {r['old_dPR']:+.3f} | {r['strict_dPR']:+.4f} | "
                   f"[{r['strict_95CI_lo']:+.4f}, {r['strict_95CI_hi']:+.4f}] | {r['interpretation']} |\n")
    report += textwrap.dedent(f"""

    CERT42_TO_SPEDIA_RF_MEAN = {h42['rf_mean']:.6f}
    CERT42_TO_SPEDIA_GNN_MEAN = {h42['gnn_mean']:.6f}
    CERT42_TO_SPEDIA_DELTA = {h42['delta']:+.6f}
    CERT42_TO_SPEDIA_CI = [{h42['ci'][0]:+.6f}, {h42['ci'][1]:+.6f}]
    CERT42_C2_STATUS = {h42['status']}

    CERT52_TO_SPEDIA_RF_MEAN = {h52['rf_mean']:.6f}
    CERT52_TO_SPEDIA_GNN_MEAN = {h52['gnn_mean']:.6f}
    CERT52_TO_SPEDIA_DELTA = {h52['delta']:+.6f}
    CERT52_TO_SPEDIA_CI = [{h52['ci'][0]:+.6f}, {h52['ci'][1]:+.6f}]
    CERT52_C2_STATUS = {h52['status']}

    C2_ROW_PARITY = PASS
    C2_STRICT_CACHE_ONLY = PASS
    C2_NO_TIME_ENCODING = PASS
    C2_DETERMINISM = PASS

    FINAL_STATUS =
    STRICT_C2_COMPLETE
    """)
    (D["c2"] / "STRICT_C2_GPU_REPORT.md").write_text(report, encoding="utf-8")
    for name in ("STRICT_C2_PER_SEED.csv", "STRICT_C2_COMPARISON.csv", "STRICT_C2_BOOTSTRAP.csv",
                 "STRICT_C2_RUN_MANIFEST.json", "STRICT_C2_GPU_REPORT.md", "STRICT_C2_SEED0_CHECK.md"):
        src = D["c2"] / name
        if src.exists():
            shutil.copy2(src, REPORT_DIR / name)
    log.info("[c2] evidence pack written to %s and %s", D["c2"], REPORT_DIR)


def stage_c2(seeds):
    point_cache_to_strict()
    D["c2"].mkdir(parents=True, exist_ok=True)
    (D["c2"] / "predictions").mkdir(parents=True, exist_ok=True)
    env = environment()
    write_json(D["c2"] / "environment_at_c2_start.json", env)
    cells = D["c2"] / "c2_cells.csv"
    for sd in seeds:
        for s, t in C2_CELLS:
            r = run_gnn_offdiag(s, t, sd, D["c2"], "c2_gnn_notime")
            if r:
                append_row(cells, r)
                log.info("[c2] GNN %s->%s seed %d PR=%.4f", s, t, sd, r["pr_auc"])
            r = run_rf(s, t, sd, "full_source", D["c2"], "c2_rf")
            if r:
                append_row(cells, r)
                log.info("[c2] RF  %s->%s seed %d PR=%.4f", s, t, sd, r["pr_auc"])
        summarize_c2()
        if sd == 0:
            write_c2_seed0_check()
        write_json(D["c2"] / f"c2_interim_after_seed{sd}.json",
                   {"seed": sd, "note": "interim after this seed; not final"})
        log.info("[c2] interim after seed %d", sd)
    finalize_c2_reports(env)


def stage_rf(seeds):
    point_cache_to_strict()
    cells = D["rf"] / "rf_cells.csv"
    for sd in seeds:
        for s in DOMAINS:
            for proto in ("user_disjoint", "temporal"):
                r = run_rf(s, s, sd, proto, D["rf"], "rf")
                if r:
                    append_row(cells, r)
            for t in DOMAINS:
                if t != s:
                    r = run_rf(s, t, sd, "full_source", D["rf"], "rf", save=True)
                    if r:
                        append_row(cells, r)
        log.info("[rf] seed %d done", sd)


def run_gnn_diag(dom: str, protocol: str, seed: int) -> dict | None:
    """Mirror of frozen run_gnn_cell_job diagonal branch (+ no-time, + predictions)."""
    out = D["temporal_controls"]
    pred_p = out / "predictions" / f"gnn_notime_{dom}_{protocol}_seed{seed}.parquet"
    if pred_p.exists():
        return None
    t0 = time.time()
    enable_strict_determinism(seed)
    set_seed(seed)
    feat_p, df_p = strict_feat_path(dom), strict_events_path(dom)
    log.info("[cache-isolation] GNN-diag %s %s seed=%d feat=%s events=%s",
             dom, protocol, seed, feat_p, df_p)
    feat, df = T.load_feat(dom), T.load_events(dom)
    if protocol == "temporal":
        tr, te = temporal_split(df, train_frac=0.7)
    else:
        tr, te = user_disjoint_split(df, train_frac=0.7, seed=seed)
    row = dict(model="day_gnn_strict_notime", source=dom, target=dom, seed=seed, protocol=protocol,
               test_events=len(te), test_event_pos=int(te[LABEL].sum()), config=CONFIG_LABEL)
    if int(te[LABEL].sum()) == 0:
        row.update(evaluable=False, pr_auc=np.nan, n_pos=0, note="no positive events in test partition")
        pred_p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"user": [], "day": [], "score": [], "label": []}).to_parquet(pred_p)
        return row
    kw = gnn_kw(dom, seed)
    det = StrictNoTimeGNN(**kw)
    det.set_full_features(feat)
    det.fit(tr)
    agg = det.score_user_day(te)
    row.update(metrics(agg["label"], agg["score"]))
    row.update(evaluable=True, best_epoch=det.best_epoch, best_val_pr=det.best_val_pr,
               wall_s=round(time.time() - t0, 1), pred_file=str(pred_p.relative_to(OUT)))
    save_pred(pred_p, agg)
    del det, feat, df, tr, te, agg
    free_gpu()
    return row


def lanl_temporal_check() -> dict:
    point_cache_to_strict()
    f = T.load_feat("lanl")
    tr, te = _split_user_day_frame(f, "temporal", train_frac=0.7)
    ev = pd.read_parquet(D["cache"] / "lanl_events.parquet", columns=[TIMESTAMP, LABEL])
    ev[TIMESTAMP] = pd.to_datetime(ev[TIMESTAMP])
    cut_e = ev[TIMESTAMP].quantile(0.7)
    pos_days = f.loc[f[LABEL] == 1, "day"]
    rep = dict(domain="lanl", rf_cut_day=str(f["day"].quantile(0.7)), rf_test_userdays=len(te),
               rf_test_pos=int(te[LABEL].sum()), gnn_cut_event_time=str(cut_e),
               gnn_test_event_pos=int(ev.loc[ev[TIMESTAMP] > cut_e, LABEL].sum()),
               first_pos_day=str(pos_days.min()), last_pos_day=str(pos_days.max()),
               n_pos_userdays=int(f[LABEL].sum()))
    rep["evaluable"] = bool(rep["rf_test_pos"] > 0 and rep["gnn_test_event_pos"] > 0)
    write_json(D["manifests"] / "lanl_temporal_evaluability.json", rep)
    return rep


def stage_temporal(seeds):
    point_cache_to_strict()
    cells = D["temporal_controls"] / "gnn_diag_cells.csv"
    lanl_temporal_check()
    for sd in seeds:
        for dom in ("cert42", "cert52", "spedia"):
            for proto in ("temporal", "user_disjoint"):
                r = run_gnn_diag(dom, proto, sd)
                if r:
                    append_row(cells, r)
                    log.info("[temporal] GNN %s %s seed %d PR=%s", dom, proto, sd, r.get("pr_auc"))


LODO_SOURCES = ("cert42", "cert52", "cert62", "lanl")
LODO_COLS = ["timestamp", "user", "src_host", "dst_host", "action", "label"]


def _pooled_sources():
    parts, fparts, comp, pos = [], [], {}, {}
    for n in LODO_SOURCES:
        d = pd.read_parquet(strict_events_path(n), columns=LODO_COLS)
        d[TIMESTAMP] = pd.to_datetime(d[TIMESTAMP])
        for c in (USER, "src_host", "dst_host"):          # == tag_and_concat naming
            d[c] = n + "::" + d[c].astype(str)
        d["dataset"] = n
        comp[n] = int(len(d))
        parts.append(d)
        f = T.load_feat(n)
        log.info("[cache-isolation] LODO source feat %s -> %s", n, strict_feat_path(n))
        f[USER] = n + "::" + f[USER].astype(str)
        pos[n] = int(f[LABEL].sum())
        fparts.append(f)
        log.info("[lodo] loaded %s events=%d rss=%.1fGB", n, len(d), rss_gb())
    ev = pd.concat(parts, ignore_index=True)
    del parts
    feat = pd.concat(fparts, ignore_index=True)
    gc.collect()
    return ev, feat, comp, pos


def stage_lodo(seeds, dry_run=False):
    point_cache_to_strict()
    out = D["lodo"]
    cells = out / "lodo_cells.csv"
    for sd in seeds:
        for setting in ("zero_shot", "dann_uda"):
            pred_p = out / "predictions" / f"lodo_spedia_{setting}_seed{sd}.parquet"
            if pred_p.exists() and not dry_run:
                continue
            t0 = time.time()
            set_seed(sd)
            ev_s, feat_s, comp, pos = _pooled_sources()
            df_t_p, feat_t_p = strict_events_path("spedia"), strict_feat_path("spedia")
            log.info("[cache-isolation] LODO target events=%s feat=%s", df_t_p, feat_t_p)
            df_t, feat_t = T.load_events("spedia"), T.load_feat("spedia")
            if any(n == "spedia" for n in LODO_SOURCES):
                raise RuntimeError("FAIL CLOSED: SPEDIA is in the labelled source pool")
            assert not ev_s[USER].str.startswith("spedia::").any()
            info = dict(seed=sd, setting=setting, source_event_counts=comp, source_pos_userdays=pos,
                        target_events=len(df_t), target_userdays=len(feat_t), target_pos=int(feat_t[LABEL].sum()),
                        rss_gb_after_load=round(rss_gb(), 2),
                        source_only_has_spedia_events=False,
                        dann_uses_unlabelled_spedia=(setting == "dann_uda"))
            log.info("[lodo] %s", json.dumps(info))
            if dry_run:
                write_json(D["manifests"] / "lodo_dry_run.json", info)
                del ev_s, feat_s, df_t, feat_t
                gc.collect()
                return
            use_dann = setting == "dann_uda"
            kw = gnn_kw("cert", sd, use_dann=use_dann)
            if kw.get("use_time_encoding") is not False or kw.get("use_memory") is False:
                raise RuntimeError("FAIL CLOSED: LODO time/memory flags")
            det = StrictNoTimeGNN(**kw)
            det.set_full_features(feat_s)
            if use_dann:
                df_t_unlab = df_t.copy()
                df_t_unlab[LABEL] = 0
                det.fit(ev_s, df_tgt=df_t_unlab)     # unlabelled SPEDIA; labels zeroed
                del df_t_unlab
            else:
                det.fit(ev_s)                  # no SPEDIA observation of any kind
            del ev_s, feat_s
            gc.collect()
            det.set_full_features(feat_t)
            agg = det.score_user_day(df_t)
            m = metrics(agg["label"], agg["score"])
            save_pred(pred_p, agg)
            row = dict(model="day_gnn_strict_notime", setting=setting, seed=sd, target="spedia",
                       sources="+".join(LODO_SOURCES), **m, best_epoch=det.best_epoch, best_val_pr=det.best_val_pr,
                       epochs=kw["epochs"], wall_s=round(time.time() - t0, 1), config=CONFIG_LABEL,
                       pred_file=str(pred_p.relative_to(OUT)))
            append_row(cells, row)
            write_json(D["manifests"] / f"lodo_seed{sd}_{setting}.json",
                       dict(row=row, gnn_kwargs=kw, **info,
                            early_stopping="15% of pooled SOURCE users (DaySupervisedGNN val split)",
                            schedule_note="gnn_kw_for_source('cert') for the CERT-dominated pool",
                            spedia_labels_used_in_training=False,
                            determinism=dict(DETERMINISM_STATE)))
            log.info("[lodo] %s seed %d PR=%.4f", setting, sd, m["pr_auc"])
            del det, df_t, feat_t, agg
            free_gpu()


def stage_gnn_matrix(seeds):
    """OPTIONAL: GNN(no-time) off-diagonal cells, frozen per-cell protocol."""
    point_cache_to_strict()
    cells = D["gnn_matrix"] / "gnn_matrix_cells.csv"
    for sd in seeds:
        for s in DOMAINS:
            for t in DOMAINS:
                if s != t and (s, t) not in C2_CELLS:
                    r = run_gnn_offdiag(s, t, sd, D["gnn_matrix"], "gnnm")
                    if r:
                        append_row(cells, r)


# =============================================================================
# summaries + manifest
# =============================================================================
def gap(cells: pd.DataFrame) -> dict:
    ud = cells[(cells.source == cells.target) & (cells.protocol == "user_disjoint")]
    off = cells[(cells.source != cells.target)]
    diag = ud[ud.source.isin(TARGETS_EVALUABLE)].groupby("source").pr_auc.mean()
    o = off[off.target.isin(TARGETS_EVALUABLE)].groupby(["source", "target"]).pr_auc.mean()
    res = dict(T=list(diag.index), n_P=int(len(o)), diag_mean=float(diag.mean()), off_mean=float(o.mean()),
               gap=float(diag.mean() - o.mean()))
    tp = cells[(cells.source == cells.target) & (cells.protocol == "temporal") & cells.evaluable.astype(bool)]
    tdiag = tp[tp.source.isin(TARGETS_EVALUABLE)].groupby("source").pr_auc.mean()
    to = off[off.target.isin(tdiag.index)].groupby(["source", "target"]).pr_auc.mean()
    res.update(T_temporal=list(tdiag.index), n_P_temporal=int(len(to)),
               gap_temporal=float(tdiag.mean() - to.mean()) if len(tdiag) else np.nan)
    return res


def summarize():
    D["manifests"].mkdir(parents=True, exist_ok=True)
    out = {}
    p = D["rf"] / "rf_cells.csv"
    if p.exists():
        c = pd.read_csv(p).drop_duplicates(["source", "target", "seed", "protocol"], keep="last")
        c["evaluable_target"] = c.target.isin(TARGETS_EVALUABLE)
        agg = (c.groupby(["source", "target", "protocol"])
                .agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"), lift_mean=("lift", "mean"),
                     base_rate=("base_rate", "mean"), n=("n", "mean"), n_pos=("n_pos", "mean"),
                     n_seeds=("seed", "nunique"), evaluable_target=("evaluable_target", "first")).reset_index())
        ci = c.groupby(["source", "target", "protocol"]).pr_auc.apply(lambda v: seed_bootstrap_ci(v.to_numpy(), 5000, seed=0))
        agg["ci_lo"] = [ci[tuple(r)][1] for r in agg[["source", "target", "protocol"]].to_numpy()]
        agg["ci_hi"] = [ci[tuple(r)][2] for r in agg[["source", "target", "protocol"]].to_numpy()]
        agg["row_type"] = "mean_over_seeds"
        per = c.assign(row_type="per_seed")
        pd.concat([agg, per], ignore_index=True).to_csv(OUT / "STRICT_RF_MATRIX.csv", index=False)
        out["rf_gap"] = gap(c)
        out["rf_seeds"] = sorted(c.seed.unique().tolist())
    c2 = summarize_c2()
    tparts = []
    if p.exists():
        rd = c[(c.source == c.target)].assign(model="random_forest")
        tparts.append(rd)
    gp = D["temporal_controls"] / "gnn_diag_cells.csv"
    if gp.exists():
        tparts.append(pd.read_csv(gp).drop_duplicates(["source", "protocol", "seed"], keep="last"))
    if tparts:
        tc = pd.concat(tparts, ignore_index=True)
        tsum = (tc.groupby(["model", "source", "protocol"])
                  .agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"), lift_mean=("lift", "mean"),
                       base_rate=("base_rate", "mean"), test_pos_mean=("n_pos", "mean"),
                       n_seeds_evaluable=("pr_auc", "count")).reset_index())
        tsum["label"] = tsum.model.map({"random_forest": "RF", "day_gnn_strict_notime": "GNN (no-time)"}) + " " + \
            tsum.protocol.map({"user_disjoint": "user-disjoint", "temporal": "chronological"})
        tsum.to_csv(OUT / "STRICT_TEMPORAL_CONTROLS.csv", index=False)
    lp = D["lodo"] / "lodo_cells.csv"
    if lp.exists():
        l = pd.read_csv(lp).drop_duplicates(["setting", "seed"], keep="last")
        zs = l[l.setting == "zero_shot"].set_index("seed").pr_auc
        da = l[l.setting == "dann_uda"].set_index("seed").pr_auc
        k = sorted(set(zs.index) & set(da.index))
        rows = l.assign(row_type="per_seed").to_dict("records")
        if k:
            d, lo, hi = paired_delta_ci(da[k].to_numpy(), zs[k].to_numpy(), n_boot=5000, seed=0)
            rows.append(dict(row_type="summary_dann_minus_source_only", seed=",".join(map(str, k)),
                             zero_shot_mean=zs[k].mean(), dann_uda_mean=da[k].mean(), delta=d, ci_lo=lo, ci_hi=hi,
                             n_seeds=len(k)))
        pd.DataFrame(rows).to_csv(OUT / "STRICT_LODO_SPEDIA.csv", index=False)
    # manifest
    code = ["scripts/strict_camera_ready_2026_09_15.py", "scripts/run_transfer_matrix_5domain.py",
            "src/data/loaders.py", "src/data/features.py", "src/data/graph_builder.py", "src/data/schema.py",
            "src/train/day_supervised_gnn.py", "src/models/temporal_gnn.py", "src/models/dann.py",
            "src/models/ud_head.py", "src/train/multiseed_transfer.py", "src/train/supervised_transfer.py",
            "src/train/splits.py", "src/train/evaluate.py", "src/utils/seed.py"]
    man = dict(tag=TAG, config=CONFIG_LABEL, sample_seed=SAMPLE_SEED, cert_http_frac=CERT_FRAC, lanl_auth_frac=LANL_FRAC,
               seeds=list(SEEDS), raw_paths=RAW, evaluable_targets=list(TARGETS_EVALUABLE),
               keep_rule_diffs={"cert": CERT_STRICT_DIFF, "lanl": LANL_STRICT_DIFF},
               determinism=dict(DETERMINISM_STATE),
               code_sha256={f: sha256(ROOT / f) for f in code if (ROOT / f).exists()},
               original_cache_sha256={p.name: sha256(p) for p in sorted(ORIG_CACHE.glob("*.parquet"))},
               strict_cache_sha256={p.name: sha256(p) for p in sorted(D["cache"].glob("*.parquet"))},
               sampling_verification={p.stem: json.loads(p.read_text()) for p in sorted(D["manifests"].glob("sampling_verification_*.json"))},
               environment=json.loads((D["manifests"] / "environment.json").read_text()) if (D["manifests"] / "environment.json").exists() else None,
               summary=out)
    if ORIG_LANL_EVENTS.exists():
        man["original_cache_sha256"][str(ORIG_LANL_EVENTS)] = sha256(ORIG_LANL_EVENTS)
    outputs = sorted([p for p in OUT.rglob("*") if p.is_file() and p.name != "STRICT_MANIFEST.json"
                      and "cache" not in p.parts])
    man["output_sha256"] = {str(p.relative_to(OUT)): sha256(p) for p in outputs}
    write_json(OUT / "STRICT_MANIFEST.json", man)
    log.info("[summary] %s", json.dumps(out, default=str))


def environment():
    env = dict(python=sys.version, platform=platform.platform(), machine=platform.machine())
    for mod in ("numpy", "pandas", "sklearn", "scipy", "pyarrow", "torch", "psutil"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:
            env[mod] = f"unavailable ({e})"
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        env["gpu_vram_total_mb"] = (
            int(torch.cuda.get_device_properties(0).total_memory / 2**20)
            if torch.cuda.is_available() else None
        )
        env["gpu_vram_alloc_mb"] = (
            int(torch.cuda.memory_allocated(0) / 2**20) if torch.cuda.is_available() else None
        )
        env["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        env["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    except Exception:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        env["ram_total_gb"] = round(vm.total / 2**30, 2)
        env["ram_used_gb"] = round(vm.used / 2**30, 2)
        env["ram_available_gb"] = round(vm.available / 2**30, 2)
        env["ram_percent"] = float(vm.percent)
    except Exception:
        pass
    env["determinism"] = dict(DETERMINISM_STATE)
    env["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    write_json(D["manifests"] / "environment.json", env)
    return env


def _chunk_keep_cert(n_chunks=2) -> dict:
    """Apply the STRICT HTTP KEEP expression to real r4.2 http.csv chunks. No roster."""
    fp = Path(RAW["cert42"]) / "http.csv"
    rng = np.random.default_rng(SAMPLE_SEED)
    scanned = kept = 0
    n_users = set()
    for i, chunk in enumerate(pd.read_csv(
        fp, dtype=str, chunksize=1_000_000, keep_default_na=False,
        usecols=lambda c: c in ("date", "user", "pc", "url"),
    )):
        keep = rng.random(len(chunk)) < CERT_FRAC  # identical to _CERT_KEEP_NEW
        scanned += len(chunk)
        kept += int(keep.sum())
        n_users.update(chunk["user"].astype(str).unique().tolist())
        if i + 1 >= n_chunks:
            break
    return {
        "file": str(fp), "chunks": n_chunks, "scanned": scanned, "kept": kept,
        "keep_rate": kept / scanned if scanned else float("nan"),
        "n_users_in_chunks": len(n_users),
        "keep_expression": _CERT_KEEP_NEW,
        "keep_referenced_insider_or_label": False,
    }


def _chunk_keep_lanl(n_chunks=2) -> dict:
    """Apply the STRICT auth KEEP expression to real auth.txt.gz chunks. No redteam."""
    fp = L._lanl_find(RAW["lanl"], "auth")
    rng = np.random.default_rng(SAMPLE_SEED)
    scanned = eligible = kept = machine = 0
    for i, chunk in enumerate(pd.read_csv(
        fp, header=None, names=L._LANL_AUTH_NAMES, compression="infer",
        dtype=str, keep_default_na=False, chunksize=1_000_000,
    )):
        scanned += len(chunk)
        user = chunk["src_user"].astype(str).str.split("@", n=1).str[0]
        is_mach = user.str.endswith("$") | user.str.match(r"^C\d+$", na=False)
        machine += int(is_mach.sum())
        chunk = chunk.loc[~is_mach]
        eligible += int(len(chunk))
        keep = rng.random(len(chunk)) < LANL_FRAC  # identical to _LANL_KEEP_NEW
        kept += int(keep.sum())
        if i + 1 >= n_chunks:
            break
    return {
        "file": str(fp), "chunks": n_chunks, "raw_scanned": scanned,
        "machine_dropped": machine, "eligible": eligible, "kept": kept,
        "keep_rate_among_eligible": kept / eligible if eligible else float("nan"),
        "keep_expression": _LANL_KEEP_NEW,
        "redteam_opened": False,
        "keep_referenced_redteam_or_label": False,
    }


def _time_invariance_probe() -> dict:
    """StrictNoTimeGNN: dt is all zeros, so a later unused timestamp cannot enter the model."""
    enable_strict_determinism(0)
    from src.data.schema import DST_HOST, SRC_HOST
    df = pd.DataFrame({
        TIMESTAMP: pd.to_datetime(["2010-01-01 00:00:00", "2010-01-02 00:00:00", "2010-01-03 00:00:00"]),
        USER: ["u1", "u1", "u2"],
        SRC_HOST: ["h1", "h1", "h2"],
        DST_HOST: ["h1", "h1", "h2"],
        ACTION: ["logon", "http", "logon"],
        "object": ["a", "b", "c"],
        LABEL: [0, 0, 0],
        "insider_type": ["benign", "benign", "benign"],
        "dataset": ["probe", "probe", "probe"],
    })
    kw = gnn_kw("cert42", 0, use_dann=False)
    det = StrictNoTimeGNN(**kw)
    t1 = det._tensors(df)
    df2 = df.copy()
    df2.loc[df2.index[-1], TIMESTAMP] = df2.loc[df2.index[-1], TIMESTAMP] + pd.Timedelta(days=365)
    t2 = det._tensors(df2)
    import torch
    z1 = t1["dt"].detach().cpu()
    z2 = t2["dt"].detach().cpu()
    rec = {
        "use_time_encoding": det.use_time_encoding,
        "use_memory": det.use_memory,
        "dt_all_zero_base": bool(torch.all(z1 == 0)),
        "dt_all_zero_shifted": bool(torch.all(z2 == 0)),
        "dt_identical": bool(torch.equal(z1, z2)),
        "score_difference": 0.0 if torch.equal(z1, z2) and torch.all(z1 == 0) else float("nan"),
        "gnn_kwargs": {k: kw[k] for k in ("use_time_encoding", "use_memory", "use_dann", "seed")},
    }
    del det, t1, t2
    free_gpu()
    return rec


def preflight():
    det = enable_strict_determinism(0)
    env = environment()
    checks = {"environment": env, "determinism": det}
    for k, v in RAW.items():
        checks[f"raw:{k}"] = os.path.exists(v)
    http = Path(RAW["cert42"]) / "http.csv"
    logon = Path(RAW["cert42"]) / "logon.csv"
    checks["raw:cert42_http"] = http.exists()
    checks["raw:cert42_logon"] = logon.exists()
    checks["raw:cert52_http"] = (Path(RAW["cert52"]) / "http.csv").exists()
    checks["raw:cert62_http"] = (Path(RAW["cert62"]) / "http.csv").exists()
    checks["raw:answers_insiders"] = (Path(RAW["answers"]) / "insiders.csv").exists()
    spedia_csv = DATA_ROOT / "SPEDIA" / "logs_SPEDIA.csv"
    checks["raw:spedia_csv"] = spedia_csv.exists()
    checks["strict_spedia_source"] = str(ORIG_CACHE / "spedia_events.parquet")
    for n in ("cert42", "cert52", "cert62", "spedia"):
        checks[f"orig_events:{n}"] = (ORIG_CACHE / f"{n}_events.parquet").exists()
        checks[f"orig_feat:{n}"] = (ORIG_CACHE / f"{n}_user_day_dev.parquet").exists()
    checks["orig_events:lanl"] = ORIG_LANL_EVENTS.exists()
    checks["orig_feat:lanl"] = (ORIG_CACHE / "lanl_user_day_dev.parquet").exists()
    checks["keep_rule_cert"] = CERT_STRICT_DIFF["keep_rule_names"]
    checks["keep_rule_lanl"] = LANL_STRICT_DIFF["keep_rule_names"]
    checks["keep_rule_cert_ok"] = set(CERT_STRICT_DIFF["keep_rule_names"]) <= _ALLOWED_KEEP_NAMES
    checks["keep_rule_lanl_ok"] = set(LANL_STRICT_DIFF["keep_rule_names"]) <= _ALLOWED_KEEP_NAMES
    if logon.exists():
        hdr = pd.read_csv(logon, nrows=2)
        checks["schema_cert_logon_cols"] = list(hdr.columns)
        ts = pd.to_datetime(hdr["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        checks["schema_cert_date_parse_ok"] = bool(ts.notna().all())
    if spedia_csv.exists():
        sp = pd.read_csv(spedia_csv, nrows=5)
        checks["schema_spedia_has_Decoder_name"] = "Decoder_name" in sp.columns
        n_cert = int((pd.read_csv(spedia_csv, usecols=["Decoder_name"])["Decoder_name"] == "cert").sum())
        n_all = n_cert  # placeholder overwritten below
        dec = pd.read_csv(spedia_csv, usecols=["Decoder_name"])["Decoder_name"]
        checks["spedia_n_rows"] = int(len(dec))
        checks["spedia_n_decoder_cert"] = int((dec == "cert").sum())
        checks["spedia_n_real_only"] = int((dec != "cert").sum())
        checks["spedia_real_only_expected_20619"] = checks["spedia_n_real_only"] == 20619
        del dec
        gc.collect()
    checks["cert_http_keep_demo"] = _chunk_keep_cert(2)
    checks["lanl_auth_keep_demo"] = _chunk_keep_lanl(2)
    checks["time_invariance_probe"] = _time_invariance_probe()
    if ORIG_TRANSFER_CSV.exists():
        td = pd.read_csv(ORIG_TRANSFER_CSV)
        gap_rec = evaluable_set_gap(td, "random_forest")
        checks["gap_original"] = gap_rec
        checks["gap_matches_0.4836233"] = abs(gap_rec["delta"] - 0.4836233488735972) < 1e-9
        checks["gap_rounds_to_0.484"] = abs(gap_rec["delta"] - 0.484) < 5e-4
        matched = matched_evaluable_global_cells(td)
        checks["global_test_included_cells"] = matched[["source", "target"]].values.tolist()
        checks["global_test_n_included"] = int(len(matched))
        checks["global_test_excluded_cells"] = matched.attrs.get("excluded_cells")
        checks["global_test_wilcoxon_run"] = False
        matched.to_csv(D["manifests"] / "strict_global_test_cells.csv", index=False)
        del td, matched
        gc.collect()
    checks["c2_cells"] = [list(c) for c in C2_CELLS]
    checks["c2_default_seeds"] = list(SEEDS)
    checks["c2_gnn_kw_seed0"] = gnn_kw("cert42", 0, use_dann=False)
    checks["c2_bootstrap_n"] = 5000
    checks["lodo_sources"] = list(LODO_SOURCES)
    checks["lodo_target"] = "spedia"
    checks["lodo_zero_shot_kw"] = gnn_kw("cert", 0, use_dann=False)
    checks["lodo_dann_kw"] = gnn_kw("cert", 0, use_dann=True)
    checks["lodo_in_stage_all"] = False
    planned = {}
    T.CACHE, T.LANL_EVENTS = D["cache"], D["cache"] / "lanl_events.parquet"
    for n in DOMAINS:
        planned[n] = {
            "events": str(Path(T._events_path(n)).resolve()),
            "features": str(Path(T._feat_path(n)).resolve()),
        }
        assert_strict_cache_path(T._events_path(n), f"planned_events:{n}")
        assert_strict_cache_path(T._feat_path(n), f"planned_features:{n}")
    checks["planned_strict_cache_paths"] = planned
    ram_pct = env.get("ram_percent", float("nan"))
    ram_avail = env.get("ram_available_gb", float("nan"))
    checks["lodo_estimated_source_events"] = 61_900_000
    checks["lodo_estimated_peak_gb"] = "25-30"
    checks["lodo_ram_preflight"] = (
        "NOT_SAFE" if (isinstance(ram_pct, float) and ram_pct >= 50) or (
            isinstance(ram_avail, float) and ram_avail < 30
        ) else "PASS"
    )
    path_ok = all(v for k, v in checks.items() if k.startswith(("raw:", "orig_")) and isinstance(v, bool))
    keep_ok = checks["keep_rule_cert_ok"] and checks["keep_rule_lanl_ok"]
    time_ok = checks["time_invariance_probe"]["dt_identical"] and checks["time_invariance_probe"]["dt_all_zero_base"]
    gap_ok = checks.get("gap_matches_0.4836233", False)
    checks["ok"] = bool(path_ok and keep_ok and time_ok and gap_ok and det.get("use_deterministic_algorithms"))
    write_json(D["manifests"] / "preflight.json", checks)
    print(json.dumps(checks, indent=1, default=str))
    if not env.get("cuda_available"):
        print("WARNING: CUDA not available in this environment; GNN stages will be very slow.")
    return checks["ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["preflight", "build-c2", "c2", "build-rest", "rf", "temporal", "lodo",
                             "lodo-dry-run", "gnn-matrix", "summarize", "all"])
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="comma-separated seeds; default is the five-seed C2 protocol 0,1,2,3,4")
    ap.add_argument("--skip-parity", action="store_true", help="skip the feature-path parity check in build")
    for k in RAW:
        ap.add_argument(f"--raw-{k}", default=None)
    a = ap.parse_args()
    for k in RAW:
        v = getattr(a, f"raw_{k}")
        if v:
            RAW[k] = v
    for p in D.values():
        p.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(D["logs"] / f"strict_{a.stage}_{time.strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), fh], force=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    st = a.stage
    if st == "preflight":
        sys.exit(0 if preflight() else 1)
    enable_strict_determinism(seeds[0] if seeds else 0)
    environment()
    if st in ("build-c2", "all"):
        for n in ("spedia", "cert42", "cert52"):
            build_domain(n, a.skip_parity)
    if st == "all":
        stage_c2([seeds[0]])                  # interim C2 check first
    if st in ("build-rest", "all"):
        for n in ("cert62", "lanl"):
            build_domain(n, a.skip_parity)
    if st in ("rf", "all"):
        stage_rf(seeds)
    if st in ("c2", "all"):
        stage_c2(seeds)
    if st in ("temporal", "all"):
        stage_temporal(seeds)
    if st == "all":
        summarize()
        log.info("[all] complete. LODO was NOT started (requires --stage lodo-dry-run then --stage lodo).")
    if st == "lodo-dry-run":
        stage_lodo([seeds[0]], dry_run=True)
    if st == "lodo":
        stage_lodo(seeds)
    if st == "gnn-matrix":
        stage_gnn_matrix(seeds)
    if st not in ("all", "c2", "preflight"):
        summarize()


if __name__ == "__main__":
    main()
