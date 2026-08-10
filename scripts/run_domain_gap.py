"""
Quantify distribution shift (proxy A-distance + MMD) across CERT + SPEDIA + LANL.

CPU-only (sklearn/numpy/pandas). Does not import torch / touch GNN training.

Usage:
  set PYTHONIOENCODING=utf-8
  set CUDA_VISIBLE_DEVICES=
  python -m scripts.run_domain_gap
  python -m scripts.run_domain_gap --seeds 0,1,2,3,4
  python -m scripts.run_domain_gap --domains spedia --cache-only
  python -m scripts.run_domain_gap --skip-load   # reuse cached feature matrices
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import sys
import time
from itertools import combinations
from pathlib import Path

# Force CPU before any accidental GPU libs; do not start CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.features import FEATURE_COLUMNS, user_day_features
from src.data.loaders import load_lanl, load_spedia
from src.data.schema import validate
from src.eval.domain_gap import pairwise_domain_gap, summarize_domain_gap
from src.eval.stream_cert_features import stream_cert_user_day_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("domain_gap")

CERT42 = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
CERT52 = r"C:\PhD\07_Projects\r5.2"
CERT62 = r"C:\PhD\07_Projects\r6.2"
ANSWERS = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2\answers"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"
LANL_DIR = r"C:\PhD\lanl"
LANL_EVENTS = Path(LANL_DIR) / "_cache_load_lanl_redteam_aware_bf002.parquet"

OUT = ROOT / "results"
CACHE = OUT / "cache" / "domain_gap"
TRANSFER_FEAT_CACHE = OUT / "cache" / "transfer_5d"

# Cap each domain to this many user-days (seed=0) before dA / MMD.
SUBSAMPLE_CAP = 50_000
SUBSAMPLE_SEED = 0

CERT_LOAD_KW = dict(
    sources=("logon", "device", "file", "email"),
    http_mode="insider_aware",
    benign_http_frac=0.05,
    answers_dir=ANSWERS,
)

RELEASE_PATHS = {
    "cert_r42": (CERT42, "4.2"),
    "cert_r52": (CERT52, "5.2"),
    "cert_r62": (CERT62, "6.2"),
}

DOMAIN_ORDER = ["cert_r42", "cert_r52", "cert_r62", "spedia", "lanl"]
NON_CERT = frozenset({"spedia", "lanl"})


def _parse_seeds(s: str):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")


def _parse_domains(s: str | None) -> list[str]:
    if not s:
        return list(DOMAIN_ORDER)
    names = [x.strip() for x in s.split(",") if x.strip()]
    bad = [n for n in names if n not in DOMAIN_ORDER]
    if bad:
        raise ValueError(f"unknown domains {bad}; choose from {DOMAIN_ORDER}")
    return names


def _require_paths_for(names: list[str]):
    missing = []
    for name in names:
        if name == "spedia":
            if not os.path.isfile(SPEDIA):
                missing.append(f"SPEDIA not found: {SPEDIA}")
            continue
        if name == "lanl":
            if not os.path.isdir(LANL_DIR):
                missing.append(f"LANL dir not found: {LANL_DIR}")
            continue
        p, rel = RELEASE_PATHS[name]
        if not os.path.isdir(p):
            missing.append(f"{name}: path not found: {p} (release={rel})")
        else:
            ok = any(
                os.path.isfile(os.path.join(p, f"{src}.csv"))
                for src in ("logon", "device", "file", "email")
            )
            if not ok:
                missing.append(f"{name}: no activity CSVs under {p}")
        if not os.path.isdir(ANSWERS):
            missing.append(f"answers_dir not found: {ANSWERS}")
    if missing:
        seen = set()
        uniq = []
        for m in missing:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        for m in uniq:
            log.error("MISSING: %s", m)
        raise FileNotFoundError(
            "Required datasets unavailable:\n  - " + "\n  - ".join(uniq)
        )
    log.info("Dataset paths OK for: %s", names)
    for name in names:
        if name == "spedia":
            log.info("  spedia -> %s", SPEDIA)
        elif name == "lanl":
            log.info("  lanl -> %s", LANL_DIR)
            if LANL_EVENTS.is_file():
                log.info("  lanl events cache -> %s", LANL_EVENTS)
        else:
            p, rel = RELEASE_PATHS[name]
            log.info("  %s -> %s (release=%s)", name, p, rel)
    if any(n not in NON_CERT for n in names):
        log.info("  answers -> %s", ANSWERS)


def _feat_cache_path(name: str) -> Path:
    return CACHE / f"{name}_user_day_dev.parquet"


def _save_features(name: str, feat: pd.DataFrame):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = _feat_cache_path(name)
    feat.to_parquet(path, index=False)
    log.info("Cached features %s -> %s (n=%d)", name, path, len(feat))


def _load_cached_features(name: str) -> pd.DataFrame | None:
    path = _feat_cache_path(name)
    if path.is_file():
        feat = pd.read_parquet(path)
        log.info("Loaded cached features %s n=%d from %s", name, len(feat), path)
        return feat
    return None


def _matrix_from_feat(feat: pd.DataFrame) -> np.ndarray:
    missing = [c for c in FEATURE_COLUMNS if c not in feat.columns]
    if missing:
        raise KeyError(f"feature columns missing: {missing[:5]}...")
    return feat[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)


def _subsample_matrix(
    X: np.ndarray,
    cap: int = SUBSAMPLE_CAP,
    seed: int = SUBSAMPLE_SEED,
) -> np.ndarray:
    """Fixed-seed row subsample (at most ``cap``) before dA / MMD."""
    n = len(X)
    if n <= cap:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=cap, replace=False)
    return X[idx]


def _load_lanl_features() -> pd.DataFrame:
    """Build LANL user-day features (shared FEATURE_COLUMNS, deviation=True).

    Prefer an existing feature parquet (domain_gap or transfer_5d). Otherwise
    reuse the load_lanl event cache if present; else call load_lanl(LANL_DIR).
    """
    alt = TRANSFER_FEAT_CACHE / "lanl_user_day_dev.parquet"
    if alt.is_file():
        log.info("Copying LANL features from transfer_5d cache %s", alt)
        CACHE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(alt, _feat_cache_path("lanl"))
        feat = pd.read_parquet(_feat_cache_path("lanl"))
        log.info("LANL features n=%d (from transfer_5d)", len(feat))
        return feat

    t0 = time.time()
    if LANL_EVENTS.is_file():
        log.info(
            "Loading LANL events from load_lanl cache %s ...", LANL_EVENTS
        )
        df = pd.read_parquet(LANL_EVENTS)
    else:
        log.info("Calling load_lanl(%r) (CPU/IO-heavy) ...", LANL_DIR)
        df = load_lanl(LANL_DIR)
    validate(df)
    log.info(
        "lanl events=%d pos_events=%d (%.1fs)",
        len(df), int(df["label"].sum()), time.time() - t0,
    )
    feat = user_day_features(df, deviation=True)
    log.info(
        "lanl user-days=%d pos_ud=%d",
        len(feat), int(feat["label"].sum()),
    )
    del df
    gc.collect()
    return feat


def build_or_load_features(
    skip_load: bool = False,
    domains: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return {domain: X} on FEATURE_COLUMNS (deviation=True)."""
    want = domains or list(DOMAIN_ORDER)
    matrices: dict[str, np.ndarray] = {}
    n_rows: dict[str, int] = {}

    all_cached = True
    cached_feats = {}
    for name in want:
        feat = _load_cached_features(name)
        if feat is None:
            all_cached = False
            break
        cached_feats[name] = feat

    if all_cached:
        log.info("Using cached feature matrices for domains=%s", want)
        for name, feat in cached_feats.items():
            matrices[name] = _matrix_from_feat(feat)
            n_rows[name] = len(feat)
        log.info("Feature row counts: %s", n_rows)
        return matrices

    def _can_satisfy(name: str) -> bool:
        if _feat_cache_path(name).is_file():
            return True
        # LANL features may already exist under transfer_5d (same builder).
        if name == "lanl" and (
            TRANSFER_FEAT_CACHE / "lanl_user_day_dev.parquet"
        ).is_file():
            return True
        return False

    if skip_load:
        missing = [n for n in want if not _can_satisfy(n)]
        if missing:
            raise FileNotFoundError(
                f"--skip-load set but cache missing for: {missing}. "
                f"Expected under {CACHE} (or transfer_5d for lanl)"
            )

    need_load = [n for n in want if _load_cached_features(n) is None]
    if need_load:
        # LANL may be satisfied from transfer_5d feature cache without raw paths.
        need_paths = [n for n in need_load if not _can_satisfy(n)]
        if need_paths:
            _require_paths_for(need_paths)

    for name in want:
        if name in NON_CERT:
            continue
        feat = _load_cached_features(name)
        if feat is not None:
            matrices[name] = _matrix_from_feat(feat)
            n_rows[name] = len(feat)
            continue

        path, release = RELEASE_PATHS[name]
        log.info(
            "Streaming %s from %s (insider_aware http, low-RAM) ...",
            name, path,
        )
        t0 = time.time()
        # Stream aggregates to avoid competing with concurrent GNN RAM use.
        # Same loader conventions (sources / http_mode / answers / seed).
        feat = stream_cert_user_day_features(
            path,
            release=release,
            answers_dir=ANSWERS,
            sources=CERT_LOAD_KW["sources"],
            http_mode=CERT_LOAD_KW["http_mode"],
            benign_http_frac=CERT_LOAD_KW["benign_http_frac"],
            seed=7,
        )
        log.info(
            "%s user-days=%d pos_ud=%d (%.1fs)",
            name, len(feat), int(feat["label"].sum()), time.time() - t0,
        )
        _save_features(name, feat)
        matrices[name] = _matrix_from_feat(feat)
        n_rows[name] = len(feat)
        del feat
        gc.collect()

    if "spedia" in want:
        name = "spedia"
        feat = _load_cached_features(name)
        if feat is None:
            log.info("Loading SPEDIA real_only from %s ...", SPEDIA)
            t0 = time.time()
            df = load_spedia(SPEDIA, real_only=True)
            validate(df)
            log.info(
                "spedia events=%d pos_events=%d (%.1fs)",
                len(df), int(df["label"].sum()), time.time() - t0,
            )
            feat = user_day_features(df, deviation=True)
            log.info(
                "spedia user-days=%d pos_ud=%d",
                len(feat), int(feat["label"].sum()),
            )
            _save_features(name, feat)
            del df
            gc.collect()
        matrices[name] = _matrix_from_feat(feat)
        n_rows[name] = len(feat)

    if "lanl" in want:
        name = "lanl"
        feat = _load_cached_features(name)
        if feat is None:
            feat = _load_lanl_features()
            _save_features(name, feat)
        matrices[name] = _matrix_from_feat(feat)
        n_rows[name] = len(feat)

    log.info("Feature row counts: %s", n_rows)
    return matrices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument(
        "--skip-load",
        action="store_true",
        help="only use cached parquet feature matrices (no CERT CSV load)",
    )
    ap.add_argument(
        "--domains",
        default="",
        help="comma-separated subset to load/cache, e.g. spedia,cert_r42,lanl",
    )
    ap.add_argument(
        "--cache-only",
        action="store_true",
        help="build/cache feature matrices only; do not run pairwise metrics",
    )
    ap.add_argument(
        "--subsample-cap",
        type=int,
        default=SUBSAMPLE_CAP,
        help="max user-days per domain (seed=0) before dA/MMD",
    )
    ap.add_argument("--domain-cap", type=int, default=20_000)
    ap.add_argument("--mmd-cap", type=int, default=4_000)
    args = ap.parse_args()
    seeds = _parse_seeds(args.seeds)
    domains = _parse_domains(args.domains or None)

    log.info("CPU-only domain-gap run | seeds=%s | domains=%s", seeds, domains)
    log.info("FEATURE_COLUMNS (%d): %s...", len(FEATURE_COLUMNS), FEATURE_COLUMNS[:4])

    OUT.mkdir(exist_ok=True)
    matrices = build_or_load_features(skip_load=args.skip_load, domains=domains)
    if args.cache_only:
        log.info("--cache-only: skipping pairwise metrics.")
        return None

    # Fixed seed-0 subsample BEFORE dA / MMD (LANL / CERT are huge).
    used: dict[str, np.ndarray] = {}
    n_used: dict[str, int] = {}
    for name, X in matrices.items():
        Xs = _subsample_matrix(X, cap=args.subsample_cap, seed=SUBSAMPLE_SEED)
        used[name] = Xs
        n_used[name] = len(Xs)
        log.info(
            "subsample %s: full=%d -> used=%d (cap=%d seed=%d)",
            name, len(X), len(Xs), args.subsample_cap, SUBSAMPLE_SEED,
        )

    detail_rows = []
    summary_rows = []
    five_rows = []
    pairs = list(combinations([d for d in DOMAIN_ORDER if d in used], 2))
    if len(pairs) == 0:
        raise RuntimeError(
            "Need ≥2 domains for pairwise metrics; "
            f"have {list(used)}. Cache more domains or drop --cache-only."
        )
    log.info("Unordered pairs (%d): %s", len(pairs), pairs)

    for a, b in pairs:
        pair = f"{a}__{b}"
        log.info(
            "==== pair %s | n_A=%d n_B=%d (post-subsample) ====",
            pair, n_used[a], n_used[b],
        )
        details = pairwise_domain_gap(
            used[a],
            used[b],
            seeds=seeds,
            domain_cap=args.domain_cap,
            mmd_cap=args.mmd_cap,
        )
        details.insert(0, "pair", pair)
        details.insert(1, "dataset_A", a)
        details.insert(2, "dataset_B", b)
        detail_rows.append(details)
        summary_rows.append(summarize_domain_gap(details, pair=pair))

        # Seed-0 point estimate for the tidy 5-domain matrix.
        seed0 = details.loc[details["seed"] == SUBSAMPLE_SEED]
        if seed0.empty:
            seed0 = details.iloc[[0]]
        r0 = seed0.iloc[0]
        five_rows.append({
            "src": a,
            "tgt": b,
            "dA": float(r0["da"]),
            "mmd_linear": float(r0["mmd_linear"]),
            "mmd_rbf": float(r0["mmd_rbf"]),
            "n_src": int(n_used[a]),
            "n_tgt": int(n_used[b]),
        })

    details_all = pd.concat(detail_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    five = pd.DataFrame(five_rows)

    tidy = summary[
        ["pair", "n_A", "n_B", "domain_AUC_LR", "domain_AUC_RF", "dA_hat", "MMD"]
    ].copy()
    tidy = tidy.rename(columns={
        "domain_AUC_LR": "domain-AUC (LR)",
        "domain_AUC_RF": "domain-AUC (RF)",
        "dA_hat": "dA_hat",
        "MMD": "MMD",
    })

    details_path = OUT / "domain_gap_details.csv"
    summary_path = OUT / "domain_gap_summary.csv"
    tidy_path = OUT / "domain_gap.csv"
    five_path = OUT / "domain_gap_5domain.csv"
    details_all.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    tidy.to_csv(tidy_path, index=False, encoding="utf-8-sig")
    five.to_csv(five_path, index=False, encoding="utf-8-sig")

    print("\n======== DOMAIN GAP (mean±std over seeds) ========")
    print(tidy.to_string(index=False))
    print("\n======== DOMAIN GAP 5-DOMAIN (seed=0; post ≤50k subsample) ========")
    print(five.to_string(index=False))
    lanl_pairs = five[(five["src"] == "lanl") | (five["tgt"] == "lanl")]
    print("\n======== LANL PAIRS (eyeball: far from CERT / SPEDIA?) ========")
    print(lanl_pairs.to_string(index=False))
    print(f"\nSaved: {tidy_path}")
    print(f"5-domain: {five_path}")
    print(f"Details: {details_path}")
    print(
        "Notes: dA = 2(1-2ε) from LogisticRegression CV error (ε=1-accuracy); "
        "mmd_rbf = RBF-MMD^2 with median-heuristic bandwidth on standardized features; "
        f"each domain subsampled to ≤{args.subsample_cap} (seed={SUBSAMPLE_SEED}) "
        "before metrics; "
        f"domain classifiers capped at {args.domain_cap}/side after balance; "
        f"MMD capped at {args.mmd_cap}/side."
    )
    return five


if __name__ == "__main__":
    main()
