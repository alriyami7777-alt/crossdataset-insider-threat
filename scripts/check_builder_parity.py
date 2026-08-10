"""
CPU-only parity check: stream_cert_user_day_features vs user_day_features(deviation=True).

Does not use GPU/CUDA. Safe to run alongside other jobs (read-only CERT IO).
"""
from __future__ import annotations

import os
import sys
import time

# Force CPU-only; never initialize CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.features import FEATURE_COLUMNS, user_day_features
from src.data.loaders import load_cert
from src.data.schema import LABEL, USER
from src.eval.stream_cert_features import stream_cert_user_day_features

CERT = os.environ.get(
    "CERT_PATH",
    r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2",
)
ANSWERS = os.path.join(CERT, "answers")
SOURCES = ("logon", "device", "file", "email")
HTTP_MODE = "skip"
RELEASE = "4.2"
MAX_USERS = int(os.environ.get("PARITY_MAX_USERS", "50"))
ATOL = 1e-6

# Integer / exact columns (counts + label)
_EXACT_SUFFIX_HINTS = ("cnt_", "n_events", "n_distinct_hosts", LABEL)
_COUNT_LIKE = {
    c for c in FEATURE_COLUMNS
    if c.startswith("cnt_") or c in ("n_events", "n_distinct_hosts")
}
_COUNT_LIKE |= {f"dev_{c}" for c in list(_COUNT_LIKE)}  # still float compare for dev_*


def _normalize_day(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d")


def _align(a: pd.DataFrame, b: pd.DataFrame):
    a = a.copy()
    b = b.copy()
    a["_day"] = _normalize_day(a["day"])
    b["_day"] = _normalize_day(b["day"])
    a[USER] = a[USER].astype(str)
    b[USER] = b[USER].astype(str)
    keys = [USER, "_day"]
    a = a.set_index(keys).sort_index()
    b = b.set_index(keys).sort_index()
    common = a.index.intersection(b.index)
    only_a = a.index.difference(b.index)
    only_b = b.index.difference(a.index)
    return a.loc[common], b.loc[common], only_a, only_b


def _is_exact_col(col: str) -> bool:
    return col in ("n_events", "n_distinct_hosts", LABEL) or col.startswith("cnt_")


def main() -> int:
    print("=" * 72)
    print("CERT builder parity check (CPU-only)")
    print("=" * 72)
    print(f"CERT={CERT}")
    print(f"answers={ANSWERS}")
    print(f"sources={SOURCES}  http_mode={HTTP_MODE}  release={RELEASE}")
    print(f"max_users={MAX_USERS}  atol={ATOL}")
    print(f"FEATURE_COLUMNS={len(FEATURE_COLUMNS)}")
    print()

    t0 = time.time()
    print("[1/4] load_cert(...) ...")
    df = load_cert(
        CERT,
        sources=SOURCES,
        http_mode=HTTP_MODE,
        release=RELEASE,
        answers_dir=ANSWERS,
        seed=7,
    )
    print(f"  events={len(df):,}  users={df[USER].nunique():,}  ({time.time()-t0:.1f}s)")

    users = sorted(df[USER].astype(str).unique())[:MAX_USERS]
    user_set = set(users)
    df = df[df[USER].astype(str).isin(user_set)].copy()
    print(f"  filtered to first {len(users)} users -> events={len(df):,}")

    t1 = time.time()
    print("[2/4] user_day_features(df, deviation=True) ...")
    feat_canon = user_day_features(df, deviation=True)
    print(
        f"  user-days={len(feat_canon):,}  pos={int(feat_canon[LABEL].sum())}  "
        f"({time.time()-t1:.1f}s)"
    )
    del df

    t2 = time.time()
    print("[3/4] stream_cert_user_day_features(...) ...")
    feat_stream = stream_cert_user_day_features(
        CERT,
        release=RELEASE,
        answers_dir=ANSWERS,
        sources=SOURCES,
        http_mode=HTTP_MODE,
        seed=7,
    )
    feat_stream = feat_stream[feat_stream[USER].astype(str).isin(user_set)].copy()
    print(
        f"  user-days (filtered)={len(feat_stream):,}  "
        f"pos={int(feat_stream[LABEL].sum())}  ({time.time()-t2:.1f}s)"
    )

    print("[4/4] align + compare FEATURE_COLUMNS ...")
    a, b, only_a, only_b = _align(feat_canon, feat_stream)
    n = len(a)
    print(f"  common user-days compared: {n:,}")
    print(f"  only in canonical: {len(only_a):,}  only in stream: {len(only_b):,}")

    missing_a = [c for c in FEATURE_COLUMNS if c not in a.columns]
    missing_b = [c for c in FEATURE_COLUMNS if c not in b.columns]
    if missing_a or missing_b:
        print(f"  MISSING cols canonical: {missing_a}")
        print(f"  MISSING cols stream:    {missing_b}")
        print()
        print("VERDICT: NO-GO (missing FEATURE_COLUMNS)")
        return 1

    # Also compare label
    cols = list(FEATURE_COLUMNS) + ([LABEL] if LABEL not in FEATURE_COLUMNS else [])
    # label is not in FEATURE_COLUMNS; compare separately for diagnostics
    compare_cols = list(FEATURE_COLUMNS)

    diffs = []
    for col in compare_cols:
        va = a[col].to_numpy(dtype=np.float64)
        vb = b[col].to_numpy(dtype=np.float64)
        if _is_exact_col(col):
            # exact for counts; still report abs max
            bad = va != vb
            # tolerate tiny float cast noise on ints stored as float
            if bad.any() and np.allclose(va, vb, rtol=0.0, atol=0.0, equal_nan=True):
                bad = np.zeros(len(va), dtype=bool)
            elif bad.any():
                # integer-valued: allow 0 atol exact; else flag
                bad = ~np.isclose(va, vb, rtol=0.0, atol=0.0, equal_nan=True)
            max_abs = float(np.max(np.abs(va - vb))) if len(va) else 0.0
            n_bad = int(bad.sum())
            ok = n_bad == 0
        else:
            close = np.isclose(va, vb, rtol=0.0, atol=ATOL, equal_nan=True)
            n_bad = int((~close).sum())
            max_abs = float(np.max(np.abs(va - vb))) if len(va) else 0.0
            ok = n_bad == 0
        diffs.append((col, ok, n_bad, max_abs))

    # label exact check
    la = a[LABEL].to_numpy(dtype=np.int64)
    lb = b[LABEL].to_numpy(dtype=np.int64)
    label_bad = int((la != lb).sum())
    label_max = float(np.max(np.abs(la.astype(np.float64) - lb.astype(np.float64)))) if n else 0.0

    print()
    print("-" * 72)
    print(f"{'column':<28} {'ok':>4} {'n_diff':>8} {'max_abs_diff':>14}")
    print("-" * 72)
    differing = []
    for col, ok, n_bad, max_abs in diffs:
        mark = "OK" if ok else "DIFF"
        print(f"{col:<28} {mark:>4} {n_bad:>8} {max_abs:>14.6g}")
        if not ok:
            differing.append((col, n_bad, max_abs))
    print(f"{'label':<28} {'OK' if label_bad == 0 else 'DIFF':>4} {label_bad:>8} {label_max:>14.6g}")
    print("-" * 72)

    if only_a.size or only_b.size:
        print()
        print("KEY MISMATCH: (user, day) sets differ — treating as NO-GO")
        if len(only_a):
            print(f"  sample only-canonical: {list(only_a[:5])}")
        if len(only_b):
            print(f"  sample only-stream:    {list(only_b[:5])}")

    go = (len(differing) == 0) and (label_bad == 0) and (only_a.size == 0) and (only_b.size == 0) and n > 0

    if differing or label_bad:
        # worst (user, day) by total abs feature diff
        abs_sum = np.zeros(n, dtype=np.float64)
        per_row_worst_feat = np.empty(n, dtype=object)
        per_row_worst_val = np.zeros(n, dtype=np.float64)
        for col, ok, n_bad, max_abs in diffs:
            if ok and col != LABEL:
                continue
            va = a[col].to_numpy(dtype=np.float64)
            vb = b[col].to_numpy(dtype=np.float64)
            d = np.abs(va - vb)
            abs_sum += d
            worse = d > per_row_worst_val
            per_row_worst_val[worse] = d[worse]
            per_row_worst_feat[worse] = col
        if label_bad:
            d = np.abs(la.astype(np.float64) - lb.astype(np.float64))
            abs_sum += d
            worse = d > per_row_worst_val
            per_row_worst_val[worse] = d[worse]
            per_row_worst_feat[worse] = LABEL

        order = np.argsort(-abs_sum)[:5]
        idx = a.index.to_numpy()
        print()
        print("5 worst-offending (user, day) rows:")
        for rank, i in enumerate(order, 1):
            u, day = idx[i]
            feat = per_row_worst_feat[i]
            print(f"  #{rank} user={u} day={day}  total_abs={abs_sum[i]:.6g}  "
                  f"worst_feat={feat} abs={per_row_worst_val[i]:.6g}")
            # side-by-side for differing columns (top few)
            shown = 0
            for col, ok, _, _ in diffs:
                if ok:
                    continue
                ca = float(a.iloc[i][col])
                cb = float(b.iloc[i][col])
                if abs(ca - cb) <= ATOL and _is_exact_col(col) is False:
                    continue
                if abs(ca - cb) == 0 and _is_exact_col(col):
                    continue
                print(f"      {col}: canonical={ca:.8g}  stream={cb:.8g}  "
                      f"diff={ca-cb:.8g}")
                shown += 1
                if shown >= 8:
                    break
            if label_bad and la[i] != lb[i]:
                print(f"      label: canonical={la[i]}  stream={lb[i]}")

    print()
    print(f"user-days compared: {n:,}")
    print(f"columns differing (FEATURE_COLUMNS): {len(differing)}")
    if differing:
        print("differing columns (name, n_diff, max_abs):")
        for col, n_bad, max_abs in sorted(differing, key=lambda x: -x[2]):
            print(f"  {col}: n_diff={n_bad}  max_abs={max_abs:.6g}")
    else:
        print("all FEATURE_COLUMNS match within tolerance")
    print(f"total wall time: {time.time()-t0:.1f}s")
    print()
    if go:
        print("VERDICT: GO — shared space is clean; domain-gap table is trustworthy.")
        return 0
    print("VERDICT: NO-GO — builders disagree; investigate before trusting domain-gap.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
