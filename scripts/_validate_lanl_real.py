"""One-off real-LANL validation of load_lanl. Print only; do not commit.

CPU/IO-only. Checks parse fidelity, red-team coverage (98 users / 176 mal
user-days), temporal vs user-disjoint positive splits, and ID PR-AUC on the
user-disjoint split (temporal is unusable: red-team confined to days 2–30).
"""
from __future__ import annotations

import gzip
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import FEATURE_COLUMNS, user_day_features, X_y
from src.data.loaders import _LANL_EPOCH, _LANL_RT_NAMES, _lanl_find, load_lanl
from src.data.schema import ACTION, LABEL, TIMESTAMP, USER, INSIDER_TYPE, validate
from src.train.evaluate import bootstrap_ci
from src.train.splits import temporal_split, user_disjoint_split

LANL_PATH = r"C:\PhD\lanl"
N_PARSE_CHECK = 100_000
LEAK_PR_AUC = 0.95
# Local cache so a failed ID-eval retry does not re-stream 7.1 GiB auth.txt.gz.
# Lives next to the LANL dump (not in the git repo). Kept after GO for reuse.
_CACHE = Path(LANL_PATH) / "_cache_load_lanl_redteam_aware_bf002.parquet"


def _rss_gb(proc: psutil.Process | None = None) -> float:
    p = proc or psutil.Process()
    return p.memory_info().rss / (1024 ** 3)


class _PeakRSS:
    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.peak_gb = _rss_gb()
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thr.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thr.join(timeout=5)

    def _run(self):
        proc = psutil.Process()
        while not self._stop.wait(self.interval):
            self.peak_gb = max(self.peak_gb, _rss_gb(proc))


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


def check_parse_fidelity(auth_fp: str, n: int = N_PARSE_CHECK) -> None:
    section("1. PARSE FIDELITY (first ~100k raw auth.txt.gz rows)")
    malformed = 0
    field_counts: dict[int, int] = {}
    scanned = 0
    opener = gzip.open if auth_fp.endswith(".gz") else open
    with opener(auth_fp, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            scanned += 1
            n_fields = len(line.rstrip("\n\r").split(","))
            field_counts[n_fields] = field_counts.get(n_fields, 0) + 1
            if n_fields != 9:
                malformed += 1
                if malformed <= 5:
                    print(f"  malformed example (fields={n_fields}): {line[:160]!r}")
    print(f"scanned_head={scanned}  malformed(!=9 fields)={malformed}")
    print(f"field_count histogram: {dict(sorted(field_counts.items()))}")
    assert malformed == 0, f"PARSE FAIL: {malformed} rows did not split into 9 fields"
    print("ASSERT (a) 9-field parse clean: PASS")


def main() -> None:
    proc = psutil.Process()
    print("LANL real-load validation (CPU/IO-only, print-only, no commit)")
    print(f"path={LANL_PATH}")
    print(f"start RSS: {_rss_gb():.2f} GiB")

    auth_fp = _lanl_find(LANL_PATH, "auth")
    rt_fp = _lanl_find(LANL_PATH, "redteam")
    print(f"auth={auth_fp}")
    print(f"redteam={rt_fp}")

    check_parse_fidelity(auth_fp)

    # Independent ground truth
    section("GROUND TRUTH redteam.txt.gz")
    rt = pd.read_csv(
        rt_fp, header=None, names=_LANL_RT_NAMES,
        compression="infer", dtype=str, keep_default_na=False,
    )
    rt_user = rt["user"].astype(str).str.split("@", n=1).str[0]
    rt_ts = _LANL_EPOCH + pd.to_timedelta(
        pd.to_numeric(rt["time"], errors="coerce"), unit="s"
    )
    rt_day = rt_ts.dt.strftime("%Y-%m-%d")
    gt_users = set(rt_user)
    gt_mal_days = set(zip(rt_user, rt_day))
    print(f"redteam events={len(rt)}")
    print(f"distinct red-team users={len(gt_users)}")
    print(f"malicious user-days={len(gt_mal_days)}")
    print(f"all U-accounts? {all(u.startswith('U') for u in gt_users)}")
    print(
        f"mal day range: {min(d for _, d in gt_mal_days)} .. "
        f"{max(d for _, d in gt_mal_days)}"
    )
    assert len(rt) == 749, len(rt)
    assert len(gt_users) == 98, len(gt_users)
    assert len(gt_mal_days) == 176, len(gt_mal_days)

    section("1b. load_lanl() defaults (redteam_aware, benign_frac=0.02)")
    rss_before = _rss_gb(proc)
    t0 = time.perf_counter()
    loaded_from_cache = False
    with _PeakRSS(interval=2.0) as peak:
        if _CACHE.exists():
            print(f"[cache] reading {_CACHE} ...", flush=True)
            df = pd.read_parquet(_CACHE)
            loaded_from_cache = True
        else:
            df = load_lanl(LANL_PATH)
            print(f"[cache] writing {_CACHE} for ID-eval retries ...", flush=True)
            df.to_parquet(_CACHE, index=False)
        peak.peak_gb = max(peak.peak_gb, _rss_gb(proc))
    load_s = time.perf_counter() - t0
    rss_after = _rss_gb(proc)
    peak_rss = max(peak.peak_gb, rss_before, rss_after)
    src = "cache" if loaded_from_cache else "load_lanl"
    print(f"\n{src} wall-clock={load_s:.1f}s ({load_s / 60:.1f} min)")
    print(f"RSS before={rss_before:.2f} GiB  after={rss_after:.2f} GiB  "
          f"peak≈{peak_rss:.2f} GiB")
    validate(df)

    print("\nACTION value_counts:")
    print(df[ACTION].value_counts().to_string())
    bad_act = sorted(set(df[ACTION].unique()) - {"logon", "logoff", "auth"})
    print(f"unexpected actions: {bad_act if bad_act else '(none)'}")
    assert not bad_act, bad_act

    ts_min, ts_max = df[TIMESTAMP].min(), df[TIMESTAMP].max()
    n_days = int(df[TIMESTAMP].dt.strftime("%Y-%m-%d").nunique())
    print(f"\nTIMESTAMP min={ts_min}  max={ts_max}")
    print(f"distinct days={n_days}  (auth spans ~58 days)")
    print(f"rows kept={len(df):,}  users={df[USER].nunique():,}")

    section("2. RED-TEAM COVERAGE")
    present = set(df[USER].astype(str))
    missing = sorted(gt_users - present)
    print(f"red-team users present in df: {len(gt_users & present)} / {len(gt_users)}")
    print(f"missing red-team users ({len(missing)}): "
          f"{missing if missing else '(none)'}")
    assert not missing, missing

    pos = df[df[LABEL] == 1]
    pos_ud = (
        pos.assign(day=pos[TIMESTAMP].dt.strftime("%Y-%m-%d"))
        [[USER, "day"]]
        .drop_duplicates()
    )
    n_mal_ud = len(pos_ud)
    print(f"malicious user-days after join={n_mal_ud}  (expect 176)")
    assert n_mal_ud == 176, n_mal_ud
    print("ASSERT (b) 98 RT users / 176 malicious user-days: PASS")

    prev_events = float(df[LABEL].mean())
    day_all = df.assign(day=df[TIMESTAMP].dt.strftime("%Y-%m-%d"))
    ud_label = day_all.groupby([USER, "day"], sort=False)[LABEL].max()
    print(f"positive prevalence events={prev_events:.6f}  "
          f"user-days={float(ud_label.mean()):.6f}")
    print(f"insider_type counts:\n{df[INSIDER_TYPE].value_counts().to_string()}")
    mal_days_sorted = sorted(pos_ud["day"].unique())
    print(f"malicious calendar days: {mal_days_sorted[0]} .. {mal_days_sorted[-1]}  "
          f"({len(mal_days_sorted)} distinct)")
    assert mal_days_sorted[0] >= "2015-01-02" and mal_days_sorted[-1] <= "2015-01-30"

    section("3. SPLIT POSITIVES (temporal vs user-disjoint)")
    print("Building user_day_features(deviation=True) on FULL event stream ...",
          flush=True)
    t_feat = time.perf_counter()
    with _PeakRSS(interval=2.0) as peak2:
        feat_full = user_day_features(df, deviation=True)
        peak_rss = max(peak_rss, peak2.peak_gb, _rss_gb(proc))
    print(f"features wall={time.perf_counter() - t_feat:.1f}s  "
          f"user-days={len(feat_full):,}  pos={int(feat_full[LABEL].sum())}  "
          f"rss={_rss_gb():.2f} GiB")

    # --- temporal (event split → user-day keys) ---
    print("\n--- temporal_split(train_frac=0.7) ---", flush=True)
    tr_t, te_t = temporal_split(df, train_frac=0.7)
    print(f"event cut: train={len(tr_t):,} test={len(te_t):,}  "
          f"cut≈{tr_t[TIMESTAMP].max()}")

    def _ud_keys(ev: pd.DataFrame) -> set:
        tmp = ev[[USER]].assign(day=ev[TIMESTAMP].dt.strftime("%Y-%m-%d"))
        return set(map(tuple, tmp.drop_duplicates().itertuples(index=False, name=None)))

    feat_full = feat_full.copy()
    feat_full["_day_str"] = pd.to_datetime(feat_full["day"]).dt.strftime("%Y-%m-%d")
    tr_keys, te_keys = _ud_keys(tr_t), _ud_keys(te_t)
    fkeys = list(zip(feat_full[USER], feat_full["_day_str"]))
    in_tr = np.fromiter((k in tr_keys for k in fkeys), dtype=bool, count=len(fkeys))
    in_te = np.fromiter((k in te_keys for k in fkeys), dtype=bool, count=len(fkeys))
    ft_tr, ft_te = feat_full.loc[in_tr], feat_full.loc[in_te]
    tr_pos_t, te_pos_t = int(ft_tr[LABEL].sum()), int(ft_te[LABEL].sum())
    print(f"[temporal] train user-days={len(ft_tr):,} pos={tr_pos_t}  |  "
          f"test user-days={len(ft_te):,} pos={te_pos_t}")

    # --- user-disjoint ---
    print("\n--- user_disjoint_split(train_frac=0.7, seed=7) ---", flush=True)
    tr_u, te_u = user_disjoint_split(df, train_frac=0.7, seed=7)
    train_users = set(tr_u[USER].unique())
    test_users = set(te_u[USER].unique())
    fu_tr = feat_full[feat_full[USER].isin(train_users)]
    fu_te = feat_full[feat_full[USER].isin(test_users)]
    tr_pos_u, te_pos_u = int(fu_tr[LABEL].sum()), int(fu_te[LABEL].sum())
    print(
        f"[user_disjoint] train users={len(train_users):,} "
        f"user-days={len(fu_tr):,} pos={tr_pos_u}  |  "
        f"test users={len(test_users):,} user-days={len(fu_te):,} pos={te_pos_u}"
    )

    if te_pos_t == 0:
        print(
            "\nNOTE: default temporal_split(train_frac=0.7) on the full 58-day "
            "timeline has 0 test positives (red-team confined to days 2–30). "
            "Also report (i) temporal cut at median malicious user-day and "
            "(ii) user-disjoint as the usable LANL ID diagonal."
        )
    assert tr_pos_u > 0 and te_pos_u > 0, (tr_pos_u, te_pos_u)
    print("ASSERT user-disjoint positives in train AND test: PASS")

    def _fit_rf(train_f, test_f, tag: str):
        Xtr, ytr = X_y(train_f)
        Xte, yte = X_y(test_f)
        base = float(yte.mean())
        print(f"\n--- RF [{tag}] ---")
        print(f"train: n={len(ytr):,} pos={int(ytr.sum())}  "
              f"test: n={len(yte):,} pos={int(yte.sum())}  base_rate={base:.6f}")
        if int(yte.sum()) == 0 or len(np.unique(yte)) < 2:
            print("SKIP: no test positives — PR-AUC undefined")
            return None
        t_rf = time.perf_counter()
        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=7,
        )
        rf.fit(Xtr, ytr)
        rf_wall = time.perf_counter() - t_rf
        scores = rf.predict_proba(Xte)[:, 1]
        pr = float(average_precision_score(yte, scores))
        roc = float(roc_auc_score(yte, scores))
        pr_pt, pr_lo, pr_hi = bootstrap_ci(
            yte, scores, metric="pr_auc", n_boot=1000, alpha=0.05, seed=7
        )
        lift = pr / max(base, 1e-12)
        print(f"RF fit wall={rf_wall:.1f}s")
        print(f"PR-AUC = {pr:.4f}  95% CI [{pr_lo:.4f}, {pr_hi:.4f}]  "
              f"(bootstrap point={pr_pt:.4f})")
        print(f"ROC-AUC = {roc:.4f}")
        print(f"base-rate lift = {lift:.2f}x  (base={base:.6f})")
        return {
            "pr": pr, "pr_lo": pr_lo, "pr_hi": pr_hi, "roc": roc,
            "lift": lift, "base": base, "rf": rf, "tag": tag,
        }

    section("4. RandomForest ID metrics (temporal RT-cut + user-disjoint)")
    # Temporal cut inside the red-team window (past RT days → train, later → test)
    pos_day_vals = pd.to_datetime(sorted(feat_full.loc[feat_full[LABEL] == 1, "day"].unique()))
    cut_med = pos_day_vals[len(pos_day_vals) // 2]
    day_ts = pd.to_datetime(feat_full["day"])
    ft_med_tr = feat_full.loc[day_ts <= cut_med]
    ft_med_te = feat_full.loc[day_ts > cut_med]
    print(f"median-malicious-day cut: {cut_med.date()}  "
          f"(pos days {pos_day_vals[0].date()} .. {pos_day_vals[-1].date()})")
    r_temp = _fit_rf(ft_med_tr, ft_med_te, "temporal cut@median-malicious-day")
    r_ud = _fit_rf(fu_tr, fu_te, "user_disjoint")
    # Headline for go/no-go: prefer temporal RT-cut if defined, else user-disjoint
    headline = r_temp if r_temp is not None else r_ud
    assert headline is not None
    pr, pr_lo, pr_hi = headline["pr"], headline["pr_lo"], headline["pr_hi"]
    roc, lift = headline["roc"], headline["lift"]
    rf = headline["rf"]
    peak_rss = max(peak_rss, _rss_gb(proc))

    print("\n--- COST ---")
    src_cost = "cache" if loaded_from_cache else "load_lanl"
    print(f"{src_cost} wall-clock = {load_s:.1f}s ({load_s / 60:.1f} min)")
    print(f"peak RSS ≈ {peak_rss:.2f} GiB  (final={_rss_gb():.2f} GiB)")
    print(f"rows kept = {len(df):,}")

    section("GO / NO-GO")
    leak = pr >= LEAK_PR_AUC
    plausible = 0.1 <= pr <= 0.5
    soft_ok = 0.05 <= pr <= 0.85
    print("(a) 9-field parse clean: GO")
    print(f"(b) 58 distinct calendar days: {'GO' if n_days == 58 else 'NO-GO'} "
          f"(got {n_days})")
    print(f"(c) all RT users + 176 mal user-days: GO "
          f"(missing={missing}, mal_ud={n_mal_ud})")
    print(f"(d) PR-AUC via [{headline['tag']}]: {pr:.4f} "
          f"[{pr_lo:.4f}, {pr_hi:.4f}]  "
          f"in ~0.1-0.5? {plausible}  leakage(~1.0)? {leak}")
    if r_temp is not None and r_ud is not None:
        print(f"    also user_disjoint PR-AUC={r_ud['pr']:.4f}  "
              f"ROC={r_ud['roc']:.4f}  lift={r_ud['lift']:.2f}x")

    if leak:
        print("\n*** PR-AUC ≈ 1.0 suggests leakage — top RF feature importances ***")
        imp = sorted(
            zip(FEATURE_COLUMNS, rf.feature_importances_),
            key=lambda x: -x[1],
        )
        for name, val in imp[:25]:
            print(f"  {val:.4f}  {name}")
        print("NO-GO: stop and investigate leakage before LANL enters the matrix.")
        raise SystemExit(2)

    if not soft_ok:
        print(f"WARNING: PR-AUC={pr:.4f} outside broad plausible band.")
        if pr_hi < 0.05 or pr_lo > 0.85:
            print("NO-GO: CI fully outside plausible band.")
            raise SystemExit(3)

    overall = (n_days == 58) and (not missing) and (n_mal_ud == 176) and soft_ok and (not leak)
    if overall:
        print(
            "\nGO: LANL loader validated on real data. "
            "Default temporal_split(0.7) is unusable for the LANL diagonal "
            "(0 test positives); use median-RT temporal cut or user-disjoint. "
            "Ready for the transfer matrix."
        )
    else:
        print(f"\nNO-GO: see flags above (headline PR-AUC={pr:.4f}).")
    if _CACHE.exists():
        print(f"[cache] kept {_CACHE} for reuse (not a git artifact)")


if __name__ == "__main__":
    main()
