"""Supplemental ID eval after discovering default temporal 0.7 has 0 test positives.

Red-team activity is 2015-01-02..2015-01-30; a 0.7 day-quantile cut on the
full 58-day timeline lands after all positives. Re-run RF with a past/future
cut *inside* the red-team window so PR-AUC is defined. Print-only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import FEATURE_COLUMNS, user_day_features, X_y
from src.data.loaders import load_lanl
from src.data.schema import LABEL, TIMESTAMP, USER
from src.train.evaluate import bootstrap_ci, compute_metrics

LANL_PATH = r"C:\PhD\lanl"
CACHE = Path(LANL_PATH) / "_cache_load_lanl_redteam_aware_bf002.parquet"
LEAK_PR_AUC = 0.95


def _load_or_cache() -> pd.DataFrame:
    if CACHE.exists():
        print(f"loading cache: {CACHE}")
        t0 = time.perf_counter()
        df = pd.read_parquet(CACHE)
        print(f"cache load: {time.perf_counter() - t0:.1f}s | rows={len(df)}")
        return df
    print("no cache — calling load_lanl (expect ~100+ min)")
    t0 = time.perf_counter()
    df = load_lanl(LANL_PATH)
    print(f"load_lanl wall-clock: {time.perf_counter() - t0:.1f}s | rows={len(df)}")
    t1 = time.perf_counter()
    df.to_parquet(CACHE, index=False)
    print(f"wrote cache {CACHE} in {time.perf_counter() - t1:.1f}s")
    return df


def _eval_at_cut(feat: pd.DataFrame, cut_day, tag: str) -> dict:
    train_f = feat[feat["day"] <= cut_day]
    test_f = feat[feat["day"] > cut_day]
    ytr = train_f[LABEL].to_numpy()
    yte = test_f[LABEL].to_numpy()
    print(
        f"\n--- {tag} | cut_day={cut_day.date() if hasattr(cut_day, 'date') else cut_day} ---"
    )
    print(
        f"train={len(train_f)} (pos={int(ytr.sum())}) | "
        f"test={len(test_f)} (pos={int(yte.sum())})"
    )
    if int(yte.sum()) == 0 or int(ytr.sum()) == 0 or len(np.unique(yte)) < 2:
        print("SKIP: need positives on both sides for PR-AUC")
        return {"pr_auc": float("nan"), "leak_flag": False, "tag": tag}

    Xtr, ytr = X_y(train_f)
    Xte, yte = X_y(test_f)
    prevalence = float(np.mean(yte))
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=0,
    )
    t0 = time.perf_counter()
    rf.fit(Xtr, ytr)
    scores = rf.predict_proba(Xte)[:, 1]
    fit_s = time.perf_counter() - t0
    m = compute_metrics(yte, scores)
    pr, pr_lo, pr_hi = bootstrap_ci(yte, scores, metric="pr_auc", n_boot=1000, seed=0)
    lift = (m["pr_auc"] / prevalence) if prevalence > 0 else float("nan")
    print(f"RF fit: {fit_s:.1f}s")
    print(f"PR-AUC: {m['pr_auc']:.4f}  (bootstrap 95% CI [{pr_lo:.4f}, {pr_hi:.4f}])")
    print(f"ROC-AUC: {m['roc_auc']:.4f}")
    print(f"test prevalence: {prevalence:.6f}")
    print(f"base-rate lift (PR-AUC / prevalence): {lift:.2f}x")
    print(
        f"dr@1%FPR={m['dr_at_1pct_fpr']:.4f} | dr@5%FPR={m['dr_at_5pct_fpr']:.4f} | "
        f"P@k={m['precision_at_k']:.4f} | n={m['n']} n_pos={m['n_pos']}"
    )
    leak = m["pr_auc"] >= LEAK_PR_AUC
    if leak:
        print("LEAKAGE FLAG: top RF feature importances:")
        imp = sorted(
            zip(FEATURE_COLUMNS, rf.feature_importances_),
            key=lambda x: -x[1],
        )
        for name, val in imp[:25]:
            print(f"  {val:.4f}  {name}")
    return {
        "tag": tag,
        "pr_auc": m["pr_auc"],
        "pr_lo": pr_lo,
        "pr_hi": pr_hi,
        "roc_auc": m["roc_auc"],
        "lift": lift,
        "prevalence": prevalence,
        "leak_flag": leak,
        "n_pos_test": int(m["n_pos"]),
        "n_pos_train": int(ytr.sum()),
    }


def main() -> None:
    print("Supplemental LANL ID eval (temporal cut inside red-team window)")
    print(f"start RSS: {psutil.Process().memory_info().rss / 1e6:.0f} MB")
    df = _load_or_cache()

    feat = user_day_features(df, deviation=True)
    feat = feat.copy()
    feat["day"] = pd.to_datetime(feat["day"])
    pos_days = sorted(feat.loc[feat[LABEL] == 1, "day"].unique())
    print(f"user-days={len(feat)} | pos={int(feat[LABEL].sum())}")
    print(f"positive day span: {pos_days[0].date()} .. {pos_days[-1].date()} ({len(pos_days)} days)")
    print(f"all day span: {feat['day'].min().date()} .. {feat['day'].max().date()}")

    # A) default-style 0.7 quantile (reproduce nan)
    cut07 = feat["day"].quantile(0.7)
    r07 = _eval_at_cut(feat, cut07, "full_timeline train_frac~0.7 (expect 0 test pos)")

    # B) cut at median of malicious user-days (past RT / future RT)
    cut_med = pd.to_datetime(np.median(np.array(pos_days, dtype="datetime64[ns]")))
    r_med = _eval_at_cut(feat, cut_med, "cut=median malicious user-day")

    # C) restrict to RT-active calendar window then 0.7 temporal
    rt_lo, rt_hi = pos_days[0], pos_days[-1]
    feat_rt = feat[(feat["day"] >= rt_lo) & (feat["day"] <= rt_hi)].copy()
    cut_rt = feat_rt["day"].quantile(0.7)
    r_win = _eval_at_cut(feat_rt, cut_rt, "RT-window-only then train_frac~0.7")

    # D) user-disjoint as secondary sanity (not the headline protocol)
    from src.train.splits import user_disjoint_split
    # user_disjoint needs TIMESTAMP — map day
    feat_ts = feat.copy()
    feat_ts[TIMESTAMP] = feat_ts["day"]
    # operate at user-day level: split users
    rng = np.random.default_rng(0)
    user_pos = feat.groupby(USER)[LABEL].max()
    pos_u = user_pos[user_pos == 1].index.to_numpy()
    neg_u = user_pos[user_pos == 0].index.to_numpy()
    rng.shuffle(pos_u)
    rng.shuffle(neg_u)

    def _take(users, frac=0.7):
        n_tr = int(round(len(users) * frac))
        if len(users) >= 2:
            n_tr = min(max(n_tr, 1), len(users) - 1)
        return set(users[:n_tr])

    train_users = _take(pos_u) | _take(neg_u)
    test_users = set(user_pos.index) - train_users
    train_f = feat[feat[USER].isin(train_users)]
    test_f = feat[feat[USER].isin(test_users)]
    print("\n--- user_disjoint (secondary sanity) ---")
    print(
        f"train={len(train_f)} (pos={int(train_f[LABEL].sum())}) | "
        f"test={len(test_f)} (pos={int(test_f[LABEL].sum())})"
    )
    Xtr, ytr = X_y(train_f)
    Xte, yte = X_y(test_f)
    prevalence = float(np.mean(yte))
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=0,
    )
    rf.fit(Xtr, ytr)
    scores = rf.predict_proba(Xte)[:, 1]
    m = compute_metrics(yte, scores)
    pr, pr_lo, pr_hi = bootstrap_ci(yte, scores, metric="pr_auc", n_boot=1000, seed=0)
    lift = m["pr_auc"] / prevalence if prevalence > 0 else float("nan")
    print(f"PR-AUC: {m['pr_auc']:.4f}  (bootstrap 95% CI [{pr_lo:.4f}, {pr_hi:.4f}])")
    print(f"ROC-AUC: {m['roc_auc']:.4f}")
    print(f"test prevalence: {prevalence:.6f}")
    print(f"base-rate lift: {lift:.2f}x")
    r_ud = {
        "pr_auc": m["pr_auc"], "pr_lo": pr_lo, "pr_hi": pr_hi,
        "roc_auc": m["roc_auc"], "lift": lift, "leak_flag": m["pr_auc"] >= LEAK_PR_AUC,
    }
    if r_ud["leak_flag"]:
        print("LEAKAGE FLAG: top RF feature importances:")
        for name, val in sorted(
            zip(FEATURE_COLUMNS, rf.feature_importances_), key=lambda x: -x[1]
        )[:25]:
            print(f"  {val:.4f}  {name}")

    print("\n" + "=" * 72)
    print("SUPPLEMENTAL GO/NO-GO for (d)")
    print("=" * 72)
    headline = r_med if not np.isnan(r_med["pr_auc"]) else r_win
    print(
        f"headline temporal (median-RT cut): "
        f"PR-AUC={r_med['pr_auc']:.4f} [{r_med.get('pr_lo', float('nan')):.4f}, "
        f"{r_med.get('pr_hi', float('nan')):.4f}]"
    )
    print(
        f"RT-window 0.7 temporal: "
        f"PR-AUC={r_win['pr_auc']:.4f} [{r_win.get('pr_lo', float('nan')):.4f}, "
        f"{r_win.get('pr_hi', float('nan')):.4f}]"
    )
    print(
        f"user_disjoint sanity: "
        f"PR-AUC={r_ud['pr_auc']:.4f} [{r_ud['pr_lo']:.4f}, {r_ud['pr_hi']:.4f}]"
    )
    pr = headline["pr_auc"]
    leak = headline.get("leak_flag", False)
    in_band = 0.1 <= pr <= 0.5
    d_ok = in_band and not leak
    print(
        f"(d) plausible real PR-AUC ~0.1-0.5: "
        f"{'GO' if d_ok else 'NO-GO'} (headline PR-AUC={pr:.4f}"
        f"{' LEAKAGE' if leak else ''})"
    )
    print(f"NOTE: default temporal_split(train_frac=0.7) on full 58 days -> "
          f"0 test positives (PR-AUC nan); RT front-loaded into first ~18 days.")


if __name__ == "__main__":
    main()
