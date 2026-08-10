"""
Multi-seed transfer across CERT releases + SPEDIA.

Tests whether day-supervised GNN > RF transfer advantage holds beyond the
original cert↔spedia pair by treating CERT r4.2 / r5.2 (/ optional r6.2) as
separate domains under the same honest protocol as the confirm run.

Phase 1 (default): day_gnn_zero_shot + random_forest on {cert42, cert52, spedia}
Phase 2 (--baselines): add static_gcn, athitd, transformer, lstm_ae, IF, OCSVM
Optional: --include-cert62 (expensive; skipped by default)

Usage:
  set PYTHONIOENCODING=utf-8
  python -m scripts.run_multiseed_releases
  python -m scripts.run_multiseed_releases --baselines
  python -m scripts.run_multiseed_releases --include-cert62 --stats-only
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.features import user_day_features
from src.data.loaders import load_cert, load_cert_releases, load_spedia
from src.data.schema import LABEL, validate
from src.train.multiseed_transfer import (
    gnn_minus_rf_table,
    print_confirmation_report,
    run_multiseed_confirm,
    target_base_rates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("multiseed_releases")

CERT42 = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2"
# Extracted trees (no local answers/); shared answers pack lives under CERT_r4.2.
CERT52 = r"C:\PhD\07_Projects\r5.2"
CERT62 = r"C:\PhD\07_Projects\r6.2"
ANSWERS = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\CERT_r4.2\answers"
SPEDIA = r"C:\PhD\04_Journal_Papers\New Paper Aug 2026\logs_SPEDIA.csv"
OUT = ROOT / "results"

CERT_LOAD_KW = dict(
    sources=("logon", "device", "file", "email"),
    http_mode="insider_aware",
    benign_http_frac=0.05,
    answers_dir=ANSWERS,
)


def _parse_seeds(s: str):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")


def _quick_base_rate(path: str, release: str, name: str) -> dict:
    """Load a release only to report user-day base rate (for skipped domains).

    Uses ``http_mode=skip`` so the stats check stays cheap (r6.2 http.csv ~90GB).
    """
    log.info(
        "Quick stats load for %s path=%s release=%s (http_mode=skip)",
        name, path, release,
    )
    kw = dict(CERT_LOAD_KW)
    kw["http_mode"] = "skip"
    df = load_cert(path, release=release, dataset_tag=name, **kw)
    feat = user_day_features(df, deviation=False)
    y = feat[LABEL].to_numpy()
    st = {
        "target": name,
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_events": int(len(df)),
        "n_pos_events": int(df[LABEL].sum()),
        "path": path,
        "release": release,
    }
    log.info(
        "STATS %s: user-day p=%.6f n=%d n_pos=%d | events=%d pos_events=%d",
        name, st["base_rate"], st["n"], st["n_pos"], st["n_events"], st["n_pos_events"],
    )
    return st


def _save_results(results: dict, dfs: dict, tag: str):
    OUT.mkdir(exist_ok=True)
    results["details"].to_csv(OUT / f"multiseed_releases_details_{tag}.csv", index=False)
    results["summary"].to_csv(OUT / f"multiseed_releases_summary_{tag}.csv", index=False)
    results["deltas"].to_csv(OUT / f"multiseed_releases_deltas_{tag}.csv", index=False)
    results["verdict"].to_csv(OUT / f"multiseed_releases_verdict_{tag}.csv", index=False)
    results["target_stats"].to_csv(
        OUT / f"multiseed_releases_target_stats_{tag}.csv", index=False,
    )

    summary = results["summary"]
    for model in summary["model"].unique():
        for protocol in ("temporal", "user_disjoint"):
            names = list(dfs.keys())
            pr = pd.DataFrame(index=names, columns=names, dtype=float)
            lf = pd.DataFrame(index=names, columns=names, dtype=float)
            sub = summary[summary["model"] == model]
            for s in names:
                for t in names:
                    if s == t:
                        row = sub[
                            (sub["source"] == s) & (sub["target"] == t)
                            & (sub["diagonal_protocol"] == protocol)
                        ]
                    else:
                        row = sub[
                            (sub["source"] == s) & (sub["target"] == t)
                            & (sub["diagonal_protocol"] == "full_source")
                        ]
                    if len(row):
                        pr.loc[s, t] = float(row.iloc[0]["pr_auc_mean"])
                        lf.loc[s, t] = float(row.iloc[0]["lift_mean"])
            safe = str(model).replace("/", "_")
            pr.to_csv(OUT / f"multiseed_releases_pr_auc_{safe}_{protocol}_{tag}.csv")
            lf.to_csv(OUT / f"multiseed_releases_lift_{safe}_{protocol}_{tag}.csv")


def _print_hold_count(deltas: pd.DataFrame):
    off = deltas[deltas["cell_kind"] == "off_diagonal"].copy()
    n = len(off)
    holds = int(off["claim_holds"].fillna(False).astype(bool).sum())
    print("\n======== OFF-DIAGONAL HOLD COUNT (day_gnn beats RF, CI excludes 0) ========")
    print(f"  {holds} / {n} off-diagonal cells HOLD")
    for _, r in off.sort_values(["source", "target"]).iterrows():
        status = "HOLDS" if r["claim_holds"] else "FAILS"
        print(
            f"  {r['source']}->{r['target']}: {status}  "
            f"GNN={r['gnn_pr_mean']:.3f} RF={r['rf_pr_mean']:.3f}  "
            f"ΔPR={r['delta_pr_mean']:.3f} "
            f"[{r['delta_pr_lo']:.3f}, {r['delta_pr_hi']:.3f}]"
        )
    return holds, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--tag", default="core", help="filename tag (default: core)")
    ap.add_argument(
        "--baselines",
        action="store_true",
        help="phase 2: also run static_gcn/athitd/transformer/lstm_ae/IF/OCSVM",
    )
    ap.add_argument("--include-cert62", action="store_true")
    ap.add_argument(
        "--stats-only",
        action="store_true",
        help="only load domains and print base rates (no training)",
    )
    ap.add_argument("--skip-gnn", action="store_true")
    ap.add_argument("--parallel-feature-seeds", type=int, default=1)
    ap.add_argument(
        "--skip-cert62-stats",
        action="store_true",
        help="skip cheap cert62 base-rate probe (use prior skipped_stats CSV)",
    )
    ap.add_argument(
        "--resume-details",
        default="",
        help="path to prior multiseed_releases_details_*.csv (or checkpoint) to resume",
    )
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    log.info("CERT42=%s", CERT42)
    log.info("CERT52=%s", CERT52)
    log.info("CERT62=%s", CERT62)
    log.info("ANSWERS=%s", ANSWERS)
    log.info("SPEDIA=%s", SPEDIA)

    release_paths = {
        "cert42": (CERT42, "4.2"),
        "cert52": (CERT52, "5.2"),
    }
    if args.include_cert62:
        release_paths["cert62"] = (CERT62, "6.2")

    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    log.info("Loading CERT releases %s ...", list(release_paths))
    dfs = load_cert_releases(release_paths, **CERT_LOAD_KW)
    for name, df in dfs.items():
        validate(df)

    log.info("Loading SPEDIA real_only ...")
    spedia = load_spedia(SPEDIA, real_only=True)
    validate(spedia)
    dfs["spedia"] = spedia
    log.info(
        "Loaded in %.1fs | %s",
        time.time() - t0,
        {k: len(v) for k, v in dfs.items()},
    )

    # Base rates from full-stream user-day features (deviation for consistency log)
    full_feats = {n: user_day_features(df, deviation=True) for n, df in dfs.items()}
    tstats = target_base_rates(full_feats)
    print("\n======== TARGET BASE RATES (user-day, deviation features built) ========")
    print(tstats.to_string(index=False))
    tstats.to_csv(OUT / f"multiseed_releases_target_stats_{args.tag}.csv", index=False)

    skipped_stats = []
    if not args.include_cert62:
        prior_skip = OUT / f"multiseed_releases_skipped_stats_{args.tag}.csv"
        if args.skip_cert62_stats and prior_skip.exists():
            prev = pd.read_csv(prior_skip)
            print(
                f"\n[cert62 SKIPPED from matrix] (cached) "
                + ", ".join(
                    f"{k}={prev.iloc[0][k]}"
                    for k in ("base_rate", "n", "n_pos")
                    if k in prev.columns
                )
            )
        else:
            # Optional: cheap-ish stats for r6.2 without putting it in the matrix.
            # Uses http_mode=skip (r6.2 http.csv ~90GB).
            try:
                st62 = _quick_base_rate(CERT62, "6.2", "cert62")
                skipped_stats.append(st62)
                pd.DataFrame(skipped_stats).to_csv(prior_skip, index=False)
                print(
                    f"\n[cert62 SKIPPED from matrix] base_rate={st62['base_rate']:.6f} "
                    f"n={st62['n']} n_pos={st62['n_pos']} "
                    f"(sparse: {st62['n_pos']} positive user-days)"
                )
            except Exception as e:
                log.warning("Could not load cert62 for stats: %s", e)
                print(f"\n[cert62 SKIPPED from matrix] stats unavailable: {e}")

    if args.stats_only:
        print("\n--stats-only: exiting before training.")
        return

    if args.baselines:
        feature_models = [
            "random_forest",
            "isolation_forest",
            "ocsvm",
            "lstm_ae",
            "transformer",
            "static_gcn",
            "athitd",
        ]
    else:
        feature_models = ["random_forest"]
        log.info("Phase 1: feature_models=%s (GNN + RF only)", feature_models)

    ckpt = OUT / f"multiseed_releases_details_checkpoint_{args.tag}.csv"
    # Prefer explicit --resume-details; else newest existing checkpoint/partial/final.
    resume = args.resume_details.strip() or None
    if not resume:
        for cand in (
            ckpt,
            OUT / f"multiseed_releases_details_continuous_{args.tag}.csv",
            OUT / f"multiseed_releases_details_partial_{args.tag}.csv",
            OUT / f"multiseed_releases_details_{args.tag}.csv",
        ):
            if cand.exists() and cand.stat().st_size > 0:
                resume = str(cand)
                log.info("Auto-resume from %s", resume)
                break
    t1 = time.time()
    results = run_multiseed_confirm(
        dfs,
        seeds=seeds,
        dann_seeds=(),
        feature_models=feature_models,
        run_gnn=not args.skip_gnn,
        run_dann=False,
        parallel_feature_seeds=args.parallel_feature_seeds,
        checkpoint_path=str(ckpt),
        resume_details_path=resume,
    )
    # Recompute deltas explicitly (already in results)
    results["deltas"] = gnn_minus_rf_table(
        results["summary"], results["details"],
        gnn_model="day_gnn_zero_shot", rf_model="random_forest",
    )
    log.info("Training finished in %.1fs", time.time() - t1)

    _save_results(results, dfs, args.tag)
    print_confirmation_report(results)
    holds, n = _print_hold_count(results["deltas"])
    print(f"\nSaved CSVs under {OUT}/multiseed_releases_*_{args.tag}.csv")
    print(f"SUMMARY: day_gnn beats RF with CI excluding 0 on {holds}/{n} off-diagonal cells.")


if __name__ == "__main__":
    main()
