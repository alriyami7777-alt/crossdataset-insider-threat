# Does It Generalize? A Cross-Dataset Study of Graph-Based Insider Threat Detection Beyond CERT

A reproducibility repository for the first systematic **zero-shot cross-dataset generalization
benchmark** for graph-based insider threat detection (ITD). Representative anomaly, sequence,
static-graph, and temporal-graph detectors are trained on one dataset and evaluated on
independently constructed datasets under a strict source-to-target protocol, to test whether
strong single-dataset performance reflects *transferable* insider behaviour or
*dataset-specific artefacts*.

> **Status:** benchmark complete for **CERT, SPEDIA, and LANL**; the accompanying paper is under
> review (IJACSA). Evaluation code, configs, and summary result tables are being added
> incrementally. Raw datasets are **not** redistributed here — obtain each from its original source.

## Datasets

**Evaluated in this study (three independently constructed families, five domains):**

- **CERT** r4.2 / r5.2 / r6.2 — synthetic enterprise activity with answer-key traitor labels.
- **SPEDIA** — real security-event data (Zenodo `10.5281/zenodo.15495572`); CERT-derived rows are
  filtered out for a clean comparison.
- **LANL** — real authentication events with labelled red-team (lateral-movement) activity.

**Named future extensions (not evaluated here):**

- **TWOS** — gated behind a SUTD data-sharing agreement; a natural breadth extension.
- **DARPA OpTC** — a feasibility probe found its host/process-centric telemetry is dominated by
  service and system principals (real-user attribution ≈5%), so it does not map cleanly onto the
  user-day prediction unit used here. Adapting the benchmark to a host-day or process-lineage unit
  is left for future work. (Its ground truth is a set of malicious eCAR record IDs, which is the
  correct labelling scheme when it is added.)

See [`DATASETS.md`](DATASETS.md) for access and labelling details, and [`METHODS.md`](METHODS.md)
for the full protocol.

## Key findings

- **Cross-dataset collapse (C1).** Strong in-distribution detectors (Random Forest PR-AUC
  0.94–0.99 on CERT r4.2, CERT r5.2, and SPEDIA) **fail to transfer**: off-diagonal PR-AUC falls
  sharply, the collapse is worst into the independently constructed real target (SPEDIA) and total
  for the intrusion-oriented domain (LANL). The aggregate generalization gap is **0.48** (reported
  as a transparency statistic; the per-cell pattern is the substantive evidence).
- **A scoped representation-class advantage (C2).** Temporal and graph representations transfer
  significantly better than the tabular baseline **only when transferring into the real target
  SPEDIA** (ΔPR ≈ **+0.13 / +0.17**, 95% bootstrap CI excluding zero, reproduced across CERT
  releases and independent runs). This is **not** universal: within the synthetic CERT family the
  tabular model is as good or better, transfer into extreme-imbalance targets is unresolved for all
  model classes, and RF is the strongest in-distribution detector.
- **Adaptation.** Pairwise domain-adversarial training (DANN) does not help and can slightly hurt;
  multi-source leave-one-domain-out adaptation improves transfer into SPEDIA from 0.36 to 0.46.

## Approach

Every dataset is mapped to a canonical event tuple `(u, h, a, t, y)` and aggregated to a
**user-day** prediction unit, with an aligned, **content-free** feature space (counts, activity
flags, timing) plus **causal per-user deviation features** computed only from strictly earlier
history. Baselines span anomaly detectors, sequence models, and static graph networks; the
proposed detector is a domain-adaptive **temporal graph neural network** combining memory-based
message passing, causal self-supervised next-action pretraining, and domain-adversarial learning,
with an explainability-based transfer diagnostic (proposed; full empirical evaluation is future
work). PR-AUC is the headline metric with base-rate lift and stratified bootstrap confidence
intervals; zero-shot and domain-adaptation results are reported in separate matrices so
target-assisted adaptation is never presented as source-only generalization.

## Repository layout

```
crossdataset-insider-threat/
├── README.md            # this file
├── DATASETS.md          # dataset access, labelling, harmonisation
├── METHODS.md           # evaluation protocol and metrics
├── CITATION.cff         # citation metadata
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT (code); datasets retain their own licences
└── (src/, scripts/, configs/, results/ — added with the evaluation release)
```

## Reproducibility notes

- **No leakage.** Feature normalisation and class weighting are fit on source training data only;
  causal deviation features use strictly earlier user history; target labels are reserved for final
  evaluation (never for training, model selection, early stopping, or thresholding).
- **Honest metrics.** PR-AUC + base-rate lift + stratified bootstrap CIs; transfer deltas are
  claimed only when the CI excludes zero.
- **In-distribution controls.** Every diagonal is reported under both a chronological temporal split
  and a user-disjoint split.
- Large data files and run artefacts are git-ignored; only code, configs, and summary tables are
  tracked.

## Citation

The accompanying paper is under review; a citation entry will be finalised on acceptance. Basic
metadata is in [`CITATION.cff`](CITATION.cff). Until then, please contact the authors before using
or reporting these results.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). Datasets are governed by their
original providers' licences (CERT, SPEDIA/Zenodo, LANL; and, for future work, TWOS and DARPA OpTC)
and are not redistributed here.
