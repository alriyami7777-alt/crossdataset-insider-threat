# Methods and Protocol

This document summarises the evaluation protocol so results in this repository are
interpretable and reproducible. It mirrors the methodology of the paper, which evaluates
**CERT, SPEDIA, and LANL** (TWOS and DARPA OpTC are named future extensions; see
[`DATASETS.md`](DATASETS.md)).

## Prediction unit and label

The common unit of analysis is the **user-day**. For user `u` and day `d`, the
binary label is positive when at least one event in that interval satisfies the
dataset-specific malicious-event or malicious-span rule. This preserves each
dataset's documented labelling semantics while enforcing a shared target.

## Canonical event and graph

Every raw event is mapped to a canonical tuple `(u, h, a, t, y)` — user, host,
action from a fixed vocabulary `A`, timestamp, label. Each domain is represented
as a heterogeneous temporal graph over user and host nodes with action-typed,
timestamped edges. Unsupported raw actions map to a reserved "unknown" class
rather than being dropped.

## Aligned, content-free features

User-day feature vectors contain counts, binary activity indicators, and timing
variables computable consistently across datasets. **Per-user causal deviation
features** express each value relative to the same user's strictly-earlier
history:

```
dev_x(u, d) = ( x_{u,d} - mean_{u,<d} ) / ( std_{u,<d} + eps )
```

computed only from days before `d` (zero on a user's first day). This is
leakage-safe under temporal, user-disjoint, and cross-dataset evaluation.
Dataset identifiers, raw user/host names, and source-specific record IDs are
excluded.

## Evaluation settings

1. **Zero-shot cross-dataset generalization (primary).** Train on a source domain
   with labelled data only; evaluate on each target with **no** access to target
   labels, unlabelled target data, target statistics, or target model-selection
   signals. Produces a source-by-target PR-AUC matrix `M[i][j]`.
2. **Unsupervised domain adaptation (UDA), reported separately.** Labelled source
   plus **unlabelled** target used for domain-adversarial training. Because the
   target distribution participates in training, these results are never included
   in the zero-shot claim.
3. **Leave-one-domain-out (LODO).** Train on the union of all domains except the
   held-out target; test on the held-out target. Tests whether source diversity
   improves transfer.

## In-distribution controls

For every dataset the diagonal is reported under both a **chronological temporal
split** and a **user-disjoint split** (no shared users between train and test).
These controls distinguish genuine in-distribution skill from temporal proximity
or user memorisation.

## Metrics

- **PR-AUC** is the headline metric because positive user-days are rare.
- **Base-rate lift** (`PR-AUC / base_rate`) is reported alongside, since class
  prevalence differs sharply across datasets (CERT ≈ 0.2–0.3%, LANL ≈ 0.04%,
  SPEDIA ≈ 41%).
- **Generalization gap**: uniformly-weighted mean of diagonal cells minus
  uniformly-weighted mean of off-diagonal cells (the degenerate CERT r6.2 diagonal
  is excluded). Reported as a transparency statistic — because it mixes targets whose
  base rates differ by ~1000×, the per-cell transfer matrix is the primary evidence.
- All metrics are accompanied by **stratified bootstrap confidence intervals**;
  transfer deltas are claimed only when the CI excludes zero.

## Domain gap

Distributional distance between domains is reported with a domain-classifier
proxy-Â-distance and MMD on the aligned feature space, as a **descriptive** indicator
of where transfer is hardest. It is not treated as a quantitative law: proxy-Â-distance
is saturated near its maximum for almost all pairs (within-CERT pairs are near-maximal
yet still transfer well, while transfers into LANL collapse at comparable distance), so
the per-cell transfer matrix remains the primary evidence.

## Explainability as a transfer diagnostic

Integrated gradients over the **aligned user-day features** produce per-domain attribution
profiles; comparing source and target profiles separates signals that stay influential
across datasets from those that behave as dataset-specific artefacts. Action-embedding
attribution and temporal-edge occlusion are complementary event-level diagnostics outlined
for future analysis rather than quantified here. These explanations are interpreted
diagnostically, not causally.
