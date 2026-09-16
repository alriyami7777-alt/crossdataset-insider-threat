# Methods and protocol (camera-ready)

This document summarises the protocol behind the camera-ready results in [`camera_ready_strict_2026_09/`](camera_ready_strict_2026_09/).

**Evaluated domains:** CERT r4.2, r5.2 and r6.2; SPEDIA; LANL.

**Not evaluated:** TWOS and DARPA OpTC are future extensions only.

## Prediction unit and labels

The unit of labelling, prediction and evaluation is the **user-day**. A user-day is positive when it contains at least one event that meets the dataset-specific malicious-event rule:

- **CERT:** an answer-key event.
- **SPEDIA:** "Highly/Midly Suspicious", or Wazuh level ≥ 8.
- **LANL:** a red-team authentication event.

## Event sampling (label-independent)

Two event sources are subsampled, with the same rate for every user and a fixed seed (7):

- **CERT web (`http.csv`):** each event is kept with probability 0.05.
- **LANL authentication (non-machine accounts):** each event is kept with probability 0.02.

**Rules**

- The keep decision depends only on the random draw. It does not use user identity, ground-truth rosters or labels.
- Labels are attached after sampling.
- All other CERT sources are kept in full.
- SPEDIA is not subsampled.

**Verification.** Row-level checks are published in `camera_ready_strict_2026_09/results/manifests/sampling_verification_*.json`.

**Observability.** For LANL, 158 of the 176 red-team user-days remain observable after sampling.

## Canonical events and features

- **Canonical events.** Every event is mapped to `(u, h, a, t, y)` using a 16-action vocabulary. No evaluated event maps to `unknown`.
- **User-day features.** Aligned, content-free counts, activity flags and timing variables.
- **Causal per-user deviation features.** For each feature:

  ```
  dev_x(u, d) = ( x_{u,d} - mean_{u,<d} ) / ( std_{u,<d} + eps )
  ```

  - Only strictly earlier days of the same user are used, via a one-day shift.
  - The first day of each user is set to 0.
- **Standardisation.** Fitted on source training users only.

## Models

- **Random Forest (reference model)**
  - 300 trees with balanced-subsample class weights.
  - Evaluated on the full source-by-target matrix and on both in-distribution splits.
- **Temporal GNN**
  - Architecture:
    - 64-dimensional user and host GRU memories, updated in chronological event order.
    - A 16-dimensional action embedding.
    - Causal self-supervised next-action pretraining.
  - Time handling: `use_time_encoding=False`, and the time channel is zeroed. Temporal information enters only through chronological memory updates.
  - Training:
    - AdamW optimiser: learning rate 3e-4, weight decay 1e-4.
    - Early stopping on 15% of the source training users.
    - Deterministic GPU settings.
  - Evaluated on:
    - the in-distribution controls for CERT r4.2, CERT r5.2 and SPEDIA;
    - the transfers CERT r4.2→SPEDIA and CERT r5.2→SPEDIA.
  - The full GNN transfer matrix was not computed.

## Evaluation

- **Zero-shot transfer**
  - Train on the complete source domain and score the complete target domain.
  - No target data, statistics or model-selection signal is used.
- **In-distribution controls**
  - **Chronological split:** the cut is at the 0.7 quantile of user-days (RF) or of event timestamps (GNN).
  - **User-disjoint split:** 70/30 by user, stratified by whether the user has any positive user-day.
- **Evaluable targets**
  - A domain is a target only if it has at least 50 positive user-days. CERT r6.2 (44) is therefore a source only.
  - LANL has no positive user-days in its chronological test period, so its chronological score is n/e.
- **Metrics**
  - PR-AUC (average precision) and base-rate lift, over five seeds (0–4).
  - Model differences: mean paired per-seed difference with a 95% percentile bootstrap interval (5,000 resamples).
  - No aggregate significance test is reported.
- **Generalization gap**
  - Formula:

    ```
    Δ = mean_{j∈T} M[j][j] − mean_{(i,j)∈P} M[i][j],   P = {(i,j): i∈S, j∈T, i≠j}
    ```

  - The diagonal uses the user-disjoint split, giving Δ = 0.409.
  - Δ is reported as a transparency statistic alongside the per-cell values and lifts.

## Out of scope for the camera-ready results

The following were not reported in the camera-ready paper:

- target-assisted domain adaptation;
- leave-one-domain-out training;
- domain-gap (proxy-A/MMD) analysis;
- component ablations;
- explainability-based diagnostics.
