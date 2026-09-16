# Does It Generalize? A Cross-Dataset Study of Graph-Based Insider Threat Detection Beyond CERT

This repository supports the camera-ready version of the IJACSA paper (manuscript MS-17-9-0257). The paper presents a **zero-shot cross-dataset benchmark** for insider threat detection across **CERT, SPEDIA and LANL**. In the zero-shot setting, a detector is trained on one source dataset and scored on an independently constructed target dataset. No labelled or unlabelled target data are used during training.

> **Authoritative evidence:** the camera-ready results come **only** from [`camera_ready_strict_2026_09/`](camera_ready_strict_2026_09/) and the driver [`scripts/strict_camera_ready_2026_09_15.py`](scripts/strict_camera_ready_2026_09_15.py). Other scripts and modules in this repository are historical development code; see [`scripts/HISTORICAL_SCRIPTS.md`](scripts/HISTORICAL_SCRIPTS.md). Raw datasets are **not** redistributed.

## Evaluated datasets

- **CERT r4.2, r5.2 and r6.2:** synthetic enterprise activity with answer-key traitor labels.
  - CERT r6.2 is used as a **training source only**. It has 44 positive user-days, below the evaluability criterion of 50, so it is not evaluable (n/e) as a target.
- **SPEDIA:** real security-event data (Zenodo `10.5281/zenodo.15495572`). CERT-derived rows (`Decoder_name == 'cert'`) are removed, leaving 20,619 real rows.
- **LANL:** real authentication events with labelled red-team activity. Machine accounts are removed.

**Future extensions (not evaluated in the paper):**

- **TWOS**
- **DARPA OpTC**

See [`DATASETS.md`](DATASETS.md).

## Camera-ready protocol (summary)

- **Label-independent uniform sampling.** Two event sources are subsampled with a fixed seed (7), at the same rate for every user:
  - CERT web events are kept with probability 0.05.
  - Eligible LANL authentication events are kept with probability 0.02.

  Ground-truth insider and red-team rosters are **not** used. Labels are attached after the sampled event universe is fixed.
- **User-day prediction unit** with a 16-action canonical vocabulary.
- **Causal per-user deviation features.** These use only strictly earlier days of the same user.
- **Models**
  - A Random Forest reference model, run on the full source-by-target matrix.
  - A memory-based temporal GNN, run on the in-distribution controls and on CERT r4.2→SPEDIA and CERT r5.2→SPEDIA.
    - It uses chronological user/host GRU memories and causal self-supervised next-action pretraining.
    - It has **no explicit time encoding**, so no target-derived timestamp statistic is used.
- **In-distribution controls:** a chronological split and a user-disjoint split.
- **Metrics:** PR-AUC with base-rate lift, five seeds (0–4), and paired seed-level bootstrap intervals (5,000 resamples).
- **Generalization gap:** computed over the evaluable targets T = {CERT r4.2, CERT r5.2, SPEDIA, LANL} using the user-disjoint diagonal. There are 16 evaluable transfer cells.

## Final results (camera-ready)

| Quantity | Value |
|---|---|
| RF mean in-distribution PR-AUC (evaluable targets) | 0.542 |
| RF mean zero-shot PR-AUC (16 evaluable cells) | 0.133 |
| Generalization gap | **0.409** |
| CERT r4.2→SPEDIA, Temporal-GNN − RF | **+0.123**, 95% CI [+0.075, +0.169] (GNN supported) |
| CERT r5.2→SPEDIA, Temporal-GNN − RF | **−0.031**, 95% CI [−0.061, −0.012] (RF supported) |

**Interpretation**

- Transfer performance falls substantially from in-distribution performance.
- The representation effect is **source-dependent**. No general graph advantage is claimed.

**Not claimed in the camera-ready paper**

- No multi-source leave-one-domain-out (LODO) result.
- No domain-adaptation result.
- No aggregate Wilcoxon test.
- No empirical explainability (XAI) result.

## Repository layout

```
camera_ready_strict_2026_09/   authoritative camera-ready summary results + protocol README
scripts/strict_camera_ready_2026_09_15.py   authoritative driver (paths use <DATA_ROOT> placeholders)
scripts/run_transfer_matrix_5domain.py      helper module imported by the driver
src/                            loaders, features, splits, RF adapter, temporal GNN (imported by the driver)
scripts/HISTORICAL_SCRIPTS.md   list of non-authoritative development scripts
DATASETS.md, METHODS.md         data access and camera-ready protocol
configs/default.yaml            legacy development config (not read by the camera-ready driver)
```

## Reproducing

See [`camera_ready_strict_2026_09/README.md`](camera_ready_strict_2026_09/README.md) for:

- the environment;
- where to place the third-party data (`ITD_DATA_ROOT`);
- the driver stages.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Datasets:** governed by their original providers' licences and not redistributed here.
