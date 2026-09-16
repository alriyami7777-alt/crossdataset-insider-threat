# Camera-ready evidence (IJACSA MS-17-9-0257), September 2026

This folder holds the **authoritative summary results** reported in the camera-ready paper. They were produced by [`../scripts/strict_camera_ready_2026_09_15.py`](../scripts/strict_camera_ready_2026_09_15.py).

- **Superseded:** earlier development results that were previously in this repository are superseded and were removed.
- **Not published:**
  - raw data and parquet caches;
  - model checkpoints;
  - prediction-level user data.
- **Path placeholders:** local absolute paths in the published files are replaced by `<REPO_ROOT>`, `<DATA_ROOT>` and `<PYTHON_EXE>`.

## What changed relative to the earlier development runs

1. **Label-independent uniform sampling.** A fixed seed (7) is used, at the same rate for every user:
   - CERT web events are kept with probability 0.05.
   - Eligible LANL authentication events are kept with probability 0.02.

   Ground-truth insider and red-team rosters are not used in the sampling decision, and labels are attached afterwards. The earlier implementation retained events at a different rate for ground-truth insider or red-team users. The affected experiments were rerun.
2. **Temporal GNN without explicit time encoding.**
   - The model runs with `use_time_encoding=False`, and the time channel is zeroed.
   - Temporal information enters only through chronological user/host GRU memory updates.
   - As a result, no timestamp statistic of the scored target stream is used.

## Files

| File | Content |
|---|---|
| `results/STRICT_RF_MATRIX.csv` | Random Forest source×target PR-AUC and lift (5 seeds; mean/sd). Rows marked `evaluable=True` enter the gap. |
| `results/STRICT_RF_PER_SEED.csv` | Random Forest per-seed cells. |
| `results/STRICT_RF_GAP.json` | Evaluable-set generalization gap: T, P, diagonal and off-diagonal means, and Δ. |
| `results/STRICT_TEMPORAL_CONTROLS.csv` | Random Forest chronological-split diagonals. |
| `results/STRICT_RF_CACHE_MANIFEST.csv` | Per-domain strict cache counts (events, user-days, positive user-days, action counts) and SHA-256 hashes. |
| `results/c2/STRICT_C2_BOOTSTRAP.csv` | Temporal-GNN vs RF on CERT r4.2→SPEDIA and CERT r5.2→SPEDIA: paired Δ, 95% bootstrap CI (5,000 resamples) and per-seed Δ. |
| `results/c2/STRICT_C2_GNN_RF.csv`, `STRICT_C2_PER_SEED.csv` | Per-model and per-seed C2 values. |
| `results/controls/STRICT_CONTROLS_TABLE.csv` | In-distribution controls (RF and GNN × user-disjoint and chronological). |
| `results/controls/STRICT_CONTROLS_SUMMARY.csv`, `STRICT_CONTROLS_PER_SEED.csv` | Control details. |
| `results/controls/evaluability.json` | Evaluability rule (≥ 50 positive user-days). Records CERT r6.2 = 44 and the LANL chronological test set = 0 positives. |
| `results/controls/chrono_split_documentation.json`, `time_invariance_probe.json` | Split construction, and a check that the GNN output does not depend on timestamps. |
| `results/manifests/sampling_verification_*.json` | Row-level verification of the uniform sampling for each domain. |
| `results/manifests/environment.json`, `results/c2/environment_at_c2_start.json` | Software and hardware. |
| `results/manifests/*_cache.json`, `build_summary.json`, `lanl_temporal_evaluability.json` | Cache build records. |

`STRICT_RF_GAP.json` contains the key `historical_gap_0.484`. It records that the pre-correction value was audited only; it is **not** a reported result.

## Headline values

| Quantity | Value |
|---|---|
| Evaluable targets T | CERT r4.2, CERT r5.2, SPEDIA, LANL (CERT r6.2 is a source only) |
| RF mean in-distribution PR-AUC (user-disjoint, over T) | 0.542 |
| RF mean zero-shot PR-AUC (16 cells) | 0.133 |
| Generalization gap Δ | 0.409 |
| CERT r4.2→SPEDIA: RF / GNN / Δ (95% CI) | 0.315 / 0.438 / +0.123 [+0.075, +0.169] |
| CERT r5.2→SPEDIA: RF / GNN / Δ (95% CI) | 0.363 / 0.332 / −0.031 [−0.061, −0.012] |

### In-distribution PR-AUC

| Domain | RF user-disjoint | RF chronological | GNN user-disjoint | GNN chronological |
|---|---|---|---|---|
| CERT r4.2 | 0.5497 | 0.6660 | 0.4443 | 0.6211 |
| CERT r5.2 | 0.6254 | 0.8150 | 0.4776 | 0.7554 |
| SPEDIA | 0.9895 | 0.9924 | 0.8072 | 0.9720 |
| LANL | 0.0036 | n/e | not run* | n/e |
| CERT r6.2 | n/e | n/e | n/e | n/e |

\* The GNN was not trained on LANL. `STRICT_CONTROLS_TABLE.csv` shows this cell as `n/e`; `STRICT_CONTROLS_SUMMARY.csv` has no GNN row for LANL.

**Interpretation**

- The representation effect is **source-dependent**. No general graph advantage is claimed.
- The camera-ready paper reports no leave-one-domain-out, domain-adaptation, full GNN-matrix, aggregate Wilcoxon or explainability result. The driver contains `lodo`, `lodo-dry-run` and `gnn-matrix` stages, but these were not used for the paper.

## Environment used

- **Software:**
  - Python 3.11.15; NumPy 2.4.4; pandas 3.0.3; scikit-learn 1.9.0; SciPy 1.17.1; pyarrow 25.0.0.
  - PyTorch 2.12.1 with CUDA 13.0.
- **Hardware:** NVIDIA GeForce RTX 5070 Laptop GPU (8 GB) and 31 GB RAM.
- **Determinism settings:**
  - `PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
  - Deterministic PyTorch/cuDNN, with cuDNN benchmarking disabled.

## Running the driver

### 1. Obtain the third-party data

Get CERT r4.2/r5.2/r6.2 with the answer keys, SPEDIA (`logs_SPEDIA.csv`) and LANL (`auth`, `redteam`) from their original providers. Arrange them as follows:

```
<DATA_ROOT>/CERT/r4.2  <DATA_ROOT>/CERT/r5.2  <DATA_ROOT>/CERT/r6.2  <DATA_ROOT>/CERT/answers
<DATA_ROOT>/SPEDIA/logs_SPEDIA.csv
<DATA_ROOT>/lanl/      (auth + redteam files)
```

### 2. Set the paths

- Set `ITD_DATA_ROOT=<DATA_ROOT>`, or pass `--raw-cert42 / --raw-cert52 / --raw-cert62 / --raw-answers / --raw-lanl`.
- Optionally set `ITD_REPORT_DIR`.

### 3. Run the stages from the repository root

```
python -m scripts.strict_camera_ready_2026_09_15 --stage preflight
python -m scripts.strict_camera_ready_2026_09_15 --stage all      # caches, C2, RF matrix, controls, summary
```

`--stage all` never starts the LODO stages.

**Individual stages**

- `build-c2`
- `c2`
- `build-rest`
- `rf`
- `temporal`
- `summarize`

**Seeds.** The default is `--seeds 0,1,2,3,4`.

### Dependency on pre-correction caches

The driver was run with read-only access to the caches of the earlier pipeline. These are:

- `results/cache/transfer_5d/`, built by `scripts/run_transfer_matrix_5domain.py`;
- the earlier LANL event cache under `<DATA_ROOT>/lanl/`.

The driver uses them in two ways:

- **SPEDIA:** the strict SPEDIA cache is a byte-identical copy of the provenance-filtered SPEDIA cache.
- **CERT and LANL:** the rebuilt events are verified row by row against the earlier caches. The results are in `results/manifests/sampling_verification_*.json`.

These caches are not distributed. `--stage preflight` reports whether they are present.
