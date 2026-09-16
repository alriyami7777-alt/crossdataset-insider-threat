# Script status

## Authoritative (camera-ready)

- `strict_camera_ready_2026_09_15.py`: the only driver behind the camera-ready results in [`../camera_ready_strict_2026_09/`](../camera_ready_strict_2026_09/).
  - It imports `run_transfer_matrix_5domain.py` (cache paths and feature alignment) and modules under `src/`.
  - Its `lodo`, `lodo-dry-run` and `gnn-matrix` stages were **not** used for the paper.

## Historical development scripts (not authoritative)

All other scripts in this folder come from earlier development phases and are kept for provenance only. They are **not** used for the camera-ready results.

**Why they are not authoritative**

- Several use the earlier sampling rule, which retained events for ground-truth insider and red-team users at a different rate.
- Several use a temporal GNN with explicit time encoding.
- Several contain local absolute paths from the development machine.

**What they produce.** Their outputs, for example domain-gap, ablation, domain-adaptation, leave-one-domain-out and explainability experiments, are **not** reported in the camera-ready paper. Such results must not be cited as findings of the paper.

**Modules under `src/`.** These modules are likewise not used by the camera-ready driver:

- `src/models/dann.py`
- `src/explain/xai.py`
- `src/eval/domain_gap.py`
- `src/train/ablation_ladder.py`
