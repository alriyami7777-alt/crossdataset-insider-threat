# Summary result tables

These are the compact, human-readable summary CSVs used to fill the paper's tables.
Large per-seed artefacts, model checkpoints, feature caches, and raw data are not tracked
(see the top-level `.gitignore`).

- `paper_numbers.csv`        — every manuscript value (token -> value) with its source file.
- `transfer_matrix_phaseA.csv` — full source x target transfer results (RF and temporal GNN).
- `transfer5d_into_spedia.csv` — into-SPEDIA cells (per seed) with CIs.
- `domain_gap.csv`, `domain_gap_summary.csv`, `domain_gap_5domain.csv` — proxy-A-distance / MMD per pair.
- `ablation_into_spedia.csv` — component ablation (into-SPEDIA); may be partial until the run completes.
