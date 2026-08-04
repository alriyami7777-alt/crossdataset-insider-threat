# Does It Generalize? A Cross-Dataset Study of Graph-Based Insider Threat Detection Beyond CERT

Reproducibility repository for the paper *"Does It Generalize? A Cross-Dataset
Study of Graph-Based Insider Threat Detection Beyond CERT"* (Q. M. M. Alriyami and
M. M. B. Mohamad, Universiti Teknologi Malaysia). Target venue: *International
Journal of Advanced Computer Science and Applications* (IJACSA).

> **Status:** scaffold / work in progress. The evaluation code, configuration
> files, and final result tables are being finalised and will be added to this
> repository at paper submission. This README documents the intended structure so
> the release is complete and reviewable on day one. No results reproduced here
> should be treated as final until the corresponding release is tagged.

## Overview

Most graph-based insider threat detectors report strong numbers on a single
version of the synthetic CERT corpus, leaving open whether they learn transferable
insider behaviour or dataset-specific artefacts. This project provides, to the best
of our knowledge, the first systematic **zero-shot cross-dataset generalization
benchmark** for graph-based insider threat detection, evaluating representative
anomaly, sequence, static-graph, and temporal-graph models across **CERT, SPEDIA,
LANL, and TWOS** under a common source-to-target protocol.

The benchmark reports source-by-target performance matrices, temporal and
user-disjoint in-distribution controls, a generalization gap, and bootstrap
confidence intervals. Unsupervised domain adaptation (which uses unlabelled target
data) is reported **separately** from zero-shot generalization so the two settings
are never conflated.

## Contributions

- **A cross-dataset generalization benchmark** for insider threat detection with a
  shared user-day prediction unit, an aligned content-free feature space, and
  source-by-target evaluation matrices.
- **A domain-adaptive temporal graph detector** combining memory-based temporal
  message passing, causal next-action self-supervised pretraining, and
  domain-adversarial learning, with an explainability layer used as a
  transfer diagnostic.

## Planned repository structure

```
crossdataset-insider-threat/
├── README.md                 # this file
├── LICENSE                   # MIT (code)
├── CITATION.cff              # how to cite this work
├── requirements.txt          # Python dependencies
├── .gitignore                # excludes raw data, model artefacts, LaTeX aux
├── docs/
│   ├── DATASETS.md           # dataset provenance, access, and licensing
│   └── METHODS.md            # schema, labels, splits, protocol, metrics
├── src/                      # loaders, models, training/evaluation (added at submission)
│   ├── data/                 # schema, loaders, feature engineering
│   ├── models/               # baselines + temporal graph detector
│   └── train/                # splits, cross-dataset runner, metrics
├── scripts/                  # thin CLI entry points (run_matrix.py, run_domain_gap.py, ...)
├── configs/                  # default.yaml and experiment configs
└── results/                  # result CSVs and figures (added at submission)
```

Raw datasets and large model artefacts are **not** stored in this repository; see
[`DATASETS.md`](DATASETS.md) for how to obtain each dataset from its
official source.

## Datasets

| Dataset | Role | Access |
|---|---|---|
| CERT (r4.2 / r5.2 / r6.2) | Synthetic source / overfitting exhibit | CMU-SEI / Kaggle mirror (public) |
| SPEDIA | Real benchmark target | Zenodo (public); provenance filter applied |
| LANL | Real enterprise stress test | LANL cyber1 mirror (public, CC0) |
| TWOS | Real masquerader + traitor | SUTD release agreement (request required) |

Datasets are used under their respective licences and are **not redistributed**
here. See [`DATASETS.md`](DATASETS.md).

## Installation

```bash
git clone https://github.com/alriyami7777-alt/crossdataset-insider-threat.git
cd crossdataset-insider-threat
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The exact package versions used for the reported experiments will be frozen
(`pip freeze`) and committed alongside the code release at submission.

## Usage (planned)

Once the code is released, experiments will be driven by thin CLI entry points
wired to `configs/default.yaml`:

```bash
python scripts/run_matrix.py      --config configs/default.yaml   # source-by-target PR-AUC matrix
python scripts/run_domain_gap.py  --config configs/default.yaml   # proxy A-distance / MMD domain-gap table
```

## Reproducibility

- **Prediction unit:** user-day, with a dataset-specific malicious label rule
  mapped to a common binary target.
- **In-distribution controls:** chronological temporal split and user-disjoint
  split (no train == test).
- **Metrics:** PR-AUC (headline, because positive user-days are rare) plus
  base-rate lift, with stratified bootstrap confidence intervals.
- **Leakage safety:** per-user deviation features are computed causally from
  strictly earlier history; normalisation is fitted on permitted training data
  only; target labels are never used for training, model selection, or
  thresholding.

See [`METHODS.md`](METHODS.md) for the full protocol.

## Citation

If you use this benchmark, please cite the paper (see [`CITATION.cff`](CITATION.cff)).
A full citation with venue and DOI will be added once the paper is published.

## License

Code in this repository is released under the [MIT License](LICENSE). Third-party
datasets remain under their own licences and are not covered by this licence.

## Contact

Qasim Mohamed Muhanna Alriyami — `mohamedmuhanna@graduate.utm.my`
Faculty of Computing, Universiti Teknologi Malaysia.
