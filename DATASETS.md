# Datasets

This project evaluates cross-dataset generalization across four insider-threat /
enterprise-security datasets. **No dataset is redistributed in this repository.**
Each must be obtained from its official source under its own licence and placed
under a local `data/` directory (which is git-ignored).

| Dataset | Insider type | Role in benchmark | Source |
|---|---|---|---|
| CERT r4.2 / r5.2 / r6.2 | Traitor (behavioural) | Synthetic source; overfitting exhibit | CMU-SEI Insider Threat Test Dataset |
| SPEDIA | Real (descriptive + Wazuh severity) | Real benchmark target | Zenodo record 15495572 |
| LANL | Masquerader / lateral movement | Real enterprise stress test | Los Alamos "Comprehensive, Multi-Source Cyber-Security Events" |
| TWOS | Masquerader + traitor | Real, independently collected | SUTD TWOS release (agreement required) |

## Access notes

- **CERT** — Synthetic corpus with answer-key labels. Positive user-days are
  built from the official `answers/insiders.csv` scenario files, gated to the
  official insider set for each release.
- **SPEDIA** — Public on Zenodo (DOI `10.5281/zenodo.15495572`). The release
  contains both real and CERT-derived rows; a **provenance filter** removes every
  row with `Decoder_name == 'cert'` so CERT-derived records do not leak into the
  target evaluation. A user-day is positive when it contains an event described as
  "Highly Suspicious" / "Midly Suspicious" or with a Wazuh level >= 8.
- **LANL** — Public mirror (CC0) of the comprehensive labelled events set
  (DOI `10.17021/1179829`); use the labelled events set, **not** the unlabelled
  2014 authentication-only set. Red-team labels are confined to a limited window,
  so LANL's in-distribution diagonal uses the **user-disjoint** split rather than
  a chronological split.
- **TWOS** — Requires acceptance of the SUTD release agreement. Treated as
  breadth (not required to reproduce the core results).

## Harmonisation

All datasets are converted to a common canonical event `(u, h, a, t, y)` — user,
host, action from a fixed vocabulary, timestamp, and label — and a shared
content-free user-day feature space. Dataset-specific tokens, URLs, email/message
content, and raw identifiers are deliberately excluded so that models cannot
exploit corpus-specific artefacts. See [`METHODS.md`](METHODS.md).
