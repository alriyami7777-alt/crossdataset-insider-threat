# Datasets

This study evaluates cross-dataset generalization across **three independently constructed
insider-threat families (five domains)**. None of the datasets are redistributed here; obtain each
from its original source and place it under a local `data/` directory (git-ignored).

## Evaluated in this study

| Dataset | Type | Source | Role |
|---|---|---|---|
| **CERT** r4.2 / r5.2 / r6.2 | synthetic | CMU-SEI Insider Threat Test Dataset (Glasser & Lindauer, 2013) | Controlled source family; answer-key traitor labels. Three releases used as separate domains. |
| **SPEDIA** | real | Zenodo `10.5281/zenodo.15495572` | Primary real target; CERT-derived rows removed for a clean comparison. |
| **LANL** | real | LANL Comprehensive Multi-Source Cyber-Security Events (Kent, 2015) | Real intrusion-oriented domain; labelled red-team (lateral-movement) activity. |

### Labelling
- **CERT** — a user-day is positive if it contains an answer-key malicious event (restricted to the
  official insiders).
- **SPEDIA** — positive if an event is "Highly/Midly Suspicious" **or** Wazuh level ≥ 8. Every row
  with `Decoder_name == 'cert'` is excluded; the audited real-only subset is 20,619 rows with zero
  `dtaa.com` occurrences, zero `cert` tokens, and zero CERT-style identifiers (auditd / PAM /
  syscheck / JSON sources only). Positive-class base rate ≈ 0.41.
- **LANL** — positive if the user-day contains a labelled red-team authentication event. Because
  red-team activity is confined to a sub-window, the in-distribution diagonal is reported under the
  **user-disjoint** split only (a chronological split leaves no positives in the test window).

Positive-class base rates differ by roughly three orders of magnitude across domains (CERT
≈ 0.2–0.3%, LANL ≈ 0.04%, SPEDIA ≈ 41%), which is why base-rate lift and per-cell PR-AUC are
reported alongside any aggregate statistic. The CERT r6.2 diagonal is **not evaluable** (only 44
positive user-days) and is excluded as a target and from the aggregate generalization gap, though
it is retained as a training source.

## Named future extensions (not evaluated here)

- **TWOS** (Harilal et al., 2017) — real gamified masquerader/traitor data, gated behind a SUTD
  data-sharing agreement. A natural breadth extension; identified as future work.
- **DARPA OpTC** — a feasibility probe on the evaluation-window eCAR archives (SysClient0201, the
  primary red-team host) found real-user attribution of only ≈5%; the telemetry is host- and
  process-centric (FLOW/PROCESS/MODULE events emitted by NETWORK SERVICE / SYSTEM), so it does not
  map cleanly onto the user-day prediction unit. Its ground truth is a set of malicious eCAR
  **record IDs** (the correct labelling scheme when added). Adapting the benchmark to a host-day or
  process-lineage unit is left for future work.

## Harmonisation

All datasets are mapped to a canonical event tuple `(u, h, a, t, y)` — user, host, action from a
fixed vocabulary, timestamp, label — and aggregated to the **user-day** unit. Raw actions are mapped
to a shared vocabulary (unsupported actions map to a reserved `unknown` class rather than being
dropped). Dataset identifiers, raw user/host names, and source-specific record identifiers are
deliberately excluded so models cannot exploit corpus fingerprints. See [`METHODS.md`](METHODS.md).
