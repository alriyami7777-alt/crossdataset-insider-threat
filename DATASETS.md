# Datasets

The camera-ready benchmark evaluates **CERT, SPEDIA and LANL**. None of the datasets are redistributed here. Obtain each one from its original source and point `ITD_DATA_ROOT` at it (see [`camera_ready_strict_2026_09/README.md`](camera_ready_strict_2026_09/README.md)).

## Evaluated in the paper

| Dataset | Type | Source | Role |
|---|---|---|---|
| **CERT** r4.2 / r5.2 / r6.2 | synthetic | CMU-SEI Insider Threat Test Dataset (Glasser & Lindauer, 2013) | Source domains. r4.2 and r5.2 are also targets. r6.2 is a source only (44 positive user-days, below the criterion of 50). |
| **SPEDIA** | real | Zenodo `10.5281/zenodo.15495572` | Target and source. CERT-derived rows are removed. |
| **LANL** | real | LANL Comprehensive Multi-Source Cyber-Security Events (Kent, 2015) | Target and source. Positives are labelled red-team authentication activity. |

## Labelling and filtering

- **CERT**
  - A user-day is positive if it contains an answer-key malicious event.
  - Web events are sampled uniformly (p = 0.05, seed 7) for every user.
- **SPEDIA**
  - A user-day is positive if an event is "Highly/Midly Suspicious" or has Wazuh level ≥ 8.
  - Every row with `Decoder_name == 'cert'` is excluded.
  - The audited real-only subset has:
    - 20,619 rows;
    - zero `dtaa.com` occurrences, zero `cert` tokens and zero CERT-style identifiers;
    - only auditd, PAM, syscheck and JSON sources.
  - It is not subsampled. There are 82 positive user-days out of 256 (base rate 0.32).
- **LANL**
  - A user-day is positive if it contains a labelled red-team authentication event.
  - Machine accounts are removed.
  - Authentication events are sampled uniformly (p = 0.02, seed 7) for every user.
  - There are 158 positive user-days out of 389,323. 158 of the 176 red-team user-days remain observable after sampling.
  - The chronological split has no positives in the test period, so LANL is evaluated in distribution under the user-disjoint split only.

## Future extensions (not evaluated in the paper)

- **TWOS** (Harilal et al., 2017): real gamified masquerader/traitor data, available under a data-sharing agreement.
- **DARPA OpTC:** host- and process-centric telemetry that does not map cleanly onto the user-day unit. A host-day or process-level unit would be needed.
