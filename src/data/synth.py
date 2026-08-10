"""
Synthetic event generator producing two *different* domains (synthA, synthB).

Purpose: let the whole pipeline run end-to-end before the real datasets arrive,
AND deliberately encode a domain shift between synthA and synthB so the
cross-dataset degradation the paper is about actually shows up in the smoke test.
This is scaffolding for wiring/CI only -- never a substitute for real data in the
paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    TIMESTAMP, USER, SRC_HOST, DST_HOST, ACTION, OBJECT, LABEL,
    INSIDER_TYPE, DATASET, CANONICAL_COLUMNS,
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate(
    dataset: str = "synthA",
    n_users: int = 60,
    n_hosts: int = 25,
    days: int = 20,
    events_per_user_day: int = 40,
    malicious_frac: float = 0.08,
    insider_type: str = "traitor",
    domain_shift: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    domain_shift in [0,1] warps the benign action distribution and the working
    hours, so a model tuned to synthA (shift=0) sees a different world in synthB
    (shift>0). Malicious behaviour is injected differently for traitor vs
    masquerader to mirror the CERT-vs-LANL/TWOS contrast.
    """
    rng = _rng(seed)
    users = [f"{dataset}_u{ i:03d}" for i in range(n_users)]
    hosts = [f"{dataset}_h{ i:03d}" for i in range(n_hosts)]

    # Benign action mix; domain_shift tilts it toward web/email vs file/auth.
    base_mix = np.array([0.14, 0.10, 0.18, 0.16, 0.10, 0.02,
                         0.03, 0.03, 0.10, 0.05, 0.05, 0.02, 0.01, 0.005, 0.005, 0.0])
    tilt = np.zeros_like(base_mix)
    tilt[[8, 9, 10]] += 0.10 * domain_shift   # http, email_send, email_recv up
    tilt[[3, 4, 2]] -= 0.10 * domain_shift    # file_read, file_write, auth down
    from .schema import ACTIONS
    mix = np.clip(base_mix + tilt, 1e-4, None)
    mix = mix / mix.sum()

    n_malicious = max(1, int(round(n_users * malicious_frac)))
    malicious_users = set(rng.choice(users, size=n_malicious, replace=False))

    rows = []
    start = np.datetime64("2025-01-01T00:00:00")
    for d in range(days):
        # working-hours center drifts with domain_shift
        center = 13 + 3 * domain_shift
        for u in users:
            is_mal = u in malicious_users
            n_ev = rng.poisson(events_per_user_day)
            for _ in range(n_ev):
                hour = int(np.clip(rng.normal(center, 2.5), 0, 23))
                minute = int(rng.integers(0, 60))
                ts = start + np.timedelta64(d, "D") + np.timedelta64(hour, "h") + np.timedelta64(minute, "m")
                a_id = rng.choice(len(ACTIONS), p=mix)
                action = ACTIONS[a_id]
                src = rng.choice(hosts)
                dst = src if rng.random() < 0.6 else rng.choice(hosts)
                obj = f"obj{int(rng.integers(0, 500))}"
                label, itype = 0, "benign"

                # Inject malicious behaviour on a subset of days for mal users.
                if is_mal and d >= days - 6 and rng.random() < 0.35:
                    label, itype = 1, insider_type
                    if insider_type == "traitor":
                        # exfil-like: heavy file_write/usb at odd hours
                        action = rng.choice(["file_write", "usb_connect", "email_send"])
                        hour = int(rng.choice([2, 3, 22, 23]))
                        ts = start + np.timedelta64(d, "D") + np.timedelta64(hour, "h")
                    else:  # masquerader: auth to unusual hosts, breadth
                        action = rng.choice(["auth", "ssh", "logon"])
                        dst = rng.choice(hosts)  # scattered targets
                rows.append((ts, u, src, dst, action, obj, label, itype, dataset))

    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    return df.sort_values(TIMESTAMP).reset_index(drop=True)


def two_domains(seed: int = 0):
    """Convenience: a source (synthA, no shift) and a target (synthB, shifted)."""
    a = generate("synthA", insider_type="traitor", domain_shift=0.0, seed=seed)
    b = generate("synthB", insider_type="masquerader", domain_shift=0.7, seed=seed + 1)
    return a, b
