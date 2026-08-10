"""
Low-RAM streaming CERT → user-day features (CPU only).

Aggregates activity CSVs chunk-by-chunk into the same BASE_FEATURE_COLUMNS as
``user_day_features``, then applies causal deviation features. Avoids materializing
the full event frame (important when a GPU job already holds large CERT loads).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from src.data.features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, _add_causal_deviations
from src.data.loaders import (
    _ACT,
    _CERT_DATE,
    _CERT_EMAIL_ACT,
    _CERT_FILE_ACT,
    _cert_malicious_userdays,
)
from src.data.schema import ACTIONS, LABEL, USER

log = logging.getLogger(__name__)


def _map_actions(chunk: pd.DataFrame, fixed_action=None, activity_map=None):
    if activity_map is not None and "activity" in chunk.columns:
        return chunk["activity"].map(activity_map).fillna(fixed_action or "unknown")
    if fixed_action is not None:
        return pd.Series(fixed_action, index=chunk.index)
    return chunk["activity"].map(_ACT).fillna("unknown")


def _aggregate_chunk(chunk: pd.DataFrame, *, fixed_action=None, activity_map=None) -> pd.DataFrame:
    """One partial user-day aggregate frame for a raw CERT activity chunk."""
    ts = pd.to_datetime(chunk["date"], format=_CERT_DATE, errors="coerce")
    ok = ts.notna()
    if not ok.any():
        return pd.DataFrame()
    d = pd.DataFrame({
        USER: chunk.loc[ok, "user"].astype(str).to_numpy(),
        "day": ts.loc[ok].dt.floor("D"),
        "hour": ts.loc[ok].dt.hour.to_numpy(dtype=np.int16),
        "action": _map_actions(chunk.loc[ok], fixed_action=fixed_action, activity_map=activity_map).to_numpy(),
        "host": (
            chunk.loc[ok, "pc"].astype(str).to_numpy()
            if "pc" in chunk.columns
            else np.full(int(ok.sum()), "unknown", dtype=object)
        ),
    })
    d["offhours"] = ((d["hour"] < 7) | (d["hour"] > 19)).astype(np.int8)

    counts = (
        d.pivot_table(index=[USER, "day"], columns="action", values="hour",
                      aggfunc="count", fill_value=0)
        .reindex(columns=ACTIONS, fill_value=0)
    )
    counts.columns = [f"cnt_{a}" for a in ACTIONS]

    agg = d.groupby([USER, "day"], sort=False).agg(
        n_events=("hour", "count"),
        frac_offhours=("offhours", "mean"),
        hour_mean=("hour", "mean"),
        hour_sum=("hour", "sum"),
        hour_sumsq=("hour", lambda s: float((s.astype(np.float64) ** 2).sum())),
        hosts=("host", lambda s: frozenset(s)),
    )
    out = counts.join(agg)
    return out.reset_index()


def _combine_partials(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge partial user-day aggregates (sum counts / union hosts / recompute means)."""
    if not parts:
        return pd.DataFrame()
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.DataFrame()
    if len(parts) == 1:
        return parts[0]
    df = pd.concat(parts, ignore_index=True)
    cnt_cols = [f"cnt_{a}" for a in ACTIONS]

    def _union_hosts(series):
        out = set()
        for x in series:
            if isinstance(x, (set, frozenset)):
                out |= set(x)
            elif pd.isna(x):
                continue
            else:
                out.add(x)
        return frozenset(out)

    df = df.copy()
    df["_off_n"] = df["frac_offhours"].astype(np.float64) * df["n_events"].astype(np.float64)
    g = df.groupby([USER, "day"], sort=False)
    summed = g[cnt_cols + ["n_events", "hour_sum", "hour_sumsq", "_off_n"]].sum()
    hosts = g["hosts"].agg(_union_hosts)
    out = summed.drop(columns=["_off_n"]).copy()
    out["frac_offhours"] = summed["_off_n"] / out["n_events"].clip(lower=1)
    out["hour_mean"] = out["hour_sum"] / out["n_events"].clip(lower=1)
    out["hosts"] = hosts
    return out.reset_index()


def _finalize(acc: pd.DataFrame, mal_days: set) -> pd.DataFrame:
    """Convert combined aggregates → FEATURE_COLUMNS with deviation=True."""
    n = acc["n_events"].to_numpy(dtype=np.float64)
    sum_h = acc["hour_sum"].to_numpy(dtype=np.float64)
    sum_h2 = acc["hour_sumsq"].to_numpy(dtype=np.float64)
    # pandas groupby .std() uses ddof=1; n==1 → 0 (fillna in user_day_features)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(n > 1, (sum_h2 - (sum_h ** 2) / n) / (n - 1), 0.0)
    hour_std = np.sqrt(np.maximum(np.nan_to_num(var, nan=0.0), 0.0))

    out = acc[[USER, "day"] + [f"cnt_{a}" for a in ACTIONS]].copy()
    out["n_events"] = acc["n_events"].astype(np.int64)
    out["n_distinct_hosts"] = acc["hosts"].map(len).astype(np.int64)
    out["frac_offhours"] = acc["frac_offhours"].astype(np.float64)
    out["hour_mean"] = acc["hour_mean"].astype(np.float64)
    out["hour_std"] = hour_std
    days = pd.to_datetime(out["day"]).dt.strftime("%Y-%m-%d")
    out[LABEL] = [
        1 if (u, d) in mal_days else 0
        for u, d in zip(out[USER].astype(str), days)
    ]
    for c in BASE_FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = 0.0
    out = _add_causal_deviations(out)
    for c in FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = 0.0
    return out


def stream_cert_user_day_features(
    path: str,
    release: str,
    answers_dir: str,
    sources=("logon", "device", "file", "email"),
    http_mode: str = "insider_aware",
    benign_http_frac: float = 0.05,
    seed: int = 7,
    chunksize: int = 1_000_000,
    combine_every: int = 8,
) -> pd.DataFrame:
    """Build deviation=True user-day features without a full event DataFrame."""
    if not os.path.isdir(answers_dir):
        raise FileNotFoundError(f"answers_dir not found: {answers_dir}")
    mal_days, insiders = _cert_malicious_userdays(answers_dir, release=release)
    log.info(
        "stream_cert: path=%s release=%s insiders=%d mal_userdays=%d http_mode=%s",
        path, release, len(insiders), len(mal_days), http_mode,
    )
    partials: list[pd.DataFrame] = []

    def _flush():
        nonlocal partials
        if len(partials) > 1:
            combined = _combine_partials(partials)
            partials = [combined]
            log.info("stream_cert: combined partials -> %d user-days", len(combined))

    def _add_part(part: pd.DataFrame):
        if part is None or len(part) == 0:
            return
        partials.append(part)
        if len(partials) >= combine_every:
            _flush()

    def _read_source(name, **kw):
        fp = os.path.join(path, f"{name}.csv")
        if not os.path.isfile(fp):
            log.info("stream_cert: skip missing %s", fp)
            return
        log.info("stream_cert: reading %s ...", fp)
        n = 0
        for chunk in pd.read_csv(fp, dtype=str, chunksize=chunksize, keep_default_na=False):
            _add_part(_aggregate_chunk(chunk, **kw))
            n += len(chunk)
        log.info("stream_cert: finished %s rows≈%d", name, n)
        _flush()

    if "logon" in sources:
        _read_source("logon")
    if "device" in sources:
        _read_source("device")
    if "file" in sources:
        _read_source("file", fixed_action="file_write", activity_map=_CERT_FILE_ACT)
    if "email" in sources:
        _read_source("email", fixed_action="email_send", activity_map=_CERT_EMAIL_ACT)

    if http_mode != "skip":
        fp = os.path.join(path, "http.csv")
        if not os.path.isfile(fp):
            log.info("stream_cert: skip missing http.csv")
        elif http_mode == "insider_aware":
            rng = np.random.default_rng(seed)
            ins_tot = ben_tot = 0
            log.info("stream_cert: reading http.csv (insider_aware) ...")
            n = 0
            for chunk in pd.read_csv(
                fp, dtype=str, chunksize=chunksize, keep_default_na=False
            ):
                is_ins = chunk["user"].isin(insiders).to_numpy()
                keep = is_ins | (rng.random(len(chunk)) < benign_http_frac)
                ins_tot += int(is_ins.sum())
                ben_tot += int((keep & ~is_ins).sum())
                if keep.any():
                    _add_part(
                        _aggregate_chunk(chunk.loc[keep], fixed_action="http")
                    )
                n += len(chunk)
                if n % 5_000_000 < chunksize:
                    log.info(
                        "stream_cert: http scanned≈%d kept_keys_pending=%d",
                        n, sum(len(p) for p in partials),
                    )
            log.info(
                "stream_cert: http scanned≈%d insider≈%d benign_kept≈%d",
                n, ins_tot, ben_tot,
            )
            _flush()
        else:
            raise ValueError(
                f"stream_cert supports http_mode skip|insider_aware, got {http_mode!r}"
            )

    acc = _combine_partials(partials)
    if acc is None or len(acc) == 0:
        raise RuntimeError(f"no user-days aggregated from {path}")
    out = _finalize(acc, mal_days)
    log.info("stream_cert: user-days=%d pos=%d", len(out), int(out[LABEL].sum()))
    return out
