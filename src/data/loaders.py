"""
Per-dataset loaders. Each returns a dataframe in the canonical schema.

Real-data column mappings are stubbed with the fields we already know and clear
TODOs where the exact native column names must be confirmed against the files
once they are in hand. Until a path is provided, `load(name)` falls back to the
synthetic generator so downstream code always has something to run on.

IMPORTANT (SPEDIA): SPEDIA mixes CERT-derived rows with real-exercise rows. For
clean cross-dataset tests, filter to the real-exercise subset via `spedia_real_only`.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .schema import (
    CANONICAL_COLUMNS, TIMESTAMP, USER, SRC_HOST, DST_HOST, ACTION, OBJECT,
    LABEL, INSIDER_TYPE, DATASET, ACTIONS,
)
from . import synth


# ---- CERT -------------------------------------------------------------------
# Verified against CERT r4.2 (2026-08). Activity file schemas:
#   logon.csv/device.csv : id,date,user,pc,activity   (Logon/Logoff ; Connect/Disconnect)
#   file.csv             : id,date,user,pc,filename,content
#   http.csv             : id,date,user,pc,url,content            (~14.5 GB -> sampled/skipped)
#   email.csv            : id,date,user,pc,to,cc,bcc,from,size,attachments,content
# Date format: MM/DD/YYYY HH:MM:SS.
# Labels: answers/insiders.csv (dataset,scenario,details,user,start,end) lists the 70 r4.2
#   insiders; answers/r4.2-<scn>/r4.2-<scn>-<USER>.csv hold the exact malicious events
#   (ragged rows: col0=type, col1=id, col2=date, col3=user, ...). We build the set of
#   malicious (user, day) pairs from those files -> a user-day is positive iff it appears there.
import os as _os
import csv as _csv
import glob as _glob
from datetime import datetime as _dt

_CERT_DATE = "%m/%d/%Y %H:%M:%S"


def _cert_official_insiders(answers_dir, release="4.2"):
    """Authoritative insider set for this release, from answers/insiders.csv."""
    fp = _os.path.join(answers_dir, "insiders.csv")
    users = set()
    with open(fp, newline="") as fh:
        r = _csv.DictReader(fh)
        for row in r:
            if str(row.get("dataset", "")).strip() == release:
                users.add(row["user"].strip())
    return users


def _cert_malicious_userdays(answers_dir, release="4.2"):
    """Malicious (user, 'YYYY-MM-DD') pairs from the observable files, restricted to
    the OFFICIAL insiders so benign context accounts (e.g. a supervisor replying in a
    scenario-3 thread) are not mislabelled.

    Supports both layouts present in the shared CERT answers pack:
      * nested dirs: ``answers/r5.2-1/r5.2-1-USER.csv`` (r4.2 / r5.2)
      * flat files:  ``answers/r6.2-1.csv`` (r6.2 aggregates users per scenario)
    """
    official = _cert_official_insiders(answers_dir, release=release)
    mal_days = set()
    nested = _glob.glob(
        _os.path.join(answers_dir, f"r{release}-*", f"r{release}-*-*.csv")
    )
    flat = _glob.glob(_os.path.join(answers_dir, f"r{release}-*.csv"))
    files = sorted(set(nested) | set(flat))
    print(
        f"[load_cert] answers release={release}: "
        f"{len(nested)} nested + {len(flat)} flat answer files "
        f"(dir={answers_dir})"
    )
    for f in files:
        with open(f, newline="") as fh:
            for row in _csv.reader(fh):
                if len(row) < 4:
                    continue
                # row = [type, {id}, date, user, pc, ...]
                date_s, user = row[2], row[3]
                if user not in official:
                    continue
                try:
                    day = _dt.strptime(date_s, _CERT_DATE).strftime("%Y-%m-%d")
                except ValueError:
                    continue
                mal_days.add((user, day))
    return mal_days, official


def _cert_source(path, name, action_of, object_col):
    """Yield canonical rows from one CERT activity csv, chunked (memory-safe)."""
    fp = _os.path.join(path, f"{name}.csv")
    if not _os.path.exists(fp):
        print(f"[load_cert] skip missing {name}.csv")
        return
    for chunk in pd.read_csv(fp, dtype=str, chunksize=1_000_000, keep_default_na=False):
        yield chunk, action_of, object_col


def load_cert(path="CERT_r4.2", sources=("logon", "device", "file", "email"),
              http_mode="skip", http_nrows=5_000_000, benign_http_frac=0.05,
              include_http=None, release="4.2", seed=7, answers_dir=None,
              dataset_tag=None):
    """Load CERT into canonical schema. http.csv is ~14.5 GB, so it is controlled by
    ``http_mode``:
      * "skip"           -> no http (default; fastest).
      * "head"           -> first ``http_nrows`` rows. WARNING: chronological head
                            biases the temporal split (early benign http floods the
                            train period). Use only for a quick smoke, not results.
      * "insider_aware"  -> keep ALL http rows for the official insiders (so the
                            scenario-1 upload and scenario-2 job-site evidence is
                            retained across the whole timeline) plus a random
                            ``benign_http_frac`` sample of everyone else. This keeps
                            size bounded WITHOUT wrecking the temporal split.
    ``include_http`` (bool) is kept for back-compat: True -> "head", False -> "skip".
    Labels come from the answers/ observables at the user-day level.

    ``answers_dir``: optional override. CERT r5.2/r6.2 event trees on disk often
    ship without a local ``answers/`` folder; pass the shared answers pack
    (e.g. from CERT_r4.2/answers) which contains r5.2-* / r6.2-* ground truth.
    ``dataset_tag``: value written to the DATASET column (default ``\"cert\"``).
    """
    if path is None:
        return synth.generate("cert", insider_type="traitor", domain_shift=0.1, seed=11)

    if answers_dir is None:
        answers_dir = _os.path.join(path, "answers")
    if not _os.path.isdir(answers_dir):
        raise FileNotFoundError(
            f"CERT answers_dir not found: {answers_dir!r} (path={path!r}, "
            f"release={release}). Pass answers_dir= to the shared answers pack."
        )
    print(f"[load_cert] path={path} release={release} answers_dir={answers_dir}")
    mal_days, insiders = _cert_malicious_userdays(answers_dir, release=release)
    print(f"[load_cert] {len(insiders)} insiders, {len(mal_days)} malicious user-days")
    ds_tag = dataset_tag or "cert"

    frames = []

    def _emit(df_raw, action_map_col, obj_col, fixed_action=None, activity_map=None):
        d = df_raw
        ts = pd.to_datetime(d["date"], format=_CERT_DATE, errors="coerce")
        if activity_map is not None and "activity" in d.columns:
            # r5.2/r6.2 file & email carry an 'activity' column (Send/View,
            # File Open/Copy/Write/Delete); r4.2 does not (falls back to fixed_action).
            action = d["activity"].map(activity_map).fillna(fixed_action or "unknown")
        elif fixed_action is not None:
            action = pd.Series(fixed_action, index=d.index)
        else:
            action = d[action_map_col].map(_ACT).fillna("unknown")
        host = d["pc"].astype(str) if "pc" in d else pd.Series("unknown", index=d.index)
        if obj_col and obj_col in d:
            obj = d[obj_col].astype(str).str.slice(0, 120)
        else:
            obj = pd.Series("unknown", index=d.index)
        out = pd.DataFrame({
            TIMESTAMP: ts, USER: d["user"].astype(str),
            SRC_HOST: host, DST_HOST: host,
            ACTION: action, OBJECT: obj,
            LABEL: 0, INSIDER_TYPE: "benign", DATASET: ds_tag,
        })
        return out.dropna(subset=[TIMESTAMP])

    # logon/device use their 'activity' column; file/http/email use a fixed action
    if "logon" in sources:
        for chunk, _, _ in _cert_source(path, "logon", None, None):
            frames.append(_emit(chunk, "activity", None))
    if "device" in sources:
        for chunk, _, _ in _cert_source(path, "device", None, None):
            frames.append(_emit(chunk, "activity", None))
    if "file" in sources:
        for chunk, _, _ in _cert_source(path, "file", None, "filename"):
            frames.append(_emit(chunk, None, "filename", fixed_action="file_write",
                                activity_map=_CERT_FILE_ACT))
    if "email" in sources:
        for chunk, _, _ in _cert_source(path, "email", None, "to"):
            frames.append(_emit(chunk, None, "to", fixed_action="email_send",
                                activity_map=_CERT_EMAIL_ACT))
    # back-compat with the old include_http flag
    if include_http is True and http_mode == "skip":
        http_mode = "head"
    elif include_http is False:
        http_mode = "skip"

    if http_mode != "skip":
        fp = _os.path.join(path, "http.csv")
        if not _os.path.exists(fp):
            print("[load_cert] skip: http.csv not found")
        elif http_mode == "head":
            n = 0
            for chunk in pd.read_csv(fp, dtype=str, chunksize=1_000_000, keep_default_na=False):
                frames.append(_emit(chunk, None, "url", fixed_action="http"))
                n += len(chunk)
                if http_nrows and n >= http_nrows:
                    print(f"[load_cert] http 'head' capped at ~{n} rows "
                          f"(WARNING: biases the temporal split)")
                    break
        elif http_mode == "insider_aware":
            rng = np.random.default_rng(seed)
            ins_tot = ben_tot = 0
            for chunk in pd.read_csv(fp, dtype=str, chunksize=1_000_000, keep_default_na=False):
                is_ins = chunk["user"].isin(insiders).to_numpy()
                keep = is_ins | (rng.random(len(chunk)) < benign_http_frac)
                ins_tot += int(is_ins.sum())
                ben_tot += int((keep & ~is_ins).sum())
                frames.append(_emit(chunk[keep], None, "url", fixed_action="http"))
            print(f"[load_cert] http insider_aware: kept ALL insider http (~{ins_tot}) "
                  f"+ {benign_http_frac:.0%} benign (~{ben_tot}); timeline preserved "
                  f"for temporal split")
        else:
            raise ValueError(f"unknown http_mode={http_mode!r}")

    df = pd.concat(frames, ignore_index=True)
    # user-day labels from answers
    day = df[TIMESTAMP].dt.strftime("%Y-%m-%d")
    key = list(zip(df[USER], day))
    mask = pd.Series([k in mal_days for k in key], index=df.index)
    df.loc[mask, LABEL] = 1
    df.loc[mask, INSIDER_TYPE] = "traitor"
    return df[CANONICAL_COLUMNS].sort_values(TIMESTAMP).reset_index(drop=True)


# CERT activity-token -> canonical action (for logon/device 'activity' column)
_ACT = {
    "Logon": "logon", "Logoff": "logoff",
    "Connect": "usb_connect", "Disconnect": "usb_disconnect",
}
# r5.2/r6.2 file & email 'activity' columns -> canonical action (r4.2 lacks these,
# so those releases fall back to file_write / email_send).
_CERT_FILE_ACT = {
    "File Open": "file_read", "File Copy": "file_write",
    "File Write": "file_write", "File Delete": "file_delete",
}
_CERT_EMAIL_ACT = {"Send": "email_send", "View": "email_recv", "Receive": "email_recv"}


def load_cert_releases(release_paths: dict, answers_dir=None, **kw):
    """Convenience hedge: load several CERT releases as SEPARATE domains.

    release_paths: {name: (path, release)} e.g.
        {"cert42": ("CERT_r4.2","4.2"), "cert52": ("CERT_r5.2","5.2"), "cert62": ("CERT_r6.2","6.2")}
    ``answers_dir``: shared answers pack used for every release when individual
    trees lack a local ``answers/`` (typical for extracted r5.2/r6.2).
    Returns {name: canonical_df}. Enables a cross-RELEASE generalization axis with
    no external access requests (frame as a secondary analysis: same generator,
    different populations/densities/scenarios)."""
    out = {}
    for name, (p, rel) in release_paths.items():
        print(f"[load_cert_releases] loading domain={name} path={p} release={rel}")
        out[name] = load_cert(
            p, release=rel, answers_dir=answers_dir, dataset_tag=name, **kw,
        )
        print(
            f"[load_cert_releases] {name}: n_events={len(out[name])} "
            f"pos_events={int(out[name][LABEL].sum())}"
        )
    return out


# ---- LANL -------------------------------------------------------------------
# LANL Comprehensive Multi-Source Cyber Security Events (Kent 2015).
# auth.txt[.gz] (no header, ~1.05B rows / ~7.2 GB gzipped):
#   time, src_user@dom, dst_user@dom, src_comp, dst_comp,
#   auth_type, logon_type, auth_orientation, success
#   time = integer seconds since start (range ~1..5e6 over 58 days).
# redteam.txt[.gz] (~749 rows): time, user@dom, src_comp, dst_comp
# Labels are at the user-day level from redteam (compromised-credential lateral
# movement ≈ masquerader). auth must be streamed + subsampled — never fully loaded.

_LANL_EPOCH = pd.Timestamp("2015-01-01")
_LANL_AUTH_NAMES = [
    "time", "src_user", "dst_user", "src_comp", "dst_comp",
    "auth_type", "logon_type", "auth_orientation", "success",
]
_LANL_RT_NAMES = ["time", "user", "src_comp", "dst_comp"]
_LANL_ORIENT = {"LogOn": "logon", "LogOff": "logoff"}


def _lanl_find(path, stem):
    """Resolve ``stem.txt.gz`` or ``stem.txt`` under ``path``."""
    for name in (f"{stem}.txt.gz", f"{stem}.txt"):
        fp = _os.path.join(path, name)
        if _os.path.exists(fp):
            return fp
    raise FileNotFoundError(
        f"LANL {stem}.txt[.gz] not found under {path!r}"
    )


def _lanl_seconds_to_ts(seconds):
    return _LANL_EPOCH + pd.to_timedelta(
        pd.to_numeric(seconds, errors="coerce"), unit="s"
    )


def load_lanl(path=None, mode="redteam_aware", benign_frac=0.02,
              nrows=None, keep_machine_accounts=False, seed=7):
    """Load LANL auth + redteam into the canonical schema.

    ``path``: directory containing ``auth.txt[.gz]`` and ``redteam.txt[.gz]``.
    ``mode``:
      * ``"redteam_aware"`` (default) — keep ALL auth rows whose src_user is a
        red-team user (normal + compromised days across the whole timeline)
        plus a random ``benign_frac`` sample of everyone else. Bounds size
        without wrecking the temporal split (mirrors CERT ``http_mode=
        "insider_aware"``).
      * ``"head"`` — first ``nrows`` rows only. WARNING: chronological head
        biases the temporal split; smoke use only.
    ``keep_machine_accounts``: if False (default), drop users ending in ``$``
    or matching ``^C\\d+$``. Red-team actors are U-accounts, so this is safe.
    """
    if path is None:
        return synth.generate(
            "lanl", insider_type="masquerader", domain_shift=0.6, seed=12
        )
    if mode not in ("redteam_aware", "head"):
        raise ValueError(
            f"unknown mode={mode!r}; expected 'redteam_aware' or 'head'"
        )

    auth_fp = _lanl_find(path, "auth")
    rt_fp = _lanl_find(path, "redteam")
    print(f"[load_lanl] path={path} mode={mode} auth={auth_fp} redteam={rt_fp}")

    # --- redteam ground truth (tiny; load whole) -----------------------------
    rt = pd.read_csv(
        rt_fp, header=None, names=_LANL_RT_NAMES,
        compression="infer", dtype=str, keep_default_na=False,
    )
    rt_user = rt["user"].astype(str).str.split("@", n=1).str[0]
    rt_ts = _lanl_seconds_to_ts(rt["time"])
    rt_day = rt_ts.dt.strftime("%Y-%m-%d")
    mal_days = set(zip(rt_user, rt_day))
    rt_users = set(rt_user)
    n_rt_events = len(rt)
    print(
        f"[load_lanl] redteam events={n_rt_events} | "
        f"{len(rt_users)} red-team users, {len(mal_days)} malicious user-days"
    )

    # --- stream auth ---------------------------------------------------------
    if mode == "head" and nrows is None:
        nrows = 5_000_000
        print(
            f"[load_lanl] mode='head' with nrows=None -> defaulting to "
            f"{nrows} (WARNING: biases the temporal split)"
        )
    elif mode == "head":
        print(
            f"[load_lanl] mode='head' capped at {nrows} rows "
            f"(WARNING: biases the temporal split)"
        )

    rng = np.random.default_rng(seed)
    frames = []
    n_kept = n_rt_kept = n_ben_kept = 0
    n_raw_scanned = n_machine = n_user_acct = 0
    chunksize = 1_000_000

    for chunk in pd.read_csv(
        auth_fp, header=None, names=_LANL_AUTH_NAMES,
        compression="infer", dtype=str, keep_default_na=False,
        chunksize=chunksize,
    ):
        n_raw_scanned += len(chunk)
        user = chunk["src_user"].astype(str).str.split("@", n=1).str[0]
        if not keep_machine_accounts:
            # endswith $ OR ^C\d+$ (LANL computer / machine accounts)
            is_mach = user.str.endswith("$") | user.str.match(r"^C\d+$", na=False)
            n_machine += int(is_mach.sum())
            n_user_acct += int((~is_mach).sum())
            chunk = chunk.loc[~is_mach]
            user = user.loc[~is_mach]
            if len(chunk) == 0:
                continue
        else:
            n_user_acct += len(chunk)

        is_rt = user.isin(rt_users).to_numpy()
        if mode == "redteam_aware":
            keep = is_rt | (rng.random(len(chunk)) < benign_frac)
        else:  # head — keep everything in chronological order until nrows
            keep = np.ones(len(chunk), dtype=bool)

        if not keep.any():
            continue
        sub = chunk.loc[keep].copy()
        sub_user = user.loc[keep]
        sub_is_rt = is_rt[keep]

        if mode == "head" and nrows is not None:
            remain = nrows - n_kept
            if remain <= 0:
                break
            if len(sub) > remain:
                sub = sub.iloc[:remain]
                sub_user = sub_user.iloc[:remain]
                sub_is_rt = sub_is_rt[:remain]

        ts = _lanl_seconds_to_ts(sub["time"])
        action = (
            sub["auth_orientation"].map(_LANL_ORIENT).fillna("auth")
        )
        obj = (
            sub["auth_type"].astype(str) + ":"
            + sub["logon_type"].astype(str) + ":"
            + sub["success"].astype(str)
        ).str.slice(0, 120)

        frames.append(pd.DataFrame({
            TIMESTAMP: ts,
            USER: sub_user.to_numpy(),
            SRC_HOST: sub["src_comp"].astype(str).to_numpy(),
            DST_HOST: sub["dst_comp"].astype(str).to_numpy(),
            ACTION: action.to_numpy(),
            OBJECT: obj.to_numpy(),
            LABEL: 0,
            INSIDER_TYPE: "benign",
            DATASET: "lanl",
        }))
        n_kept += len(sub)
        n_rt_kept += int(sub_is_rt.sum())
        n_ben_kept += int((~sub_is_rt).sum())

        if mode == "head" and nrows is not None and n_kept >= nrows:
            break

    if not frames:
        raise RuntimeError(
            f"[load_lanl] no auth rows kept (path={path!r}, mode={mode!r})"
        )

    df = pd.concat(frames, ignore_index=True).dropna(subset=[TIMESTAMP])
    day = df[TIMESTAMP].dt.strftime("%Y-%m-%d")
    key = list(zip(df[USER], day))
    mask = pd.Series([k in mal_days for k in key], index=df.index)
    df.loc[mask, LABEL] = 1
    df.loc[mask, INSIDER_TYPE] = "masquerader"

    mach_ratio = (n_machine / n_user_acct) if n_user_acct else float("inf")
    print(
        f"[load_lanl] auth scan: raw={n_raw_scanned} | "
        f"machine_dropped={n_machine} | user_kept={n_user_acct} | "
        f"machine:user drop ratio = {mach_ratio:.2f}:1"
    )
    ratio = (n_ben_kept / n_rt_kept) if n_rt_kept else float("inf")
    print(
        f"[load_lanl] {len(rt_users)} red-team users, "
        f"{len(mal_days)} malicious user-days, "
        f"{len(df)} rows kept "
        f"(benign:red-team row ratio = {ratio:.1f}:1; "
        f"ben={n_ben_kept}, rt={n_rt_kept})"
    )
    if mode == "redteam_aware":
        print(
            f"[load_lanl] redteam_aware: kept ALL red-team-user auth "
            f"(~{n_rt_kept}) + {benign_frac:.0%} benign (~{n_ben_kept}); "
            f"timeline preserved for temporal split"
        )
    return df[CANONICAL_COLUMNS].sort_values(TIMESTAMP).reset_index(drop=True)


# ---- OpTC -------------------------------------------------------------------
# DARPA OpTC (FiveDirections eCAR JSON-lines, gzipped shards).
# Use ONLY the evaluation window (red-team days) + 1-2 benign days — not the full ~1TB.
# Expected layout under ``path``:
#   ecar/evaluation/*.json.gz   (or evaluation/)
#   ecar/benign/*.json.gz       (or benign/)
#   <ground-truth CSV>          host + timestamp ranges (see ``_optc_load_ground_truth``)
# eCAR record: object, action, actorID, timestamp, hostname, principal, properties{...}.
# Labels: event positive if (hostname, timestamp) falls in a red-team GT interval;
#   insider_type="apt_redteam". user_day_features() then marks a user-day positive
#   if it contains >=1 red-team event.
# Note on timestamps: parse to UTC. If the OpTC clock anchor differs from wall
#   time, relative ordering is preserved; causal per-user deviation features
#   absorb a constant offset (same note as LANL).

_OPTC_SERVICE_USERS = frozenset({
    "SYSTEM", "LOCAL SERVICE", "NETWORK SERVICE", "ANONYMOUS LOGON",
    "LOCAL SYSTEM", "SERVICE", "DWM-1", "DWM-2", "UMFD-0", "UMFD-1",
})
# (object, action) -> canonical ACTIONS token. Lookup is case-insensitive.
_OPTC_ACTION = {
    ("FILE", "CREATE"): "file_write",
    ("FILE", "WRITE"): "file_write",
    ("FILE", "MODIFY"): "file_write",
    ("FILE", "RENAME"): "file_write",
    ("FILE", "READ"): "file_read",
    ("FILE", "DELETE"): "file_delete",
    ("PROCESS", "CREATE"): "process_start",
    ("PROCESS", "OPEN"): "process_start",
    ("SHELL", "COMMAND"): "cmd_exec",
    ("USER_SESSION", "LOGIN"): "logon",
    ("USER_SESSION", "INTERACTIVE"): "logon",
    ("USER_SESSION", "REMOTE"): "logon",
    ("USER_SESSION", "RDP"): "logon",
    ("USER_SESSION", "UNLOCK"): "logon",
    ("USER_SESSION", "GRANT"): "logon",
    ("USER_SESSION", "LOGOUT"): "logoff",
    ("FLOW", "START"): "http",
    ("FLOW", "MESSAGE"): "http",
    ("FLOW", "OPEN"): "http",
}
_OPTC_LOGON_ACTIONS = frozenset({
    "LOGIN", "INTERACTIVE", "REMOTE", "RDP", "UNLOCK", "GRANT",
})
_OPTC_GT_NAMES = (
    "OpTCRedTeamGroundTruth.csv",
    "redteam_intervals.csv",
    "optc_redteam_intervals.csv",
    "ground_truth.csv",
    "redteam.csv",
)


def _optc_cache_path(root, benign_frac):
    tag = f"{int(round(benign_frac * 100)):03d}"
    return _os.path.join(root, f"_cache_load_optc_bf{tag}.parquet")


def _optc_norm_host(h):
    """Strip domain suffix; OpTC hosts are SysClient####[.systemia.com]."""
    s = str(h).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    return s.split(".", 1)[0]


def _optc_norm_user(raw):
    """Return bare username from DOMAIN\\user / user@dom / plain user; else ''."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "null", "unknown", ""):
        return ""
    if "\\" in s:
        s = s.split("\\")[-1]
    elif "@" in s:
        s = s.split("@", 1)[0]
    return s.strip()


def _optc_is_machine_account(user):
    """True for Windows service / machine accounts (dropped like LANL ``$`` / C#)."""
    if user is None:
        return True
    s = str(user).strip()
    if not s:
        return True
    up = s.upper()
    if up.startswith("NT AUTHORITY\\"):
        return True
    bare = _optc_norm_user(s)
    if not bare:
        return True
    bu = bare.upper()
    if bu in _OPTC_SERVICE_USERS:
        return True
    if bare.endswith("$"):
        return True
    # SysClient####$ / computer accounts
    if bu.startswith("SYSCLIENT") and bare.endswith("$"):
        return True
    return False


def _optc_parse_ts(val):
    """Parse eCAR timestamp (ISO-8601 with tz, or epoch ms/s) -> UTC Timestamp."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return pd.NaT
    if isinstance(val, (int, float, np.integer, np.floating)):
        x = float(val)
        # ms since epoch if large; else seconds
        if x > 1e12:
            return pd.to_datetime(x, unit="ms", utc=True)
        if x > 1e9:
            return pd.to_datetime(x, unit="s", utc=True)
        return pd.NaT
    s = str(val).strip()
    if not s:
        return pd.NaT
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return ts


def _optc_map_action(obj, act, props=None):
    """Map (object, action) to ACTIONS; refine FLOW by dest_port when present."""
    o = str(obj or "").strip().upper()
    a = str(act or "").strip().upper()
    if o == "FLOW" and props:
        try:
            port = int(props.get("dest_port") or props.get("dst_port") or -1)
        except (TypeError, ValueError):
            port = -1
        if port == 22:
            return "ssh"
        if port == 21:
            return "ftp"
        if port in (80, 443, 8080, 8443):
            return "http"
    return _OPTC_ACTION.get((o, a), "unknown")


def _optc_object_str(obj, props):
    """Best-effort object identifier from properties (path / cmdline / etc.)."""
    props = props or {}
    o = str(obj or "").strip().upper()
    keys = {
        "FILE": ("file_path", "new_path", "image_path"),
        "PROCESS": ("command_line", "image_path"),
        "SHELL": ("payload", "command_line", "image_path"),
        "FLOW": ("dest_ip", "src_ip", "image_path"),
        "MODULE": ("module_path", "image_path"),
        "REGISTRY": ("key", "value", "image_path"),
        "TASK": ("task_name", "path", "image_path"),
        "SERVICE": ("name", "image_path"),
        "USER_SESSION": ("requesting_user", "logon_id", "image_path"),
    }.get(o, ("image_path", "command_line", "file_path"))
    for k in keys:
        v = props.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()[:120]
    return "unknown"


def _optc_dst_host(hostname, obj, props):
    """src=hostname; dst=remote for FLOW else hostname."""
    host = str(hostname or "").strip() or "unknown"
    o = str(obj or "").strip().upper()
    props = props or {}
    if o != "FLOW":
        return host
    direction = str(props.get("direction", "")).strip().lower()
    src_ip = str(props.get("src_ip") or "").strip()
    dest_ip = str(props.get("dest_ip") or "").strip()
    if direction == "outbound" and dest_ip:
        return dest_ip
    if direction == "inbound" and src_ip:
        return src_ip
    # Fall back: prefer dest_ip if it differs from empty
    if dest_ip:
        return dest_ip
    if src_ip:
        return src_ip
    return host


def _optc_find_ground_truth(path, ground_truth=None):
    if ground_truth is not None:
        if not _os.path.exists(ground_truth):
            raise FileNotFoundError(
                f"OpTC ground_truth not found: {ground_truth!r}"
            )
        return ground_truth
    for name in _OPTC_GT_NAMES:
        fp = _os.path.join(path, name)
        if _os.path.exists(fp):
            return fp
    # Also check one level up when path points at ecar/
    parent = _os.path.dirname(_os.path.abspath(path))
    for name in _OPTC_GT_NAMES:
        fp = _os.path.join(parent, name)
        if _os.path.exists(fp):
            return fp
    raise FileNotFoundError(
        f"OpTC ground-truth CSV not found under {path!r}. "
        f"Pass ground_truth= or place one of {_OPTC_GT_NAMES} in the root. "
        f"Expected columns: hostname, start, end (ISO or epoch)."
    )


def _optc_load_ground_truth(fp):
    """Load red-team (host, start, end) intervals.

    Accepts CSV with flexible column names:
      hostname|host|src_host , start|start_time|begin , end|end_time|finish
    Optional ``day`` column (YYYY-MM-DD) expands to a full UTC day when
    start/end are absent.
    Returns list of (norm_host, start_utc, end_utc) and the set of norm hosts.
    """
    gt = pd.read_csv(fp, dtype=str, keep_default_na=False)
    cols = {c.lower().strip(): c for c in gt.columns}

    def _col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    hcol = _col("hostname", "host", "src_host", "src", "sysclient")
    scol = _col("start", "start_time", "begin", "start_ts", "t_start")
    ecol = _col("end", "end_time", "finish", "end_ts", "t_end")
    dcol = _col("day", "date")
    if hcol is None:
        raise ValueError(
            f"OpTC ground truth {fp!r} missing hostname column; "
            f"have {list(gt.columns)}"
        )
    intervals = []
    for _, row in gt.iterrows():
        host = _optc_norm_host(row[hcol])
        if not host:
            continue
        if scol and ecol and str(row[scol]).strip() and str(row[ecol]).strip():
            start = _optc_parse_ts(row[scol])
            end = _optc_parse_ts(row[ecol])
        elif dcol and str(row[dcol]).strip():
            day = pd.to_datetime(str(row[dcol]).strip(), utc=True, errors="coerce")
            if pd.isna(day):
                continue
            start = day.normalize()
            end = start + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        else:
            continue
        if pd.isna(start) or pd.isna(end):
            continue
        if end < start:
            start, end = end, start
        intervals.append((host, start, end))
    if not intervals:
        raise ValueError(
            f"OpTC ground truth {fp!r} produced 0 intervals "
            f"(need hostname + start/end or day)"
        )
    hosts = {h for h, _, _ in intervals}
    return intervals, hosts


def _optc_is_redteam(norm_host, ts, intervals_by_host):
    if not norm_host or pd.isna(ts):
        return False
    for start, end in intervals_by_host.get(norm_host, ()):
        if start <= ts <= end:
            return True
    return False


def _optc_iter_shards(path):
    """Yield gzipped eCAR jsonl shard paths (evaluation + benign; skip short/)."""
    candidates = []
    # Prefer the official ecar/ tree; fall back to flat / evaluation|benign roots.
    search_roots = []
    for sub in (
        _os.path.join(path, "ecar", "evaluation"),
        _os.path.join(path, "ecar", "benign"),
        _os.path.join(path, "evaluation"),
        _os.path.join(path, "benign"),
    ):
        if _os.path.isdir(sub):
            search_roots.append(sub)
    if not search_roots:
        if _os.path.isdir(path):
            search_roots = [path]
        else:
            raise FileNotFoundError(
                f"OpTC eCAR root not found under {path!r} "
                f"(expected ecar/evaluation + ecar/benign)"
            )

    for root in search_roots:
        # Skip the documented incomplete "short" split if it appears
        if _os.path.basename(root.rstrip("\\/")).lower() == "short":
            continue
        for dirpath, dirnames, filenames in _os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() != "short"]
            # Prefer full ecar over ecar-bro when walking a broad root
            if "ecar-bro" in dirpath.replace("\\", "/").lower():
                continue
            for fn in filenames:
                low = fn.lower()
                if low.endswith(".json.gz") or low.endswith(".jsonl.gz"):
                    candidates.append(_os.path.join(dirpath, fn))
                elif low.endswith(".json") or low.endswith(".jsonl"):
                    candidates.append(_os.path.join(dirpath, fn))
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            f"No eCAR json[.gz] shards found under {path!r}"
        )
    return candidates


def _optc_open_text(fp):
    if fp.lower().endswith(".gz"):
        return gzip.open(fp, "rt", encoding="utf-8", errors="replace")
    return open(fp, "rt", encoding="utf-8", errors="replace")


def _optc_extract_principal(rec, props, keep_machine_accounts=False):
    """Acting user from principal / properties.user / requesting_user."""
    props = props or {}
    for raw in (
        rec.get("principal"),
        props.get("user"),
        props.get("user_name"),
        props.get("requesting_user"),
        props.get("username"),
    ):
        u = _optc_norm_user(raw)
        if not u:
            continue
        if (not keep_machine_accounts) and _optc_is_machine_account(
            raw if raw is not None else u
        ):
            continue
        return u
    return ""


def _optc_build_hostday_users(shards):
    """Pass 1: dominant real user per (norm_host, day) from USER_SESSION logons."""
    counts = defaultdict(Counter)
    n_session = 0
    for fp in shards:
        with _optc_open_text(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # Cheap pre-filter before json.loads
                if '"USER_SESSION"' not in line and '"user_session"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(rec.get("object", "")).upper() != "USER_SESSION":
                    continue
                if str(rec.get("action", "")).upper() not in _OPTC_LOGON_ACTIONS:
                    continue
                props = rec.get("properties") or {}
                if not isinstance(props, dict):
                    props = {}
                user = _optc_extract_principal(rec, props)
                if not user:
                    continue
                host = _optc_norm_host(rec.get("hostname", ""))
                ts = _optc_parse_ts(rec.get("timestamp"))
                if not host or pd.isna(ts):
                    continue
                day = ts.strftime("%Y-%m-%d")
                counts[(host, day)][user] += 1
                n_session += 1
    dominant = {
        k: ctr.most_common(1)[0][0] for k, ctr in counts.items() if ctr
    }
    print(
        f"[load_optc] host-day dominant users: {len(dominant)} "
        f"(from {n_session} USER_SESSION logon records)"
    )
    return dominant


def load_optc(path=None, ground_truth=None, benign_frac=0.02,
              keep_machine_accounts=False, seed=7, use_cache=True):
    """Load DARPA OpTC eCAR shards into the canonical schema.

    ``path``: OpTC root containing ``ecar/evaluation`` + ``ecar/benign`` (or
    those folders directly) and a red-team ground-truth CSV of host + time
    ranges. We intentionally support the eval-window slice only — not the
    full ~1TB release.
    ``ground_truth``: optional explicit path to the intervals CSV.
    ``benign_frac``: keep ALL events whose (host, ts) hits a red-team interval,
    plus a random ``benign_frac`` sample of everything else (mirrors
    ``load_lanl`` redteam_aware). Default 0.02.
    ``keep_machine_accounts``: if False (default), treat SYSTEM / LOCAL SERVICE
    / NETWORK SERVICE / ``$`` accounts as absent principals (fall through to
    host-day dominant user / ``host:<hostname>``); benign rows that still
    resolve only to a machine identity are dropped.
    Cache: ``<path>/_cache_load_optc_bf002.parquet`` (tag tracks benign_frac).
    """
    if path is None:
        return synth.generate(
            "optc", insider_type="apt_redteam", domain_shift=0.5, seed=15
        )

    cache_fp = _optc_cache_path(path, benign_frac)
    if use_cache and _os.path.exists(cache_fp):
        print(f"[load_optc] cache hit: {cache_fp}")
        df = pd.read_parquet(cache_fp)
        # Light validation reprint from cache
        n_rt = int(df[LABEL].sum())
        rt_users = set(df.loc[df[LABEL] == 1, USER].astype(str))
        day = df[TIMESTAMP].dt.strftime("%Y-%m-%d")
        mal_days = set(zip(df.loc[df[LABEL] == 1, USER].astype(str), day[df[LABEL] == 1]))
        rt_hosts = set(df.loc[df[LABEL] == 1, SRC_HOST].map(_optc_norm_host))
        print(
            f"[load_optc] cached: total={len(df)} | red-team events={n_rt} | "
            f"red-team hosts={len(rt_hosts)} | red-team users={len(rt_users)} | "
            f"malicious user-days={len(mal_days)}"
        )
        assert len(rt_users) > 0, "[load_optc] #red-team users == 0 (cache)"
        assert len(mal_days) > 0, "[load_optc] #malicious user-days == 0 (cache)"
        return df[CANONICAL_COLUMNS].sort_values(TIMESTAMP).reset_index(drop=True)

    gt_fp = _optc_find_ground_truth(path, ground_truth)
    intervals, rt_hosts_gt = _optc_load_ground_truth(gt_fp)
    intervals_by_host = defaultdict(list)
    for h, s, e in intervals:
        intervals_by_host[h].append((s, e))
    for h in intervals_by_host:
        intervals_by_host[h].sort()
    print(
        f"[load_optc] path={path} ground_truth={gt_fp} | "
        f"{len(intervals)} intervals on {len(rt_hosts_gt)} red-team hosts | "
        f"benign_frac={benign_frac}"
    )

    shards = _optc_iter_shards(path)
    print(f"[load_optc] {len(shards)} eCAR shards")
    hostday_user = _optc_build_hostday_users(shards)

    rng = np.random.default_rng(seed)
    frames = []
    n_raw = n_kept = n_rt_kept = n_ben_kept = 0
    n_machine_drop = n_pseudo = n_dom_fill = n_direct = 0

    for fp in shards:
        batch_ts, batch_user, batch_src, batch_dst = [], [], [], []
        batch_act, batch_obj, batch_lab, batch_itype = [], [], [], []
        with _optc_open_text(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_raw += 1
                props = rec.get("properties") or {}
                if not isinstance(props, dict):
                    props = {}
                ts = _optc_parse_ts(rec.get("timestamp"))
                if pd.isna(ts):
                    continue
                hostname = str(rec.get("hostname") or "").strip() or "unknown"
                norm_host = _optc_norm_host(hostname) or hostname
                obj = rec.get("object", "")
                act = rec.get("action", "")
                is_rt = _optc_is_redteam(norm_host, ts, intervals_by_host)

                if is_rt:
                    keep = True
                else:
                    keep = bool(rng.random() < benign_frac)
                if not keep:
                    continue

                user = _optc_extract_principal(
                    rec, props, keep_machine_accounts=keep_machine_accounts
                )
                attrib = "direct"
                if user:
                    pass
                else:
                    day = ts.strftime("%Y-%m-%d")
                    user = hostday_user.get((norm_host, day), "")
                    if user:
                        attrib = "dom"
                    else:
                        user = f"host:{norm_host}"
                        attrib = "pseudo"

                if not keep_machine_accounts and _optc_is_machine_account(user):
                    # Rare after principal filtering; drop benign only
                    if not is_rt:
                        n_machine_drop += 1
                        continue
                    user = f"host:{norm_host}"
                    attrib = "pseudo"

                if attrib == "direct":
                    n_direct += 1
                elif attrib == "dom":
                    n_dom_fill += 1
                else:
                    n_pseudo += 1

                action = _optc_map_action(obj, act, props)
                dst = _optc_dst_host(hostname, obj, props)
                batch_ts.append(ts)
                batch_user.append(user)
                batch_src.append(hostname)
                batch_dst.append(dst)
                batch_act.append(action)
                batch_obj.append(_optc_object_str(obj, props))
                batch_lab.append(1 if is_rt else 0)
                batch_itype.append("apt_redteam" if is_rt else "benign")
                n_kept += 1
                if is_rt:
                    n_rt_kept += 1
                else:
                    n_ben_kept += 1

        if batch_ts:
            frames.append(pd.DataFrame({
                TIMESTAMP: batch_ts,
                USER: batch_user,
                SRC_HOST: batch_src,
                DST_HOST: batch_dst,
                ACTION: batch_act,
                OBJECT: batch_obj,
                LABEL: batch_lab,
                INSIDER_TYPE: batch_itype,
                DATASET: "optc",
            }))

    if not frames:
        raise RuntimeError(
            f"[load_optc] no eCAR rows kept (path={path!r}, "
            f"benign_frac={benign_frac})"
        )

    df = pd.concat(frames, ignore_index=True).dropna(subset=[TIMESTAMP])
    # Ensure UTC datetime64
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], utc=True)
    # Strip tz for parity with other loaders (naive UTC wall clock)
    try:
        df[TIMESTAMP] = df[TIMESTAMP].dt.tz_localize(None)
    except TypeError:
        pass

    day = df[TIMESTAMP].dt.strftime("%Y-%m-%d")
    mal_mask = df[LABEL] == 1
    rt_users = set(df.loc[mal_mask, USER].astype(str))
    mal_days = set(zip(df.loc[mal_mask, USER].astype(str), day[mal_mask]))
    rt_hosts_obs = set(df.loc[mal_mask, SRC_HOST].map(_optc_norm_host))
    pseudo_rate = (n_pseudo / n_kept) if n_kept else 0.0

    print(
        f"[load_optc] scan: raw={n_raw} | kept={n_kept} "
        f"(benign subsample={n_ben_kept}, red-team={n_rt_kept}) | "
        f"machine_dropped={n_machine_drop}"
    )
    print(
        f"[load_optc] total events={len(df)} | red-team events={n_rt_kept} | "
        f"red-team hosts={len(rt_hosts_obs)} | red-team users={len(rt_users)} | "
        f"malicious user-days={len(mal_days)} | "
        f"benign subsample size={n_ben_kept} | "
        f"pseudo-user fallback rate={pseudo_rate:.2%} "
        f"(direct={n_direct}, host-day fill={n_dom_fill}, pseudo={n_pseudo})"
    )
    assert len(rt_users) > 0, (
        "[load_optc] #red-team users == 0 — check ground-truth host/time match"
    )
    assert len(mal_days) > 0, (
        "[load_optc] #malicious user-days == 0 — check ground-truth intervals"
    )

    out = df[CANONICAL_COLUMNS].sort_values(TIMESTAMP).reset_index(drop=True)
    if use_cache:
        try:
            out.to_parquet(cache_fp, index=False)
            print(f"[load_optc] wrote cache {cache_fp}")
        except Exception as exc:  # noqa: BLE001 — cache is best-effort
            print(f"[load_optc] cache write skipped: {exc}")
    return out


# ---- TWOS -------------------------------------------------------------------
def load_twos(path=None):
    """TWOS: keystroke/mouse/host/network/email/logon per user; masquerader+traitor.
    TODO: unify per-source logs on timestamp+user; label from provided instance spans."""
    if path is None:
        return synth.generate("twos", insider_type="masquerader", domain_shift=0.4, seed=13)
    raise NotImplementedError("Implement TWOS multi-source unification.")


# ---- SPEDIA -----------------------------------------------------------------
# Verified against the real logs_SPEDIA.csv (72,250 rows, 22 columns):
#   Agent_name,User,Timestamp,Decoder_name,Description,Full_log,Content,Url,To,Cc,
#   Bcc,From,Attachments,Size,Size_before,Size_after,Filename,Path,Command,Level,
#   Activity,Action
# ORIGIN flag: Decoder_name == 'cert' marks CERT-derived rows (51,631 of them, 71%);
#   real-exercise rows use auditd/pam/syscheck/json decoders (20,619 rows).
# LABEL: SPEDIA grades command events in `Description` as
#   {Non|Lowly|Midly|Highly} Suspicious, correlated with Wazuh `Level`.
#   Default binary = (Highly|Midly Suspicious) OR Level>=8  -> ~24.8% positive on
#   the real-only subset. Adjust `mal_threshold` to match the companion paper.

def _spedia_action(activity, action):
    a = str(activity).lower(); act = str(action).lower()
    if a == "command":
        return "cmd_exec"
    if a == "http":
        return "http"
    if a == "email":
        return "email_send"
    if a == "session":
        if "logoff" in act:
            return "logoff"
        if "fail" in act:
            return "auth"
        return "logon"
    if a == "file":
        if "modif" in act or "added" in act:
            return "file_write"
        if "delet" in act:
            return "file_delete"
        return "file_read"
    if a == "device":
        return "usb_disconnect" if "disconnect" in act else "usb_connect"
    return "unknown"


def load_spedia(path="logs_SPEDIA.csv", real_only=False, mal_threshold="midly"):
    """Load the real SPEDIA CSV into canonical schema.

    real_only: drop CERT-derived rows (Decoder_name=='cert') for a clean
               cross-dataset comparison against CERT.
    mal_threshold: 'highly' | 'midly' | 'lowly' -- lowest suspicious grade counted
               as malicious (default 'midly': Highly+Midly). Level>=8 also counts.
    """
    if path is None:
        return synth.generate("spedia", insider_type="traitor", domain_shift=0.2, seed=14)
    raw = pd.read_csv(path, low_memory=False)

    if real_only:
        raw = raw[raw["Decoder_name"] != "cert"].reset_index(drop=True)

    desc = raw["Description"].fillna("").astype(str)
    grades = {"lowly": ["Highly", "Midly", "Lowly"],
              "midly": ["Highly", "Midly"],
              "highly": ["Highly"]}[mal_threshold]
    susp = desc.str.contains("|".join(f"{g} Suspicious" for g in grades), case=False)
    level = pd.to_numeric(raw["Level"], errors="coerce").fillna(0)
    label = (susp | (level >= 8)).astype(int)

    host = raw["Agent_name"].astype(str)
    obj = (raw.get("Command").fillna("") if "Command" in raw else "").astype(str)
    for alt in ("Url", "Filename", "Path"):
        if alt in raw:
            obj = obj.where(obj.str.len() > 0, raw[alt].fillna("").astype(str))

    df = pd.DataFrame({
        TIMESTAMP: pd.to_datetime(raw["Timestamp"], errors="coerce"),
        USER: raw["User"].fillna("unknown").astype(str),
        SRC_HOST: host,
        DST_HOST: host,
        ACTION: [_spedia_action(a, ac) for a, ac in zip(raw["Activity"], raw["Action"])],
        OBJECT: obj.replace("", "unknown"),
        LABEL: label,
        INSIDER_TYPE: "benign",
        DATASET: "spedia",
    })
    df.loc[df[LABEL] == 1, INSIDER_TYPE] = "traitor"
    return df[CANONICAL_COLUMNS].dropna(subset=[TIMESTAMP]).reset_index(drop=True)


LOADERS = {
    "cert": load_cert, "lanl": load_lanl, "optc": load_optc,
    "twos": load_twos, "spedia": load_spedia,
}


def load(name, path=None):
    if name in ("synthA", "synthB"):
        a, b = synth.two_domains()
        return a if name == "synthA" else b
    return LOADERS[name](path) if path else LOADERS[name]()
