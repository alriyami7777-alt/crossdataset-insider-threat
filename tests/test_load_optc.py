"""Synthetic eCAR shard smoke test for load_optc (no real OpTC dump required)."""
from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

import pandas as pd

from src.data.features import user_day_features
from src.data.loaders import load_optc
from src.data.schema import (
    ACTIONS, ACTION, CANONICAL_COLUMNS, DATASET, INSIDER_TYPE, LABEL,
    SRC_HOST, TIMESTAMP, USER, validate,
)


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _ecar(hostname, obj, action, ts, principal, props=None, **extra):
    rec = {
        "timestamp": ts,
        "id": extra.pop("id", "00000000-0000-0000-0000-000000000001"),
        "hostname": hostname,
        "object": obj,
        "action": action,
        "actorID": "00000000-0000-0000-0000-000000000002",
        "principal": principal,
        "properties": props or {},
    }
    rec.update(extra)
    return rec


def test_load_optc_synthetic_ecar():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        eval_dir = root / "ecar" / "evaluation"
        ben_dir = root / "ecar" / "benign"

        # Red-team day on SysClient0201; benign day on same + another host
        rt_host = "SysClient0201.systemia.com"
        other = "SysClient0099.systemia.com"
        user = r"SYSTEMIACOM\alice"
        system = r"NT AUTHORITY\SYSTEM"

        eval_recs = [
            _ecar(rt_host, "USER_SESSION", "LOGIN",
                  "2019-09-23T09:00:00.000-04:00", user,
                  {"requesting_user": user, "user": user}),
            _ecar(rt_host, "PROCESS", "CREATE",
                  "2019-09-23T10:15:00.000-04:00", user,
                  {"command_line": "powershell.exe -enc AA==",
                   "image_path": r"\Device\HarddiskVolume1\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
                   "user": user}),
            _ecar(rt_host, "FLOW", "START",
                  "2019-09-23T10:16:00.000-04:00", user,
                  {"src_ip": "142.20.56.10", "dest_ip": "132.197.158.98",
                   "src_port": "49152", "dest_port": "80", "l4protocol": "6",
                   "direction": "outbound", "user": user}),
            # SYSTEM event inside red-team window -> host-day / pseudo attribution
            _ecar(rt_host, "FILE", "WRITE",
                  "2019-09-23T11:00:00.000-04:00", system,
                  {"file_path": r"C:\Users\alice\exfil.dat", "user": system}),
            # Unmapped object/action must become "unknown", never dropped
            _ecar(rt_host, "REGISTRY", "EDIT",
                  "2019-09-23T11:05:00.000-04:00", user,
                  {"key": r"HKLM\Software\Evil", "user": user}),
        ]
        ben_recs = [
            _ecar(rt_host, "USER_SESSION", "LOGIN",
                  "2019-09-17T09:00:00.000-04:00", user,
                  {"user": user}),
            _ecar(rt_host, "PROCESS", "CREATE",
                  "2019-09-17T10:00:00.000-04:00", user,
                  {"command_line": "notepad.exe", "user": user}),
            _ecar(other, "PROCESS", "CREATE",
                  "2019-09-17T10:30:00.000-04:00", r"SYSTEMIACOM\bob",
                  {"command_line": "calc.exe", "user": r"SYSTEMIACOM\bob"}),
            # Pure SYSTEM benign -> should be dropped (machine account, no keep)
            _ecar(other, "PROCESS", "CREATE",
                  "2019-09-17T11:00:00.000-04:00", system,
                  {"command_line": "svchost.exe", "user": system}),
        ]
        # Repeat benign PROCESS rows so benign_frac=1.0 keeps a stable set
        for i in range(20):
            ben_recs.append(_ecar(
                other, "FILE", "READ",
                f"2019-09-17T12:{i:02d}:00.000-04:00", r"SYSTEMIACOM\bob",
                {"file_path": f"C:\\tmp\\f{i}.txt", "user": r"SYSTEMIACOM\bob"},
            ))

        _write_jsonl_gz(eval_dir / "SysClient0201.json.gz", eval_recs)
        _write_jsonl_gz(ben_dir / "benign.json.gz", ben_recs)

        gt = pd.DataFrame({
            "hostname": ["SysClient0201"],
            "start": ["2019-09-23T00:00:00-04:00"],
            "end": ["2019-09-23T23:59:59-04:00"],
        })
        gt.to_csv(root / "OpTCRedTeamGroundTruth.csv", index=False)

        df = load_optc(
            str(root), benign_frac=1.0, seed=7, use_cache=True,
        )
        assert list(df.columns) == CANONICAL_COLUMNS
        assert validate(df)
        assert (df[DATASET] == "optc").all()
        assert set(df[ACTION]).issubset(set(ACTIONS))
        assert "unknown" in set(df[ACTION])  # REGISTRY/EDIT
        assert int(df[LABEL].sum()) >= 1
        assert (df.loc[df[LABEL] == 1, INSIDER_TYPE] == "apt_redteam").all()

        # FLOW remote dst differs from src
        flow = df[df[ACTION] == "http"]
        assert len(flow) >= 1
        assert (flow[SRC_HOST] != flow["dst_host"]).any()

        # SYSTEM file write on red-team day attributed to alice (host-day) or host:
        rt_users = set(df.loc[df[LABEL] == 1, USER])
        assert len(rt_users) > 0
        assert "alice" in rt_users or any(u.startswith("host:") for u in rt_users)

        feat = user_day_features(df, deviation=False)
        assert feat["label"].sum() >= 1

        # Cache reuse
        df2 = load_optc(str(root), benign_frac=1.0, seed=7, use_cache=True)
        assert len(df2) == len(df)


def test_load_optc_synth_fallback():
    df = load_optc(None)
    assert validate(df)
    assert (df[DATASET] == "optc").all()
