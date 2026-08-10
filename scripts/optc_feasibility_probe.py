"""OpTC acquisition + bounded SysClient0201 feasibility probe (no loaders.py)."""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import sys
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import gdown
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "data" / "optc_probe"
ECAR_DIR = PROBE / "ecar"
MANIFEST = PROBE / "manifest.txt"
RESULTS_CSV = ROOT / "results" / "optc_probe.csv"
GT_CSV = PROBE / "optc_redteam.csv"
FOLDER_ID = "1n3kkS3KR31KUegn42yk3-e6JkZvf0Caa"
HOST = "SysClient0201"
MAX_SINGLE_BYTES = 8 * 1024**3
PROGRESS_EVERY = 50_000

SERVICE_USERS = {"SYSTEM", "LOCAL SERVICE", "NETWORK SERVICE"}


def norm_host(h: str) -> str:
    s = str(h or "").strip()
    return s.split(".", 1)[0] if s else ""


def norm_user(raw) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "null", "unknown"):
        return ""
    if "\\" in s:
        s = s.split("\\")[-1]
    elif "@" in s:
        s = s.split("@", 1)[0]
    return s.strip()


def is_service_or_machine(user: str) -> bool:
    if not user:
        return True
    up = user.upper()
    if up.startswith("NT AUTHORITY\\"):
        bare = norm_user(user).upper()
        return bare in SERVICE_USERS or bare.endswith("$") or not bare
    bare = norm_user(user)
    bu = bare.upper()
    if bu in SERVICE_USERS:
        return True
    if bare.endswith("$"):
        return True
    return False


def probe_size(file_id: str) -> int | None:
    """Best-effort size via Drive uc endpoint (follow confirm= for real CL)."""
    sess = requests.Session()
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    try:
        r = sess.get(url, stream=True, timeout=60, allow_redirects=True)
        ctype = r.headers.get("Content-Type") or ""
        cl = r.headers.get("Content-Length")
        if "text/html" in ctype:
            text_head = r.text[:20000]
            r.close()
            # Prefer explicit size in warning page, e.g. "(105.2M)" / "110M"
            m = re.search(
                r"\((\d+(?:\.\d+)?)\s*([KMG])i?B?\)", text_head, re.I
            ) or re.search(
                r"download.*?(\d+(?:\.\d+)?)\s*([KMG])B", text_head, re.I
            )
            if m:
                val = float(m.group(1))
                unit = m.group(2).upper()
                mult = {"K": 1024, "M": 1024**2, "G": 1024**3}[unit]
                return int(val * mult)
            m = re.search(r"confirm=([0-9A-Za-z_-]+)", text_head)
            uuid_m = re.search(r"uuid=([0-9a-fA-F-]+)", text_head)
            if m:
                conf_url = (
                    f"https://drive.google.com/uc?export=download&id={file_id}"
                    f"&confirm={m.group(1)}"
                )
                if uuid_m:
                    conf_url += f"&uuid={uuid_m.group(1)}"
                r2 = sess.get(conf_url, stream=True, timeout=60, allow_redirects=True)
                cl = r2.headers.get("Content-Length")
                # Don't download body
                r2.close()
                if cl and int(cl) > 10_000:  # ignore tiny HTML masquerading as CL
                    return int(cl)
            return None
        r.close()
        if cl and int(cl) > 10_000:
            return int(cl)
    except Exception as e:
        print(f"[size] {file_id}: {e}")
    return None


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    for u, s in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if n >= u:
            return f"{n / u:.2f} {s}"
    return f"{n} B"


def step1() -> None:
    print(f"STEP1 gdown.__version__ = {gdown.__version__}")
    # Ensure folder API path works (skip_download)
    assert hasattr(gdown, "download_folder"), "gdown.download_folder missing"
    from gdown.download_folder import download_folder as _df  # noqa: F401
    print("STEP1 OK: download_folder importable (no _parse_google_drive_file needed)")


def _load_manifest() -> list[dict] | None:
    if not MANIFEST.exists():
        return None
    rows = []
    with MANIFEST.open(encoding="utf-8") as fh:
        header = fh.readline()
        if not header.lower().startswith("name"):
            return None
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            size = None
            if len(parts) >= 3 and parts[2].strip():
                try:
                    size = int(parts[2])
                except ValueError:
                    size = None
            rows.append({"name": parts[0], "file_id": parts[1], "size": size})
    return rows or None


def step2_list(reuse_manifest: bool = True) -> list[dict]:
    if reuse_manifest:
        cached = _load_manifest()
        if cached and any("201-225" in r["name"] for r in cached):
            print(f"STEP2 reusing cached manifest ({len(cached)} rows) at {MANIFEST}")
            # Re-probe 201-225 sizes (previous HTML CL probes were wrong)
            for r in cached:
                if "201-225" in r["name"]:
                    print(f"STEP2 probing size for {r['name']} …")
                    r["size"] = probe_size(r["file_id"])
                    print(f"  -> {fmt_bytes(r['size'])}")
            with MANIFEST.open("w", encoding="utf-8") as fh:
                fh.write("name\tfile_id\tsize\n")
                for r in cached:
                    fh.write(
                        f"{r['name']}\t{r['file_id']}\t"
                        f"{r['size'] if r['size'] is not None else ''}\n"
                    )
            for r in cached:
                if "201-225" in r["name"]:
                    print(f"  {r['name']}  id={r['file_id']}  size={fmt_bytes(r['size'])}")
            return cached

    print("STEP2 listing Drive folder (skip_download=True)…")
    files = gdown.download_folder(id=FOLDER_ID, skip_download=True, quiet=False)
    print(f"STEP2 total files in tree: {len(files)}")

    # evaluation eCAR only (sizes filled for 201-225 candidates below)
    rows = []
    for f in files:
        path = f.path.replace("/", "\\")
        parts = path.split("\\")
        if len(parts) < 2:
            continue
        if parts[0].lower() != "ecar":
            continue
        if parts[1].lower() != "evaluation":
            continue
        rows.append({"name": path, "file_id": f.id, "size": None})

    # Probe sizes only for AIA-201-225 (SysClient0201 range) — avoid 139 HEAD round-trips
    for r in rows:
        if "201-225" in r["name"]:
            print(f"STEP2 probing size for {r['name']} …")
            r["size"] = probe_size(r["file_id"])
            print(f"  -> {fmt_bytes(r['size'])}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as fh:
        fh.write("name\tfile_id\tsize\n")
        for r in rows:
            fh.write(f"{r['name']}\t{r['file_id']}\t{r['size'] if r['size'] is not None else ''}\n")
    print(f"STEP2 wrote {MANIFEST} ({len(rows)} evaluation eCAR files)")
    for r in rows:
        print(f"  {r['name']}  id={r['file_id']}  size={fmt_bytes(r['size'])}")
    return rows


def step3_select(rows: list[dict]) -> list[dict]:
    # Match AIA-201-225 (covers SysClient0201) and eval red-team days.
    # Exclude 23Sep-Night (overnight benign). Day-1 primary → 23Sep19-red first.
    day_order = [
        ("23sep19-red", "2019-09-23"),
        ("24sep19", "2019-09-24"),
        ("25sept", "2019-09-25"),
    ]
    by_day: dict[str, list[dict]] = {k: [] for k, _ in day_order}
    for r in rows:
        p = r["name"].lower().replace("/", "\\")
        if "201-225" not in p:
            continue
        segs = p.split("\\")
        for key, _ in day_order:
            if key in segs:
                by_day[key].append(r)
                break

    print("STEP3 candidates by day:")
    for key, label in day_order:
        for r in by_day[key]:
            print(f"  [{label}/{key}] {r['name']}  size={fmt_bytes(r['size'])}")

    chosen: list[dict] = []
    for key, _ in day_order:
        for r in by_day[key]:
            if r not in chosen:
                chosen.append(r)
    if not chosen:
        raise SystemExit("STEP3: no AIA-201-225 evaluation archives matched")

    oversized = [r for r in chosen if r["size"] is not None and r["size"] > MAX_SINGLE_BYTES]
    first_key, first = "23sep19-red", by_day["23sep19-red"]
    if not first:
        for key, _ in day_order:
            if by_day[key]:
                first_key, first = key, by_day[key]
                break

    if oversized:
        print(
            f"STEP3 GUARDRAIL: archive(s) > 8 GiB detected "
            f"({', '.join(fmt_bytes(r['size']) for r in oversized)}). "
            f"Downloading ONLY first eval day ({first_key})."
        )
        chosen = first
    elif any(r["size"] is None for r in chosen) and len(chosen) > len(first):
        print(
            "STEP3 GUARDRAIL: size unknown for some multi-day archives; "
            f"downloading ONLY first eval day ({first_key}) to stay bounded."
        )
        chosen = first

    print("STEP3 CHOSEN before download:")
    for r in chosen:
        print(f"  {r['name']}  id={r['file_id']}  size={fmt_bytes(r['size'])}")
    return chosen


def step4_download(chosen: list[dict]) -> tuple[list[Path], int]:
    """Download into data/optc_probe/ecar/<day>/... to avoid ecar-last name collisions."""
    ECAR_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    total = 0
    for r in chosen:
        rel = Path(r["name"].replace("\\", "/"))
        # keep evaluation/<day>/filename (drop leading ecar/)
        parts = rel.parts
        if len(parts) >= 3 and parts[0].lower() == "ecar":
            out = ECAR_DIR.joinpath(*parts[1:])  # evaluation/<day>/AIA-...
        else:
            out = ECAR_DIR / rel.name
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"STEP4 downloading {r['file_id']} -> {out}")
        gdown.download(id=r["file_id"], output=str(out), quiet=False, resume=True)
        nbytes = out.stat().st_size if out.exists() else 0
        # refresh size in chosen row for reporting
        r["size"] = nbytes
        total += nbytes
        print(f"STEP4 {out}: {nbytes} bytes ({fmt_bytes(nbytes)})")
        paths.append(out)
    print(f"STEP4 TOTAL bytes: {total} ({fmt_bytes(total)})")
    return paths, total


def iter_json_lines(path: Path):
    """Yield JSON objects from .json/.json.gz or archives containing them."""
    name = path.name.lower()
    if name.endswith(".tar") or name.endswith(".tar.gz") or name.endswith(".tgz"):
        mode = "r:gz" if name.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(path, mode) as tf:
            for m in tf:
                if not m.isfile():
                    continue
                if not re.search(r"\.json(\.gz)?$", m.name, re.I):
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                if m.name.lower().endswith(".gz"):
                    import io

                    with gzip.GzipFile(fileobj=f) as gz:
                        for line in gz:
                            yield line
                else:
                    for line in f:
                        yield line
        return
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if not re.search(r"\.json(\.gz)?$", info.filename, re.I):
                    continue
                with zf.open(info) as f:
                    if info.filename.lower().endswith(".gz"):
                        with gzip.GzipFile(fileobj=f) as gz:
                            for line in gz:
                                yield line
                    else:
                        for line in f:
                            yield line
        return
    # plain json.gz / json
    if name.endswith(".gz"):
        opener = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        opener = open(path, "rt", encoding="utf-8", errors="replace")
    with opener as fh:
        for line in fh:
            yield line.encode("utf-8") if isinstance(line, str) else line


def step5_probe(paths: list[Path], bytes_downloaded: int, t0: float) -> None:
    print("STEP5 loading redteam id set…")
    gt = pd.read_csv(GT_CSV)
    pos_ids = set(gt["id"].astype(str))
    print(f"STEP5 positive ids: {len(pos_ids)}")

    oa_hist: Counter = Counter()
    principal_dist: Counter = Counter()
    n_lines = 0
    n_host = 0
    n_pos = 0
    pos_user_days: set[tuple[str, str]] = set()
    samples: list[dict] = []
    key_types: dict[str, set] = {}
    principal_fields_seen: set[str] = set()
    n_service = 0
    n_real = 0
    n_empty_prin = 0
    flow_remote_keys: Counter = Counter()
    flow_samples = 0

    def open_text_stream(path: Path):
        name = path.name.lower()
        if name.endswith(".gz") and not name.endswith((".tar.gz", ".tgz")):
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if name.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
            return None  # handled separately
        return open(path, "rt", encoding="utf-8", errors="replace")

    def handle_rec(rec: dict) -> None:
        nonlocal n_host, n_pos, n_service, n_real, n_empty_prin, flow_samples
        if norm_host(rec.get("hostname", "")) != HOST:
            return
        n_host += 1
        for k, v in rec.items():
            key_types.setdefault(k, set()).add(type(v).__name__)
        if len(samples) < 3:
            samples.append(rec)
        obj = str(rec.get("object", "")).strip()
        act = str(rec.get("action", "")).strip()
        oa_hist[f"{obj} x {act}"] += 1

        props = rec.get("properties") or {}
        if not isinstance(props, dict):
            props = {}
        # principal fields
        for fld in ("principal",):
            if fld in rec and rec.get(fld) is not None:
                principal_fields_seen.add(fld)
        for fld in ("user", "requesting_user"):
            if fld in props and props.get(fld) is not None:
                principal_fields_seen.add(f"properties.{fld}")

        raw_prin = rec.get("principal")
        if raw_prin is None or str(raw_prin).strip() == "":
            raw_prin = props.get("user") or props.get("requesting_user") or ""
        bare = norm_user(raw_prin)
        if not bare:
            n_empty_prin += 1
            principal_dist["<empty>"] += 1
        elif is_service_or_machine(str(raw_prin)):
            n_service += 1
            label = bare.upper() if bare else "<service>"
            if str(raw_prin).strip().upper().startswith("NT AUTHORITY\\"):
                label = bare.upper()
            elif bare.endswith("$"):
                label = bare  # machine
            principal_dist[label] += 1
        else:
            n_real += 1
            principal_dist[bare] += 1

        rid = str(rec.get("id", ""))
        if rid in pos_ids:
            n_pos += 1
            ts = rec.get("timestamp")
            day = ""
            if ts is not None:
                try:
                    day = str(pd.to_datetime(ts, utc=True, errors="coerce").date())
                except Exception:
                    day = str(ts)[:10]
            user_for_pair = bare if bare and not is_service_or_machine(str(raw_prin)) else (
                bare if bare else "<empty>"
            )
            # count (user,day) for positives — use real user when available else bare/service
            pos_user_days.add((user_for_pair, day))

        if str(obj).upper() == "FLOW" and flow_samples < 200:
            flow_samples += 1
            for k in ("dest_ip", "dst_ip", "dst_host", "dest_host", "remote_ip", "src_ip"):
                if k in props and props.get(k) not in (None, ""):
                    flow_remote_keys[k] += 1

    for path in paths:
        print(f"STEP5 scanning {path} …")
        text_fh = open_text_stream(path)
        if text_fh is not None:
            with text_fh as fh:
                for line in fh:
                    n_lines += 1
                    if n_lines % PROGRESS_EVERY == 0:
                        print(
                            f"  progress lines={n_lines:,} host_hits={n_host:,} "
                            f"pos={n_pos:,}"
                        )
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    handle_rec(rec)
        else:
            for raw in iter_json_lines(path):
                n_lines += 1
                if n_lines % PROGRESS_EVERY == 0:
                    print(
                        f"  progress lines={n_lines:,} host_hits={n_host:,} "
                        f"pos={n_pos:,}"
                    )
                try:
                    line = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    continue
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                handle_rec(rec)

    # Write histogram immediately
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["object_x_action", "count"])
        for k, c in oa_hist.most_common(20):
            w.writerow([k, c])
        fh.flush()
        os.fsync(fh.fileno())
    print(f"STEP5 wrote {RESULTS_CSV}")

    print("\n=== REPORT ===")
    print("1) top-level JSON keys + types:")
    for k in sorted(key_types):
        print(f"   {k}: {sorted(key_types[k])}")
    print("   sample records (3):")
    for i, s in enumerate(samples):
        print(f"   --- sample {i+1} ---")
        print(json.dumps(s, indent=2, default=str)[:2000])

    print("\n2) object x action histogram (top 20):")
    for k, c in oa_hist.most_common(20):
        print(f"   {c:10d}  {k}")

    print("\n3) principal fields present:", sorted(principal_fields_seen) or ["(none)"])
    print("   principal distribution (top 30):")
    for k, c in principal_dist.most_common(30):
        print(f"   {c:10d}  {k}")
    denom = max(n_host, 1)
    frac_service = n_service / denom
    frac_real = n_real / denom
    frac_empty = n_empty_prin / denom
    # pseudo ≈ empty / missing real attribution among non-real
    pseudo_rate = (n_empty_prin + n_service) / denom
    print(f"   host events: {n_host}")
    print(f"   fraction service/machine: {frac_service:.4f} ({n_service})")
    print(f"   fraction real users: {frac_real:.4f} ({n_real})")
    print(f"   fraction empty principal: {frac_empty:.4f} ({n_empty_prin})")
    print(f"   real-user attribution rate: {frac_real:.4f}")
    print(f"   pseudo/service rate: {pseudo_rate:.4f}")

    # positive user-days: prefer real-user pairs
    real_pos_ud = {(u, d) for (u, d) in pos_user_days if u and u != "<empty>" and u.upper() not in SERVICE_USERS and not u.endswith("$")}
    print(f"\n4) LABELS BY RECORD ID:")
    print(f"   #positive events on {HOST}: {n_pos}")
    print(f"   #positive (user,day) pairs: {len(pos_user_days)}")
    print(f"   #positive REAL-user (user,day) pairs: {len(real_pos_ud)}")
    if real_pos_ud:
        print("   examples:", list(sorted(real_pos_ud))[:10])

    print("\n5) FLOW remote-host property keys (among FLOW samples):")
    if flow_remote_keys:
        for k, c in flow_remote_keys.most_common():
            print(f"   {k}: {c}")
        best = flow_remote_keys.most_common(1)[0][0]
        print(f"   => dst_host from properties['{best}'] (with direction-aware logic)")
    else:
        print("   (no FLOW records seen on this host in downloaded shards)")

    # VERDICT
    dominated = frac_real < 0.5  # real-user attribution dominated by SYSTEM/service noise
    n_pos_ud = len(real_pos_ud) if real_pos_ud else len(pos_user_days)
    # require >=1 positive user-day; prefer real-user pairs for GO
    ok_labels = len(real_pos_ud) >= 1 or (n_pos >= 1 and len(pos_user_days) >= 1 and not dominated)
    # Strict per instructions: #positive user-days >= 1 AND not dominated by service noise
    if dominated:
        verdict = "NO-GO"
        reason = (
            f"real-user attribution dominated by SYSTEM/service noise "
            f"(real={frac_real:.2%}, service/pseudo={pseudo_rate:.2%})"
        )
    elif len(real_pos_ud) < 1 and len(pos_user_days) < 1:
        verdict = "NO-GO"
        reason = f"#positive user-days on {HOST} = 0 (pos events={n_pos})"
    elif len(real_pos_ud) < 1 and n_pos >= 1:
        # positives exist but only under service/empty principals
        verdict = "NO-GO"
        reason = (
            f"#positive events={n_pos} but no real-user (user,day) pairs "
            f"(pairs={len(pos_user_days)} are service/empty)"
        )
    else:
        verdict = "GO"
        reason = (
            f"real-user rate={frac_real:.2%}; "
            f"positive real user-days={len(real_pos_ud)} on {HOST}"
        )

    wall = time.time() - t0
    print(f"\nVERDICT {verdict}" + (f" — {reason}" if verdict == "NO-GO" else f" — {reason}"))
    print(f"TOTAL wall-clock: {wall:.1f}s")
    print(f"TOTAL bytes downloaded: {bytes_downloaded} ({fmt_bytes(bytes_downloaded)})")
    print(f"lines scanned={n_lines:,} host_events={n_host:,}")


def main() -> None:
    t0 = time.time()
    os.chdir(ROOT)
    step1()
    rows = step2_list()
    if not rows:
        print("STEP2 listing returned 0 evaluation files — abort")
        sys.exit(2)
    chosen = step3_select(rows)
    # Extra guardrail print if day1 file(s) > 8GB
    for r in chosen:
        if r["size"] is not None and r["size"] > MAX_SINGLE_BYTES:
            print(
                f"NOTE: chosen archive {r['name']} is {fmt_bytes(r['size'])} > 8 GiB; "
                "still downloading first eval day only as required."
            )
    paths, total = step4_download(chosen)
    step5_probe(paths, total, t0)


if __name__ == "__main__":
    main()
