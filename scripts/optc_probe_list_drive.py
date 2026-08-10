"""List public Google Drive OpTC folder contents (no full download)."""
from __future__ import annotations

import json
import re
import sys

import requests

ROOT = "1n3kkS3KR31KUegn42yk3-e6JkZvf0Caa"


def list_folder(folder_id: str, depth: int = 0, max_depth: int = 3) -> None:
    sess = requests.Session()
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    r = sess.get(url, timeout=60)
    print(f"\n{'  '*depth}FOLDER {folder_id} status={r.status_code} bytes={len(r.text)}")
    # Embedded JSON blobs with file metadata
    # Pattern used by Drive UI: ["NAME",["FILE_ID"],...]
    # Also look for _DRIVE_ivd payloads
    names = re.findall(r'\["([^"]+)",\["([a-zA-Z0-9_-]{20,})"\]', r.text)
    seen = set()
    entries = []
    for name, fid in names:
        if fid in seen:
            continue
        seen.add(fid)
        entries.append((name, fid))
    # Alternate scrape: data-id + title nearby
    if not entries:
        for m in re.finditer(
            r'data-id="([a-zA-Z0-9_-]{20,})"[^>]*>.*?<div[^>]*class="[^"]*Q5txwe[^"]*"[^>]*>([^<]+)',
            r.text,
            re.S,
        ):
            fid, name = m.group(1), m.group(2)
            if fid not in seen:
                seen.add(fid)
                entries.append((name, fid))

    print(f"{'  '*depth}entries={len(entries)}")
    for name, fid in entries[:80]:
        print(f"{'  '*depth}- {name}  id={fid}")

    # Recurse into likely subfolders by name heuristics
    if depth >= max_depth:
        return
    for name, fid in entries:
        lname = name.lower()
        if any(k in lname for k in ("ecar", "benign", "evaluation", "short", "sep", "aia", "20-23", "23sep", "24sep", "25sep")):
            # Avoid recursing into giant leaf files (have extensions)
            if "." in name and not name.endswith("/"):
                continue
            list_folder(fid, depth + 1, max_depth)


def try_gdown_parse(folder_id: str) -> None:
    try:
        from gdown.download_folder import _parse_google_drive_file, parse_google_drive_file
    except Exception as e:
        print("gdown parse import failed", e)
        return
    try:
        # newer gdown
        from gdown.download_folder import _download_and_parse_google_drive_link
        is_folder, gdrive_file, id_name_map = _download_and_parse_google_drive_link(
            requests.Session(),
            f"https://drive.google.com/drive/folders/{folder_id}",
            quiet=False,
            remaining_ok=True,
        )
        print("gdown is_folder", is_folder)
        print("gdrive_file", gdrive_file)
        print("id_name_map count", len(id_name_map) if id_name_map else None)
        if id_name_map:
            for i, (k, v) in enumerate(list(id_name_map.items())[:50]):
                print(i, k, v)
    except Exception as e:
        print("gdown parse failed:", type(e), e)


if __name__ == "__main__":
    try_gdown_parse(ROOT)
    list_folder(ROOT, 0, 2)
