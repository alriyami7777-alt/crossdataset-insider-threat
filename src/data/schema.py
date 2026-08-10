"""
Unified event schema shared across all datasets (CERT, LANL, OpTC, TWOS, SPEDIA).

The whole paper hinges on comparing datasets on equal footing, so every loader
MUST map its native columns into this canonical schema before anything else
touches the data. If a field is missing in a given dataset, fill with a sentinel
(NA / "unknown") rather than dropping the column, so the graph/feature code can
assume a fixed set of columns everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

# Canonical column names. Order matters only for readability.
TIMESTAMP = "timestamp"     # pandas datetime64[ns]
USER = "user"               # actor identity (string id)
SRC_HOST = "src_host"       # host the action originates from
DST_HOST = "dst_host"       # host/resource the action targets (may == src_host)
ACTION = "action"           # canonical action type (see ACTIONS)
OBJECT = "object"           # object touched: file path, url, email dst, etc.
LABEL = "label"             # 1 = malicious insider event, 0 = benign
INSIDER_TYPE = "insider_type"  # 'traitor' | 'masquerader' | 'benign'
DATASET = "dataset"         # source dataset tag, e.g. 'cert','lanl','optc','twos','spedia'

CANONICAL_COLUMNS = [
    TIMESTAMP, USER, SRC_HOST, DST_HOST, ACTION, OBJECT,
    LABEL, INSIDER_TYPE, DATASET,
]

# Canonical action vocabulary. Native actions are mapped onto these so a model
# trained on one dataset sees the same action tokens on another.
ACTIONS = [
    "logon", "logoff", "auth", "file_read", "file_write", "file_delete",
    "usb_connect", "usb_disconnect", "http", "email_send", "email_recv",
    "cmd_exec", "ssh", "ftp", "process_start", "unknown",
]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}


@dataclass(frozen=True)
class DatasetMeta:
    """Book-keeping for each dataset so results tables stay honest."""
    name: str
    is_real: bool               # real user activity vs synthetic
    has_cert_overlap: bool      # True for SPEDIA (mixes CERT-derived rows)
    insider_types: tuple        # which insider types are present
    note: str = ""


DATASET_META = {
    "cert":   DatasetMeta("cert",   is_real=False, has_cert_overlap=True,
                          insider_types=("traitor",), note="Synthetic; overfitting exhibit."),
    "lanl":   DatasetMeta("lanl",   is_real=True,  has_cert_overlap=False,
                          insider_types=("masquerader",), note="Real auth; red-team = lateral movement."),
    "optc":   DatasetMeta("optc",   is_real=True,  has_cert_overlap=False,
                          insider_types=("apt_redteam",),
                          note="DARPA OpTC eCAR host telemetry; APT red-team eval window."),
    "twos":   DatasetMeta("twos",   is_real=True,  has_cert_overlap=False,
                          insider_types=("masquerader", "traitor"), note="Real users, gamified."),
    "spedia": DatasetMeta("spedia", is_real=True,  has_cert_overlap=True,
                          insider_types=("traitor", "masquerader"),
                          note="Partly CERT-derived; isolate real-exercise rows for clean cross-dataset."),
    "synthA": DatasetMeta("synthA", is_real=False, has_cert_overlap=False, insider_types=("traitor",)),
    "synthB": DatasetMeta("synthB", is_real=False, has_cert_overlap=False, insider_types=("masquerader",)),
}


def validate(df):
    """Raise if a dataframe is not in canonical schema. Cheap insurance."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical columns: {missing}")
    bad = set(df[ACTION].unique()) - set(ACTIONS)
    if bad:
        raise ValueError(f"Non-canonical actions present: {sorted(bad)}")
    return True
