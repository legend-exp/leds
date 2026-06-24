"""Flatten the evt-tier tables for a single event into field/value DataFrames.

First pass: every leaf field becomes one row (``field``, ``value``), with nested
structs flattened to dotted names and vector-of-vectors shown as the per-hit
list. Reshaping into per-hit column tables / trimming fields comes later.
"""

from __future__ import annotations

from datetime import UTC, datetime

import lgdo
import lh5
import pandas as pd

#: evt tables shown in the Event details tab (display order).
TABLES = ("trigger", "coincident", "geds", "spms")

#: leaf field names to drop from a given table's flattened view.
EXCLUDE = {
    "spms": {"energy", "hit_idx", "is_trig_coin_pulse", "is_physical", "rawid", "t0"},
}

#: summary rows: (label, evt table, field path within the table). The timestamp
#: is handled separately (shown both UTC-formatted and as the raw value).
SUMMARY = (
    ("detector name", "geds", "detector_name"),
    ("energy", "geds", "energy_sum"),
    ("coincident pulser", "coincident", "puls"),
    ("trigger is_forced", "trigger", "is_forced"),
    ("coincident muon", "coincident", "muon"),
    ("geds.quality.is_bb_like", "geds", "quality.is_bb_like"),
    ("multiplicity", "geds", "multiplicity"),
    ("spms coincident", "coincident", "spms"),
    ("geds.psd.is_bb_like", "geds", "psd.is_bb_like"),
)


def _to_py(value):
    """Bytes -> str, recursively (for nested per-hit lists)."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    if isinstance(value, list):
        return [_to_py(v) for v in value]
    return value


def _flatten(obj, rows, prefix=""):
    for key in obj:
        child = obj[key]
        name = prefix + key
        if isinstance(child, (lgdo.Table, lgdo.Struct)):
            _flatten(child, rows, name + ".")
        elif isinstance(child, lgdo.Array):
            rows.append((name, _to_py(child.nda[0])))
        else:  # VectorOfVectors / ArrayOfEqualSizedArrays -> per-hit list
            rows.append((name, _to_py(child.view_as("ak")[0].to_list())))


def _read_field(ev, table, path):
    obj = lh5.read(
        f"{ev.tier}/{table}/{path.replace('.', '/')}",
        str(ev.current_file),
        start_row=ev.index,
        n_rows=1,
    )
    if isinstance(obj, lgdo.Array):
        return _to_py(obj.nda[0])
    return _to_py(obj.view_as("ak")[0].to_list())


def summary_dataframe(ev):
    """One-row-per-quantity summary of the current event."""
    timestamp = _read_field(ev, "trigger", "timestamp")
    rows = [
        ("period", ev.period),
        ("run", ev.run),
        (
            "timestamp (UTC)",
            datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        ),
        ("timestamp (original)", timestamp),
    ]
    rows += [(label, _read_field(ev, table, path)) for label, table, path in SUMMARY]
    return pd.DataFrame(
        [(field, str(value)) for field, value in rows], columns=["field", "value"]
    )


def table_dataframe(ev, table):
    """``field | value`` DataFrame for one evt table at the current event."""
    obj = lh5.read(
        f"{ev.tier}/{table}", str(ev.current_file), start_row=ev.index, n_rows=1
    )
    rows: list = []
    _flatten(obj, rows)
    excluded = EXCLUDE.get(table, set())
    return pd.DataFrame(
        [
            (field, str(value))
            for field, value in rows
            if field.split(".")[-1] not in excluded
        ],
        columns=["field", "value"],
    )
