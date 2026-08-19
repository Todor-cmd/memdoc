from __future__ import annotations

import json
from typing import Any

import pandas as pd

# Same column order as ``prepare_data.question_evidence_ids.iter_evidence_dicts_from_row``.
_EVIDENCE_COLUMNS = (
    "memory_evidence",
    "evidence_list",
    "golden_memory_evidence",
    "golden_document_evidence",
)


def _iter_evidence_dicts_from_row(row: pd.Series) -> list[dict]:
    out: list[dict] = []
    for col in _EVIDENCE_COLUMNS:
        if col not in row.index:
            continue
        for ev in _parse_evidence_cell(row[col]):
            if isinstance(ev, dict):
                out.append(ev)
    return out


def _parse_evidence_cell(val: Any) -> list:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if callable(getattr(val, "tolist", None)) and not isinstance(
        val, (dict, str, bytes)
    ):
        val = val.tolist()
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        return json.loads(s)
    return []


def evidence_dicts_union_for_question(row: pd.Series) -> list[dict[str, Any]]:
    """All gold evidence dicts for *Q* (deduped by ``evidence_id``), any column order."""
    by_eid: dict[str, dict[str, Any]] = {}
    for ev in _iter_evidence_dicts_from_row(row):
        if not isinstance(ev, dict):
            continue
        eid = ev.get("evidence_id")
        if eid is None or (isinstance(eid, float) and pd.isna(eid)):
            continue
        key = str(eid).strip()
        if key:
            by_eid[key] = ev
    return list(by_eid.values())


def integrated_memory_evidence(row: pd.Series) -> list[dict[str, Any]]:
    """Evidence assigned to the memory channel for integrated variant."""
    if "memory_evidence" in row.index:
        mem = _parse_evidence_cell(row["memory_evidence"])
        if mem:
            return [e for e in mem if isinstance(e, dict)]
    return [e for e in _parse_evidence_cell(row.get("golden_memory_evidence")) if isinstance(e, dict)]


def integrated_document_evidence(row: pd.Series) -> list[dict[str, Any]]:
    """Evidence assigned to the document channel for integrated variant."""
    if "evidence_list" in row.index:
        docs = _parse_evidence_cell(row["evidence_list"])
        if docs:
            return [e for e in docs if isinstance(e, dict)]
    return [e for e in _parse_evidence_cell(row.get("golden_document_evidence")) if isinstance(e, dict)]


def urls_from_evidence_dicts(items: list[dict[str, Any]]) -> frozenset[str]:
    out: set[str] = set()
    for ev in items:
        u = ev.get("url")
        if u is None or (isinstance(u, float) and pd.isna(u)):
            continue
        s = str(u).strip()
        if s:
            out.add(s)
    return frozenset(out)
