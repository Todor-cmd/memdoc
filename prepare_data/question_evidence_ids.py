"""Stable question_id (query hash) and scoped evidence_id for dataset prep + batch jobs.

evidence_id = first EVIDENCE_ID_HEX_LENGTH hex chars of:
  SHA256(source_question_id + "\\n" + canonical_json(evidence_subset))

Canonical payload uses a fixed key order; only those keys participate (extras ignored).
Changing EVIDENCE_CANONICAL_KEYS or the separator is a breaking change for id stability.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

# Keys aligned with inference / EvidenceItem-style dicts; unknown keys on dict are ignored.
EVIDENCE_CANONICAL_KEYS: tuple[str, ...] = (
    "author",
    "category",
    "fact",
    "published_at",
    "source",
    "title",
    "url",
)
ID_PAYLOAD_SEPARATOR = "\n"
EVIDENCE_ID_HEX_LENGTH = 32


def question_id_from_query(query: Any) -> str:
    """Stable global id: SHA-256 hex of UTF-8 stripped query."""
    if query is None or (isinstance(query, float) and pd.isna(query)):
        q = ""
    else:
        q = str(query).strip()
    return hashlib.sha256(q.encode("utf-8")).hexdigest()


def _norm_field(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def canonical_evidence_json(evidence: dict) -> str:
    """Stable JSON for hashing; ``evidence_id`` on the dict is not included."""
    payload = {k: _norm_field(evidence.get(k)) for k in EVIDENCE_CANONICAL_KEYS}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def source_question_id_for_row(row: pd.Series, df_index: Any) -> str:
    """Resolve stable question id: prefer ``question_id``, then ``question_idx``, else hash of ``query``."""
    if "question_id" in row.index:
        v = row["question_id"]
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return str(v).strip()
    if "question_idx" in row.index:
        v = row["question_idx"]
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            if isinstance(v, str):
                return v.strip()
            try:
                f = float(v)
                if f.is_integer():
                    return str(int(f))
            except (TypeError, ValueError):
                return str(v).strip()
    if "query" in row.index:
        return question_id_from_query(row["query"])
    return str(df_index)


def scoped_evidence_id(source_question_id: str, evidence: dict) -> str:
    """Globally unique id for one (question, evidence payload) session unit."""
    sid = str(source_question_id).strip()
    body = sid + ID_PAYLOAD_SEPARATOR + canonical_evidence_json(evidence)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:EVIDENCE_ID_HEX_LENGTH]


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


EVIDENCE_COLUMNS = (
    "memory_evidence",
    "evidence_list",
    "golden_memory_evidence",
    "golden_document_evidence",
)


def iter_evidence_dicts_from_row(row: pd.Series) -> list[dict]:
    """Flatten evidence dicts from row columns (same column order as batch union)."""
    out: list[dict] = []
    for col in EVIDENCE_COLUMNS:
        if col not in row.index:
            continue
        for ev in _parse_evidence_cell(row[col]):
            if isinstance(ev, dict):
                out.append(ev)
    return out


def attach_evidence_ids_to_dataframe(
    df: pd.DataFrame,
    *,
    source_question_id_col: str = "question_idx",
) -> None:
    """Mutate evidence list cells in-place: each dict gets ``evidence_id``."""
    if source_question_id_col not in df.columns:
        raise KeyError(f"Missing column {source_question_id_col!r}")
    cols = [c for c in EVIDENCE_COLUMNS if c in df.columns]
    if not cols:
        return
    for idx in df.index:
        sid = df.at[idx, source_question_id_col]
        if sid is None or (isinstance(sid, float) and pd.isna(sid)):
            raise ValueError(f"Missing {source_question_id_col} for row index {idx!r}")
        sid_str = str(sid).strip()
        for col in cols:
            items = _parse_evidence_cell(df.at[idx, col])
            for ev in items:
                if isinstance(ev, dict):
                    ev["evidence_id"] = scoped_evidence_id(sid_str, ev)
            df.at[idx, col] = items
