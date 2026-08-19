"""Strict full-context evidence-hit recall."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from analysis.retrieval.audit_map import SessionAuditEntry, load_audit_map
from analysis.retrieval.doc_evidence import DocEvidence, load_doc_evidence_by_url
from analysis.retrieval.pair_index import PairIndexCache, load_session_turns
from analysis.retrieval.textnorm import normalize

DOC_DISTS = frozenset({"document_only", "integrated"})
MEM_DISTS = frozenset({"memory_only", "integrated"})

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_AUDIT = _REPO_ROOT / "data" / "session_audit" / "evidence_field_locations.jsonl"
_DEFAULT_MANIFEST = (
    _REPO_ROOT
    / "data"
    / "batch_jobs"
    / "experiment_sessions"
    / "batch_manifest_20260607_145849.json"
)
_DEFAULT_MEMORY = _REPO_ROOT / "data" / "memory_collection"


def _parse_list(val: Any) -> list:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _parse_retrieved(val: Any) -> list[dict[str, Any]]:
    items = _parse_list(val)
    return [x for x in items if isinstance(x, dict)]


def gold_ids_for_row(row: pd.Series) -> list[str]:
    """Dist-conditioned gold evidence IDs (same as recall_at_10 gold)."""
    gold: list[str] = []
    dist = row.get("dist")
    if dist in DOC_DISTS:
        for u in _parse_list(row.get("golden_document_urls")):
            gold.append(f"doc::{u}")
    if dist in MEM_DISTS:
        for s in _parse_list(row.get("golden_memory_session_ids")):
            gold.append(f"mem::{s}")
    return gold


def _sorted_doc_chunks(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """All doc chunks by score desc — no URL dedupe."""
    return sorted(docs, key=lambda d: d.get("score", 0) or 0, reverse=True)


def _text_covers(needle: str, haystack: str) -> bool:
    n = normalize(needle)
    h = normalize(haystack)
    return bool(n) and n in h


def _pool_covers(needle: str, texts: list[str]) -> bool:
    if not texts:
        return False
    if any(_text_covers(needle, t) for t in texts):
        return True
    return _text_covers(needle, "\n".join(texts))


def strict_doc_hit(
    url: str,
    retrieved_docs: list[dict[str, Any]],
    evidence_by_url: dict[str, DocEvidence],
) -> bool:
    """True if gold fact for ``url`` is covered by same-URL retrieved chunks."""
    pool = [
        str(d.get("text") or "")
        for d in _sorted_doc_chunks(retrieved_docs)
        if str(d.get("url") or "").strip() == url
    ]
    if not pool:
        return False

    ev = evidence_by_url.get(url)
    if ev is None:
        return False

    fact = ev.fact
    if fact and len(fact) >= 20:
        return _pool_covers(fact, pool)

    # Short / missing fact: require title and source in the same-URL pool
    title_ok = bool(ev.title) and _pool_covers(ev.title, pool)
    source_ok = bool(ev.source) and _pool_covers(ev.source, pool)
    if ev.title and ev.source:
        return title_ok and source_ok
    return title_ok or source_ok


def strict_mem_hit(
    session_id: str,
    retrieved_mem: list[dict[str, Any]],
    audit: dict[str, SessionAuditEntry],
    pair_cache: PairIndexCache,
) -> bool:
    """True if a retrieved chunk from ``session_id`` maps to a key_information turn."""
    entry = audit.get(session_id)
    if entry is None or not entry.key_information_turns:
        return False
    gold_turns = set(entry.key_information_turns)

    for m in retrieved_mem:
        if str(m.get("session_id") or "").strip() != session_id:
            continue
        idx = pair_cache.index_for(session_id, str(m.get("text") or ""))
        if idx is not None and idx in gold_turns:
            return True
    return False


def evidence_hit(
    gold_id: str,
    *,
    retrieved_docs: list[dict[str, Any]],
    retrieved_mem: list[dict[str, Any]],
    audit: dict[str, SessionAuditEntry],
    evidence_by_url: dict[str, DocEvidence],
    pair_cache: PairIndexCache,
) -> bool:
    if gold_id.startswith("doc::"):
        return strict_doc_hit(gold_id[5:], retrieved_docs, evidence_by_url)
    if gold_id.startswith("mem::"):
        return strict_mem_hit(gold_id[5:], retrieved_mem, audit, pair_cache)
    return False


def strict_recall_for_row(
    row: pd.Series,
    *,
    audit: dict[str, SessionAuditEntry],
    evidence_by_url: dict[str, DocEvidence],
    pair_cache: PairIndexCache,
) -> float:
    gold = gold_ids_for_row(row)
    if not gold:
        return float(np.nan)

    docs = _parse_retrieved(row.get("retrieved_documents"))
    mems = _parse_retrieved(row.get("retrieved_memory"))
    hits = sum(
        1
        for g in gold
        if evidence_hit(
            g,
            retrieved_docs=docs,
            retrieved_mem=mems,
            audit=audit,
            evidence_by_url=evidence_by_url,
            pair_cache=pair_cache,
        )
    )
    return hits / len(gold)


def add_strict_recall(
    df: pd.DataFrame,
    *,
    audit_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    memory_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Add ``recall_full_context_strict`` column (does not modify ``recall_at_10``)."""
    audit_path = audit_path or _DEFAULT_AUDIT
    manifest_path = manifest_path or _DEFAULT_MANIFEST
    memory_dir = memory_dir or _DEFAULT_MEMORY

    audit = load_audit_map(audit_path)
    evidence_by_url = load_doc_evidence_by_url(manifest_path)
    sessions = load_session_turns(memory_dir)
    pair_cache = PairIndexCache(sessions)

    print(
        f"  Strict recall resources: audit={len(audit)} sessions, "
        f"doc_evidence_urls={len(evidence_by_url)}, "
        f"memory_sessions={len(sessions)}"
    )

    values = [
        strict_recall_for_row(
            row,
            audit=audit,
            evidence_by_url=evidence_by_url,
            pair_cache=pair_cache,
        )
        for _, row in df.iterrows()
    ]
    out = df.copy()
    out["recall_full_context_strict"] = values
    return out
