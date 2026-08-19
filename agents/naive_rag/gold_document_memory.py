"""Swap gold memory sessions for corpus documents chunked like the Chroma index."""

from __future__ import annotations

import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from experiment.paths import DEFAULT_EVIDENCE_ID_TO_URL

from .indexing import ChunkTuple, corpus_doc_to_chunks


class MemoryGoldSource(str, Enum):
    SESSION = "session"
    GOLD_DOCUMENT = "gold_document"


def parse_memory_gold_source(value: str | MemoryGoldSource) -> MemoryGoldSource:
    if isinstance(value, MemoryGoldSource):
        return value
    return MemoryGoldSource(str(value).strip().lower())


def evidence_id_from_session_id(session_id: str) -> str | None:
    """Parse ``ev-{evidence_id}`` from a harness session id."""
    for part in str(session_id).split("__"):
        if part.startswith("ev-") and len(part) > 3:
            return part[3:]
    return None


@lru_cache(maxsize=4)
def load_evidence_id_to_url(path: str | None = None) -> dict[str, str]:
    p = Path(path) if path else DEFAULT_EVIDENCE_ID_TO_URL
    mapping: dict[str, str] = {}
    if not p.exists():
        return mapping
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            eid = str(rec.get("evidence_id") or "").strip()
            url = str(rec.get("url") or "").strip()
            if eid and url:
                mapping[eid] = url
    return mapping


def url_for_session_id(
    session_id: str,
    *,
    evidence_id: str | None = None,
    eid_to_url: dict[str, str] | None = None,
) -> str | None:
    eid = (evidence_id or evidence_id_from_session_id(session_id) or "").strip()
    if not eid:
        return None
    mapping = eid_to_url if eid_to_url is not None else load_evidence_id_to_url()
    url = mapping.get(eid)
    return url.strip() if url else None


def gold_document_memory_chunks(
    session_id: str,
    corpus_by_url: dict[str, dict[str, Any]],
    *,
    evidence_id: str | None = None,
    eid_to_url: dict[str, str] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> list[ChunkTuple]:
    """Corpus article chunks tagged as memory for ``session_id``.

    Chunk text uses the same splitter as the document collection. Chunk ids are
    prefixed so they do not collide with document-store ids in a unified index.
    """
    url = url_for_session_id(
        session_id, evidence_id=evidence_id, eid_to_url=eid_to_url
    )
    if not url:
        return []
    doc = corpus_by_url.get(url)
    if not doc:
        return []

    out: list[ChunkTuple] = []
    for i, (_cid, text, meta) in enumerate(corpus_doc_to_chunks(doc)):
        chunk_id = f"{session_id}__golddoc_{i}"
        mem_meta = dict(meta)
        mem_meta["session_id"] = session_id
        mem_meta["url"] = url
        if extra_meta:
            mem_meta.update(extra_meta)
        out.append((chunk_id, text, mem_meta))
    return out
