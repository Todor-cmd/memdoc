"""Helpers to turn persona sessions and corpus documents into Chroma-indexable chunks.

Supports configurable memory granularity (SESSION / PAIR / TURN) and
metadata-enriched document chunking (256 tok / 32 overlap, sentence-aware).
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

ChunkTuple = tuple[str, str, dict[str, Any]]


class MemoryGranularity(str, Enum):
    SESSION = "session"
    PAIR = "pair"
    TURN = "turn"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_persona_sessions(persona_json: Path) -> list[dict[str, Any]]:
    raw = json.loads(persona_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array in {persona_json}")
    return raw


def load_corpus_jsonl(corpus_path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


# ---------------------------------------------------------------------------
# Memory session chunking
# ---------------------------------------------------------------------------

def _base_session_meta(session: dict[str, Any], sid: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "session_id": sid,
        "source": session.get("source", ""),
        "date": session.get("date", ""),
    }
    if session.get("evidence_id"):
        meta["evidence_id"] = str(session["evidence_id"])
    return meta


def _session_full_text(session: dict[str, Any]) -> str:
    """Flatten all turns into a single string."""
    turns = session.get("session", [])
    return "\n".join(f"{t.get('role', 'unknown')}: {t.get('content', '')}" for t in turns)


def session_to_chunks(
    session: dict[str, Any],
    granularity: MemoryGranularity = MemoryGranularity.SESSION,
) -> list[ChunkTuple]:
    """Split a session into one or more ``(chunk_id, text, metadata)`` tuples.

    * ``SESSION`` — single chunk with all turns concatenated.
    * ``PAIR``    — one chunk per user+assistant message pair.
    * ``TURN``    — one chunk per user message only.
    """
    sid = str(session.get("session_id", ""))

    if granularity == MemoryGranularity.SESSION:
        text = _session_full_text(session)
        return [(sid, text, _base_session_meta(session, sid))]

    turns = session.get("session", [])
    chunks: list[ChunkTuple] = []

    if granularity == MemoryGranularity.PAIR:
        pair_idx = 0
        i = 0
        while i < len(turns):
            user_turn = turns[i]
            if user_turn.get("role") != "user":
                i += 1
                continue
            user_text = f"user: {user_turn.get('content', '')}"
            if i + 1 < len(turns) and turns[i + 1].get("role") == "assistant":
                asst_text = f"assistant: {turns[i + 1].get('content', '')}"
                text = f"{user_text}\n{asst_text}"
                i += 2
            else:
                text = user_text
                i += 1
            chunk_id = f"{sid}__pair_{pair_idx}"
            meta = _base_session_meta(session, sid)
            meta["chunk_index"] = pair_idx
            chunks.append((chunk_id, text, meta))
            pair_idx += 1

    elif granularity == MemoryGranularity.TURN:
        turn_idx = 0
        for turn in turns:
            if turn.get("role") != "user":
                continue
            text = turn.get("content", "")
            chunk_id = f"{sid}__turn_{turn_idx}"
            meta = _base_session_meta(session, sid)
            meta["chunk_index"] = turn_idx
            chunks.append((chunk_id, text, meta))
            turn_idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Document corpus chunking
# ---------------------------------------------------------------------------

_DOC_SPLITTER: RecursiveCharacterTextSplitter | None = None


def _get_doc_splitter(
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> RecursiveCharacterTextSplitter:
    global _DOC_SPLITTER
    if _DOC_SPLITTER is None or (
        _DOC_SPLITTER._chunk_size != chunk_size
        or _DOC_SPLITTER._chunk_overlap != chunk_overlap
    ):
        _DOC_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return _DOC_SPLITTER


def _doc_metadata_prefix(doc: dict[str, Any]) -> str:
    title = str(doc.get("title", "")).strip()
    author = str(doc.get("author", "")).strip()
    published = str(doc.get("published_at", "")).strip()
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if author:
        parts.append(f"Author: {author}")
    if published:
        parts.append(f"Published: {published}")
    return " | ".join(parts)


def _base_doc_meta(doc: dict[str, Any], url: str) -> dict[str, Any]:
    return {
        "url": url,
        "title": str(doc.get("title", "")).strip(),
        "category": str(doc.get("category", "")),
        "author": str(doc.get("author", "")),
        "source": str(doc.get("source", "")),
        "published_at": str(doc.get("published_at", "")),
    }


def corpus_doc_to_chunks(
    doc: dict[str, Any],
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> list[ChunkTuple]:
    """Split a corpus article into metadata-enriched chunks.

    Each chunk's text is prefixed with ``Title | Author | Published`` and the
    same fields are stored as Chroma metadata for filtering.
    """
    url = str(doc.get("url", "")).strip()
    title = str(doc.get("title", "")).strip()
    body = str(doc.get("body", "")).strip()
    full_text = f"{title}\n\n{body}" if title else body

    prefix = _doc_metadata_prefix(doc)
    text_to_split = f"{prefix}\n\n{full_text}" if prefix else full_text

    splitter = _get_doc_splitter(chunk_size, chunk_overlap)
    split_texts = splitter.split_text(text_to_split)

    chunks: list[ChunkTuple] = []
    for i, chunk_text in enumerate(split_texts):
        chunk_id = f"{url}__chunk_{i}"
        meta = _base_doc_meta(doc, url)
        meta["chunk_index"] = i
        chunks.append((chunk_id, chunk_text, meta))

    return chunks


# ---------------------------------------------------------------------------
# Backward-compatible single-item helpers (used by old code paths)
# ---------------------------------------------------------------------------

def session_to_text(session: dict[str, Any]) -> str:
    """Flatten a multi-turn session into a single string for embedding."""
    return _session_full_text(session)


def session_to_document(session: dict[str, Any]) -> ChunkTuple:
    """Return single ``(doc_id, text, metadata)`` for a session (SESSION granularity)."""
    return session_to_chunks(session, MemoryGranularity.SESSION)[0]


def corpus_doc_to_document(doc: dict[str, Any]) -> ChunkTuple:
    """Return single ``(doc_id, text, metadata)`` for a corpus article (no splitting)."""
    url = str(doc.get("url", "")).strip()
    title = str(doc.get("title", "")).strip()
    body = str(doc.get("body", "")).strip()
    text = f"{title}\n\n{body}" if title else body
    return url, text, _base_doc_meta(doc, url)
