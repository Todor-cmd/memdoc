"""Batch ``custom_id`` / memory ``session_id`` for evidence sessions (see ``session_generation/create_batch_job.py``)."""

from __future__ import annotations

import hashlib


def query_fingerprint(normalized_query: str) -> str:
    """First 16 hex chars of SHA-256 of UTF-8 query (strip whitespace)."""
    q = normalized_query.strip().encode("utf-8")
    return hashlib.sha256(q).hexdigest()[:16]


def evidence_session_id(query: str, evidence_id: str, persona_id: str | None = None) -> str:
    """Stable session id for an evidence session.

    New format (cross-persona): ``q-{fp}__ev-{evidence_id}__p-{persona_id}``
    Legacy format (single persona): ``q-{fp}__ev-{evidence_id}``

    When ``persona_id`` is provided, uses the new format.
    """
    fp = query_fingerprint(query)
    eid = str(evidence_id).strip()
    if persona_id:
        return f"q-{fp}__ev-{eid}__p-{persona_id}"
    return f"q-{fp}__ev-{eid}"
