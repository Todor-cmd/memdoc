"""Join memory_collection sessions to batch-manifest evidence records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class AuditRow:
    """One session + intended evidence ready for the faithfulness judge."""

    session_id: str
    evidence_id: str
    source_question_id: str
    is_temporal: bool
    evidence: dict[str, Any]
    session_turns: list[dict[str, Any]]
    persona_file: str


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load batch manifest keyed by custom_id (== session_id)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected manifest object at {path}, got {type(data)}")
    return data


def iter_memory_sessions(memory_dir: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (persona_filename, session_record) from ``*.json`` arrays."""
    for path in sorted(memory_dir.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            sessions = json.load(f)
        if not isinstance(sessions, list):
            raise ValueError(f"Expected JSON array in {path}")
        for rec in sessions:
            if isinstance(rec, dict):
                yield path.name, rec


def build_audit_rows(
    memory_dir: Path,
    manifest: dict[str, dict[str, Any]],
) -> tuple[list[AuditRow], list[str]]:
    """Join sessions to manifest evidence.

    Returns
    -------
    rows :
        Successfully joined audit inputs.
    orphans :
        ``session_id`` values with no matching manifest entry.
    """
    rows: list[AuditRow] = []
    orphans: list[str] = []

    for persona_file, rec in iter_memory_sessions(memory_dir):
        session_id = str(rec.get("session_id", "")).strip()
        if not session_id:
            continue

        man = manifest.get(session_id)
        if man is None:
            orphans.append(session_id)
            continue

        evidence = man.get("evidence")
        if not isinstance(evidence, dict):
            orphans.append(session_id)
            continue

        turns = rec.get("session")
        if not isinstance(turns, list):
            orphans.append(session_id)
            continue

        is_temporal = bool(man.get("is_temporal"))
        if not is_temporal and str(man.get("question_type", "")) == "temporal_query":
            is_temporal = True

        evidence_id = str(
            man.get("evidence_id") or rec.get("evidence_id") or ""
        ).strip()
        source_qid = str(
            man.get("source_question_id") or rec.get("source_question_id") or ""
        ).strip()

        rows.append(
            AuditRow(
                session_id=session_id,
                evidence_id=evidence_id,
                source_question_id=source_qid,
                is_temporal=is_temporal,
                evidence=evidence,
                session_turns=turns,
                persona_file=persona_file,
            )
        )

    return rows, orphans
