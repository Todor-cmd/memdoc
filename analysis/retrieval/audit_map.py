"""Load session-audit JSONL into key_information turn maps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionAuditEntry:
    """Audit localization for one evidence session."""

    session_id: str
    key_information_turns: tuple[int, ...]
    all_required_present: bool
    is_temporal: bool
    notes: str = ""


def load_audit_map(path: Path) -> dict[str, SessionAuditEntry]:
    """Map ``session_id`` → audit entry (``key_information`` turn indices)."""
    out: dict[str, SessionAuditEntry] = {}
    if not path.is_file():
        return out

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec: dict[str, Any] = json.loads(line)
            if rec.get("error"):
                continue
            sid = str(rec.get("session_id", "")).strip()
            if not sid:
                continue
            key_turns = rec.get("key_information") or []
            if not isinstance(key_turns, list):
                key_turns = []
            turns = tuple(int(t) for t in key_turns)
            out[sid] = SessionAuditEntry(
                session_id=sid,
                key_information_turns=turns,
                all_required_present=bool(rec.get("all_required_present")),
                is_temporal=bool(rec.get("is_temporal")),
                notes=str(rec.get("notes") or "").strip(),
            )
    return out
