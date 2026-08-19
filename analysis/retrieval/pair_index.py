"""Rebuild PAIR chunks and match retrieved memory text → pair index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from analysis.retrieval.textnorm import normalize


def load_session_turns(memory_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """``session_id`` → turn list from ``memory_collection/*.json``."""
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(memory_dir.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            sessions = json.load(f)
        if not isinstance(sessions, list):
            continue
        for rec in sessions:
            if not isinstance(rec, dict):
                continue
            sid = str(rec.get("session_id", "")).strip()
            turns = rec.get("session")
            if sid and isinstance(turns, list):
                out[sid] = turns
    return out


def pair_texts_for_session(turns: list[dict[str, Any]]) -> list[str]:
    """Rebuild PAIR chunk texts (mirrors ``session_to_chunks(..., PAIR)``).

    Index ``i`` aligns with audit user-turn ids ``[Ui]``.
    """
    chunks: list[str] = []
    i = 0
    while i < len(turns):
        turn = turns[i]
        if str(turn.get("role", "")).strip().lower() != "user":
            i += 1
            continue
        user_text = f"user: {turn.get('content', '')}"
        if i + 1 < len(turns) and str(turns[i + 1].get("role", "")).strip().lower() == "assistant":
            asst_text = f"assistant: {turns[i + 1].get('content', '')}"
            chunks.append(f"{user_text}\n{asst_text}")
            i += 2
        else:
            chunks.append(user_text)
            i += 1
    return chunks


def match_pair_index(retrieved_text: str, pair_texts: list[str]) -> Optional[int]:
    """Map retrieved memory text to a unique PAIR index, or None if ambiguous/missing."""
    if not pair_texts:
        return None
    norm_ret = normalize(retrieved_text)
    if not norm_ret:
        return None

    exact = [i for i, p in enumerate(pair_texts) if normalize(p) == norm_ret]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None

    candidates: list[tuple[int, int]] = []
    for i, p in enumerate(pair_texts):
        np = normalize(p)
        if not np:
            continue
        if norm_ret in np or np in norm_ret:
            candidates.append((i, min(len(norm_ret), len(np))))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]

    best_len = max(c[1] for c in candidates)
    best = [c[0] for c in candidates if c[1] == best_len]
    return best[0] if len(best) == 1 else None


class PairIndexCache:
    """Lazy cache: session_id → pair texts."""

    def __init__(self, sessions_by_id: dict[str, list[dict[str, Any]]]):
        self._sessions = sessions_by_id
        self._pairs: dict[str, list[str]] = {}

    def get_pairs(self, session_id: str) -> list[str]:
        if session_id not in self._pairs:
            turns = self._sessions.get(session_id)
            if not turns:
                self._pairs[session_id] = []
            else:
                self._pairs[session_id] = pair_texts_for_session(turns)
        return self._pairs[session_id]

    def index_for(self, session_id: str, retrieved_text: str) -> Optional[int]:
        return match_pair_index(retrieved_text, self.get_pairs(session_id))
