"""Prompts for evidence-field localization in generated memory sessions."""

from __future__ import annotations

from datetime import datetime
from typing import Any


SYSTEM_PROMPT = """\
You audit whether a generated user–assistant conversation embeds intended \
news-article evidence fields in the USER's messages.

Rules:
- Credit a field ONLY if the USER states it. Assistant-only mentions do not count.
- Semantic / paraphrased presence is enough; verbatim copy is not required.
- Return 0-based indices of the numbered user turns ([U0], [U1], …) where each \
field appears. A field may appear in multiple user turns — list all of them.
- Use an empty list when the field is absent from every user turn.
- Keep notes brief.
"""


def format_transcript(session_turns: list[dict[str, Any]]) -> str:
    """Render the session with [U{i}] labels on user turns only."""
    lines: list[str] = []
    user_i = 0
    for turn in session_turns:
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if role == "user":
            lines.append(f"[U{user_i}] {content}")
            user_i += 1
        else:
            label = "A" if role == "assistant" else role.upper() or "OTHER"
            lines.append(f"[{label}] {content}")
    return "\n\n".join(lines) if lines else "(empty session)"


def _format_published_at(raw: str) -> str:
    """Human-readable published_at matching session-generation prompts when possible."""
    s = str(raw).strip()
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y at %I:%M %p UTC")
    except ValueError:
        return s


def build_user_prompt(
    *,
    evidence: dict[str, Any],
    session_turns: list[dict[str, Any]],
    is_temporal: bool,
) -> str:
    """Build the audit user message with gold fields and labeled transcript."""
    title = str(evidence.get("title", "")).strip()
    source = str(evidence.get("source", "")).strip()
    fact = str(evidence.get("fact", "")).strip()

    evidence_lines = [
        f"- Topic (title): {title}",
        f"- Source: {source}",
        f"- Key information: {fact}",
    ]
    if is_temporal:
        raw_pub = evidence.get("published_at", "")
        pretty = _format_published_at(str(raw_pub)) if raw_pub else ""
        evidence_lines.append(f"- Published at (raw): {raw_pub}")
        if pretty and pretty != str(raw_pub).strip():
            evidence_lines.append(f"- Published at (readable form used in generation): {pretty}")

    required = "topic, source, key_information"
    if is_temporal:
        required += ", published_at"

    return (
        "INTENDED EVIDENCE FIELDS:\n"
        + "\n".join(evidence_lines)
        + "\n\n"
        "CONVERSATION (credit fields only in [U*] user turns):\n"
        + format_transcript(session_turns)
        + "\n\n"
        f"Locate each required field ({required}) in the user turns and return "
        "the structured localization."
    )
