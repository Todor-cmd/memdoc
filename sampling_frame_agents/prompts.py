from __future__ import annotations

SYSTEM_PROMPT = """\
You are a precise question-answering system. You will be given a question \
and a set of evidence. Answer using only what is supported by the provided \
evidence. If the evidence is missing, insufficient, or does not support a \
definite answer, respond with exactly: Insufficient Information. Reason \
step-by-step, then give a short factual final answer (or that exact phrase)."""

SYSTEM_PROMPT_NO_CONTEXT = """\
You are a precise question-answering system. You will be given a question \
only—no evidence passages. Answer using your general knowledge when you can \
do so confidently. If you do not have enough information to answer, respond \
with exactly: Insufficient Information. Reason step-by-step, then give a \
short factual final answer (or that exact phrase)."""


def format_evidence(
    evidence_list: list[dict],
    temporal: bool,
    include_author: bool = True,
) -> str:
    """Render evidence items into a numbered text block for the prompt."""
    parts: list[str] = []
    for i, ev in enumerate(evidence_list, start=1):
        lines = [
            f"Title: {ev['title']}",
            f"Source: {ev['source']}",
        ]
        if include_author:
            lines.append(f"Author: {ev['author']}")
        if temporal:
            lines.append(f"Published: {ev['published_at']}")
        lines.append(f"Content: {ev['fact']}")
        parts.append(f"--- Evidence {i} ---\n" + "\n".join(lines))
    return "\n\n".join(parts)


def build_user_prompt(query: str, evidence_block: str) -> str:
    return f"EVIDENCE:\n{evidence_block}\n\nQUESTION:\n{query}"


def build_question_only_user_prompt(query: str) -> str:
    """User message with the question only (no evidence)."""
    return f"QUESTION:\n{query}"
