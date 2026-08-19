"""Prompts for the ChromaDB hybrid RAG agent."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a precise question-answering system with access to two knowledge \
sources: conversation memory and a document corpus. You will be given a \
question and retrieved passages from both sources. Answer using only what \
is supported by the provided passages. If the passages are missing, \
insufficient, or do not support a definite answer, respond with exactly: \
Insufficient Information. Reason step-by-step, then give a short factual \
final answer (or that exact phrase)."""


def format_retrieved_passages(
    memory_passages: list[str],
    document_passages: list[str],
) -> str:
    parts: list[str] = []

    if memory_passages:
        parts.append("=== Retrieved from Conversation Memory ===")
        for i, p in enumerate(memory_passages, 1):
            parts.append(f"--- Memory Passage {i} ---\n{p}")

    if document_passages:
        parts.append("=== Retrieved from Document Corpus ===")
        for i, p in enumerate(document_passages, 1):
            parts.append(f"--- Document Passage {i} ---\n{p}")

    if not parts:
        return "(No passages retrieved.)"
    return "\n\n".join(parts)


def build_user_prompt(query: str, passages_block: str) -> str:
    return f"PASSAGES:\n{passages_block}\n\nQUESTION:\n{query}"
