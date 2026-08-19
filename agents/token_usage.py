"""Helpers for extracting provider-reported token counts from LLM responses."""

from __future__ import annotations

from typing import Any


def extract_token_usage(response: Any) -> tuple[int, int, int]:
    """Return ``(input_tokens, output_tokens, total_tokens)`` from a response.

    Handles both a raw LangChain ``AIMessage`` and the dict returned by
    ``with_structured_output(include_raw=True)`` (which nests the message under
    ``"raw"``). Returns zeros when usage metadata is unavailable.
    """
    msg = response.get("raw") if isinstance(response, dict) else response
    meta = getattr(msg, "usage_metadata", None)
    if not isinstance(meta, dict):
        return 0, 0, 0
    return (
        int(meta.get("input_tokens") or 0),
        int(meta.get("output_tokens") or 0),
        int(meta.get("total_tokens") or 0),
    )
