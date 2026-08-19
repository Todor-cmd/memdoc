"""Normalize agent answer_fn return values for experiment output rows."""

from __future__ import annotations

from typing import Any


def coerce_answer_result(raw: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Split an answer_fn result into ``(prediction, extra_fields)``.

    Naive agents return a prediction string. Corrective RAG returns a dict with
    ``prediction`` plus loop metadata (``rewrite_count``, etc.).
    """
    if isinstance(raw, str):
        return raw, {}

    if not isinstance(raw, dict):
        raise TypeError(
            f"answer_fn must return str or dict, got {type(raw).__name__}"
        )

    if "prediction" in raw:
        prediction = str(raw["prediction"])
        extra = {k: v for k, v in raw.items() if k != "prediction"}
        return prediction, extra

    if "final_answer" in raw:
        prediction = str(raw["final_answer"])
        extra = {k: v for k, v in raw.items() if k != "final_answer"}
        return prediction, extra

    raise ValueError(
        "answer_fn dict must include 'prediction' or 'final_answer'; "
        f"got keys: {sorted(raw)}"
    )
