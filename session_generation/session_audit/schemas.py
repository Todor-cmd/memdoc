"""Structured-output schemas for evidence-field → user-turn localization."""

from __future__ import annotations

from pydantic import BaseModel, Field

_TURN_INDEX_HELP = (
    "Integers N matching transcript labels [UN] "
    "(e.g. [U0] → 0, [U1] → 1). User messages only; "
    "empty list if the field is absent."
)


class EvidenceFieldLocations(BaseModel):
    """Non-temporal audit: topic, source, and key information in user turns.

    Each list holds integers N for transcript labels [UN] (first user message
    is [U0] → 0). Assistant turns are never indexed. Empty list = not found.
    """

    topic: list[int] = Field(
        description=(
            "User-turn indices where the article title / topic appears. "
            + _TURN_INDEX_HELP
        ),
    )
    source: list[int] = Field(
        description=(
            "User-turn indices where the source / outlet name appears. "
            + _TURN_INDEX_HELP
        ),
    )
    key_information: list[int] = Field(
        description=(
            "User-turn indices where key factual details from the evidence "
            "appear (may span multiple turns). "
            + _TURN_INDEX_HELP
        ),
    )
    notes: str = Field(
        default="",
        description="Brief rationale for the localization decisions.",
    )


class EvidenceFieldLocationsTemporal(EvidenceFieldLocations):
    """Temporal audit: also localize publication time in user turns."""

    published_at: list[int] = Field(
        description=(
            "User-turn indices where the publication time / date appears. "
            + _TURN_INDEX_HELP
        ),
    )


def all_required_present(
    locations: EvidenceFieldLocations,
    *,
    is_temporal: bool,
) -> bool:
    """True when every required field has at least one user-turn index."""
    if not locations.topic or not locations.source or not locations.key_information:
        return False
    if is_temporal:
        published = getattr(locations, "published_at", None)
        if not published:
            return False
    return True
