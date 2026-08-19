from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class EvidenceItem(BaseModel):
    """A single evidence item from MultiHop-RAG."""

    fact: str
    title: str
    url: str
    source: str
    author: str
    published_at: str
    category: str


class Turn(BaseModel):
    role: str = Field(description="Either 'user' or 'assistant'")
    content: str


class Session(BaseModel):
    """Structured output schema for the session-generation LLM call."""

    turns: list[Turn] = Field(
        description="Alternating user/assistant conversation turns. In total 4 to 10 turns.",
        min_length=4,
        max_length=10
    )


class SessionRecord(BaseModel):
    """History-compiler-compatible session record."""

    session_id: str
    session: list[dict[str, str]]


class SessionMetadata(BaseModel):
    """Sidecar metadata for traceability and oracle construction."""

    session_id: str
    source_question_id: str
    evidence_index: int
    source_evidence: dict
    persona_id: str
    generation_model: str
    generation_timestamp: str
    status: str
    flagged_for_review: bool = False
    flag_reasons: list[str] = Field(default_factory=list)
    attempts: int = 1
    fact_coverage_result: Optional[dict] = None
    hallucination_result: Optional[dict] = None
    scenario: Optional[str] = None
    scenario_category: Optional[str] = None
    turn_range: Optional[list[int]] = None
    evidence_placement: Optional[str] = None
    topic_drift: Optional[str] = None
    evidence_density: Optional[str] = None


class FactCoverageResult(BaseModel):
    """LLM-as-judge output for fact-coverage verification."""

    claims_present: list[str] = Field(
        description="Atomic claims from the evidence found in the session"
    )
    claims_missing: list[str] = Field(
        description="Atomic claims from the evidence NOT found in the session"
    )
    all_covered: bool = Field(
        description="True if all key claims are present in the session"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


class HallucinationCheckResult(BaseModel):
    """LLM-as-judge output for hallucination detection."""

    hallucinated_claims: list[str] = Field(
        description=(
            "Factual claims in the session not grounded in the provided evidence"
        )
    )
    is_clean: bool = Field(
        description="True if no hallucinated facts were found"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")
