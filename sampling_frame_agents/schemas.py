from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentAnswer(BaseModel):
    """Structured output produced by every agent / baseline run."""

    reasoning: str = Field(description="Chain-of-thought reasoning")
    final_answer: str = Field(
        max_length=70,
        description="Short factual answer for EM/F1 scoring",
    )


class TokenUsage(BaseModel):
    """Provider-reported token counts (not model-generated)."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class InferenceResult(BaseModel):
    """Answer from the model plus optional API telemetry."""

    answer: AgentAnswer
    usage: Optional[TokenUsage] = None
    #: Set when the API or structured output failed; exclude from headline EM/F1.
    inference_error: Optional[str] = None
    #: Gold evidence assigned to the memory store (from the dataset row).
    golden_memory_evidence: Optional[List[Dict[str, Any]]] = None
    #: Gold evidence assigned to the document / corpus store (from the dataset row).
    golden_document_evidence: Optional[List[Dict[str, Any]]] = None
    #: Evidence texts actually passed into the model prompt for this run (retrieval
    #: or oracle). Empty when nothing is retrieved or shown.
    retrieved_evidence: Optional[List[Dict[str, Any]]] = None
