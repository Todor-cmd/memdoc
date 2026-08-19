"""LangGraph node functions for the per-chunk corrective RAG pipeline.

Flow: retrieve → grade_per_chunk → sufficiency → (gap-driven rewrite ↻ retrieve → grade → sufficiency) → generate

Architecture:
- Per-chunk LLM grading (binary reranker) via batched plain-text call
- Sufficiency assessment with knowledge-gap identification (structured output)
- Gap-driven dual source-targeted query rewrites (structured output)
- Cross-round chunk accumulation with full exclusion of all seen IDs
- 2x over-fetch to compensate for chunks graded irrelevant
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, Protocol, TypedDict

import backoff
import groq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from sampling_frame_agents.schemas import AgentAnswer
from agents.naive_rag.prompts import format_retrieved_passages
from agents.token_usage import extract_token_usage

from .prompts import (
    GENERATE_SYSTEM,
    GENERATE_USER,
    GRADING_SYSTEM,
    GRADING_USER,
    REWRITE_SYSTEM,
    REWRITE_USER,
    SUFFICIENCY_SYSTEM,
    SUFFICIENCY_USER,
)


# ---------------------------------------------------------------------------
# Structured output schemas for sufficiency and rewrite nodes
# ---------------------------------------------------------------------------

class SufficiencyAssessment(BaseModel):
    sufficient: bool = Field(description="Whether the evidence is sufficient to answer the question")
    knowledge_gap: str | None = Field(
        default=None,
        description="When insufficient: one sentence describing the specific information still missing",
    )


class DualQueryRewrite(BaseModel):
    memory_query: str = Field(
        description="Rewritten query targeting personal conversation memory (informal, contextual)",
    )
    document_query: str = Field(
        description="Rewritten query targeting a news/document corpus (formal, factual)",
    )


# ---------------------------------------------------------------------------
# Store protocol
# ---------------------------------------------------------------------------

class _StoreHooks(Protocol):
    def query_memory(self, query: str, k: int | None = ...) -> list[str]: ...
    def query_documents(self, query: str, k: int | None = ...) -> list[str]: ...
    def query_memory_scored(
        self, query: str, k: int | None = ..., exclude_ids: set[str] | None = ...,
    ) -> list[tuple[str, str, float]]: ...
    def query_documents_scored(
        self, query: str, k: int | None = ..., exclude_ids: set[str] | None = ...,
    ) -> list[tuple[str, str, float]]: ...


# ---------------------------------------------------------------------------
# State reducers
# ---------------------------------------------------------------------------

def _dedup_chunks(
    existing: list[tuple[str, str]], new: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Reducer: accumulate (chunk_id, text) pairs, deduplicating by chunk_id."""
    seen_ids = {cid for cid, _ in existing}
    return existing + [(cid, text) for cid, text in new if cid not in seen_ids]


def _dedup_ids(existing: list[str], new: list[str]) -> list[str]:
    """Reducer: accumulate string IDs, deduplicating."""
    seen = set(existing)
    return existing + [x for x in new if x not in seen]


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    memory_query: str
    document_query: str
    # LLM-graded relevant chunks only
    retained_memory: Annotated[list[tuple[str, str]], _dedup_chunks]
    retained_documents: Annotated[list[tuple[str, str]], _dedup_chunks]
    # All retrieved chunk IDs (relevant + irrelevant) for exclusion
    seen_memory_ids: Annotated[list[str], _dedup_ids]
    seen_document_ids: Annotated[list[str], _dedup_ids]
    # Pending chunks from latest retrieval (replaced each round, not accumulated)
    pending_memory: list[tuple[str, str, float]]
    pending_documents: list[tuple[str, str, float]]
    evidence_sufficient: bool
    knowledge_gap: str
    retrieval_history: Annotated[list[dict], operator.add]
    final_answer: str
    reasoning: str
    attempts: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


MAX_REWRITE_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_bad_request(exc: BaseException) -> bool:
    return isinstance(exc, groq.BadRequestError)


def _accumulate_tokens(state: AgentState, response: Any) -> dict[str, int]:
    """Add this response's token usage to the running totals in ``state``."""
    in_t, out_t, tot_t = extract_token_usage(response)
    return {
        "input_tokens": state.get("input_tokens", 0) + in_t,
        "output_tokens": state.get("output_tokens", 0) + out_t,
        "total_tokens": state.get("total_tokens", 0) + tot_t,
    }


@backoff.on_exception(
    backoff.expo,
    (groq.RateLimitError, groq.APIError),
    max_tries=5,
    giveup=_is_bad_request,
)
def _invoke_llm(llm: Any, messages: list) -> Any:
    return llm.invoke(messages)


# ---------------------------------------------------------------------------
# Retrieve node
# ---------------------------------------------------------------------------

def make_retrieve_node(store: _StoreHooks, memory_k: int = 10, document_k: int = 10):
    """Retrieve with dual queries, 2x over-fetch, and full exclusion via seen_*_ids."""

    def retrieve(state: AgentState) -> dict[str, Any]:
        mem_q = state.get("memory_query") or state["question"]
        doc_q = state.get("document_query") or state["question"]

        mem_budget = max(0, memory_k - len(state.get("retained_memory", [])))
        doc_budget = max(0, document_k - len(state.get("retained_documents", [])))

        fetch_mem_k = mem_budget * 2
        fetch_doc_k = doc_budget * 2

        seen_mem = set(state.get("seen_memory_ids", []))
        seen_doc = set(state.get("seen_document_ids", []))

        new_mem = (
            store.query_memory_scored(mem_q, k=fetch_mem_k, exclude_ids=seen_mem)
            if fetch_mem_k > 0 else []
        )
        new_doc = (
            store.query_documents_scored(doc_q, k=fetch_doc_k, exclude_ids=seen_doc)
            if fetch_doc_k > 0 else []
        )

        return {
            "pending_memory": new_mem,
            "pending_documents": new_doc,
            "seen_memory_ids": [cid for cid, _, _ in new_mem],
            "seen_document_ids": [cid for cid, _, _ in new_doc],
        }

    return retrieve


# ---------------------------------------------------------------------------
# Grade node (per-chunk, plain text parsing)
# ---------------------------------------------------------------------------

def make_grade_node(llm: ChatGroq):
    """Per-chunk LLM grading: list relevant IDs from all pending chunks."""

    def grade(state: AgentState) -> dict[str, Any]:
        q = state.get("memory_query") or state["question"]
        pending_mem = state.get("pending_memory", [])
        pending_doc = state.get("pending_documents", [])

        if not pending_mem and not pending_doc:
            return {"retrieval_history": [_make_round_log(state, [], [])]}

        numbered, id_map = _format_numbered_passages(pending_mem, pending_doc)

        messages = [
            SystemMessage(content=GRADING_SYSTEM),
            HumanMessage(content=GRADING_USER.format(
                question=state["question"], numbered_passages=numbered,
            )),
        ]
        try:
            resp = _invoke_llm(llm, messages)
            tokens = _accumulate_tokens(state, resp)
            relevant_ids = _parse_relevant_ids(resp.content, id_map)
        except Exception:
            relevant_ids = set(id_map.keys())
            tokens = {}

        new_retained_mem = []
        new_retained_doc = []
        mem_log = []
        doc_log = []

        for i, (cid, text, score) in enumerate(pending_mem):
            tag = f"M{i + 1}"
            is_relevant = tag in relevant_ids
            if is_relevant:
                new_retained_mem.append((cid, text))
            mem_log.append({"id": cid, "score": round(score, 4), "relevant": is_relevant})

        for i, (cid, text, score) in enumerate(pending_doc):
            tag = f"D{i + 1}"
            is_relevant = tag in relevant_ids
            if is_relevant:
                new_retained_doc.append((cid, text))
            doc_log.append({"id": cid, "score": round(score, 4), "relevant": is_relevant})

        round_log = _make_round_log(state, mem_log, doc_log)

        return {
            "retained_memory": new_retained_mem,
            "retained_documents": new_retained_doc,
            "retrieval_history": [round_log],
            **tokens,
        }

    return grade


def _format_numbered_passages(
    pending_mem: list[tuple[str, str, float]],
    pending_doc: list[tuple[str, str, float]],
) -> tuple[str, dict[str, str]]:
    """Format passages with [M1], [D1] tags. Returns formatted text and tag->chunk_id map."""
    lines = []
    id_map: dict[str, str] = {}

    for i, (cid, text, _) in enumerate(pending_mem):
        tag = f"M{i + 1}"
        id_map[tag] = cid
        lines.append(f"[{tag}] {text}")

    for i, (cid, text, _) in enumerate(pending_doc):
        tag = f"D{i + 1}"
        id_map[tag] = cid
        lines.append(f"[{tag}] {text}")

    return "\n\n".join(lines), id_map


def _parse_relevant_ids(content: str, id_map: dict[str, str]) -> set[str]:
    """Parse comma-separated IDs from LLM response. Returns set of tags (M1, D2, etc.)."""
    content = content.strip().lower()
    if "none" in content and len(content) < 20:
        return set()
    found = set()
    for tag in id_map:
        if tag.lower() in content:
            found.add(tag)
    if not found:
        pattern = re.compile(r"[MD]\d+", re.IGNORECASE)
        matches = pattern.findall(content)
        found = {m.upper() for m in matches if m.upper() in id_map}
    return found


def _make_round_log(state: AgentState, mem_log: list, doc_log: list) -> dict:
    return {
        "round": state.get("attempts", 0) + 1,
        "memory_query": state.get("memory_query") or state["question"],
        "document_query": state.get("document_query") or state["question"],
        "memory_retrieved": mem_log,
        "document_retrieved": doc_log,
        "retained_memory_count": len(state.get("retained_memory", [])) + sum(
            1 for e in mem_log if e.get("relevant")
        ),
        "retained_documents_count": len(state.get("retained_documents", [])) + sum(
            1 for e in doc_log if e.get("relevant")
        ),
    }


# ---------------------------------------------------------------------------
# Sufficiency node (structured output)
# ---------------------------------------------------------------------------

def make_sufficiency_node(llm: ChatGroq):
    """Assess whether retained evidence is sufficient; identify knowledge gap if not."""
    structured_llm = llm.with_structured_output(SufficiencyAssessment, include_raw=True)

    def assess_sufficiency(state: AgentState) -> dict[str, Any]:
        mem_texts = [text for _, text in state.get("retained_memory", [])]
        doc_texts = [text for _, text in state.get("retained_documents", [])]
        context = format_retrieved_passages(mem_texts, doc_texts)

        messages = [
            SystemMessage(content=SUFFICIENCY_SYSTEM),
            HumanMessage(content=SUFFICIENCY_USER.format(
                question=state["question"], context=context,
            )),
        ]
        try:
            out = _invoke_llm(structured_llm, messages)
            tokens = _accumulate_tokens(state, out)
            if out["parsing_error"] is not None:
                return {"evidence_sufficient": False, "knowledge_gap": "Failed to assess", **tokens}
            parsed: SufficiencyAssessment = out["parsed"]
            return {
                "evidence_sufficient": parsed.sufficient,
                "knowledge_gap": parsed.knowledge_gap or "",
                **tokens,
            }
        except Exception:
            return {"evidence_sufficient": False, "knowledge_gap": "Assessment failed"}

    return assess_sufficiency


# ---------------------------------------------------------------------------
# Rewrite node (gap-driven dual-query, structured output)
# ---------------------------------------------------------------------------

def make_rewrite_node(llm: ChatGroq):
    """Produce dual source-targeted queries driven by the identified knowledge gap."""
    structured_llm = llm.with_structured_output(DualQueryRewrite, include_raw=True)

    def rewrite(state: AgentState) -> dict[str, Any]:
        gap = state.get("knowledge_gap", "") or "General information needed"

        messages = [
            SystemMessage(content=REWRITE_SYSTEM),
            HumanMessage(content=REWRITE_USER.format(
                question=state["question"], knowledge_gap=gap,
            )),
        ]
        try:
            out = _invoke_llm(structured_llm, messages)
            tokens = _accumulate_tokens(state, out)
            if out["parsing_error"] is not None:
                return {
                    "memory_query": state["question"],
                    "document_query": state["question"],
                    "attempts": state.get("attempts", 0) + 1,
                    **tokens,
                }
            parsed: DualQueryRewrite = out["parsed"]
            return {
                "memory_query": parsed.memory_query,
                "document_query": parsed.document_query,
                "attempts": state.get("attempts", 0) + 1,
                **tokens,
            }
        except Exception:
            return {
                "memory_query": state["question"],
                "document_query": state["question"],
                "attempts": state.get("attempts", 0) + 1,
            }

    return rewrite


# ---------------------------------------------------------------------------
# Generate node
# ---------------------------------------------------------------------------

def make_generate_node(structured_llm: Any):
    """Generate a structured final answer using all retained-relevant chunks."""

    def generate(state: AgentState) -> dict[str, Any]:
        q = state.get("memory_query") or state["question"]
        mem_texts = [text for _, text in state.get("retained_memory", [])]
        doc_texts = [text for _, text in state.get("retained_documents", [])]
        context = format_retrieved_passages(mem_texts, doc_texts)

        messages = [
            SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(content=GENERATE_USER.format(context=context, question=state["question"])),
        ]
        try:
            out = _invoke_llm(structured_llm, messages)
            tokens = _accumulate_tokens(state, out)
            if out["parsing_error"] is not None:
                return {"final_answer": "[INFERENCE_FAILED]", "reasoning": "", **tokens}
            parsed: AgentAnswer | None = out["parsed"]
            if parsed is None:
                return {"final_answer": "[INFERENCE_FAILED]", "reasoning": "", **tokens}
            return {
                "final_answer": parsed.final_answer,
                "reasoning": parsed.reasoning,
                **tokens,
            }
        except Exception:
            return {"final_answer": "[INFERENCE_FAILED]", "reasoning": ""}

    return generate


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_generate_or_rewrite(state: AgentState) -> str:
    """Conditional edge: generate if evidence sufficient or budget exhausted."""
    if state.get("evidence_sufficient", False):
        return "generate"
    if state.get("attempts", 0) >= MAX_REWRITE_ATTEMPTS:
        return "generate"
    return "rewrite"
