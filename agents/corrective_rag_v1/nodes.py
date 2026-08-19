"""LangGraph node functions for the agentic RAG pipeline.

Flow: retrieve → grade → (rewrite ↻ retrieve → grade) → generate
"""

from __future__ import annotations

from typing import Any, TypedDict

import backoff
import groq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from sampling_frame_agents.schemas import AgentAnswer
from agents.naive_rag.prompts import format_retrieved_passages
from agents.token_usage import extract_token_usage
from typing import Protocol

class _StoreHooks(Protocol):
    def query_memory(self, query: str, k: int | None = ...) -> list[str]: ...
    def query_documents(self, query: str, k: int | None = ...) -> list[str]: ...

from .prompts import (
    GENERATE_SYSTEM,
    GENERATE_USER,
    GRADING_SYSTEM,
    GRADING_USER,
    REWRITE_SYSTEM,
    REWRITE_USER,
)


class AgentState(TypedDict):
    question: str
    rewritten_question: str
    memory_passages: list[str]
    document_passages: list[str]
    documents_relevant: bool
    final_answer: str
    reasoning: str
    attempts: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


MAX_REWRITE_ATTEMPTS = 2


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


def make_retrieve_node(store: _StoreHooks):
    """Retrieve passages from both memory and document stores."""

    def retrieve(state: AgentState) -> dict[str, Any]:
        q = state.get("rewritten_question") or state["question"]
        return {
            "memory_passages": store.query_memory(q),
            "document_passages": store.query_documents(q),
        }

    return retrieve


def make_grade_node(llm: ChatGroq):
    """LLM-based relevance grading of retrieved passages."""

    def grade_documents(state: AgentState) -> dict[str, Any]:
        q = state.get("rewritten_question") or state["question"]
        context = format_retrieved_passages(
            state.get("memory_passages", []),
            state.get("document_passages", []),
        )
        messages = [
            SystemMessage(content=GRADING_SYSTEM),
            HumanMessage(content=GRADING_USER.format(context=context, question=q)),
        ]
        try:
            resp = _invoke_llm(llm, messages)
            relevant = resp.content.strip().lower().startswith("yes")
            tokens = _accumulate_tokens(state, resp)
        except Exception:
            relevant = True
            tokens = {}
        return {"documents_relevant": relevant, **tokens}

    return grade_documents


def make_rewrite_node(llm: ChatGroq):
    """Rewrite the query to improve retrieval on the next attempt."""

    def rewrite(state: AgentState) -> dict[str, Any]:
        q = state.get("rewritten_question") or state["question"]
        messages = [
            SystemMessage(content=REWRITE_SYSTEM),
            HumanMessage(content=REWRITE_USER.format(question=q)),
        ]
        try:
            resp = _invoke_llm(llm, messages)
            new_q = resp.content.strip() or q
            tokens = _accumulate_tokens(state, resp)
        except Exception:
            new_q = q
            tokens = {}
        return {
            "rewritten_question": new_q,
            "attempts": state.get("attempts", 0) + 1,
            **tokens,
        }

    return rewrite


def make_generate_node(structured_llm: Any):
    """Generate a structured final answer from the retrieved passages."""

    def generate(state: AgentState) -> dict[str, Any]:
        q = state.get("rewritten_question") or state["question"]
        context = format_retrieved_passages(
            state.get("memory_passages", []),
            state.get("document_passages", []),
        )
        messages = [
            SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(content=GENERATE_USER.format(context=context, question=q)),
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


def should_generate_or_rewrite(state: AgentState) -> str:
    """Conditional edge: generate if relevant or retries exhausted, else rewrite."""
    if state.get("documents_relevant", False):
        return "generate"
    if state.get("attempts", 0) >= MAX_REWRITE_ATTEMPTS:
        return "generate"
    return "rewrite"
