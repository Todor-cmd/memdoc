"""CorrectiveRAGAgent: per-chunk corrective RAG with gap-driven dual-query rewrites."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from sampling_frame_agents.schemas import AgentAnswer
from experiment.paths import DEFAULT_DOCUMENT_CORPUS, DEFAULT_MEMORY_DIR

from agents.naive_rag.embeddings import DEFAULT_EMBEDDING_MODEL
from agents.naive_rag.indexing import MemoryGranularity
from agents.naive_rag.store import ChromaStoreHooks, RetrieverType

from .nodes import (
    AgentState,
    make_generate_node,
    make_grade_node,
    make_retrieve_node,
    make_rewrite_node,
    make_sufficiency_node,
    should_generate_or_rewrite,
)


@dataclass(frozen=True)
class CorrectiveRAGResult:
    """Inference output plus corrective-loop metadata."""

    final_answer: str
    reasoning: str
    rewrite_count: int
    retrieval_passes: int
    evidence_sufficient: bool
    knowledge_gap: str
    retained_memory_count: int
    retained_documents_count: int
    rewritten_memory_query: str
    rewritten_document_query: str
    retrieval_history: list[dict] = field(default_factory=list)
    final_memory_ids: list[str] = field(default_factory=list)
    final_document_ids: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def to_answer_fn_output(self) -> dict[str, Any]:
        return {
            "prediction": self.final_answer,
            "reasoning": self.reasoning,
            "rewrite_count": self.rewrite_count,
            "retrieval_passes": self.retrieval_passes,
            "evidence_sufficient": self.evidence_sufficient,
            "knowledge_gap": self.knowledge_gap,
            "retained_memory_count": self.retained_memory_count,
            "retained_documents_count": self.retained_documents_count,
            "rewritten_memory_query": self.rewritten_memory_query,
            "rewritten_document_query": self.rewritten_document_query,
            "retrieval_history": self.retrieval_history,
            "final_memory_ids": self.final_memory_ids,
            "final_document_ids": self.final_document_ids,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class CorrectiveRAGAgent:
    """Per-chunk corrective RAG: retrieve → grade_per_chunk → sufficiency → (rewrite ↻) → generate.

    Architecture:
    - Per-chunk LLM grading (binary reranker via batched plain-text call)
    - Sufficiency assessment with knowledge-gap identification (structured output)
    - Gap-driven dual source-targeted query rewrites (structured output)
    - Cross-round chunk accumulation with full exclusion of all seen IDs
    - 2x over-fetch to compensate for chunks graded irrelevant

    Reuses :class:`ChromaStoreHooks` for store management and provides
    ``hooks_factory`` / ``answer_fn`` for :func:`experiment.runner.run_experiment`.
    """

    def __init__(
        self,
        *,
        model_name: str = "llama-3.3-70b-versatile",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        temperature: float = 0.0,
        seed: int = 42,
        memory_k: int = 10,
        document_k: int = 10,
        memory_dir: Path | str = DEFAULT_MEMORY_DIR,
        corpus_path: Path | str = DEFAULT_DOCUMENT_CORPUS,
        memory_granularity: MemoryGranularity = MemoryGranularity.PAIR,
        retriever_type: RetrieverType = RetrieverType.DENSE,
        hybrid_weights: tuple[float, float] = (0.5, 0.5),
        store_backend: str = "chroma",
        min_score: float | None = None,
    ) -> None:
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.seed = seed
        self.memory_k = memory_k
        self.document_k = document_k
        self.memory_dir = Path(memory_dir)
        self.corpus_path = Path(corpus_path)
        self.memory_granularity = memory_granularity
        self.retriever_type = retriever_type
        self.hybrid_weights = hybrid_weights
        self.store_backend = store_backend
        self.min_score = min_score

        self._plain_llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            seed=seed,
        )
        self._structured_llm = self._plain_llm.with_structured_output(
            AgentAnswer, include_raw=True,
        )

        self._store: ChromaStoreHooks | None = None
        self._graph: StateGraph | None = None

    def hooks_factory(self):
        """Create a store hooks instance based on ``store_backend``."""
        if self.store_backend == "mempalace":
            from agents.mempalace_store.store import MemPalaceStoreHooks

            self._store = MemPalaceStoreHooks(
                memory_dir=self.memory_dir,
                corpus_path=self.corpus_path,
                memory_k=self.memory_k,
                document_k=self.document_k,
                memory_granularity=self.memory_granularity,
                retriever_type=self.retriever_type,
                hybrid_weights=self.hybrid_weights,
            )
        else:
            self._store = ChromaStoreHooks(
                embedding_model=self.embedding_model,
                memory_dir=self.memory_dir,
                corpus_path=self.corpus_path,
                memory_k=self.memory_k,
                document_k=self.document_k,
                memory_granularity=self.memory_granularity,
                retriever_type=self.retriever_type,
                hybrid_weights=self.hybrid_weights,
                min_score=self.min_score,
            )
        self._graph = self._build_graph(self._store)
        return self._store

    def _build_graph(self, store: ChromaStoreHooks) -> StateGraph:
        """Construct the 5-node LangGraph."""
        builder = StateGraph(AgentState)

        builder.add_node("retrieve", make_retrieve_node(store, self.memory_k, self.document_k))
        builder.add_node("grade", make_grade_node(self._plain_llm))
        builder.add_node("sufficiency", make_sufficiency_node(self._plain_llm))
        builder.add_node("rewrite", make_rewrite_node(self._plain_llm))
        builder.add_node("generate", make_generate_node(self._structured_llm))

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_edge("grade", "sufficiency")
        builder.add_conditional_edges(
            "sufficiency",
            should_generate_or_rewrite,
            {"generate": "generate", "rewrite": "rewrite"},
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("generate", END)

        return builder.compile()

    def run_inference(self, query: str) -> CorrectiveRAGResult:
        """Run the corrective RAG graph and return answer plus loop metadata."""
        if self._store is None or self._graph is None:
            raise RuntimeError("hooks_factory() must be called before run_inference()")

        initial_state: AgentState = {
            "question": query,
            "memory_query": "",
            "document_query": "",
            "retained_memory": [],
            "retained_documents": [],
            "seen_memory_ids": [],
            "seen_document_ids": [],
            "pending_memory": [],
            "pending_documents": [],
            "evidence_sufficient": False,
            "knowledge_gap": "",
            "retrieval_history": [],
            "final_answer": "",
            "reasoning": "",
            "attempts": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        result = self._graph.invoke(initial_state)
        rewrite_count = int(result.get("attempts", 0))

        final_mem_ids = [cid for cid, _ in result.get("retained_memory", [])]
        final_doc_ids = [cid for cid, _ in result.get("retained_documents", [])]

        return CorrectiveRAGResult(
            final_answer=result.get("final_answer", "[INFERENCE_FAILED]"),
            reasoning=result.get("reasoning", "") or "",
            rewrite_count=rewrite_count,
            retrieval_passes=rewrite_count + 1,
            evidence_sufficient=bool(result.get("evidence_sufficient", False)),
            knowledge_gap=result.get("knowledge_gap", ""),
            retained_memory_count=len(final_mem_ids),
            retained_documents_count=len(final_doc_ids),
            rewritten_memory_query=(result.get("memory_query") or query).strip() or query,
            rewritten_document_query=(result.get("document_query") or query).strip() or query,
            retrieval_history=result.get("retrieval_history", []),
            final_memory_ids=final_mem_ids,
            final_document_ids=final_doc_ids,
            input_tokens=int(result.get("input_tokens", 0)),
            output_tokens=int(result.get("output_tokens", 0)),
            total_tokens=int(result.get("total_tokens", 0)),
        )

    def answer_fn(self, query: str) -> dict[str, Any]:
        """Run inference; returns prediction plus corrective-loop metadata,
        retrieval history, and final IDs passed to generate."""
        result = self.run_inference(query)
        return result.to_answer_fn_output()
