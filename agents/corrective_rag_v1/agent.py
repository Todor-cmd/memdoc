"""CorrectiveRAGAgent: LangGraph-based corrective RAG with grade/rewrite loop."""

from __future__ import annotations

from dataclasses import dataclass
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
    should_generate_or_rewrite,
)


@dataclass(frozen=True)
class CorrectiveRAGResult:
    """Inference output plus corrective-loop metadata."""

    final_answer: str
    reasoning: str
    rewrite_count: int
    retrieval_passes: int
    documents_relevant: bool
    rewritten_question: str
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def to_answer_fn_output(self) -> dict[str, Any]:
        return {
            "prediction": self.final_answer,
            "reasoning": self.reasoning,
            "rewrite_count": self.rewrite_count,
            "retrieval_passes": self.retrieval_passes,
            "documents_relevant": self.documents_relevant,
            "rewritten_question": self.rewritten_question,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class CorrectiveRAGAgent:
    """Corrective RAG agent: retrieve → grade → (rewrite ↻) → generate.

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

        self._plain_llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            seed=seed,
        )
        self._structured_llm = self._plain_llm.with_structured_output(
            AgentAnswer, include_raw=True,
        )

        self._store = None
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
            )
        self._graph = self._build_graph(self._store)
        return self._store

    def _build_graph(self, store: ChromaStoreHooks) -> StateGraph:
        """Construct the LangGraph: retrieve → grade → conditional → generate/rewrite."""
        builder = StateGraph(AgentState)

        builder.add_node("retrieve", make_retrieve_node(store))
        builder.add_node("grade", make_grade_node(self._plain_llm))
        builder.add_node("rewrite", make_rewrite_node(self._plain_llm))
        builder.add_node("generate", make_generate_node(self._structured_llm))

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            should_generate_or_rewrite,
            {"generate": "generate", "rewrite": "rewrite"},
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("generate", END)

        return builder.compile()

    def run_inference(self, query: str) -> CorrectiveRAGResult:
        """Run the agentic RAG graph and return answer plus loop metadata."""
        if self._store is None or self._graph is None:
            raise RuntimeError("hooks_factory() must be called before run_inference()")

        initial_state: AgentState = {
            "question": query,
            "rewritten_question": "",
            "memory_passages": [],
            "document_passages": [],
            "documents_relevant": False,
            "final_answer": "",
            "reasoning": "",
            "attempts": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        result = self._graph.invoke(initial_state)
        rewrite_count = int(result.get("attempts", 0))
        rewritten = (result.get("rewritten_question") or query).strip() or query
        return CorrectiveRAGResult(
            final_answer=result.get("final_answer", "[INFERENCE_FAILED]"),
            reasoning=result.get("reasoning", "") or "",
            rewrite_count=rewrite_count,
            retrieval_passes=rewrite_count + 1,
            documents_relevant=bool(result.get("documents_relevant", False)),
            rewritten_question=rewritten,
            input_tokens=int(result.get("input_tokens", 0)),
            output_tokens=int(result.get("output_tokens", 0)),
            total_tokens=int(result.get("total_tokens", 0)),
        )

    def answer_fn(self, query: str) -> dict[str, Any]:
        """Run inference; returns prediction plus corrective-loop metadata and
        retrieved passages with metadata for the retrieval log."""
        result = self.run_inference(query)
        out = result.to_answer_fn_output()

        effective_query = result.rewritten_question if result.rewrite_count > 0 else query
        out["retrieved_memory"] = self._store.query_memory_with_metadata(effective_query)
        out["retrieved_documents"] = self._store.query_documents_with_metadata(effective_query)
        return out
