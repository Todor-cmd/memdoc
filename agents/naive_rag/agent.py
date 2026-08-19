"""NaiveRAGAgent: single-pass retrieval from memory + document stores, answered via Groq."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import backoff
import groq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from sampling_frame_agents.schemas import AgentAnswer
from experiment.paths import DEFAULT_MEMORY_DIR, DEFAULT_DOCUMENT_CORPUS

from agents.token_usage import extract_token_usage

from .embeddings import DEFAULT_EMBEDDING_MODEL
from .gold_document_memory import MemoryGoldSource, parse_memory_gold_source
from .indexing import MemoryGranularity
from .prompts import SYSTEM_PROMPT, build_user_prompt, format_retrieved_passages
from .store import ChromaStoreHooks, RetrieverType

StoreHooksType = ChromaStoreHooks  # will be unioned with MemPalaceStoreHooks lazily


def _groq_giveup_bad_request(exc: BaseException) -> bool:
    return isinstance(exc, groq.BadRequestError)


class NaiveRAGAgent:
    """Naive RAG agent: retrieve once from both stores, combine context, and generate.

    Provides ``hooks_factory`` and ``answer_fn`` for :func:`experiment.runner.run_experiment`.
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
        memory_gold_source: str | MemoryGoldSource = MemoryGoldSource.SESSION,
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
        self.memory_gold_source = parse_memory_gold_source(memory_gold_source)

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            seed=seed,
        ).with_structured_output(AgentAnswer, include_raw=True)

        self._store: Any = None

    def hooks_factory(self) -> Any:
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
                memory_gold_source=self.memory_gold_source,
            )
        elif self.store_backend == "unified":
            from agents.naive_rag.unified_store import UnifiedChromaStoreHooks

            self._store = UnifiedChromaStoreHooks(
                embedding_model=self.embedding_model,
                memory_dir=self.memory_dir,
                corpus_path=self.corpus_path,
                memory_k=self.memory_k,
                document_k=self.document_k,
                memory_granularity=self.memory_granularity,
                retriever_type=self.retriever_type,
                hybrid_weights=self.hybrid_weights,
                memory_gold_source=self.memory_gold_source,
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
                memory_gold_source=self.memory_gold_source,
            )
        return self._store

    def answer_fn(self, query: str) -> dict[str, Any]:
        """Retrieve from both stores, build prompt, call LLM.

        Returns a dict with ``prediction``, token counts, and retrieved passages
        with metadata for the retrieval log.
        """
        store = self._store
        if store is None:
            raise RuntimeError("hooks_factory() must be called before answer_fn()")

        mem_passages = store.query_memory(query)
        doc_passages = store.query_documents(query)

        passages_block = format_retrieved_passages(mem_passages, doc_passages)
        user_msg = build_user_prompt(query, passages_block)

        prediction, reasoning, (input_tokens, output_tokens, total_tokens) = (
            self._call_llm(user_msg)
        )

        mem_meta = store.query_memory_with_metadata(query)
        doc_meta = store.query_documents_with_metadata(query)

        return {
            "prediction": prediction,
            "reasoning": reasoning,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "retrieved_memory": mem_meta,
            "retrieved_documents": doc_meta,
        }

    @backoff.on_exception(
        backoff.expo,
        (groq.RateLimitError, groq.APIError),
        max_tries=5,
        giveup=_groq_giveup_bad_request,
    )
    def _call_llm(self, user_msg: str) -> tuple[str, str | None, tuple[int, int, int]]:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
        try:
            out = self.llm.invoke(messages)
            tokens = extract_token_usage(out)
            if out["parsing_error"] is not None:
                return "[INFERENCE_FAILED]", None, tokens
            parsed: AgentAnswer | None = out["parsed"]
            if parsed is None:
                return "[INFERENCE_FAILED]", None, tokens
            return parsed.final_answer, parsed.reasoning, tokens
        except Exception:
            return "[INFERENCE_FAILED]", None, (0, 0, 0)
