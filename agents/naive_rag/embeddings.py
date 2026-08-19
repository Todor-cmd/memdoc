"""Shared embedding helpers for retrieval agents."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ChromaDefaultEmbeddings(Embeddings):
    """Wraps ChromaDB's built-in ONNXMiniLM_L6_V2 as a LangChain Embeddings.

    Uses ``all-MiniLM-L6-v2`` locally (no API calls). This is the default across
    all experiment agents so retrieval comparisons are not confounded by embedding
    model choice.
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._ef = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._ef([text])[0]


def resolve_embeddings(
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embeddings: Embeddings | None = None,
) -> Embeddings:
    """Return an embeddings instance for the given model name."""
    if embeddings is not None:
        return embeddings
    if embedding_model == DEFAULT_EMBEDDING_MODEL:
        return ChromaDefaultEmbeddings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=embedding_model)
