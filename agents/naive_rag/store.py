"""ChromaDB-backed StoreHooksProtocol with configurable chunking and retriever."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from tqdm import tqdm

load_dotenv()

from .embeddings import DEFAULT_EMBEDDING_MODEL, resolve_embeddings
from experiment.prepare_spec import PrepareSpec
from experiment.paths import DEFAULT_MEMORY_DIR, DEFAULT_DOCUMENT_CORPUS

from .indexing import (
    ChunkTuple,
    MemoryGranularity,
    corpus_doc_to_chunks,
    load_corpus_jsonl,
    load_persona_sessions,
    session_to_chunks,
)
from .gold_document_memory import (
    MemoryGoldSource,
    gold_document_memory_chunks,
    parse_memory_gold_source,
)

MEMORY_COLLECTION = "memory_sessions"
DOCUMENT_COLLECTION = "document_corpus"


def _passage_meta(doc: Document, id_key: str, score: float | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "text": doc.page_content,
        id_key: doc.metadata.get(id_key),
        "score": score,
    }
    url = doc.metadata.get("url")
    if url and id_key != "url":
        entry["url"] = url
    return entry


class RetrieverType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class ChromaStoreHooks:
    """Implements ``StoreHooksProtocol`` with two in-memory Chroma collections.

    Supports configurable memory granularity and retriever type (dense / sparse / hybrid).
    """

    def __init__(
        self,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        memory_dir: Path | str = DEFAULT_MEMORY_DIR,
        corpus_path: Path | str = DEFAULT_DOCUMENT_CORPUS,
        memory_k: int = 10,
        document_k: int = 10,
        memory_granularity: MemoryGranularity = MemoryGranularity.PAIR,
        retriever_type: RetrieverType = RetrieverType.DENSE,
        hybrid_weights: tuple[float, float] = (0.5, 0.5),
        embeddings: Any | None = None,
        min_score: float | None = None,
        memory_gold_source: str | MemoryGoldSource = MemoryGoldSource.SESSION,
    ) -> None:
        self.embedding_model = embedding_model
        self.memory_dir = Path(memory_dir)
        self.corpus_path = Path(corpus_path)
        self.memory_k = memory_k
        self.document_k = document_k
        self.memory_granularity = memory_granularity
        self.retriever_type = retriever_type
        self.hybrid_weights = hybrid_weights
        self.min_score = min_score
        self.memory_gold_source = parse_memory_gold_source(memory_gold_source)

        self._embeddings = resolve_embeddings(
            embedding_model=embedding_model,
            embeddings=embeddings,
        )
        self._client = chromadb.Client()

        self._memory_store: Chroma | None = None
        self._doc_store: Chroma | None = None

        self._memory_bm25: BM25Retriever | None = None
        self._doc_bm25: BM25Retriever | None = None

        self._corpus_cache: list[dict[str, Any]] | None = None
        self._corpus_by_url: dict[str, dict[str, Any]] = {}

        # Per-question mutation tracking — stores lists of chunk tuples per parent entity
        self._removed_doc_chunks: dict[str, list[ChunkTuple]] = {}
        self._injected_chunk_ids: set[str] = set()
        self._stripped_chunks: dict[str, list[dict[str, Any]]] = {}

        self._init_document_store()

    # ------------------------------------------------------------------
    # Document store init
    # ------------------------------------------------------------------

    _EMBED_BATCH = 256

    def _add_texts_batched(
        self, store: Chroma, ids: list, texts: list, metas: list, desc: str
    ) -> None:
        """Add texts in batches with a progress bar over chunks embedded."""
        n = len(ids)
        with tqdm(total=n, desc=desc, unit="chunk", leave=False) as pbar:
            for start in range(0, n, self._EMBED_BATCH):
                end = start + self._EMBED_BATCH
                store.add_texts(
                    texts=texts[start:end],
                    metadatas=metas[start:end],
                    ids=ids[start:end],
                )
                pbar.update(min(self._EMBED_BATCH, n - start))

    def _init_document_store(self) -> None:
        if self._corpus_cache is None:
            self._corpus_cache = load_corpus_jsonl(self.corpus_path)
            for doc in self._corpus_cache:
                url = str(doc.get("url", "")).strip()
                if url:
                    self._corpus_by_url[url] = doc

        try:
            self._client.delete_collection(DOCUMENT_COLLECTION)
        except Exception:
            pass
        self._doc_store = Chroma(
            client=self._client,
            collection_name=DOCUMENT_COLLECTION,
            embedding_function=self._embeddings,
        )
        ids, texts, metas = [], [], []
        for doc in self._corpus_cache:
            for chunk_id, text, meta in corpus_doc_to_chunks(doc):
                if chunk_id:
                    ids.append(chunk_id)
                    texts.append(text)
                    metas.append(meta)
        if ids:
            self._add_texts_batched(self._doc_store, ids, texts, metas, "Embedding documents")

        if self.retriever_type in (RetrieverType.SPARSE, RetrieverType.HYBRID):
            self._doc_bm25 = self._build_bm25(texts, metas, ids)

    # ------------------------------------------------------------------
    # Memory rebuild
    # ------------------------------------------------------------------

    def rebuild_memory(self, persona_id: str) -> None:
        try:
            self._client.delete_collection(MEMORY_COLLECTION)
        except Exception:
            pass
        self._memory_store = Chroma(
            client=self._client,
            collection_name=MEMORY_COLLECTION,
            embedding_function=self._embeddings,
        )
        persona_json = self.memory_dir / f"{persona_id}.json"
        sessions = load_persona_sessions(persona_json)
        ids, texts, metas = [], [], []
        for sess in sessions:
            for chunk_id, text, meta in session_to_chunks(sess, self.memory_granularity):
                if chunk_id:
                    ids.append(chunk_id)
                    texts.append(text)
                    metas.append(meta)
        if ids:
            self._add_texts_batched(
                self._memory_store, ids, texts, metas, f"Embedding memory ({persona_id})"
            )

        if self.retriever_type in (RetrieverType.SPARSE, RetrieverType.HYBRID):
            self._memory_bm25 = self._build_bm25(texts, metas, ids)

        self._injected_chunk_ids.clear()
        self._stripped_chunks.clear()

    # ------------------------------------------------------------------
    # Prepare / restore for per-question store mutations
    # ------------------------------------------------------------------

    def prepare_stores_for_question(self, spec: PrepareSpec) -> None:
        self._removed_doc_chunks.clear()
        self._injected_chunk_ids.clear()
        self._stripped_chunks.clear()

        if spec.document_urls_to_exclude and self._doc_store is not None:
            self._exclude_doc_chunks(spec.document_urls_to_exclude)

        if spec.memory_session_ids_to_strip and self._memory_store is not None:
            self._strip_memory_chunks(spec.memory_session_ids_to_strip)

        if spec.memory_session_ids_to_ensure and self._memory_store is not None:
            self._ensure_memory_chunks(spec.memory_session_ids_to_ensure, spec)

        self._rebuild_bm25_if_needed()

    def _exclude_doc_chunks(self, urls: set[str]) -> None:
        """Remove all chunks belonging to the given document URLs."""
        coll = self._doc_store._collection
        for url in urls:
            result = coll.get(where={"url": url})
            if result and result.get("ids"):
                chunk_tuples: list[ChunkTuple] = []
                for i, cid in enumerate(result["ids"]):
                    text = result["documents"][i] if result.get("documents") else ""
                    meta = result["metadatas"][i] if result.get("metadatas") else {}
                    chunk_tuples.append((cid, text, meta))
                coll.delete(ids=result["ids"])
                self._removed_doc_chunks[url] = chunk_tuples

    def _strip_memory_chunks(self, session_ids: set[str]) -> None:
        """Remove all chunks belonging to the given session IDs, saving them for restore."""
        coll = self._memory_store._collection
        for sid in session_ids:
            result = coll.get(where={"session_id": sid})
            if result and result.get("ids"):
                saved: list[dict[str, Any]] = []
                for i, cid in enumerate(result["ids"]):
                    saved.append({
                        "id": cid,
                        "text": result["documents"][i] if result.get("documents") else "",
                        "meta": result["metadatas"][i] if result.get("metadatas") else {},
                    })
                coll.delete(ids=result["ids"])
                self._stripped_chunks[sid] = saved

    def _ensure_memory_chunks(self, session_ids: set[str], spec: PrepareSpec) -> None:
        """Ensure gold memory items exist; optionally swap sessions for gold documents."""
        if self.memory_gold_source is MemoryGoldSource.GOLD_DOCUMENT:
            for sid in session_ids:
                if gold_document_memory_chunks(sid, self._corpus_by_url):
                    self._strip_memory_chunks({sid})
                    self._inject_gold_document_chunks(sid)
                else:
                    result = self._memory_store._collection.get(where={"session_id": sid})
                    if not (result and result.get("ids")):
                        self._inject_session_chunks(sid, spec)
            return

        coll = self._memory_store._collection
        for sid in session_ids:
            result = coll.get(where={"session_id": sid})
            already_present = bool(result and result.get("ids"))
            if not already_present:
                self._inject_session_chunks(sid, spec)

    def _inject_gold_document_chunks(self, session_id: str) -> None:
        if self._memory_store is None:
            return
        for chunk_id, text, meta in gold_document_memory_chunks(
            session_id, self._corpus_by_url
        ):
            self._memory_store.add_texts(texts=[text], metadatas=[meta], ids=[chunk_id])
            self._injected_chunk_ids.add(chunk_id)

    def _inject_session_chunks(self, session_id: str, spec: PrepareSpec) -> None:
        """Find a session by ID in the persona JSON and inject its chunks."""
        persona_json = self.memory_dir / f"{spec.persona}.json"
        if not persona_json.exists():
            return
        all_sessions = load_persona_sessions(persona_json)
        by_id = {str(s.get("session_id", "")): s for s in all_sessions}
        sess = by_id.get(session_id)
        if not sess:
            return
        for chunk_id, text, meta in session_to_chunks(sess, self.memory_granularity):
            if chunk_id and self._memory_store is not None:
                self._memory_store.add_texts(texts=[text], metadatas=[meta], ids=[chunk_id])
                self._injected_chunk_ids.add(chunk_id)

    def restore_stores_after_question(self, spec: PrepareSpec) -> None:
        # Re-add removed document chunks
        if self._removed_doc_chunks and self._doc_store is not None:
            ids, texts, metas = [], [], []
            for chunk_list in self._removed_doc_chunks.values():
                for cid, text, meta in chunk_list:
                    ids.append(cid)
                    texts.append(text)
                    metas.append(meta)
            if ids:
                self._doc_store.add_texts(texts=texts, metadatas=metas, ids=ids)
            self._removed_doc_chunks.clear()

        # Remove injected chunks
        if self._injected_chunk_ids and self._memory_store is not None:
            self._memory_store.delete(ids=list(self._injected_chunk_ids))
            self._injected_chunk_ids.clear()

        # Re-add stripped memory chunks
        if self._stripped_chunks and self._memory_store is not None:
            ids, texts, metas = [], [], []
            for saved_list in self._stripped_chunks.values():
                for entry in saved_list:
                    ids.append(entry["id"])
                    texts.append(entry["text"])
                    metas.append(entry["meta"])
            if ids:
                self._memory_store.add_texts(texts=texts, metadatas=metas, ids=ids)
            self._stripped_chunks.clear()

        self._rebuild_bm25_if_needed()

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bm25(
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> BM25Retriever:
        docs = [
            Document(page_content=t, metadata={**m, "_id": i})
            for t, m, i in zip(texts, metadatas, ids)
        ]
        retriever = BM25Retriever.from_documents(docs)
        return retriever

    def _rebuild_bm25_from_collection(self, collection: Any) -> BM25Retriever | None:
        """Read all documents from a Chroma collection and build a BM25 index."""
        result = collection.get()
        if not result or not result.get("ids"):
            return None
        return self._build_bm25(
            result["documents"],
            result["metadatas"],
            result["ids"],
        )

    def _rebuild_bm25_if_needed(self) -> None:
        if self.retriever_type not in (RetrieverType.SPARSE, RetrieverType.HYBRID):
            return
        if self._memory_store is not None:
            self._memory_bm25 = self._rebuild_bm25_from_collection(
                self._memory_store._collection,
            )
        if self._doc_store is not None:
            self._doc_bm25 = self._rebuild_bm25_from_collection(
                self._doc_store._collection,
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _query_store(
        self,
        query: str,
        chroma_store: Chroma | None,
        bm25: BM25Retriever | None,
        k: int,
    ) -> list[str]:
        if chroma_store is None:
            return []

        if self.retriever_type == RetrieverType.DENSE:
            if self.min_score is not None:
                results = chroma_store.similarity_search_with_relevance_scores(query, k=k)
                return [doc.page_content for doc, score in results if score >= self.min_score]
            results = chroma_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]

        if self.retriever_type == RetrieverType.SPARSE:
            if bm25 is None:
                return []
            bm25.k = k
            results = bm25.invoke(query)
            return [doc.page_content for doc in results[:k]]

        # HYBRID
        if bm25 is None:
            if self.min_score is not None:
                results = chroma_store.similarity_search_with_relevance_scores(query, k=k)
                return [doc.page_content for doc, score in results if score >= self.min_score]
            results = chroma_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]

        dense_retriever = chroma_store.as_retriever(search_kwargs={"k": k})
        bm25.k = k
        ensemble = EnsembleRetriever(
            retrievers=[dense_retriever, bm25],
            weights=list(self.hybrid_weights),
        )
        results = ensemble.invoke(query)
        return [doc.page_content for doc in results[:k]]

    def query_memory(self, query: str, k: int | None = None) -> list[str]:
        return self._query_store(
            query, self._memory_store, self._memory_bm25, k or self.memory_k,
        )

    def query_documents(self, query: str, k: int | None = None) -> list[str]:
        return self._query_store(
            query, self._doc_store, self._doc_bm25, k or self.document_k,
        )

    # ------------------------------------------------------------------
    # Metadata-rich retrieval (for logging / analysis)
    # ------------------------------------------------------------------

    def _query_store_with_metadata(
        self,
        query: str,
        chroma_store: Chroma | None,
        bm25: BM25Retriever | None,
        k: int,
        id_key: str,
    ) -> list[dict[str, Any]]:
        """Retrieve with metadata. ``id_key`` is the metadata field to extract as source id."""
        if chroma_store is None:
            return []

        if self.retriever_type == RetrieverType.DENSE:
            results = chroma_store.similarity_search_with_relevance_scores(query, k=k)
            return [
                _passage_meta(doc, id_key, round(score, 4))
                for doc, score in results
            ]

        if self.retriever_type == RetrieverType.SPARSE:
            if bm25 is None:
                return []
            bm25.k = k
            results = bm25.invoke(query)
            return [_passage_meta(doc, id_key, None) for doc in results[:k]]

        # HYBRID
        if bm25 is None:
            results = chroma_store.similarity_search_with_relevance_scores(query, k=k)
            return [
                _passage_meta(doc, id_key, round(score, 4))
                for doc, score in results
            ]

        dense_retriever = chroma_store.as_retriever(search_kwargs={"k": k})
        bm25.k = k
        ensemble = EnsembleRetriever(
            retrievers=[dense_retriever, bm25],
            weights=list(self.hybrid_weights),
        )
        results = ensemble.invoke(query)
        return [_passage_meta(doc, id_key, None) for doc in results[:k]]

    def query_memory_with_metadata(
        self, query: str, k: int | None = None
    ) -> list[dict[str, Any]]:
        return self._query_store_with_metadata(
            query, self._memory_store, self._memory_bm25, k or self.memory_k, "session_id",
        )

    def query_documents_with_metadata(
        self, query: str, k: int | None = None
    ) -> list[dict[str, Any]]:
        return self._query_store_with_metadata(
            query, self._doc_store, self._doc_bm25, k or self.document_k, "url",
        )

    # ------------------------------------------------------------------
    # Scored retrieval with exclusion (for corrective RAG accumulation)
    # ------------------------------------------------------------------

    def _query_store_scored(
        self,
        query: str,
        chroma_store: Chroma | None,
        k: int,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        """Retrieve (chunk_id, text, score) tuples with min_score filtering and ID exclusion.

        Over-fetches by len(exclude_ids) to compensate for post-retrieval ID filtering,
        then trims to requested k. Only supports dense retrieval.
        """
        if chroma_store is None or k <= 0:
            return []

        fetch_k = k + (len(exclude_ids) if exclude_ids else 0)
        results = chroma_store.similarity_search_with_relevance_scores(query, k=fetch_k)

        out: list[tuple[str, str, float]] = []
        for doc, score in results:
            chunk_id = doc.id if doc.id else doc.metadata.get("chunk_id", "")
            if exclude_ids and chunk_id in exclude_ids:
                continue
            if self.min_score is not None and score < self.min_score:
                continue
            out.append((chunk_id, doc.page_content, score))
            if len(out) >= k:
                break
        return out

    def query_memory_scored(
        self, query: str, k: int | None = None, exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        """Return (chunk_id, text, score) tuples from memory, excluding specified IDs."""
        effective_k = self.memory_k if k is None else k
        return self._query_store_scored(
            query, self._memory_store, effective_k, exclude_ids,
        )

    def query_documents_scored(
        self, query: str, k: int | None = None, exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        """Return (chunk_id, text, score) tuples from documents, excluding specified IDs."""
        effective_k = self.document_k if k is None else k
        return self._query_store_scored(
            query, self._doc_store, effective_k, exclude_ids,
        )
