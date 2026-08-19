"""Unified ChromaDB store — memory sessions and documents in a single collection.

This represents a baseline "single vector store" approach where the agent
does not distinguish between conversational memory and document corpus at
retrieval time. Both sources compete in the same embedding space.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
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
from .store import RetrieverType
from .gold_document_memory import (
    MemoryGoldSource,
    gold_document_memory_chunks,
    parse_memory_gold_source,
)

UNIFIED_COLLECTION = "unified"


class UnifiedChromaStoreHooks:
    """Single-collection store: memory + documents indexed together.

    Implements the same interface as ChromaStoreHooks (StoreHooksProtocol +
    query_memory / query_documents) so NaiveRAGAgent can use it unchanged.
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
        self.memory_gold_source = parse_memory_gold_source(memory_gold_source)

        self._embeddings = resolve_embeddings(
            embedding_model=embedding_model,
            embeddings=embeddings,
        )
        self._client = chromadb.Client()

        self._store: Chroma | None = None
        self._bm25: BM25Retriever | None = None

        self._corpus_cache: list[dict[str, Any]] | None = None
        self._corpus_by_url: dict[str, dict[str, Any]] = {}
        self._doc_chunks_cache: list[ChunkTuple] | None = None

        # Per-question mutation tracking
        self._removed_doc_chunks: dict[str, list[ChunkTuple]] = {}
        self._injected_chunk_ids: set[str] = set()
        self._stripped_chunks: dict[str, list[dict[str, Any]]] = {}

        self._load_corpus()

    # ------------------------------------------------------------------
    # Corpus loading (cached, parsed once)
    # ------------------------------------------------------------------

    def _load_corpus(self) -> None:
        if self._corpus_cache is None:
            self._corpus_cache = load_corpus_jsonl(self.corpus_path)
            for doc in self._corpus_cache:
                url = str(doc.get("url", "")).strip()
                if url:
                    self._corpus_by_url[url] = doc

        if self._doc_chunks_cache is None:
            self._doc_chunks_cache = []
            for doc in self._corpus_cache:
                for chunk_id, text, meta in corpus_doc_to_chunks(doc):
                    if chunk_id:
                        meta["source_type"] = "document"
                        self._doc_chunks_cache.append((chunk_id, text, meta))

    # ------------------------------------------------------------------
    # Rebuild: re-create the unified collection with docs + persona memory
    # ------------------------------------------------------------------

    def rebuild_memory(self, persona_id: str) -> None:
        try:
            self._client.delete_collection(UNIFIED_COLLECTION)
        except Exception:
            pass

        self._store = Chroma(
            client=self._client,
            collection_name=UNIFIED_COLLECTION,
            embedding_function=self._embeddings,
        )

        ids, texts, metas = [], [], []

        # Add document chunks
        for chunk_id, text, meta in self._doc_chunks_cache:
            ids.append(chunk_id)
            texts.append(text)
            metas.append(meta)

        # Add memory session chunks
        persona_json = self.memory_dir / f"{persona_id}.json"
        sessions = load_persona_sessions(persona_json)
        for sess in sessions:
            for chunk_id, text, meta in session_to_chunks(sess, self.memory_granularity):
                if chunk_id:
                    meta["source_type"] = "memory"
                    ids.append(chunk_id)
                    texts.append(text)
                    metas.append(meta)

        if ids:
            batch = 256
            n = len(ids)
            with tqdm(total=n, desc=f"Embedding unified ({persona_id})", unit="chunk", leave=False) as pbar:
                for start in range(0, n, batch):
                    end = start + batch
                    self._store.add_texts(
                        texts=texts[start:end],
                        metadatas=metas[start:end],
                        ids=ids[start:end],
                    )
                    pbar.update(min(batch, n - start))

        if self.retriever_type in (RetrieverType.SPARSE, RetrieverType.HYBRID):
            self._bm25 = self._build_bm25(texts, metas, ids)

        self._injected_chunk_ids.clear()
        self._stripped_chunks.clear()

    # ------------------------------------------------------------------
    # Prepare / restore per-question mutations
    # ------------------------------------------------------------------

    def prepare_stores_for_question(self, spec: PrepareSpec) -> None:
        self._removed_doc_chunks.clear()
        self._injected_chunk_ids.clear()
        self._stripped_chunks.clear()

        if spec.document_urls_to_exclude and self._store is not None:
            self._exclude_doc_chunks(spec.document_urls_to_exclude)

        if spec.memory_session_ids_to_strip and self._store is not None:
            self._strip_memory_chunks(spec.memory_session_ids_to_strip)

        if spec.memory_session_ids_to_ensure and self._store is not None:
            self._ensure_memory_chunks(spec.memory_session_ids_to_ensure, spec)

        self._rebuild_bm25_if_needed()

    def _exclude_doc_chunks(self, urls: set[str]) -> None:
        coll = self._store._collection
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
        coll = self._store._collection
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
        if self.memory_gold_source is MemoryGoldSource.GOLD_DOCUMENT:
            for sid in session_ids:
                if gold_document_memory_chunks(
                    sid,
                    self._corpus_by_url,
                    extra_meta={"source_type": "memory"},
                ):
                    self._strip_memory_chunks({sid})
                    self._inject_gold_document_chunks(sid)
                else:
                    result = self._store._collection.get(where={"session_id": sid})
                    if not (result and result.get("ids")):
                        self._inject_session_chunks(sid, spec)
            return

        coll = self._store._collection
        for sid in session_ids:
            result = coll.get(where={"session_id": sid})
            already_present = bool(result and result.get("ids"))
            if not already_present:
                self._inject_session_chunks(sid, spec)

    def _inject_gold_document_chunks(self, session_id: str) -> None:
        if self._store is None:
            return
        for chunk_id, text, meta in gold_document_memory_chunks(
            session_id,
            self._corpus_by_url,
            extra_meta={"source_type": "memory"},
        ):
            self._store.add_texts(texts=[text], metadatas=[meta], ids=[chunk_id])
            self._injected_chunk_ids.add(chunk_id)

    def _inject_session_chunks(self, session_id: str, spec: PrepareSpec) -> None:
        persona_json = self.memory_dir / f"{spec.eval_persona}.json"
        if not persona_json.exists():
            return
        all_sessions = load_persona_sessions(persona_json)
        by_id = {str(s.get("session_id", "")): s for s in all_sessions}
        sess = by_id.get(session_id)
        if not sess:
            return
        for chunk_id, text, meta in session_to_chunks(sess, self.memory_granularity):
            if chunk_id and self._store is not None:
                meta["source_type"] = "memory"
                self._store.add_texts(texts=[text], metadatas=[meta], ids=[chunk_id])
                self._injected_chunk_ids.add(chunk_id)

    def restore_stores_after_question(self, spec: PrepareSpec) -> None:
        if self._removed_doc_chunks and self._store is not None:
            ids, texts, metas = [], [], []
            for chunk_list in self._removed_doc_chunks.values():
                for cid, text, meta in chunk_list:
                    ids.append(cid)
                    texts.append(text)
                    metas.append(meta)
            if ids:
                self._store.add_texts(texts=texts, metadatas=metas, ids=ids)
            self._removed_doc_chunks.clear()

        if self._injected_chunk_ids and self._store is not None:
            self._store.delete(ids=list(self._injected_chunk_ids))
            self._injected_chunk_ids.clear()

        if self._stripped_chunks and self._store is not None:
            ids, texts, metas = [], [], []
            for saved_list in self._stripped_chunks.values():
                for entry in saved_list:
                    ids.append(entry["id"])
                    texts.append(entry["text"])
                    metas.append(entry["meta"])
            if ids:
                self._store.add_texts(texts=texts, metadatas=metas, ids=ids)
            self._stripped_chunks.clear()

        self._rebuild_bm25_if_needed()

    # ------------------------------------------------------------------
    # BM25
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
        return BM25Retriever.from_documents(docs)

    def _rebuild_bm25_if_needed(self) -> None:
        if self.retriever_type not in (RetrieverType.SPARSE, RetrieverType.HYBRID):
            return
        if self._store is not None:
            result = self._store._collection.get()
            if result and result.get("ids"):
                self._bm25 = self._build_bm25(
                    result["documents"], result["metadatas"], result["ids"]
                )

    # ------------------------------------------------------------------
    # Retrieval — both methods query the same unified collection
    # ------------------------------------------------------------------

    def _query_unified(self, query: str, k: int) -> list[str]:
        if self._store is None:
            return []

        if self.retriever_type == RetrieverType.DENSE:
            results = self._store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]

        if self.retriever_type == RetrieverType.SPARSE:
            if self._bm25 is None:
                return []
            self._bm25.k = k
            results = self._bm25.invoke(query)
            return [doc.page_content for doc in results[:k]]

        # HYBRID
        if self._bm25 is None:
            results = self._store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]

        dense_retriever = self._store.as_retriever(search_kwargs={"k": k})
        self._bm25.k = k
        ensemble = EnsembleRetriever(
            retrievers=[dense_retriever, self._bm25],
            weights=list(self.hybrid_weights),
        )
        results = ensemble.invoke(query)
        return [doc.page_content for doc in results[:k]]

    def query_memory(self, query: str, k: int | None = None) -> list[str]:
        return self._query_unified(query, k or self.memory_k)

    def query_documents(self, query: str, k: int | None = None) -> list[str]:
        return self._query_unified(query, k or self.document_k)

    # ------------------------------------------------------------------
    # Metadata-rich retrieval (for logging / analysis)
    # ------------------------------------------------------------------

    def _query_unified_with_metadata(
        self, query: str, k: int
    ) -> list[dict[str, Any]]:
        if self._store is None:
            return []

        if self.retriever_type == RetrieverType.DENSE:
            results = self._store.similarity_search_with_relevance_scores(query, k=k)
            out = []
            for doc, score in results:
                source_type = doc.metadata.get("source_type", "")
                entry: dict[str, Any] = {
                    "text": doc.page_content,
                    "score": round(score, 4),
                    "source_type": source_type,
                }
                if source_type == "memory":
                    entry["session_id"] = doc.metadata.get("session_id")
                    if doc.metadata.get("url"):
                        entry["url"] = doc.metadata.get("url")
                else:
                    entry["url"] = doc.metadata.get("url")
                out.append(entry)
            return out

        if self.retriever_type == RetrieverType.SPARSE:
            if self._bm25 is None:
                return []
            self._bm25.k = k
            results = self._bm25.invoke(query)
            out = []
            for doc in results[:k]:
                source_type = doc.metadata.get("source_type", "")
                entry: dict[str, Any] = {
                    "text": doc.page_content,
                    "score": None,
                    "source_type": source_type,
                }
                if source_type == "memory":
                    entry["session_id"] = doc.metadata.get("session_id")
                    if doc.metadata.get("url"):
                        entry["url"] = doc.metadata.get("url")
                else:
                    entry["url"] = doc.metadata.get("url")
                out.append(entry)
            return out

        # HYBRID
        if self._bm25 is None:
            results = self._store.similarity_search_with_relevance_scores(query, k=k)
            out = []
            for doc, score in results:
                source_type = doc.metadata.get("source_type", "")
                entry: dict[str, Any] = {
                    "text": doc.page_content,
                    "score": round(score, 4),
                    "source_type": source_type,
                }
                if source_type == "memory":
                    entry["session_id"] = doc.metadata.get("session_id")
                    if doc.metadata.get("url"):
                        entry["url"] = doc.metadata.get("url")
                else:
                    entry["url"] = doc.metadata.get("url")
                out.append(entry)
            return out

        dense_retriever = self._store.as_retriever(search_kwargs={"k": k})
        self._bm25.k = k
        ensemble = EnsembleRetriever(
            retrievers=[dense_retriever, self._bm25],
            weights=list(self.hybrid_weights),
        )
        results = ensemble.invoke(query)
        out = []
        for doc in results[:k]:
            source_type = doc.metadata.get("source_type", "")
            entry: dict[str, Any] = {
                "text": doc.page_content,
                "score": None,
                "source_type": source_type,
            }
            if source_type == "memory":
                entry["session_id"] = doc.metadata.get("session_id")
                if doc.metadata.get("url"):
                    entry["url"] = doc.metadata.get("url")
            else:
                entry["url"] = doc.metadata.get("url")
            out.append(entry)
        return out

    def query_memory_with_metadata(
        self, query: str, k: int | None = None
    ) -> list[dict[str, Any]]:
        results = self._query_unified_with_metadata(query, k or self.memory_k)
        return [r for r in results if r.get("source_type") == "memory"]

    def query_documents_with_metadata(
        self, query: str, k: int | None = None
    ) -> list[dict[str, Any]]:
        results = self._query_unified_with_metadata(query, k or self.document_k)
        return [r for r in results if r.get("source_type") != "memory"]
