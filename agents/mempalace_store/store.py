"""MemPalace-backed StoreHooksProtocol — uses mempalace's own API for memory,
delegates document operations to ChromaStoreHooks.

Memory ingestion goes through mempalace's ``palace.get_collection()`` with
closet indexing via ``build_closet_lines()`` / ``upsert_closet_lines()``.
Memory search uses ``searcher.search_memories()`` for the full hybrid pipeline
(vector + BM25 re-rank + closet boost + drawer-grep enrichment).

Documents are handled entirely by the existing ``ChromaStoreHooks`` — no
document logic is reimplemented here.

Both memory and documents use ``all-MiniLM-L6-v2`` (MemPalace's default,
also ChromaDB's built-in ONNX model).  No OpenAI embedding calls needed.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from mempalace.palace import (
    build_closet_lines,
    get_closets_collection,
    get_collection,
    purge_file_closets,
    upsert_closet_lines,
)
from mempalace.searcher import search_memories

from experiment.prepare_spec import PrepareSpec
from experiment.paths import DEFAULT_DOCUMENT_CORPUS, DEFAULT_MEMORY_DIR

from tqdm import tqdm

from agents.naive_rag.indexing import (
    MemoryGranularity,
    load_persona_sessions,
    session_to_chunks,
)
from agents.naive_rag.embeddings import ChromaDefaultEmbeddings
from agents.naive_rag.store import ChromaStoreHooks, RetrieverType


class MemPalaceStoreHooks:
    """Implements ``StoreHooksProtocol`` with MemPalace for memory and
    ``ChromaStoreHooks`` for documents.

    The palace is created in a temp directory and rebuilt from scratch on each
    ``rebuild_memory()`` call.  Search goes through ``search_memories()`` so we
    get MemPalace's full hybrid pipeline (drawer vector search -> closet topic
    boost -> BM25 re-rank -> drawer-grep enrichment).

    Document operations are delegated entirely to an internal
    ``ChromaStoreHooks`` instance, avoiding any code duplication.
    """

    def __init__(
        self,
        *,
        memory_dir: Path | str = DEFAULT_MEMORY_DIR,
        corpus_path: Path | str = DEFAULT_DOCUMENT_CORPUS,
        memory_k: int = 10,
        document_k: int = 10,
        memory_granularity: MemoryGranularity = MemoryGranularity.PAIR,
        retriever_type: RetrieverType = RetrieverType.DENSE,
        hybrid_weights: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.corpus_path = Path(corpus_path)
        self.memory_k = memory_k
        self.document_k = document_k
        self.memory_granularity = memory_granularity

        self._palace_path: str | None = None
        self._palace_tmp_root: str | None = None

        # Python-side operational index (invisible to MemPalace search).
        # Maps session_id -> list of chunk IDs stored in the palace drawers.
        self._sid_to_chunks: dict[str, list[str]] = defaultdict(list)
        # chunk_id -> (text, palace_metadata) for each ingested chunk.
        self._chunk_data: dict[str, tuple[str, dict[str, Any]]] = {}
        # session_id -> source_file path used in palace metadata.
        self._sid_to_source_file: dict[str, str] = {}

        # Per-question mutation caches for memory
        self._injected_chunk_ids: set[str] = set()
        self._stripped_chunks: dict[str, list[dict[str, Any]]] = {}
        self._injected_source_files: set[str] = set()

        # Delegate all document operations to ChromaStoreHooks with the
        # same all-MiniLM-L6-v2 embedding that MemPalace uses for drawers.
        self._doc_hooks = ChromaStoreHooks(
            embeddings=ChromaDefaultEmbeddings(),
            memory_dir=self.memory_dir,
            corpus_path=self.corpus_path,
            memory_k=memory_k,
            document_k=document_k,
            memory_granularity=memory_granularity,
            retriever_type=retriever_type,
            hybrid_weights=hybrid_weights,
        )

    # ------------------------------------------------------------------
    # Memory rebuild — create a fresh MemPalace palace
    # ------------------------------------------------------------------

    def rebuild_memory(self, persona_id: str) -> None:
        if self._palace_tmp_root is not None:
            shutil.rmtree(self._palace_tmp_root, ignore_errors=True)

        self._palace_tmp_root = tempfile.mkdtemp(prefix="mempalace_")
        self._palace_path = self._palace_tmp_root
        self._sid_to_chunks.clear()
        self._chunk_data.clear()
        self._sid_to_source_file.clear()
        self._injected_chunk_ids.clear()
        self._stripped_chunks.clear()
        self._injected_source_files.clear()

        drawers_col = get_collection(self._palace_path, create=True)
        closets_col = get_closets_collection(self._palace_path, create=True)

        persona_json = self.memory_dir / f"{persona_id}.json"
        sessions = load_persona_sessions(persona_json)

        all_ids: list[str] = []
        all_texts: list[str] = []
        all_metas: list[dict[str, Any]] = []

        for sess_idx, sess in enumerate(sessions):
            sid = str(sess.get("session_id", ""))
            date = str(sess.get("date", ""))
            source_file = f"persona_{persona_id}/sess_{sess_idx:06d}.jsonl"
            self._sid_to_source_file[sid] = source_file

            for chunk_id, text, full_meta in session_to_chunks(
                sess, self.memory_granularity
            ):
                if not chunk_id:
                    continue

                # Only date is visible to MemPalace search. Operational
                # metadata (session_id, source, evidence_id) lives
                # exclusively in _sid_to_chunks / _chunk_data.
                palace_meta = {
                    "wing": persona_id,
                    "room": "general",
                    "source_file": source_file,
                    "chunk_index": full_meta.get("chunk_index", 0),
                    "date": date,
                }

                all_ids.append(chunk_id)
                all_texts.append(text)
                all_metas.append(palace_meta)
                self._sid_to_chunks[sid].append(chunk_id)
                self._chunk_data[chunk_id] = (text, palace_meta)

            # Build closet lines per session for topic-pointer search boost
            session_chunk_ids = self._sid_to_chunks.get(sid, [])
            if session_chunk_ids:
                combined_text = "\n\n".join(
                    self._chunk_data[cid][0] for cid in session_chunk_ids
                )
                closet_lines = build_closet_lines(
                    source_file, session_chunk_ids[:3], combined_text,
                    persona_id, "general",
                )
                closet_base = hashlib.sha256(
                    source_file.encode()
                ).hexdigest()[:12]
                upsert_closet_lines(
                    closets_col, closet_base, closet_lines,
                    {"wing": persona_id, "room": "general",
                     "source_file": source_file},
                )

        if all_ids:
            batch = 256
            n = len(all_ids)
            with tqdm(total=n, desc=f"Embedding memory ({persona_id})", unit="chunk", leave=False) as pbar:
                for start in range(0, n, batch):
                    end = start + batch
                    drawers_col.add(
                        documents=all_texts[start:end],
                        ids=all_ids[start:end],
                        metadatas=all_metas[start:end],
                    )
                    pbar.update(min(batch, n - start))

    # ------------------------------------------------------------------
    # Prepare / restore — memory via palace, documents via ChromaStoreHooks
    # ------------------------------------------------------------------

    def prepare_stores_for_question(self, spec: PrepareSpec) -> None:
        self._injected_chunk_ids.clear()
        self._injected_source_files.clear()
        self._stripped_chunks.clear()

        # Document mutations are delegated entirely
        self._doc_hooks._removed_doc_chunks.clear()
        if spec.document_urls_to_exclude and self._doc_hooks._doc_store is not None:
            self._doc_hooks._exclude_doc_chunks(spec.document_urls_to_exclude)

        # Memory mutations use the MemPalace drawers collection
        if spec.memory_session_ids_to_strip and self._palace_path is not None:
            self._strip_memory_chunks(spec.memory_session_ids_to_strip)

        if spec.memory_session_ids_to_ensure and self._palace_path is not None:
            self._ensure_memory_chunks(spec.memory_session_ids_to_ensure, spec)

    def _strip_memory_chunks(self, session_ids: set[str]) -> None:
        drawers_col = get_collection(self._palace_path, create=False)
        closets_col = get_closets_collection(self._palace_path, create=False)
        for sid in session_ids:
            chunk_ids = self._sid_to_chunks.get(sid, [])
            if not chunk_ids:
                continue
            saved: list[dict[str, Any]] = []
            for cid in chunk_ids:
                if cid in self._chunk_data:
                    text, meta = self._chunk_data[cid]
                    saved.append({"id": cid, "text": text, "meta": meta})
            if saved:
                drawers_col.delete(ids=chunk_ids)
                source_file = self._sid_to_source_file.get(sid)
                if source_file:
                    purge_file_closets(closets_col, source_file)
                self._stripped_chunks[sid] = saved

    def _ensure_memory_chunks(
        self, session_ids: set[str], spec: PrepareSpec
    ) -> None:
        for sid in session_ids:
            if self._sid_to_chunks.get(sid):
                continue
            self._inject_session_chunks(sid, spec)

    def _inject_session_chunks(
        self, session_id: str, spec: PrepareSpec
    ) -> None:
        persona_json = self.memory_dir / f"{spec.persona}.json"
        if not persona_json.exists():
            return
        all_sessions = load_persona_sessions(persona_json)
        by_id = {str(s.get("session_id", "")): s for s in all_sessions}
        sess = by_id.get(session_id)
        if not sess:
            return

        drawers_col = get_collection(self._palace_path, create=False)
        date = str(sess.get("date", ""))
        source_file = f"injected/{session_id}.jsonl"

        injected_texts: list[str] = []
        for chunk_id, text, full_meta in session_to_chunks(
            sess, self.memory_granularity
        ):
            if not chunk_id:
                continue
            palace_meta = {
                "wing": spec.persona,
                "room": "general",
                "source_file": source_file,
                "chunk_index": full_meta.get("chunk_index", 0),
                "date": date,
            }
            drawers_col.add(
                documents=[text], ids=[chunk_id], metadatas=[palace_meta]
            )
            self._sid_to_chunks[session_id].append(chunk_id)
            self._chunk_data[chunk_id] = (text, palace_meta)
            self._injected_chunk_ids.add(chunk_id)
            injected_texts.append(text)

        # Build closet lines so injected memories get the full hybrid
        # search treatment (closet boost + drawer-grep enrichment).
        chunk_ids = self._sid_to_chunks.get(session_id, [])
        if chunk_ids and injected_texts:
            combined_text = "\n\n".join(injected_texts)
            closet_lines = build_closet_lines(
                source_file, chunk_ids[:3], combined_text,
                spec.persona, "general",
            )
            closet_base = hashlib.sha256(
                source_file.encode()
            ).hexdigest()[:12]
            closets_col = get_closets_collection(self._palace_path, create=False)
            upsert_closet_lines(
                closets_col, closet_base, closet_lines,
                {"wing": spec.persona, "room": "general",
                 "source_file": source_file},
            )
            self._injected_source_files.add(source_file)

    def restore_stores_after_question(self, spec: PrepareSpec) -> None:
        # Restore document chunks via the delegate
        if self._doc_hooks._removed_doc_chunks and self._doc_hooks._doc_store is not None:
            ids, texts, metas = [], [], []
            for chunk_list in self._doc_hooks._removed_doc_chunks.values():
                for cid, text, meta in chunk_list:
                    ids.append(cid)
                    texts.append(text)
                    metas.append(meta)
            if ids:
                self._doc_hooks._doc_store.add_texts(
                    texts=texts, metadatas=metas, ids=ids,
                )
            self._doc_hooks._removed_doc_chunks.clear()

        if self._palace_path is None:
            return

        drawers_col = get_collection(self._palace_path, create=False)
        closets_col = get_closets_collection(self._palace_path, create=False)

        # Remove injected memory chunks and their closets
        if self._injected_chunk_ids:
            inject_list = list(self._injected_chunk_ids)
            drawers_col.delete(ids=inject_list)
            for source_file in self._injected_source_files:
                purge_file_closets(closets_col, source_file)
            for cid in inject_list:
                self._chunk_data.pop(cid, None)
            for sid in list(self._sid_to_chunks):
                self._sid_to_chunks[sid] = [
                    c for c in self._sid_to_chunks[sid]
                    if c not in self._injected_chunk_ids
                ]
                if not self._sid_to_chunks[sid]:
                    del self._sid_to_chunks[sid]
            self._injected_chunk_ids.clear()
            self._injected_source_files.clear()

        # Re-add stripped memory chunks and rebuild their closets
        if self._stripped_chunks:
            ids, texts, metas = [], [], []
            for saved_list in self._stripped_chunks.values():
                for entry in saved_list:
                    ids.append(entry["id"])
                    texts.append(entry["text"])
                    metas.append(entry["meta"])
            if ids:
                drawers_col.add(documents=texts, ids=ids, metadatas=metas)

            # Rebuild closets for each restored session
            for sid, saved_list in self._stripped_chunks.items():
                source_file = self._sid_to_source_file.get(sid)
                if not source_file or not saved_list:
                    continue
                chunk_ids = [entry["id"] for entry in saved_list]
                combined_text = "\n\n".join(entry["text"] for entry in saved_list)
                wing = saved_list[0]["meta"].get("wing", "")
                closet_lines = build_closet_lines(
                    source_file, chunk_ids[:3], combined_text,
                    wing, "general",
                )
                closet_base = hashlib.sha256(
                    source_file.encode()
                ).hexdigest()[:12]
                upsert_closet_lines(
                    closets_col, closet_base, closet_lines,
                    {"wing": wing, "room": "general",
                     "source_file": source_file},
                )
            self._stripped_chunks.clear()

    # ------------------------------------------------------------------
    # Retrieval — memory via MemPalace, documents via ChromaStoreHooks
    # ------------------------------------------------------------------

    # Cosine distance beyond which results are filtered before ranking.
    # 1.5 aligns with MemPalace's internal CLOSET_DISTANCE_CAP — results
    # past this threshold have strongly negative cosine similarity and
    # contribute zero vector signal to the hybrid re-ranker.
    _MAX_DISTANCE = 1.5

    def query_memory(self, query: str, k: int | None = None) -> list[str]:
        if self._palace_path is None:
            return []

        n = k or self.memory_k
        result = search_memories(
            query=query,
            palace_path=self._palace_path,
            n_results=n,
            max_distance=self._MAX_DISTANCE,
        )
        if "error" in result:
            return []

        hits = result.get("results", [])
        return [h["text"] for h in hits if h.get("text")]

    def query_documents(self, query: str, k: int | None = None) -> list[str]:
        return self._doc_hooks.query_documents(query, k)

    # ------------------------------------------------------------------
    # Metadata-rich retrieval (for logging / analysis)
    # ------------------------------------------------------------------

    def _source_basename_to_sid(self) -> dict[str, str]:
        """Build reverse map: source_file basename -> session_id."""
        reverse: dict[str, str] = {}
        for sid, source_file in self._sid_to_source_file.items():
            basename = Path(source_file).name
            reverse[basename] = sid
        return reverse

    def query_memory_with_metadata(
        self, query: str, k: int | None = None
    ) -> list[dict[str, Any]]:
        if self._palace_path is None:
            return []

        n = k or self.memory_k
        result = search_memories(
            query=query,
            palace_path=self._palace_path,
            n_results=n,
            max_distance=self._MAX_DISTANCE,
        )
        if "error" in result:
            return []

        basename_to_sid = self._source_basename_to_sid()
        hits = result.get("results", [])
        out: list[dict[str, Any]] = []
        for h in hits:
            if not h.get("text"):
                continue
            source_basename = h.get("source_file", "")
            session_id = basename_to_sid.get(source_basename)
            out.append({
                "text": h["text"],
                "session_id": session_id,
                "score": h.get("similarity"),
            })
        return out

    def query_documents_with_metadata(
        self, query: str, k: int | None = None
    ) -> list[dict[str, Any]]:
        return self._doc_hooks.query_documents_with_metadata(query, k)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        if self._palace_tmp_root is not None:
            shutil.rmtree(self._palace_tmp_root, ignore_errors=True)
