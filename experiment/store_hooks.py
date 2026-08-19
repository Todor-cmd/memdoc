from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .prepare_spec import PrepareSpec


@runtime_checkable
class StoreHooksProtocol(Protocol):
    """Plugs in later: vector DB / file-backed indices. ``answer`` only receives ``query``."""

    def rebuild_memory(self, persona_id: str) -> None:
        """Reset memory store to static background for ``persona_id`` (e.g. load ``persona_k.json``)."""
        ...

    def prepare_stores_for_question(self, spec: PrepareSpec) -> None:
        """Apply variant-specific patches for this question (docs + memory)."""
        ...

    def restore_stores_after_question(self, spec: PrepareSpec) -> None:
        """Undo ``prepare_stores_for_question`` for this ``spec``."""
        ...

    def query_memory_with_metadata(
        self, query: str, k: int | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve memory passages with source identifiers and scores.

        Each dict has keys: ``text``, ``session_id`` (str | None), ``score`` (float | None).
        """
        ...

    def query_documents_with_metadata(
        self, query: str, k: int | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve document passages with source identifiers and scores.

        Each dict has keys: ``text``, ``url`` (str | None), ``score`` (float | None).
        """
        ...


class NoOpStoreHooks:
    """Default hooks that perform no I/O (useful for dry runs and tests)."""

    def __init__(self) -> None:
        self.current_persona: str | None = None
        self.prepare_calls: list[PrepareSpec] = []
        self.restore_calls: list[PrepareSpec] = []
        self.rebuild_calls: list[str] = []

    def rebuild_memory(self, persona_id: str) -> None:
        self.current_persona = persona_id
        self.rebuild_calls.append(persona_id)

    def prepare_stores_for_question(self, spec: PrepareSpec) -> None:
        self.prepare_calls.append(spec)

    def restore_stores_after_question(self, spec: PrepareSpec) -> None:
        self.restore_calls.append(spec)

    def query_memory_with_metadata(
        self, query: str, k: int | None = None
    ) -> list[dict[str, Any]]:
        return []

    def query_documents_with_metadata(
        self, query: str, k: int | None = None
    ) -> list[dict[str, Any]]:
        return []
