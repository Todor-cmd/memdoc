from __future__ import annotations

from enum import Enum


class DatasetVariant(str, Enum):
    MEMORY_ONLY = "memory_only"
    DOCUMENT_ONLY = "document_only"
    INTEGRATED = "integrated"
