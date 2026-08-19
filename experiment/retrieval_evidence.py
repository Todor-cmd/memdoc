"""Helpers for extracting golden evidence metadata per variant.

Used by the runner to enrich each inference row with the ground-truth
evidence identifiers needed for retrieval analysis (recall@K, etc.).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .evidence import (
    evidence_dicts_union_for_question,
    integrated_document_evidence,
    urls_from_evidence_dicts,
)
from .variants import DatasetVariant


def golden_evidence_for_variant(
    row: pd.Series,
    variant: DatasetVariant,
) -> dict[str, Any]:
    """Extract golden evidence identifiers appropriate for the given variant.

    Returns a dict with:
        golden_document_urls: list[str] — document URLs that contain the evidence
    """
    e_all = evidence_dicts_union_for_question(row)

    if variant is DatasetVariant.MEMORY_ONLY:
        return {"golden_document_urls": sorted(urls_from_evidence_dicts(e_all))}

    if variant is DatasetVariant.DOCUMENT_ONLY:
        return {"golden_document_urls": sorted(urls_from_evidence_dicts(e_all))}

    if variant is DatasetVariant.INTEGRATED:
        e_doc = integrated_document_evidence(row)
        return {"golden_document_urls": sorted(urls_from_evidence_dicts(e_doc))}

    return {"golden_document_urls": []}
