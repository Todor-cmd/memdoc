from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .evidence import (
    evidence_dicts_union_for_question,
    integrated_document_evidence,
    integrated_memory_evidence,
    urls_from_evidence_dicts,
)
from .session_ids import evidence_session_id
from .variants import DatasetVariant


@dataclass(frozen=True)
class PrepareSpec:
    """Pure description of store mutations for one question step.

    Variant is chosen by the runner / filename, not duplicated here.
    Store hooks interpret this; ``answer(query)`` receives only the query.
    """

    persona: str
    eval_persona: str
    question_idx: str
    query: str
    question_type: str | None
    # Document index: exclude these URLs for this question (empty = full corpus).
    document_urls_to_exclude: frozenset[str] = field(default_factory=frozenset)
    # Memory: these session ids must be present after prepare (inject if missing from background).
    memory_session_ids_to_ensure: frozenset[str] = field(default_factory=frozenset)
    # Memory: temporarily remove these session ids during inference if present.
    memory_session_ids_to_strip: frozenset[str] = field(default_factory=frozenset)


def _session_ids_for_evidence(
    query: str, items: list[dict[str, Any]], eval_persona: str | None = None
) -> frozenset[str]:
    out: set[str] = set()
    for ev in items:
        eid = ev.get("evidence_id")
        if eid is None or (isinstance(eid, float) and pd.isna(eid)):
            continue
        key = str(eid).strip()
        if key:
            out.add(evidence_session_id(query, key, persona_id=eval_persona))
    return frozenset(out)


def build_prepare_spec(
    variant: DatasetVariant,
    row: pd.Series,
    *,
    eval_persona: str | None = None,
    strict_integrated_memory_strip: bool = False,
) -> PrepareSpec:
    """Build a :class:`PrepareSpec` from one dataframe row.

    Parameters
    ----------
    variant : DatasetVariant
    row : pd.Series
        Must contain ``query`` and either ``persona`` or ``original_persona``.
    eval_persona : str | None
        The persona whose corpus is used for this evaluation. When None,
        defaults to the row's persona (legacy single-persona mode). In the
        cross-persona design, this is the persona whose memory is loaded.
    strict_integrated_memory_strip : bool
        Whether to strip document-channel evidence sessions from memory in
        the integrated variant.
    """
    query = str(row["query"]).strip()
    # original_persona: which domain the question belongs to
    original_persona = str(
        row.get("original_persona", row.get("persona", ""))
    ).strip()
    # eval_persona: which corpus to use (may differ from original_persona)
    ep = eval_persona if eval_persona else original_persona
    qidx = str(row.get("question_idx", row.get("question_id", ""))).strip()
    qtype = row.get("question_type")
    qt: str | None = None if qtype is None or (isinstance(qtype, float) and pd.isna(qtype)) else str(qtype)

    e_all = evidence_dicts_union_for_question(row)
    if qt == "null_query" or not e_all:
        return PrepareSpec(
            persona=original_persona,
            eval_persona=ep,
            question_idx=qidx,
            query=query,
            question_type=qt,
        )

    if variant is DatasetVariant.MEMORY_ONLY:
        sessions = _session_ids_for_evidence(query, e_all, eval_persona=ep)
        return PrepareSpec(
            persona=original_persona,
            eval_persona=ep,
            question_idx=qidx,
            query=query,
            question_type=qt,
            document_urls_to_exclude=urls_from_evidence_dicts(e_all),
            memory_session_ids_to_ensure=sessions,
        )

    if variant is DatasetVariant.DOCUMENT_ONLY:
        sessions = _session_ids_for_evidence(query, e_all, eval_persona=ep)
        return PrepareSpec(
            persona=original_persona,
            eval_persona=ep,
            question_idx=qidx,
            query=query,
            question_type=qt,
            memory_session_ids_to_strip=sessions,
        )

    if variant is DatasetVariant.INTEGRATED:
        e_mem = integrated_memory_evidence(row)
        e_doc = integrated_document_evidence(row)
        ensure = _session_ids_for_evidence(query, e_mem, eval_persona=ep)
        strip = _session_ids_for_evidence(query, e_doc, eval_persona=ep) if strict_integrated_memory_strip else frozenset()
        return PrepareSpec(
            persona=original_persona,
            eval_persona=ep,
            question_idx=qidx,
            query=query,
            question_type=qt,
            document_urls_to_exclude=urls_from_evidence_dicts(e_mem),
            memory_session_ids_to_ensure=ensure,
            memory_session_ids_to_strip=strip,
        )

    raise ValueError(f"Unknown variant: {variant!r}")
