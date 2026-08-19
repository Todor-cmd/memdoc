"""Compute ``is_in_domain`` for experiment result rows.

The label indicates whether gold evidence assigned to the **memory channel**
falls entirely within the eval persona's domain categories. ``document_only``
rows are always in-domain because no memory retrieval is required.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from session_generation.persona import PERSONA_DOMAINS

from .evidence import evidence_dicts_union_for_question, integrated_memory_evidence
from .variants import DatasetVariant


def _normalize_variant(variant: DatasetVariant | str) -> DatasetVariant:
    if isinstance(variant, DatasetVariant):
        return variant
    return DatasetVariant(str(variant).strip())


def _memory_evidence_categories(row: pd.Series, variant: DatasetVariant) -> set[str]:
    if variant is DatasetVariant.MEMORY_ONLY:
        evidence = evidence_dicts_union_for_question(row)
    elif variant is DatasetVariant.INTEGRATED:
        evidence = integrated_memory_evidence(row)
    else:
        return set()

    categories: set[str] = set()
    for ev in evidence:
        cat = ev.get("category")
        if cat and isinstance(cat, str) and cat.strip():
            categories.add(cat.strip().lower())
    return categories


def compute_is_in_domain(
    question_row: pd.Series,
    eval_persona: str,
    variant: DatasetVariant | str,
) -> bool:
    """Return whether memory-channel gold evidence is in-domain for *eval_persona*."""
    variant = _normalize_variant(variant)

    if variant is DatasetVariant.DOCUMENT_ONLY:
        return True

    memory_cats = _memory_evidence_categories(question_row, variant)
    if not memory_cats:
        return True

    domains = PERSONA_DOMAINS.get(eval_persona, frozenset())
    return memory_cats.issubset(domains)


def label_dataframe(
    runs_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    *,
    block_id_col: str = "block_id",
) -> pd.DataFrame:
    """Add or replace ``is_in_domain`` on experiment run rows.

    Joins each run row to the questions table via ``block_id`` and applies
    ``compute_is_in_domain`` using the question's evidence metadata.
    """
    if block_id_col not in runs_df.columns:
        raise ValueError(f"runs_df missing required column {block_id_col!r}")

    questions = questions_df.reset_index(drop=True).copy()
    if block_id_col not in questions.columns:
        questions[block_id_col] = questions.index + 1

    questions_by_block = questions.set_index(block_id_col, drop=False)

    out = runs_df.copy()
    if "is_in_domain" in out.columns:
        out = out.drop(columns=["is_in_domain"])

    persona_col = "eval_persona" if "eval_persona" in out.columns else "persona"
    variant_col = "variant" if "variant" in out.columns else "dist"
    if persona_col not in out.columns:
        raise ValueError("runs_df must contain eval_persona or persona")
    if variant_col not in out.columns:
        raise ValueError("runs_df must contain variant or dist")

    labels: list[bool] = []
    for _, run_row in out.iterrows():
        block_id = run_row[block_id_col]
        if block_id not in questions_by_block.index:
            raise KeyError(f"No question row for block_id={block_id!r}")
        question_row = questions_by_block.loc[block_id]
        if isinstance(question_row, pd.DataFrame):
            question_row = question_row.iloc[0]
        labels.append(
            compute_is_in_domain(
                question_row,
                str(run_row[persona_col]).strip(),
                run_row[variant_col],
            )
        )

    out["is_in_domain"] = labels
    return out


def label_records(
    records: list[dict[str, Any]],
    questions_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return copies of *records* with ``is_in_domain`` added or replaced."""
    if not records:
        return []
    labeled = label_dataframe(pd.DataFrame(records), questions_df)
    return labeled.to_dict(orient="records")
