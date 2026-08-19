"""Dist-invariant topical relevance of a question relative to an eval persona.

The label compares the union of all golden-evidence categories C_q to the
persona topic set T_p. It does not depend on evidence distribution.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from session_generation.persona import PERSONA_DOMAINS

from .evidence import evidence_dicts_union_for_question

IN_DOMAIN = "in_domain"
PARTIAL = "partial"
OUT_OF_DOMAIN = "out_of_domain"


def evidence_in_domain(category: str | None, eval_persona: str) -> bool:
    """Whether a single evidence item's category sits in the persona topic set ``T_p``."""
    domains = PERSONA_DOMAINS.get(str(eval_persona).strip(), frozenset())
    if not domains:
        return False
    if not category or not isinstance(category, str) or not category.strip():
        return False
    return category.strip().lower() in domains


def evidence_categories_for_question(question_row: pd.Series) -> set[str]:
    """Union of category tags across all gold evidence items of the question."""
    categories: set[str] = set()
    for ev in evidence_dicts_union_for_question(question_row):
        cat = ev.get("category")
        if cat and isinstance(cat, str) and cat.strip():
            categories.add(cat.strip().lower())
    return categories


def compute_topical_relevance(question_row: pd.Series, eval_persona: str) -> str:
    """Return ``in_domain``, ``partial``, or ``out_of_domain`` for *(q, persona)*."""
    cats = evidence_categories_for_question(question_row)
    domains = PERSONA_DOMAINS.get(str(eval_persona).strip(), frozenset())
    if not cats or not domains:
        return OUT_OF_DOMAIN
    if cats.issubset(domains):
        return IN_DOMAIN
    if cats.isdisjoint(domains):
        return OUT_OF_DOMAIN
    return PARTIAL


def label_dataframe(
    runs_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    *,
    block_id_col: str = "block_id",
) -> pd.DataFrame:
    """Add or replace ``topical_relevance`` on experiment run rows."""
    if block_id_col not in runs_df.columns:
        raise ValueError(f"runs_df missing required column {block_id_col!r}")

    questions = questions_df.reset_index(drop=True).copy()
    if block_id_col not in questions.columns:
        questions[block_id_col] = questions.index + 1

    questions_by_block = questions.set_index(block_id_col, drop=False)

    out = runs_df.copy()
    if "topical_relevance" in out.columns:
        out = out.drop(columns=["topical_relevance"])

    persona_col = "eval_persona" if "eval_persona" in out.columns else "persona"
    if persona_col not in out.columns:
        raise ValueError("runs_df must contain eval_persona or persona")

    labels: list[str] = []
    for _, run_row in out.iterrows():
        block_id = run_row[block_id_col]
        if block_id not in questions_by_block.index:
            raise KeyError(f"No question row for block_id={block_id!r}")
        question_row = questions_by_block.loc[block_id]
        if isinstance(question_row, pd.DataFrame):
            question_row = question_row.iloc[0]
        labels.append(
            compute_topical_relevance(question_row, str(run_row[persona_col]).strip())
        )

    out["topical_relevance"] = labels
    return out


def label_records(
    records: list[dict[str, Any]],
    questions_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return copies of *records* with ``topical_relevance`` added or replaced."""
    if not records:
        return []
    labeled = label_dataframe(pd.DataFrame(records), questions_df)
    return labeled.to_dict(orient="records")
