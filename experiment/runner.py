from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .answer_result import coerce_answer_result
from .data import sort_questions_by_persona
from .io import VARIANT_PLACEHOLDER, IncrementalInferenceWriter, format_output_path_for_variant
from .prepare_spec import build_prepare_spec
from .store_hooks import StoreHooksProtocol
from .variants import DatasetVariant

_DEFAULT_EVAL_PERSONAS = ["persona_1", "persona_2", "persona_3"]

PERSONA_PLACEHOLDER = "{eval_persona}"


def _variants_tuple(
    variants: DatasetVariant | Sequence[DatasetVariant],
) -> tuple[DatasetVariant, ...]:
    if isinstance(variants, DatasetVariant):
        return (variants,)
    t = tuple(variants)
    if not t:
        raise ValueError("variants must be non-empty")
    return t


def _inference_row_dict(
    row: pd.Series, prediction: str, eval_persona: str
) -> dict[str, Any]:
    qt = row.get("question_type")
    qt_out: str
    if qt is None or (isinstance(qt, float) and pd.isna(qt)):
        qt_out = ""
    else:
        qt_out = str(qt)
    qidx = str(row.get("question_idx", row.get("question_id", ""))).strip()
    original_persona = str(
        row.get("original_persona", row.get("persona", ""))
    ).strip()

    return {
        "question_idx": qidx,
        "original_persona": original_persona,
        "eval_persona": eval_persona,
        "query": row.get("query"),
        "prediction": prediction,
        "gold_answer": row.get("gold_answer"),
        "question_type": qt_out,
    }


def _format_output_path(
    template: str, variant: DatasetVariant, eval_persona: str
) -> Path:
    """Replace {variant} and {eval_persona} placeholders in template."""
    s = template
    if VARIANT_PLACEHOLDER in s:
        s = s.replace(VARIANT_PLACEHOLDER, variant.value)
    if PERSONA_PLACEHOLDER in s:
        s = s.replace(PERSONA_PLACEHOLDER, eval_persona)
    return Path(s).expanduser().resolve()


def run_experiment(
    df: pd.DataFrame,
    variants: DatasetVariant | Sequence[DatasetVariant],
    hooks_factory: Callable[[], StoreHooksProtocol],
    answer_fn: Callable[[str], str | dict[str, Any]],
    output_path: Path | str,
    *,
    eval_personas: list[str] | None = None,
    flush_every: int = 5,
    sort_by_persona: bool = True,
    strict_integrated_memory_strip: bool = False,
) -> list[dict[str, Any]]:
    """Run one or more variants over ``df``, crossing with eval personas.

    ``output_path`` is a template that can contain ``{variant}`` and/or
    ``{eval_persona}`` placeholders. For example:
    ``Path("runs/{variant}_{eval_persona}.jsonl")``.

    Parameters
    ----------
    df : pd.DataFrame
        Questions dataframe. Must include ``original_persona`` or ``persona``.
    variants : DatasetVariant or sequence thereof
    hooks_factory : callable
        Creates a fresh StoreHooksProtocol for each (variant, eval_persona) pair.
    answer_fn : callable
        Given a query string, returns a prediction string or a dict with
        ``prediction`` plus optional metadata fields.
    output_path : Path | str
        Template path with optional {variant} and {eval_persona} placeholders.
    eval_personas : list[str] | None
        Personas to evaluate each question against. Defaults to
        ["persona_1", "persona_2", "persona_3"]. Each question is evaluated
        against every eval persona's corpus.
    flush_every : int
    sort_by_persona : bool
    strict_integrated_memory_strip : bool
    """
    variant_list = _variants_tuple(variants)
    if eval_personas is None:
        eval_personas = list(_DEFAULT_EVAL_PERSONAS)

    tpl = str(output_path)
    needs_variant = len(variant_list) > 1
    needs_persona = len(eval_personas) > 1
    if needs_variant and VARIANT_PLACEHOLDER not in tpl:
        raise ValueError(
            f'output_path must include {VARIANT_PLACEHOLDER!r} when running '
            f'multiple variants; got {tpl!r}'
        )
    if needs_persona and PERSONA_PLACEHOLDER not in tpl:
        raise ValueError(
            f'output_path must include {PERSONA_PLACEHOLDER!r} when running '
            f'multiple eval personas; got {tpl!r}'
        )

    work = sort_questions_by_persona(df) if sort_by_persona else df.reset_index(drop=True)
    rows_out: list[dict[str, Any]] = []

    for variant in variant_list:
        for ep in eval_personas:
            resolved = _format_output_path(tpl, variant, ep)
            hooks = hooks_factory()
            hooks.rebuild_memory(ep)

            with IncrementalInferenceWriter(resolved, flush_every=flush_every) as writer:
                for _, row in work.iterrows():
                    spec = build_prepare_spec(
                        variant,
                        row,
                        eval_persona=ep,
                        strict_integrated_memory_strip=strict_integrated_memory_strip,
                    )
                    hooks.prepare_stores_for_question(spec)
                    start = time.perf_counter()
                    try:
                        raw = answer_fn(str(row["query"]).strip())
                    finally:
                        elapsed = time.perf_counter() - start
                        hooks.restore_stores_after_question(spec)

                    pred, extra = coerce_answer_result(raw)
                    rec = _inference_row_dict(row, pred, ep)
                    rec.update(
                        {
                            "latency_s": round(elapsed, 3),
                            "input_tokens": None,
                            "output_tokens": None,
                            "total_tokens": None,
                        }
                    )
                    rec.update(extra)
                    writer.write_row(rec)
                    rows_out.append(rec)

    return rows_out
