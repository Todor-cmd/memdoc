"""Design-matrix-driven experiment runner.

Executes only the (question, variant, persona) triples specified by the
D-optimal design CSV for a single agent, maintaining 3 persistent persona
stores simultaneously.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .answer_result import coerce_answer_result
from .io import IncrementalInferenceWriter
from .prepare_spec import build_prepare_spec
from .retrieval_evidence import golden_evidence_for_variant
from .store_hooks import StoreHooksProtocol
from .variants import DatasetVariant

_DEFAULT_EVAL_PERSONAS = ["persona_1", "persona_2", "persona_3"]


def load_design_matrix(
    design_csv: Path | str,
    questions_pickle: Path | str,
) -> pd.DataFrame:
    """Load the design CSV and join to the questions pickle.

    The design CSV has columns: Block1, dist, agent, persona.
    Block1 is the 1-indexed row position in the questions pickle.

    Returns a merged DataFrame with all question columns plus design columns.
    """
    design = pd.read_csv(design_csv)
    questions = pd.read_pickle(questions_pickle)

    questions = questions.reset_index(drop=True)
    questions["block_id"] = questions.index + 1

    merged = design.merge(questions, left_on="Block1", right_on="block_id", how="left")
    if merged["query"].isna().any():
        n_missing = merged["query"].isna().sum()
        raise ValueError(
            f"{n_missing} design rows could not be joined to questions "
            f"(Block1 values out of range?)"
        )
    return merged


def filter_design_for_agent(design_df: pd.DataFrame, agent_id: str) -> pd.DataFrame:
    """Filter the merged design matrix to rows for a specific agent."""
    filtered = design_df[design_df["agent"] == agent_id].copy()
    if filtered.empty:
        available = sorted(design_df["agent"].unique())
        raise ValueError(
            f"No design rows for agent {agent_id!r}. Available: {available}"
        )
    return filtered.reset_index(drop=True)


def _design_inference_row(
    row: pd.Series,
    prediction: str,
    eval_persona: str,
    variant: DatasetVariant,
    agent_id: str,
) -> dict[str, Any]:
    """Build an output record for one design-matrix inference."""
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
        "block_id": int(row.get("block_id", row.get("Block1", 0))),
        "original_persona": original_persona,
        "eval_persona": eval_persona,
        "variant": variant.value,
        "agent": agent_id,
        "query": row.get("query"),
        "prediction": prediction,
        "gold_answer": row.get("gold_answer"),
        "question_type": qt_out,
    }


def run_design_experiment(
    design_df: pd.DataFrame,
    agent_id: str,
    hooks_factory: Callable[[], StoreHooksProtocol],
    answer_fn: Callable[[str], str | dict[str, Any]],
    output_path: Path | str,
    *,
    eval_personas: list[str] | None = None,
    flush_every: int = 5,
    strict_integrated_memory_strip: bool = True,
) -> list[dict[str, Any]]:
    """Run the design-matrix experiment for one agent.

    Maintains 3 persistent persona stores and iterates grouped by persona
    for cache locality.

    Parameters
    ----------
    design_df : pd.DataFrame
        Merged design matrix filtered to one agent (from
        ``filter_design_for_agent``). Must contain question columns
        (query, gold_answer, etc.) and design columns (dist, persona, Block1).
    agent_id : str
        Agent identifier (for output metadata).
    hooks_factory : callable
        Creates a StoreHooksProtocol instance. Called once (documents are
        shared); ``rebuild_memory`` is called per persona to swap memory.
    answer_fn : callable
        Given a query string, returns a prediction string or a dict with
        ``prediction`` plus optional metadata fields (e.g. ``rewrite_count``).
    output_path : Path | str
        Output file path (.jsonl or .csv).
    eval_personas : list[str] | None
        Persona IDs to maintain stores for. Defaults to persona_1..3.
    flush_every : int
        Flush output every N rows.
    strict_integrated_memory_strip : bool
        Strip document-channel evidence sessions from memory in the
        integrated variant to prevent leakage (default True).
    """
    if eval_personas is None:
        eval_personas = list(_DEFAULT_EVAL_PERSONAS)

    out_path = Path(output_path).expanduser().resolve()

    # Build a single store instance (documents are shared across personas)
    # and rebuild only the memory layer per persona.
    store = hooks_factory()

    rows_out: list[dict[str, Any]] = []
    total = len(design_df)

    pbar = tqdm(total=total, desc="Inferences", unit="q")
    with IncrementalInferenceWriter(out_path, flush_every=flush_every) as writer:
        for ep in eval_personas:
            persona_work = design_df[design_df["persona"] == ep]
            if persona_work.empty:
                continue

            pbar.set_postfix(persona=ep, status="rebuilding memory")
            store.rebuild_memory(ep)
            pbar.set_postfix(persona=ep)

            for _, row in persona_work.iterrows():
                variant = DatasetVariant(row["dist"])

                spec = build_prepare_spec(
                    variant,
                    row,
                    eval_persona=ep,
                    strict_integrated_memory_strip=strict_integrated_memory_strip,
                )
                store.prepare_stores_for_question(spec)
                start = time.perf_counter()
                try:
                    raw = answer_fn(str(row["query"]).strip())
                finally:
                    elapsed = time.perf_counter() - start
                    store.restore_stores_after_question(spec)

                pred, extra = coerce_answer_result(raw)
                rec = _design_inference_row(row, pred, ep, variant, agent_id)
                rec.update(
                    {
                        "latency_s": round(elapsed, 3),
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                    }
                )
                rec.update(extra)

                # Add golden evidence for retrieval analysis
                golden = golden_evidence_for_variant(row, variant)
                rec["golden_memory_session_ids"] = sorted(spec.memory_session_ids_to_ensure)
                rec.update(golden)

                writer.write_row(rec)
                rows_out.append(rec)
                pbar.update(1)

    pbar.close()
    print(f"\nDone — {total} inferences written to {out_path}")
    return rows_out


def default_design_output_path(
    out_dir: Path | str,
    agent_id: str,
    *,
    suffix: str = ".jsonl",
) -> Path:
    """Generate output path: ``<out_dir>/<agent_id><suffix>``."""
    d = Path(out_dir).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{agent_id}{suffix}"
