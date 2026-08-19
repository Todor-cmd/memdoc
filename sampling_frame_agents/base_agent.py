from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Any, Dict, List, Optional

import groq
import json
import time
from datetime import datetime, timezone

import pandas as pd
from pandas.api.types import is_scalar
from tqdm import tqdm
import os

from dotenv import load_dotenv

from .schemas import AgentAnswer, InferenceResult

load_dotenv()


def groq_giveup_bad_request(exc: BaseException) -> bool:
    """If True, backoff stops retrying (e.g. 400 tool_use_failed will not recover)."""
    return isinstance(exc, groq.BadRequestError)


def inference_result_from_failure(
    exc: BaseException,
    *,
    golden_memory_evidence: List[Dict[str, Any]],
    golden_document_evidence: List[Dict[str, Any]],
    retrieved_evidence: List[Dict[str, Any]],
) -> InferenceResult:
    """Placeholder row when invoke/parse fails so batch runs continue."""
    msg = f"{type(exc).__name__}: {exc}"
    if len(msg) > 2000:
        msg = msg[:2000] + "..."
    return InferenceResult(
        answer=AgentAnswer(
            reasoning="[Inference failed: see inference_error column.]",
            final_answer="[INFERENCE_FAILED]",
        ),
        usage=None,
        inference_error=msg,
        golden_memory_evidence=golden_memory_evidence,
        golden_document_evidence=golden_document_evidence,
        retrieved_evidence=retrieved_evidence,
    )


def coalesce_evidence(value: object) -> List[Dict[str, Any]]:
    """Normalize evidence columns from DataFrame rows (handles None, NaN, empty lists).

    ``null_query`` rows and similar may use empty lists or missing/NaN cells; this
    always returns a list so concatenation and JSON export never raise.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if is_scalar(value) and pd.isna(value):
        return []
    try:
        return list(value)
    except TypeError:
        return []


def evidence_list_to_json(evidence: Optional[List[Dict[str, Any]]]) -> str:
    """Serialize evidence dicts for CSV (handles non-JSON-native values)."""
    if not evidence:
        return "[]"
    return json.dumps(evidence, ensure_ascii=False, default=str)


def normalize_question_idx(value: Any) -> Any:
    """Align CSV/pickle index types (e.g. ``int``, ``int64``, ``174.0``)."""
    if value is None:
        return value
    try:
        if pd.isna(value):
            return value
    except TypeError:
        pass
    if isinstance(value, bool):
        return value
    try:
        f = float(value)
        if f.is_integer():
            return int(f)
    except (TypeError, ValueError):
        pass
    return value


def load_existing_inference_rows(output_csv_path: str) -> Dict[Any, dict]:
    """Load completed rows from a prior run, keyed by ``question_idx``.

    Returns an empty dict if the file is missing, empty, or invalid. Duplicate
    indices in the file: last row wins.
    """
    if not os.path.isfile(output_csv_path):
        return {}
    try:
        df = pd.read_csv(output_csv_path)
    except (pd.errors.EmptyDataError, FileNotFoundError, UnicodeDecodeError):
        return {}
    if df.empty or "question_idx" not in df.columns:
        return {}
    out: Dict[Any, dict] = {}
    for _, r in df.iterrows():
        key = normalize_question_idx(r["question_idx"])
        out[key] = r.to_dict()
    return out


def pick_test_mode_questions(df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """Subset *df* for smoke tests: one ``null_query``, one ``temporal_query``, one other.

    The third row is chosen uniformly at random among rows whose ``question_type``
    is neither ``null_query`` nor ``temporal_query``. If that pool is empty, any
    remaining row not already picked is used. Reproducible via *seed*.
    """
    if df.empty:
        return df
    rng = random.Random(seed)
    qt = df["question_type"].astype(str)
    picked: list[Any] = []

    def first_of(kind: str) -> None:
        mask = qt == kind
        for idx in df.index[mask]:
            if idx not in picked:
                picked.append(idx)
                return

    first_of("null_query")
    first_of("temporal_query")

    other_mask = ~qt.isin(["null_query", "temporal_query"])
    remaining = [i for i in df.index[other_mask] if i not in picked]
    if not remaining:
        remaining = [i for i in df.index if i not in picked]
    if remaining:
        picked.append(rng.choice(remaining))

    if not picked:
        return df.head(min(3, len(df)))

    return df.loc[picked]


class BaseAgent(ABC):
    """Abstract base class for all QA agents / baselines.

    Subclasses implement ``answer_question`` with their own retrieval or
    prompting strategy.  The ``run`` method handles batch I/O: loading the
    question DataFrame, iterating with a progress bar, and writing results
    to a CSV.

    Parameters
    ----------
    test_mode:
        If True, ``run`` processes a small fixed sample: one ``null_query``,
        one ``temporal_query``, and one random row with another question type
        (see :func:`pick_test_mode_questions`). ``save_every`` is capped so each
        step can persist.
    """

    def __init__(self, *, test_mode: bool = False) -> None:
        self.test_mode = test_mode

    @abstractmethod
    def answer_question(self, question_data: dict) -> InferenceResult:
        """Return an :class:`InferenceResult` for a single question.

        Parameters
        ----------
        question_data:
            A dict representing one row of the sampled-questions DataFrame.
            Expected keys: ``query``, ``answer``, ``question_type``,
            ``evidence_list``, ``memory_evidence``.
        """

    def run(
        self,
        questions_pkl_path: str,
        output_csv_path: str,
        save_every: int = 10,
    ) -> pd.DataFrame:
        """Load questions, run the agent on each, and write results to CSV.

        If *output_csv_path* already exists and contains a ``question_idx``
        column, rows for those indices are reused and only missing questions are
        inferred (resume).

        Parameters
        ----------
        save_every:
            Persist intermediate results to *output_csv_path* every N **appended**
            rows so progress is not lost on failure. In test mode this is capped so
            each inference triggers a save.

        Returns the results DataFrame for convenience.
        """
        questions_df = pd.read_pickle(questions_pkl_path)
        if self.test_mode:
            questions_df = pick_test_mode_questions(questions_df)
            save_every = min(save_every, max(1, len(questions_df)))

        out_dir = os.path.dirname(output_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        existing_by_idx = load_existing_inference_rows(output_csv_path)
        n_skip = sum(
            1 for i in questions_df.index if normalize_question_idx(i) in existing_by_idx
        )
        if n_skip:
            tqdm.write(
                f"{self.__class__.__name__}: resuming — {n_skip} row(s) already in "
                f"{output_csv_path!r}, skipping inference for those."
            )

        results: list[dict] = []
        for idx, row in tqdm(
            questions_df.iterrows(),
            total=len(questions_df),
            desc=f"Running {self.__class__.__name__}",
        ):
            idx_key = normalize_question_idx(idx)
            if idx_key in existing_by_idx:
                results.append(existing_by_idx[idx_key])
                if len(results) % save_every == 0:
                    pd.DataFrame(results).to_csv(output_csv_path, index=False)
                continue

            start = time.perf_counter()
            inference = self.answer_question(row.to_dict())
            elapsed = time.perf_counter() - start
            ans = inference.answer

            row_out: dict = {
                "question_idx": idx,
                "query": row["query"],
                "question_type": row["question_type"],
                "gold_answer": row["answer"],
                "reasoning": ans.reasoning,
                "final_answer": ans.final_answer,
                "inference_error": inference.inference_error,
                "golden_memory_evidence": evidence_list_to_json(
                    inference.golden_memory_evidence
                ),
                "golden_document_evidence": evidence_list_to_json(
                    inference.golden_document_evidence
                ),
                "retrieved_evidence": evidence_list_to_json(
                    inference.retrieved_evidence
                ),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "latency_s": round(elapsed, 3),
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }
            if inference.usage is not None:
                u = inference.usage
                row_out["input_tokens"] = u.input_tokens
                row_out["output_tokens"] = u.output_tokens
                row_out["total_tokens"] = u.total_tokens

            results.append(row_out)

            if len(results) % save_every == 0:
                pd.DataFrame(results).to_csv(output_csv_path, index=False)

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv_path, index=False)
        return results_df
