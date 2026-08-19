"""Split question evidence between memory and document channels for integrated eval."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np

from prepare_data.question_evidence_ids import (
    _parse_evidence_cell,
    question_id_from_query,
)


def all_golden_evidence_for_row(row: pd.Series) -> list[dict]:
    """Union golden evidence dicts for one question, deduped by ``evidence_id``."""
    by_eid: dict[str, dict] = {}
    cols = ("golden_memory_evidence", "golden_document_evidence", "evidence_list")
    for col in cols:
        if col not in row.index:
            continue
        for ev in _parse_evidence_cell(row[col]):
            if not isinstance(ev, dict):
                continue
            eid = ev.get("evidence_id")
            if eid is None or (isinstance(eid, float) and pd.isna(eid)):
                continue
            key = str(eid).strip()
            if key:
                by_eid[key] = ev
    return list(by_eid.values())


def _row_rng(row: pd.Series, df_index, base_seed: int) -> np.random.RandomState:
    """Per-question RNG so split is stable regardless of dataframe row order."""
    if "question_id" in row.index:
        v = row["question_id"]
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            qid = str(v).strip()
        else:
            qid = question_id_from_query(row.get("query"))
    elif "question_idx" in row.index:
        v = row["question_idx"]
        qid = str(v).strip() if v is not None else question_id_from_query(row.get("query"))
    elif "query" in row.index:
        qid = question_id_from_query(row["query"])
    else:
        qid = str(df_index)
    digest = int(hashlib.sha256(f"{base_seed}:{qid}".encode()).hexdigest()[:8], 16)
    return np.random.RandomState(digest)


def split_evidence_fifty_fifty(
    evidence: list[dict],
    rng: np.random.RandomState,
) -> tuple[list[dict], list[dict]]:
    """Partition evidence ~50-50 between memory and document channels.

    For odd counts, the extra item is assigned to memory or document with
    equal probability. For a single item, it is assigned entirely to one
    channel (50/50).
    """
    n = len(evidence)
    if n == 0:
        return [], []
    if n == 1:
        if rng.random() < 0.5:
            return [evidence[0]], []
        return [], [evidence[0]]

    n_mem = n // 2
    if n % 2 == 1 and rng.random() < 0.5:
        n_mem += 1

    perm = rng.permutation(n)
    mem_idxs = set(perm[:n_mem])
    memory = [evidence[i] for i in range(n) if i in mem_idxs]
    document = [evidence[i] for i in range(n) if i not in mem_idxs]
    return memory, document


def apply_memory_evidence_split(
    dataset: pd.DataFrame,
    random_seed: int = 42,
    *,
    evidence_source: str = "golden",
) -> pd.DataFrame:
    """Write ``memory_evidence`` and ``evidence_list`` using a 50-50 partition.

    Parameters
    ----------
    dataset : pd.DataFrame
        One row per question.
    random_seed : int
        Base seed; each row derives a stable per-question seed from ``question_id``
        or query hash.
    evidence_source : str
        ``"golden"`` — union ``golden_memory_evidence`` + ``golden_document_evidence``
        (and ``evidence_list`` if golden columns are absent).
        ``"evidence_list"`` — split only the ``evidence_list`` column (legacy).
    """
    memory_evidence_list: list[list] = []
    updated_evidence_list: list[list] = []

    for idx, row in dataset.iterrows():
        if evidence_source == "evidence_list":
            evidence = _parse_evidence_cell(row.get("evidence_list"))
            evidence = [e for e in evidence if isinstance(e, dict)]
        else:
            evidence = all_golden_evidence_for_row(row)

        rng = _row_rng(row, idx, random_seed)
        memory, document = split_evidence_fifty_fifty(evidence, rng)
        memory_evidence_list.append(memory)
        updated_evidence_list.append(document)

    out = dataset.copy()
    out["memory_evidence"] = memory_evidence_list
    out["evidence_list"] = updated_evidence_list
    return out


def sample_questions_with_memory_evidence(
    dataset: pd.DataFrame, n_sample: int, random_seed: int = 42
) -> pd.DataFrame:
    """Sample questions and apply a 50-50 memory/document split."""
    random_sample_qs = dataset.sample(n=n_sample, random_state=random_seed).reset_index(
        drop=True
    )
    return apply_memory_evidence_split(random_sample_qs, random_seed=random_seed)


if __name__ == "__main__":
    from datasets import load_dataset

    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Split MultiHopRAG evidence into memory vs context and save pickles."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=repo_root / "data" / "questions",
        help="Directory for sampled_questions.pkl and rest_questions.pkl",
    )
    args = parser.parse_args()
    out_dir: Path = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    multihop_dataset = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")["train"].to_pandas()

    n_sample = 500
    sample_seed = 42
    sampled_idx = multihop_dataset.sample(n=n_sample, random_state=sample_seed).index
    sampled_df = multihop_dataset.loc[sampled_idx].reset_index(drop=True)
    rest_df = multihop_dataset.drop(sampled_idx).reset_index(drop=True)

    op3_qs = apply_memory_evidence_split(sampled_df, random_seed=sample_seed, evidence_source="evidence_list")
    rest_qs = apply_memory_evidence_split(rest_df, random_seed=sample_seed + 1, evidence_source="evidence_list")

    op3_qs.to_pickle(out_dir / "sampled_questions.pkl")
    rest_qs.to_pickle(out_dir / "rest_questions.pkl")
