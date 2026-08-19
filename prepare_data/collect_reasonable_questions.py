from __future__ import annotations
import re
import string
import sys
from typing import Any
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prepare_data.question_evidence_ids import (
    attach_evidence_ids_to_dataframe,
    question_id_from_query,
)

def get_artifacts():
    repo_root = Path.cwd()
    path_sampled_golden = repo_root / "data" / "golden_context_agent_inferences" / "sampled" /"llama-3.3-70b-versatile_no_author.csv"
    path_rest_golden_t = repo_root / "data" / "golden_context_agent_inferences" / "rest" /"llama-3.3-70b-versatile_no_author.csv"
    path_sampled_contextless = repo_root / "data" / "no_context_agent_inferences" / "sampled" / "llama-3.3-70b-versatile.csv"
    path_rest_contextless = repo_root / "data" / "no_context_agent_inferences" / "rest" / "llama-3.3-70b-versatile.csv"


    if not (path_sampled_golden.exists() and path_rest_golden_t.exists() and path_sampled_contextless.exists() and path_rest_contextless.exists()):
        raise FileNotFoundError("One or more of the paths do not exist")

    print("Files found")

    sampled_golden = pd.read_csv(path_sampled_golden)
    rest_golden = pd.read_csv(path_rest_golden_t)
    sampled_contextless = pd.read_csv(path_sampled_contextless)
    rest_contextless = pd.read_csv(path_rest_contextless)

    if not (sampled_golden.shape[0] == sampled_contextless.shape[0] and rest_golden.shape[0] == rest_contextless.shape[0]):
        print(f"Sampled golden: {sampled_golden.shape[0]}, Sampled contextless: {sampled_contextless.shape[0]}")
        print(f"Rest golden: {rest_golden.shape[0]}, Rest contextless: {rest_contextless.shape[0]}")
        raise ValueError("Sampled and rest golden and contextless have different numbers of rows")


    return sampled_golden, rest_golden, sampled_contextless, rest_contextless

def normalize_answer(s: Any) -> str:
    """SQuAD-style normalization for EM/F1 (string must be comparable after this)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    text = str(s).strip().lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def exact_match(pred: Any, gold: Any) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)

def f1_score(pred: Any, gold: Any) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = {}
    for t in gold_toks:
        common[t] = common.get(t, 0) + 1
    tp = 0
    for t in pred_toks:
        if common.get(t, 0) > 0:
            tp += 1
            common[t] -= 1
    precision = tp / len(pred_toks)
    recall = tp / len(gold_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
    
def evaluate_final_vs_gold(
    df: pd.DataFrame,
    *,
    pred_col: str = "final_answer",
    gold_col: str = "gold_answer",
) -> dict[str, float]:
    """Per-row EM (0/1) and token F1; returns dataset-level means (micro-average)."""
    if pred_col not in df.columns or gold_col not in df.columns:
        raise KeyError(f"Expected columns {pred_col!r} and {gold_col!r}; got {list(df.columns)}")
    ems: list[float] = []
    f1s: list[float] = []
    for _, row in df.iterrows():
        p, g = row[pred_col], row[gold_col]
        ems.append(1.0 if exact_match(p, g) else 0.0)
        f1s.append(f1_score(p, g))
    n = len(ems)
    return {
        "n": float(n),
        "exact_match": sum(ems) / n if n else float("nan"),
        "f1": sum(f1s) / n if n else float("nan"),
    }

def get_reasonable_question_idx(df_golden, df_contextless):
    em_no_context = df_contextless.apply(
        lambda r: exact_match(r["final_answer"], r["gold_answer"]),
        axis=1,
    )
    correct_idx_no_context = df_contextless.loc[em_no_context, "question_idx"]

    em_golden = df_golden.apply(
        lambda r: exact_match(r["final_answer"], r["gold_answer"]),
        axis=1,
    )
    correct_idx_golden = df_golden.loc[em_golden, "question_idx"]

    reasonable_question_idx = set(correct_idx_golden) - set(correct_idx_no_context)

    print("No Context Correct: ", len(correct_idx_no_context))
    print("Golden Context Correct: ", len(correct_idx_golden))
    print("Reasonable: ", len(reasonable_question_idx))
    return list(reasonable_question_idx)

def reasonable_questions():
    sampled_golden, rest_golden, sampled_contextless, rest_contextless = get_artifacts()

    print("Getting reasonable questions for sampled")
    reasonable_sampled_idx = get_reasonable_question_idx(sampled_golden, sampled_contextless)
    print("Getting reasonable questions for rest")
    reasonable_rest_idx = get_reasonable_question_idx(rest_golden, rest_contextless)

    reasonable_sampled_questions = sampled_golden[
        sampled_golden["question_idx"].isin(reasonable_sampled_idx)
    ].copy()
    reasonable_rest_questions = rest_golden[
        rest_golden["question_idx"].isin(reasonable_rest_idx)
    ].copy()

    for part in (reasonable_sampled_questions, reasonable_rest_questions):
        part["question_idx"] = part["query"].map(question_id_from_query)
        part["question_id"] = part["question_idx"]
        attach_evidence_ids_to_dataframe(part)

    print(f"Reasonable sampled: {len(reasonable_sampled_questions)}, Reasonable rest: {len(reasonable_rest_questions)}")
    return reasonable_sampled_questions, reasonable_rest_questions

if __name__ == "__main__":
    reasonable_sampled_questions, reasonable_rest_questions = reasonable_questions()

    out_dir = Path.cwd() / "data" / "questions" 
    out_dir.mkdir(parents=True, exist_ok=True)

    reasonable_sampled_questions.to_pickle(out_dir / "sampled_reasonable.pkl")
    reasonable_rest_questions.to_pickle(out_dir / "rest_reasonable.pkl")

    full_reasonable = pd.concat(
        [reasonable_sampled_questions, reasonable_rest_questions],
        axis=0,
        ignore_index=True,
    )
    dup_ids = int(full_reasonable["question_idx"].duplicated().sum())
    if dup_ids:
        raise ValueError(
            f"Expected unique question_idx after query hashing; found {dup_ids} duplicate rows "
            "(duplicate query strings across sampled/rest?)."
        )
    full_reasonable.to_pickle(out_dir / "full_reasonable.pkl")
