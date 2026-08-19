#!/usr/bin/env python3
"""Collect evaluation metadata for experiment runs.

Merges raw agent run files with question metadata and computes:
  - ``is_in_domain`` / ``in_domain``  – whether memory-channel gold evidence
    falls within the eval persona's domain categories.
  - ``correct``  – SQuAD-style exact-match (EM) scoring.
  - ``hop_count`` – joined from the questions pickle.

Usage:

    python3 analysis/collect_evaluation_metadata.py \\
        --runs-dir data/experiment_runs \\
        --questions-pkl data/questions/experiment_210_split.pkl \\
        --output data/experiment_runs/labeled_runs.csv

    python3 analysis/collect_evaluation_metadata.py \\
        --input data/experiment_runs/agent_1.jsonl \\
        --questions-pkl data/questions/experiment_210_split.pkl \\
        --output data/experiment_runs/agent_1_labeled.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.in_domain import label_dataframe

# ---------------------------------------------------------------------------
# SQuAD-style exact-match scoring
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_answer(s: str | None) -> str:
    """Lowercase, strip punctuation & articles, collapse whitespace."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    s = _PUNCT_RE.sub("", s)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def exact_match(prediction: str | None, gold: str | None) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_questions(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)


def _load_runs(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(records)
    return pd.read_csv(path)


def _load_runs_dir(runs_dir: Path) -> pd.DataFrame:
    files = sorted(runs_dir.glob("agent_*.jsonl")) + sorted(runs_dir.glob("agent_*.csv"))
    if not files:
        raise FileNotFoundError(f"No agent run files found in {runs_dir}")
    return pd.concat([_load_runs(path) for path in files], ignore_index=True)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for rec in df.to_dict(orient="records"):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect(runs: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    """Label, score, and normalise column names for *runs*."""

    # 1. is_in_domain
    df = label_dataframe(runs, questions)

    # 2. Join hop_count from questions if missing
    if "hop_count" not in df.columns and "block_id" in df.columns:
        q = questions.reset_index(drop=True).copy()
        if "block_id" not in q.columns:
            q["block_id"] = q.index + 1
        hop_lookup = q[["block_id", "hop_count"]].drop_duplicates("block_id")
        df = df.merge(hop_lookup, on="block_id", how="left")

    # 3. Exact-match scoring
    df["correct"] = [
        int(exact_match(p, g))
        for p, g in zip(df["prediction"], df["gold_answer"])
    ]

    # 4. Column renaming (harness outputs eval_persona / variant)
    if "eval_persona" in df.columns and "persona" not in df.columns:
        df = df.rename(columns={"eval_persona": "persona"})
    if "variant" in df.columns and "dist" not in df.columns:
        df = df.rename(columns={"variant": "dist"})

    # 5. Friendly in_domain string label
    df["in_domain"] = df["is_in_domain"].map(
        {True: "in_domain", False: "out_of_domain"}
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Single run file (.jsonl or .csv)")
    src.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory of agent_*.jsonl / agent_*.csv files to merge and label",
    )
    parser.add_argument(
        "--questions-pkl",
        type=Path,
        required=True,
        help="Questions pickle joined by block_id",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output path (.csv or .jsonl)")
    args = parser.parse_args()

    questions = _load_questions(args.questions_pkl)
    runs = _load_runs(args.input) if args.input else _load_runs_dir(args.runs_dir)

    df = collect(runs, questions)
    _write(df, args.output)

    n_in = int(df["is_in_domain"].sum())
    n_correct = int(df["correct"].sum())
    print(
        f"Collected {len(df)} rows -> {args.output}\n"
        f"  is_in_domain: {n_in} in-domain, {len(df) - n_in} out-of-domain\n"
        f"  EM accuracy:  {n_correct}/{len(df)} = {n_correct / len(df):.3f}"
    )


if __name__ == "__main__":
    main()
