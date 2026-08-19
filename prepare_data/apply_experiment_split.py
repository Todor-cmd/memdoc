"""Apply 50-50 memory/document split to the stratified experiment question pickle.

Reads the experiment sample (default ``experiment_210.pkl``), partitions each
question's golden evidence into ``memory_evidence`` and ``evidence_list``, and
writes ``experiment_210_split.pkl`` for batch generation and eval.

Usage:
    python -m prepare_data.apply_experiment_split
    python -m prepare_data.apply_experiment_split -i data/questions/experiment_210.pkl -o out.pkl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prepare_data.knowledge_split import apply_memory_evidence_split

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INPUT = _REPO_ROOT / "data" / "questions" / "experiment_210.pkl"
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "questions" / "experiment_210_split.pkl"


def _hop_stats(df: pd.DataFrame) -> None:
    mem_lens = df["memory_evidence"].map(len)
    doc_lens = df["evidence_list"].map(len)
    total = mem_lens + doc_lens
    print(f"  memory hops:  mean={mem_lens.mean():.2f}  min={mem_lens.min()}  max={mem_lens.max()}")
    print(f"  document hops: mean={doc_lens.mean():.2f}  min={doc_lens.min()}  max={doc_lens.max()}")
    print(f"  total hops:    mean={total.mean():.2f}")
    single = int((total == 1).sum())
    if single:
        print(f"  single-hop questions: {single} (assigned entirely to memory or document)")


def main() -> None:
    parser = argparse.ArgumentParser(description="50-50 split on experiment question pickle.")
    parser.add_argument("-i", "--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(
            f"Input not found: {inp}. Run prepare_data.sample_experiment_questions first."
        )

    print(f"Loading {inp}")
    df = pd.read_pickle(inp)
    print(f"  {len(df)} questions")

    out_df = apply_memory_evidence_split(df, random_seed=args.seed, evidence_source="golden")
    print("Split statistics:")
    _hop_stats(out_df)

    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(out_path)
    print(f"Written → {out_path}")


if __name__ == "__main__":
    main()
