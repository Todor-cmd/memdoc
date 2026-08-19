"""Snapshot MultiHop-RAG corpus + evidence_id→url map for offline evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prepare_data.question_evidence_ids import (
    iter_evidence_dicts_from_row,
    source_question_id_for_row,
)

DEFAULT_DATASET = "yixuantt/MultiHopRAG"
DEFAULT_CORPUS_SPLIT = "corpus"
DEFAULT_CORPUS_JSONL = "multihop_corpus.jsonl"
DEFAULT_MAP_JSONL = "evidence_id_to_url.jsonl"


def _jsonable_value(v: Any) -> Any:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def corpus_row_to_obj(row: pd.Series) -> dict[str, Any]:
    return {str(k): _jsonable_value(row[k]) for k in row.index}


def load_corpus_dataframe(
    *,
    dataset: str,
    config_name: str,
    revision: str | None,
) -> pd.DataFrame:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": dataset,
        "name": config_name,
        "split": "train",
    }
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset(**kwargs)
    return ds.to_pandas()


def write_corpus_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = "url"
    if key not in df.columns:
        raise KeyError(f"Corpus DataFrame must include {key!r}; got {list(df.columns)}")
    dfc = df.copy()
    dfc["_sort_url"] = dfc[key].map(lambda u: "" if pd.isna(u) else str(u))
    dfc = dfc.sort_values("_sort_url", kind="stable").drop(columns=["_sort_url"])
    urls = dfc[key].astype(str)
    dup = int(urls.duplicated().sum())
    if dup:
        raise ValueError(f"Corpus contains {dup} duplicate {key} rows; refuse to write ambiguous snapshot")

    with path.open("w", encoding="utf-8") as f:
        for _, row in dfc.iterrows():
            f.write(json.dumps(corpus_row_to_obj(row), ensure_ascii=False))
            f.write("\n")


def build_evidence_id_url_records(df: pd.DataFrame) -> list[dict[str, str]]:
    """One record per distinct evidence_id with url and question_idx (for debugging)."""
    by_id: dict[str, dict[str, str]] = {}
    for idx, row in df.iterrows():
        q_source = source_question_id_for_row(row, idx)
        question_idx_val = row["question_idx"] if "question_idx" in row.index else q_source
        if question_idx_val is not None and not (
            isinstance(question_idx_val, float) and pd.isna(question_idx_val)
        ):
            q_disp = str(question_idx_val).strip()
        else:
            q_disp = str(q_source).strip()

        for ev in iter_evidence_dicts_from_row(row):
            if not isinstance(ev, dict):
                continue
            eid = ev.get("evidence_id")
            url = ev.get("url")
            if eid is None or (isinstance(eid, float) and pd.isna(eid)):
                raise ValueError(
                    f"Evidence dict missing evidence_id (row index {idx!r}, question_idx={q_disp!r}). "
                    "Run attach_evidence_ids_to_dataframe on the questions frame first."
                )
            eid_s = str(eid).strip()
            if not eid_s:
                raise ValueError(f"Empty evidence_id (row index {idx!r})")
            if url is None or (isinstance(url, float) and pd.isna(url)) or not str(url).strip():
                raise ValueError(f"Evidence {eid_s!r} missing url (row index {idx!r})")
            url_s = str(url).strip()
            prev = by_id.get(eid_s)
            if prev is not None:
                if prev["url"] != url_s:
                    raise ValueError(
                        f"evidence_id {eid_s!r} maps to two urls: {prev['url']!r} vs {url_s!r}"
                    )
                continue
            by_id[eid_s] = {
                "evidence_id": eid_s,
                "url": url_s,
                "question_idx": q_disp,
            }
    return [by_id[k] for k in sorted(by_id.keys())]


def write_records_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def main(argv: list[str] | None = None) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Export MultiHop-RAG corpus to JSONL and evidence_id→url map from reasonable questions pickle.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=repo_root / "data" / "document_collection",
        help="Output directory (default: <repo>/data/document_collection)",
    )
    parser.add_argument(
        "--questions-pickle",
        type=Path,
        default=repo_root / "data" / "questions" / "full_reasonable.pkl",
        help="Questions dataframe with evidence_id on each evidence dict "
        "(default: data/questions/full_reasonable.pkl)",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset id (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--corpus-split",
        default=DEFAULT_CORPUS_SPLIT,
        help=f"Dataset config name for articles (default: {DEFAULT_CORPUS_SPLIT})",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision/commit id for reproducible snapshots.",
    )
    parser.add_argument(
        "--corpus-jsonl-name",
        default=DEFAULT_CORPUS_JSONL,
        help=f"Filename under output-dir for corpus JSONL (default: {DEFAULT_CORPUS_JSONL})",
    )
    parser.add_argument(
        "--map-jsonl-name",
        default=DEFAULT_MAP_JSONL,
        help=f"Filename under output-dir for evidence map JSONL (default: {DEFAULT_MAP_JSONL})",
    )
    args = parser.parse_args(argv)

    out_dir = args.output_dir.expanduser().resolve()
    questions_path = args.questions_pickle.expanduser().resolve()
    corpus_path = out_dir / args.corpus_jsonl_name
    map_path = out_dir / args.map_jsonl_name

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions pickle not found: {questions_path}")

    df_q = pd.read_pickle(questions_path)
    print(f"Loaded {len(df_q)} rows from {questions_path}")

    df_corpus = load_corpus_dataframe(
        dataset=args.dataset,
        config_name=args.corpus_split,
        revision=args.revision,
    )
    print(f"Loaded corpus: {len(df_corpus)} rows from {args.dataset} ({args.corpus_split})")

    write_corpus_jsonl(df_corpus, corpus_path)
    print(f"Wrote corpus JSONL: {corpus_path}")

    evidence_records = build_evidence_id_url_records(df_q)
    write_records_jsonl(evidence_records, map_path)
    print(f"Wrote evidence_id→url map ({len(evidence_records)} entries): {map_path}")


if __name__ == "__main__":
    main()
