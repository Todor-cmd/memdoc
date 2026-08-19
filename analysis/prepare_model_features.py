"""Export per-row retrieval features and a session-level audit table.

Produces:
  1. data/analysis/model_features_full.csv
     All 4 primary agents, all dist levels.
     Adds: recall_at_10, recall_full_context_strict, topical_relevance.

  2. data/analysis/model_features_memory.csv
     memory_only + integrated rows (legacy aggregated diversity).

  3. data/analysis/model_features_audit.csv
     One row per memory-only gold session x run, with unaggregated diversity
     attributes, a strict session-level retrieval hit, and a binary
     evidence_in_domain flag (this item's category vs the eval persona).

Usage:
    python3 analysis/prepare_model_features.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.retrieval import add_strict_recall  # noqa: E402
from analysis.retrieval.audit_map import load_audit_map  # noqa: E402
from analysis.retrieval.pair_index import PairIndexCache, load_session_turns  # noqa: E402
from analysis.retrieval.strict_recall import (  # noqa: E402
    _DEFAULT_AUDIT,
    _DEFAULT_MEMORY,
    _parse_retrieved,
    strict_mem_hit,
)
from experiment.topical_relevance import (  # noqa: E402
    evidence_in_domain,
    label_dataframe as label_topical_relevance,
)

LABELED_RUNS = REPO_ROOT / "data" / "experiment_runs" / "labeled_runs.csv"
QUESTIONS_PKL = REPO_ROOT / "data" / "questions" / "experiment_210_split.pkl"
MANIFEST_PATH = (
    REPO_ROOT / "data" / "batch_jobs" / "experiment_sessions"
    / "batch_manifest_20260607_145849.json"
)
BATCH_OUTPUT_PATH = (
    REPO_ROOT / "data" / "batch_jobs" / "experiment_sessions"
    / "batch_output.jsonl"
)
MEMORY_COLLECTION_DIR = REPO_ROOT / "data" / "memory_collection"
OUT_DIR = REPO_ROOT / "data" / "analysis"

DOC_DISTS = {"document_only", "integrated"}
MEM_DISTS = {"memory_only", "integrated"}


def _parse_list(val: Any) -> list:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def compute_recall_at_k(retrieved: list, gold: list, k: int) -> float:
    if not gold:
        return np.nan
    return len(set(retrieved[:k]) & set(gold)) / len(gold)


def add_recall(df: pd.DataFrame) -> pd.DataFrame:
    """Add recall_at_10 column using unified evidence gold set."""
    recalls = []
    for _, r in df.iterrows():
        gold = []
        if r["dist"] in DOC_DISTS and pd.notna(r["golden_document_urls"]):
            gold += [f"doc::{u}" for u in ast.literal_eval(r["golden_document_urls"])]
        if r["dist"] in MEM_DISTS and pd.notna(r["golden_memory_session_ids"]):
            gold += [f"mem::{s}" for s in _parse_list(r["golden_memory_session_ids"])]

        if not gold:
            recalls.append(np.nan)
            continue

        items = []
        if pd.notna(r["retrieved_documents"]):
            docs = ast.literal_eval(r["retrieved_documents"])
            seen_urls: set = set()
            for d in sorted(docs, key=lambda x: x.get("score", 0), reverse=True):
                if d["url"] not in seen_urls:
                    seen_urls.add(d["url"])
                    items.append((d.get("score", 0), f"doc::{d['url']}"))

        if pd.notna(r["retrieved_memory"]):
            mems = ast.literal_eval(r["retrieved_memory"])
            for m in mems:
                items.append((m.get("score", 0), f"mem::{m['session_id']}"))

        items.sort(key=lambda x: x[0], reverse=True)
        retrieved = [eid for _, eid in items]
        recalls.append(compute_recall_at_k(retrieved, gold, 10))

    df = df.copy()
    df["recall_at_10"] = recalls
    return df


def _build_actual_turns_lookup() -> dict[str, int]:
    """Build session_id -> actual user-turn count from generated sessions."""
    turns_lookup: dict[str, int] = {}

    for path in sorted(MEMORY_COLLECTION_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            sessions = json.load(f)
        for s in sessions:
            sid = s.get("session_id")
            if sid and "session" in s:
                turns_lookup[sid] = len(s["session"]) // 2

    with open(BATCH_OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            sid = entry.get("custom_id")
            if not sid or sid in turns_lookup:
                continue
            try:
                output = entry["response"]["body"]["output"]
                text = output[0]["content"][0]["text"]
                session_data = json.loads(text)
                turns = session_data.get("turns", [])
                turns_lookup[sid] = len(turns) // 2
            except (KeyError, IndexError, json.JSONDecodeError):
                pass

    return turns_lookup


def add_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Add aggregated diversity features (legacy memory-subset CSV)."""
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    turns_lookup = _build_actual_turns_lookup()
    print(f"  Actual-turns lookup: {len(turns_lookup)} sessions indexed")

    diversity_lookup: dict[str, dict] = {}
    for cid, meta in manifest.items():
        div = meta.get("diversity", {})
        diversity_lookup[cid] = {
            "evidence_placement": div.get("evidence_placement"),
            "topic_drift": div.get("topic_drift"),
            "evidence_density": div.get("evidence_density"),
            "actual_turns": turns_lookup.get(cid),
        }

    records = []
    for _, row in df.iterrows():
        sids = _parse_list(row["golden_memory_session_ids"])
        matched = [diversity_lookup[sid] for sid in sids if sid in diversity_lookup]
        n = len(matched)
        if n == 0:
            records.append({k: np.nan for k in [
                "n_gold_sessions", "frac_late_placement", "frac_mid_placement",
                "frac_high_drift", "frac_multi_message",
                "max_actual_turns", "range_actual_turns",
            ]})
            continue

        actual_turn_vals = [r["actual_turns"] for r in matched if r["actual_turns"] is not None]
        if actual_turn_vals:
            max_turns = max(actual_turn_vals)
            range_turns = max(actual_turn_vals) - min(actual_turn_vals)
        else:
            max_turns = range_turns = np.nan
        records.append({
            "n_gold_sessions": n,
            "frac_late_placement": sum(1 for r in matched if r["evidence_placement"] == "late") / n,
            "frac_mid_placement": sum(1 for r in matched if r["evidence_placement"] == "middle") / n,
            "frac_high_drift": sum(1 for r in matched if r["topic_drift"] in ("moderate", "wide")) / n,
            "frac_multi_message": sum(1 for r in matched if r["evidence_density"] == "multi_message") / n,
            "max_actual_turns": max_turns,
            "range_actual_turns": range_turns,
        })

    div_df = pd.DataFrame(records, index=df.index)
    return pd.concat([df, div_df], axis=1)


def _load_questions() -> pd.DataFrame:
    questions = pd.read_pickle(QUESTIONS_PKL).reset_index(drop=True)
    if "block_id" not in questions.columns:
        questions["block_id"] = questions.index + 1
    return questions


def add_topical_relevance(df: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    out = label_topical_relevance(df, questions)
    counts = out["topical_relevance"].value_counts(dropna=False)
    print("  topical_relevance: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return out


def build_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per memory-only gold session x experimental run."""
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    turns_lookup = _build_actual_turns_lookup()
    audit = load_audit_map(_DEFAULT_AUDIT)
    pair_cache = PairIndexCache(load_session_turns(_DEFAULT_MEMORY))

    mem_only = df[df["dist"] == "memory_only"].copy()
    records: list[dict[str, Any]] = []
    for _, row in mem_only.iterrows():
        sids = [str(s).strip() for s in _parse_list(row.get("golden_memory_session_ids")) if str(s).strip()]
        retrieved_mem = _parse_retrieved(row.get("retrieved_memory"))
        for sid in sids:
            meta = manifest.get(sid, {})
            div = meta.get("diversity", {}) if isinstance(meta, dict) else {}
            entry = audit.get(sid)
            if entry is None:
                present = pd.NA
            else:
                present = bool(entry.all_required_present)
            ev = meta.get("evidence") if isinstance(meta, dict) else None
            category = None
            if isinstance(ev, dict):
                cat = ev.get("category")
                if isinstance(cat, str) and cat.strip():
                    category = cat.strip().lower()
            persona = row["persona"]
            records.append({
                "block_id": row["block_id"],
                "session_id": sid,
                "evidence_id": str(meta.get("evidence_id") or "").strip() or None,
                "persona": persona,
                "agent": row["agent"],
                "evidence_category": category,
                "evidence_in_domain": (
                    bool(evidence_in_domain(category, str(persona)))
                    if category
                    else pd.NA
                ),
                "retrieved_strict": int(strict_mem_hit(sid, retrieved_mem, audit, pair_cache)),
                "evidence_placement": div.get("evidence_placement"),
                "topic_drift": div.get("topic_drift"),
                "evidence_density": div.get("evidence_density"),
                "actual_turns": turns_lookup.get(sid),
                "all_required_present": present,
            })

    out = pd.DataFrame.from_records(records)
    n_incomplete = int((out["all_required_present"] == False).sum()) if len(out) else 0
    n_in = int((out["evidence_in_domain"] == True).sum()) if len(out) else 0
    n_out = int((out["evidence_in_domain"] == False).sum()) if len(out) else 0
    print(
        f"  Audit table: {len(out)} rows | "
        f"strict_hit={int(out['retrieved_strict'].sum()) if len(out) else 0} | "
        f"incomplete_rows={n_incomplete} | "
        f"evidence_in_domain: in={n_in}, out={n_out}"
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LABELED_RUNS)
    df = df[df["agent"].isin(["agent_1", "agent_2", "agent_3", "agent_4"])].reset_index(drop=True)
    print(f"Loaded {len(df)} rows ({df['agent'].nunique()} agents)")

    questions = _load_questions()
    df = add_topical_relevance(df, questions)

    df = add_recall(df)
    print(f"recall_at_10: mean={df['recall_at_10'].mean():.3f}, "
          f"non-NA={df['recall_at_10'].notna().sum()}")

    df = add_strict_recall(df)
    print(
        f"recall_full_context_strict: mean={df['recall_full_context_strict'].mean():.3f}, "
        f"non-NA={df['recall_full_context_strict'].notna().sum()}"
    )

    keep_full = [
        "question_idx", "block_id", "persona", "dist", "agent",
        "question_type", "hop_count", "correct", "topical_relevance",
        "recall_at_10", "recall_full_context_strict",
        "latency_s", "input_tokens", "total_tokens",
    ]
    df_full = df[[c for c in keep_full if c in df.columns]].copy()
    full_path = OUT_DIR / "model_features_full.csv"
    df_full.to_csv(full_path, index=False)
    print(f"\nFull-sample: {full_path} ({len(df_full)} rows)")

    df_mem = df[df["dist"].isin(["memory_only", "integrated"])].copy()
    df_mem = add_diversity(df_mem)
    df_mem = df_mem[df_mem["n_gold_sessions"].notna() & (df_mem["n_gold_sessions"] > 0)]

    keep_mem = keep_full + [
        "n_gold_sessions", "frac_late_placement", "frac_mid_placement",
        "frac_high_drift", "frac_multi_message",
        "max_actual_turns", "range_actual_turns",
    ]
    df_mem_out = df_mem[[c for c in keep_mem if c in df_mem.columns]].copy()
    mem_path = OUT_DIR / "model_features_memory.csv"
    df_mem_out.to_csv(mem_path, index=False)
    print(f"Memory-subset: {mem_path} ({len(df_mem_out)} rows)")

    audit = build_audit_table(df)
    audit_path = OUT_DIR / "model_features_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"Audit sessions: {audit_path} ({len(audit)} rows)")


if __name__ == "__main__":
    main()
