"""Prepare diversity-augmented dataset for the sensitivity GLMM.

Joins session-generation diversity attributes from the batch manifest to
labeled_runs.csv, producing a CSV with run-level aggregated features.

Only memory_only and integrated rows are included (document_only has no
generated sessions).

Output: data/analysis/labeled_runs_with_diversity.csv
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    REPO_ROOT / "data" / "batch_jobs" / "experiment_sessions"
    / "batch_manifest_20260607_145849.json"
)
LABELED_RUNS_PATH = REPO_ROOT / "data" / "experiment_runs" / "labeled_runs.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "analysis" / "labeled_runs_with_diversity.csv"


def _parse_session_ids(val) -> list[str]:
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def main() -> None:
    # Load manifest
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Manifest entries: {len(manifest)}")

    # Build diversity lookup
    diversity_lookup: dict[str, dict] = {}
    for cid, meta in manifest.items():
        div = meta.get("diversity", {})
        diversity_lookup[cid] = {
            "scenario_category": div.get("scenario_category"),
            "turn_range_min": div["turn_range"][0] if "turn_range" in div else None,
            "turn_range_max": div["turn_range"][1] if "turn_range" in div else None,
            "evidence_placement": div.get("evidence_placement"),
            "topic_drift": div.get("topic_drift"),
            "evidence_density": div.get("evidence_density"),
        }

    # Load labeled runs, filter to memory variants
    df = pd.read_csv(LABELED_RUNS_PATH)
    df_mem = df[df["dist"].isin(["memory_only", "integrated"])].copy()
    print(f"Memory-variant rows: {len(df_mem)}")

    df_mem["_gold_session_ids"] = df_mem["golden_memory_session_ids"].apply(
        _parse_session_ids
    )

    # Aggregate diversity features per run
    records = []
    for idx, row in df_mem.iterrows():
        sids = row["_gold_session_ids"]
        matched = [diversity_lookup[sid] for sid in sids if sid in diversity_lookup]
        n = len(matched)
        if n == 0:
            records.append({
                "n_gold_sessions": 0,
                "mean_turn_range_min": np.nan,
                "mean_turn_range_max": np.nan,
                "max_turn_range_max": np.nan,
                "frac_late_placement": np.nan,
                "frac_mid_placement": np.nan,
                "frac_high_drift": np.nan,
                "frac_wide_drift": np.nan,
                "frac_multi_message": np.nan,
                "frac_professional": np.nan,
                "frac_casual": np.nan,
                "frac_transactional": np.nan,
            })
            continue

        records.append({
            "n_gold_sessions": n,
            "mean_turn_range_min": np.mean([r["turn_range_min"] for r in matched if r["turn_range_min"] is not None]),
            "mean_turn_range_max": np.mean([r["turn_range_max"] for r in matched if r["turn_range_max"] is not None]),
            "max_turn_range_max": max((r["turn_range_max"] for r in matched if r["turn_range_max"] is not None), default=np.nan),
            "frac_late_placement": sum(1 for r in matched if r["evidence_placement"] == "late") / n,
            "frac_mid_placement": sum(1 for r in matched if r["evidence_placement"] == "middle") / n,
            "frac_high_drift": sum(1 for r in matched if r["topic_drift"] in ("moderate", "wide")) / n,
            "frac_wide_drift": sum(1 for r in matched if r["topic_drift"] == "wide") / n,
            "frac_multi_message": sum(1 for r in matched if r["evidence_density"] == "multi_message") / n,
            "frac_professional": sum(1 for r in matched if r["scenario_category"] == "professional") / n,
            "frac_casual": sum(1 for r in matched if r["scenario_category"] == "casual") / n,
            "frac_transactional": sum(1 for r in matched if r["scenario_category"] == "transactional") / n,
        })

    div_df = pd.DataFrame(records, index=df_mem.index)
    df_out = pd.concat([df_mem.drop(columns=["_gold_session_ids"]), div_df], axis=1)

    # Drop rows with no diversity data
    n_before = len(df_out)
    df_out = df_out[df_out["n_gold_sessions"] > 0]
    print(f"Rows with diversity data: {len(df_out)} / {n_before}")

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"Written: {OUTPUT_PATH}")
    print(f"Columns: {list(df_out.columns)}")


if __name__ == "__main__":
    main()
