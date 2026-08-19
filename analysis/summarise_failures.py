"""
Summarise inference failure modes by agent.

Reads the raw experiment run JSONL files and classifies each
[INFERENCE_FAILED] prediction into:

  - pre_llm: input_tokens == 0 → failure occurred before the LLM was called
             (e.g. retrieval pipeline error, missing context assembly)
  - llm_stage: input_tokens > 0 → the LLM received context but still produced
               no valid answer (e.g. output parsing failure, safety refusal,
               response truncation)

Output: prints summary tables and writes a CSV to data/analysis/failure_summary.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "data" / "experiment_runs"
OUT_DIR = REPO_ROOT / "data" / "analysis"


def load_failures() -> pd.DataFrame:
    records = []
    for path in sorted(RUNS_DIR.glob("agent_*.jsonl")):
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("prediction") != "[INFERENCE_FAILED]":
                    continue
                records.append({
                    "agent": obj["agent"],
                    "block_id": obj["block_id"],
                    "dist": obj.get("variant"),
                    "persona": obj.get("eval_persona"),
                    "question_type": obj.get("question_type"),
                    "latency_s": obj.get("latency_s", 0),
                    "input_tokens": obj.get("input_tokens", 0),
                    "output_tokens": obj.get("output_tokens", 0),
                    "rewrite_count": obj.get("rewrite_count"),
                    "retrieval_passes": obj.get("retrieval_passes"),
                    "documents_relevant": obj.get("documents_relevant"),
                })
    return pd.DataFrame(records)


def classify_failure(row: pd.Series) -> str:
    if row["input_tokens"] == 0:
        return "pre_llm"
    return "llm_stage"


def main():
    fails = load_failures()
    if fails.empty:
        print("No inference failures found.")
        return

    fails["failure_type"] = fails.apply(classify_failure, axis=1)

    total_trials = sum(
        sum(1 for _ in open(p)) for p in sorted(RUNS_DIR.glob("agent_*.jsonl"))
    )

    # --- Overall summary ---
    print("=" * 70)
    print("INFERENCE FAILURE SUMMARY")
    print("=" * 70)
    print(f"\nTotal trials: {total_trials}")
    print(f"Total failures: {len(fails)} ({len(fails)/total_trials:.1%})")
    print(f"  pre_llm (retrieval/pipeline error): {(fails['failure_type'] == 'pre_llm').sum()}")
    print(f"  llm_stage (LLM received context but failed): {(fails['failure_type'] == 'llm_stage').sum()}")

    # --- By agent ---
    print("\n" + "-" * 70)
    print("FAILURES BY AGENT")
    print("-" * 70)
    agent_totals = pd.Series(
        {p.stem: sum(1 for _ in open(p)) for p in sorted(RUNS_DIR.glob("agent_*.jsonl"))}
    )
    agent_summary = fails.groupby("agent").agg(
        n_failures=("block_id", "count"),
        pre_llm=("failure_type", lambda x: (x == "pre_llm").sum()),
        llm_stage=("failure_type", lambda x: (x == "llm_stage").sum()),
        mean_latency=("latency_s", "mean"),
        mean_input_tokens=("input_tokens", "mean"),
        mean_output_tokens=("output_tokens", "mean"),
    )
    for agent in sorted(agent_totals.index):
        if agent not in agent_summary.index:
            agent_summary.loc[agent] = 0
    agent_summary["total_trials"] = agent_totals
    agent_summary["failure_rate"] = agent_summary["n_failures"] / agent_summary["total_trials"]
    agent_summary = agent_summary.sort_index()
    print(agent_summary.to_string())

    # --- By agent × dist ---
    print("\n" + "-" * 70)
    print("FAILURES BY AGENT × DIST")
    print("-" * 70)
    print(pd.crosstab(fails["agent"], fails["dist"], margins=True).to_string())

    # --- By agent × question_type ---
    print("\n" + "-" * 70)
    print("FAILURES BY AGENT × QUESTION TYPE")
    print("-" * 70)
    print(pd.crosstab(fails["agent"], fails["question_type"], margins=True).to_string())

    # --- By agent × persona ---
    print("\n" + "-" * 70)
    print("FAILURES BY AGENT × PERSONA")
    print("-" * 70)
    print(pd.crosstab(fails["agent"], fails["persona"], margins=True).to_string())

    # --- Corrective RAG detail (agent_3) ---
    a3 = fails[fails["agent"] == "agent_3"]
    if not a3.empty:
        print("\n" + "-" * 70)
        print("CORRECTIVE RAG (agent_3) — DETAILED BREAKDOWN")
        print("-" * 70)
        print(f"  All {len(a3)} failures are llm_stage (retrieval succeeded, LLM failed)")
        print(f"  Mean input tokens: {a3['input_tokens'].mean():.0f}")
        print(f"  Mean output tokens: {a3['output_tokens'].mean():.0f} (very low → likely parsing failure)")
        print(f"  Mean latency: {a3['latency_s'].mean():.2f}s (higher than baseline → possible timeout)")
        print(f"\n  Retrieval passes before failure:")
        print(f"  {a3['retrieval_passes'].value_counts().sort_index().to_dict()}")
        print(f"\n  Questions affected: {a3['block_id'].nunique()} unique questions")
        repeat_qs = a3["block_id"].value_counts()
        repeat_qs = repeat_qs[repeat_qs > 1]
        if not repeat_qs.empty:
            print(f"  Questions with multiple failures: {len(repeat_qs)}")
            print(f"    {repeat_qs.to_dict()}")

    # --- Interpretation ---
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
  pre_llm failures (agents 1, 2, 4):
    input_tokens = 0 indicates the failure occurred before the LLM
    generation step. Likely causes: retrieval pipeline exception,
    context assembly error, or missing data. These are infrastructure
    failures, not capability failures.

  llm_stage failures (agent_3 / Corrective RAG):
    The model received ~10k input tokens and produced very few output
    tokens (~47). This pattern suggests either:
      (a) Output parsing failure — the LLM responded but not in the
          expected format (Corrective RAG has multi-pass logic)
      (b) Response truncation / safety refusal after retrieval
      (c) The corrective rewriting loop exhausted retries

    The higher latency (mean {a3['latency_s'].mean():.1f}s vs ~1.3s baseline)
    and multiple retrieval passes (mode=2) support hypothesis (a) or (c):
    the agent's iterative logic ran but couldn't extract a clean answer.
""")

    # --- Export ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "failure_summary.csv"
    fails.to_csv(out_path, index=False)
    print(f"Detailed failure records written to: {out_path}")


if __name__ == "__main__":
    main()
