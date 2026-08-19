"""CLI entry point: run one agent's portion of the D-optimal design matrix.

Usage:
    python -m experiment.run_agent --agent agent_1
    python -m experiment.run_agent --agent agent_3 --output runs/agent_3.jsonl
    python -m experiment.run_agent --agent agent_1 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.agent_registry import available_agents, get_agent_spec
from experiment.design_runner import (
    default_design_output_path,
    filter_design_for_agent,
    load_design_matrix,
    run_design_experiment,
)
from experiment.paths import (
    DEFAULT_EXPERIMENT_DESIGN_CSV,
    DEFAULT_EXPERIMENT_OUTPUT_DIR,
    DEFAULT_EXPERIMENT_QUESTIONS_PICKLE,
)
from experiment.store_hooks import NoOpStoreHooks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one agent through its design-matrix assignments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=available_agents(),
        help="Agent identifier from the design matrix.",
    )
    parser.add_argument(
        "--design-csv",
        type=Path,
        default=DEFAULT_EXPERIMENT_DESIGN_CSV,
        help="Path to experiment_design.csv.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_EXPERIMENT_QUESTIONS_PICKLE,
        help="Path to experiment questions pickle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (.jsonl or .csv). Auto-generated if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_OUTPUT_DIR,
        help="Output directory (used when --output is not specified).",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=5,
        help="Flush output every N rows.",
    )
    parser.add_argument(
        "--strict-integrated-strip",
        action="store_true",
        help="Strip document-channel evidence sessions from memory in integrated variant.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions (for quick sanity checks).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use NoOpStoreHooks and dummy answer_fn (no model calls).",
    )
    return parser.parse_args()


DESIGN_AGENT_ALIAS: dict[str, str] = {
    "agent_3_v2": "agent_3",
}


def main() -> None:
    args = parse_args()

    print(f"Loading design matrix from {args.design_csv} …")
    design = load_design_matrix(args.design_csv, args.questions)
    print(f"  Total design rows: {len(design)}")

    design_agent = DESIGN_AGENT_ALIAS.get(args.agent, args.agent)
    agent_design = filter_design_for_agent(design, design_agent)
    if design_agent != args.agent:
        print(f"  (using design rows for {design_agent!r} with {args.agent!r} architecture)")
    if args.limit:
        agent_design = agent_design.head(args.limit)
    print(f"  Rows for {args.agent}: {len(agent_design)}")
    print(f"  Variants: {sorted(agent_design['dist'].unique())}")
    print(f"  Personas: {sorted(agent_design['persona'].unique())}")
    print()

    if args.dry_run:
        print("DRY RUN — using NoOpStoreHooks + dummy answer_fn\n")

        def hooks_factory() -> NoOpStoreHooks:
            return NoOpStoreHooks()

        def answer_fn(query: str) -> str:
            return "[DRY_RUN]"
    else:
        spec = get_agent_spec(args.agent)
        print(f"Building agent: {spec.agent_class} from {spec.agent_module}")
        print(f"  kwargs: {spec.kwargs}\n")
        agent = spec.build()
        hooks_factory = agent.hooks_factory
        answer_fn = agent.answer_fn

    output_path = args.output or default_design_output_path(
        args.output_dir, args.agent
    )

    results = run_design_experiment(
        agent_design,
        args.agent,
        hooks_factory,
        answer_fn,
        output_path,
        flush_every=args.flush_every,
        strict_integrated_memory_strip=args.strict_integrated_strip,
    )

    # Summary
    print(f"\nResults summary:")
    print(f"  Total inferences: {len(results)}")
    by_variant = {}
    for r in results:
        by_variant[r["variant"]] = by_variant.get(r["variant"], 0) + 1
    print(f"  By variant: {by_variant}")


if __name__ == "__main__":
    main()
