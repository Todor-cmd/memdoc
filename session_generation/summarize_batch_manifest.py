"""Summarise diversity and composition attributes per target persona from a batch manifest.

Usage:
    python -m session_generation.summarize_batch_manifest \\
        --manifest data/batch_jobs/experiment_sessions/batch_manifest_20260607_124012.json

    python -m session_generation.summarize_batch_manifest \\
        --job-dir data/batch_jobs/experiment_sessions
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from memory_curation.diversity_sampling import diversity_summary
from session_generation.persona import PERSONA_DIVERSITY_PROFILES

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_JOB_DIR = _REPO_ROOT / "data" / "batch_jobs" / "experiment_sessions"

DIVERSITY_ATTRS = (
    "scenario_category",
    "turn_range",
    "topic_drift",
    "evidence_placement",
    "evidence_density",
)

COMPOSITION_ATTRS = (
    "question_type",
    "evidence_category",
    "is_temporal",
    "is_off_topic",
)


def _resolve_manifest(path: Path | None, job_dir: Path) -> Path:
    if path is not None:
        p = path.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Manifest not found: {p}")
        return p

    job_dir = job_dir.expanduser().resolve()
    candidates = sorted(job_dir.glob("batch_manifest_*.json"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No batch_manifest_*.json under {job_dir}")
    return candidates[0]


def load_manifest_records(manifest_path: Path) -> list[dict]:
    with manifest_path.open(encoding="utf-8") as f:
        raw: dict[str, dict] = json.load(f)
    records: list[dict] = []
    for custom_id, meta in raw.items():
        rec = dict(meta)
        rec["custom_id"] = custom_id
        records.append(rec)
    return records


def _turn_range_label(val) -> str:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return f"{val[0]}-{val[1]}"
    return str(val)


def _counter_table(counter: Counter, total: int) -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []
    for label, count in counter.most_common():
        pct = 100.0 * count / total if total else 0.0
        rows.append((str(label), count, pct))
    return rows


def _print_counter_block(title: str, counter: Counter, total: int, indent: str = "  ") -> None:
    print(f"{indent}{title}:")
    if not counter:
        print(f"{indent}  (empty)")
        return
    width = max(len(label) for label in counter)
    for label, count, pct in _counter_table(counter, total):
        print(f"{indent}  {label:<{width}}  {count:>5}  ({pct:5.1f}%)")


def _extract_counter(records: list[dict], attr: str) -> Counter:
    c: Counter[str] = Counter()
    for rec in records:
        if attr in ("is_temporal", "is_off_topic"):
            c[str(bool(rec.get(attr)))] += 1
            continue
        if attr == "evidence_category":
            ev = rec.get("evidence") or {}
            c[str(ev.get("category") or "unknown")] += 1
            continue
        if attr == "turn_range":
            d = rec.get("diversity") or {}
            c[_turn_range_label(d.get("turn_range"))] += 1
            continue
        if attr in DIVERSITY_ATTRS:
            d = rec.get("diversity") or {}
            c[str(d.get(attr) or "unknown")] += 1
            continue
        c[str(rec.get(attr) or "unknown")] += 1
    return c


def _profile_targets(persona_id: str) -> dict[str, object]:
    profile = PERSONA_DIVERSITY_PROFILES.get(persona_id) or {}
    out: dict[str, object] = {}
    if "scenario_category_weights" in profile:
        out["scenario_category"] = profile["scenario_category_weights"]
    if "turn_range_weights" in profile:
        from session_generation.persona import TURN_RANGES

        weights = profile["turn_range_weights"]
        out["turn_range"] = {
            f"{a}-{b}": weights[i]
            for i, (a, b) in enumerate(TURN_RANGES)
            if i < len(weights)
        }
    return out


def summarize_by_persona(records: list[dict]) -> dict[str, list[dict]]:
    by_persona: dict[str, list[dict]] = {}
    for rec in records:
        pid = str(rec.get("target_persona_id") or "unknown")
        by_persona.setdefault(pid, []).append(rec)
    return dict(sorted(by_persona.items()))


def print_summary(manifest_path: Path, records: list[dict]) -> None:
    print(f"Manifest: {manifest_path}")
    print(f"Total batch requests: {len(records)}")

    n_questions = len({rec.get("source_question_id") for rec in records})
    n_evidence = len({rec.get("evidence_id") for rec in records})
    print(f"Unique source questions: {n_questions}")
    print(f"Unique evidence items: {n_evidence}")

    by_persona = summarize_by_persona(records)

    for persona_id, subset in by_persona.items():
        n = len(subset)
        print(f"\n{'=' * 72}")
        print(f"{persona_id}  ({n} sessions)")
        print(f"{'=' * 72}")

        div = diversity_summary(subset)
        for attr in DIVERSITY_ATTRS:
            if attr == "turn_range":
                counter = _extract_counter(subset, attr)
            elif attr == "scenario_category":
                counter = div["scenario_category"]
            elif attr == "topic_drift":
                counter = div["topic_drift"]
            else:
                counter = _extract_counter(subset, attr)
            _print_counter_block(attr, counter, n)

            targets = _profile_targets(persona_id)
            if attr in targets:
                print(f"    target weights: {targets[attr]}")

        print()
        for attr in COMPOSITION_ATTRS:
            _print_counter_block(attr, _extract_counter(subset, attr), n)

        orig = Counter(str(rec.get("original_persona_id") or "unknown") for rec in subset)
        _print_counter_block("original_persona_id", orig, n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise batch manifest diversity attributes per target persona."
    )
    parser.add_argument(
        "--manifest", "-m", type=Path, default=None,
        help="Path to batch_manifest_*.json (default: newest under --job-dir)",
    )
    parser.add_argument(
        "--job-dir", type=Path, default=_DEFAULT_JOB_DIR,
        help="Directory to search for manifests when --manifest is omitted",
    )
    args = parser.parse_args()

    manifest_path = _resolve_manifest(args.manifest, args.job_dir)
    records = load_manifest_records(manifest_path)
    print_summary(manifest_path, records)


if __name__ == "__main__":
    main()
