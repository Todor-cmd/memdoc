"""Build a batch job to generate evidence sessions for corpus shortfall.

Reads primary + supplement batch inventories, identifies per-persona
in-domain / off-topic gaps vs the 500-session corpus targets, then samples
off-domain evidence from ``full_reasonable.pkl`` and emits batch JSONL.

Usage:
    python -m memory_curation.build_supplement_batch --inventory-only
    python -m memory_curation.build_supplement_batch --domain off_topic --persona persona_3
    python -m memory_curation.build_supplement_batch --no-submit
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from memory_curation.session_pools import (  # noqa: E402
    group_records_by_persona,
    inventory_persona_pools,
    is_in_domain_for_persona,
    load_valid_batch_records,
)
from prepare_data.question_evidence_ids import (  # noqa: E402
    iter_evidence_dicts_from_row,
    scoped_evidence_id,
    source_question_id_for_row,
)
from session_generation.create_batch_job import (  # noqa: E402
    build_batch_requests,
    load_persona_map_by_query,
    submit_batch,
)
from session_generation.persona import PERSONA_DOMAINS  # noqa: E402

_DATA = REPO_ROOT / "data"
DEFAULT_PRIMARY_MANIFEST = _DATA / "batch_jobs/experiment_sessions/batch_manifest.json"
DEFAULT_PRIMARY_JSONL = _DATA / "batch_jobs/experiment_sessions/batch_output.jsonl"
DEFAULT_SUPPLEMENT_MANIFEST = _DATA / "batch_jobs/reasonable_sessions/batch_manifest_20260424_153151.json"
DEFAULT_SUPPLEMENT_JSONL = _DATA / "batch_jobs/reasonable_sessions/batch_output.jsonl"
DEFAULT_QUESTIONS = _DATA / "questions" / "full_reasonable.pkl"
DEFAULT_PERSONA_CSV = _DATA / "persona_metadata" / "q_2_personas.csv"
DEFAULT_OUT_DIR = _DATA / "batch_jobs" / "corpus_supplement"

TARGET_PERSONAS = ["persona_1", "persona_2", "persona_3"]
TOTAL_SESSIONS = 500
IN_DOMAIN_RATIO = 0.75


def _existing_evidence_for_persona(
    primary_by_persona: dict[str, list[dict]],
    supplement_by_persona: dict[str, list[dict]],
    persona_id: str,
) -> set[str]:
    out: set[str] = set()
    for pool in (
        primary_by_persona.get(persona_id, []),
        supplement_by_persona.get(persona_id, []),
    ):
        for rec in pool:
            eid = rec.get("evidence_id")
            if eid:
                out.add(str(eid).strip())
    return out


def _evidence_matches_domain(evidence: dict, persona_id: str, want_in_domain: bool) -> bool:
    cat = evidence.get("category", "")
    domains = PERSONA_DOMAINS.get(persona_id, frozenset())
    if not isinstance(cat, str) or not cat.strip():
        return not want_in_domain
    in_dom = cat.strip().lower() in domains
    return in_dom if want_in_domain else not in_dom


def _collect_shortfall_evidence(
    df: pd.DataFrame,
    persona_id: str,
    *,
    want_in_domain: bool,
    n_needed: int,
    exclude_eids: set[str],
    rng: random.Random,
) -> list[tuple[pd.Series, dict]]:
    """Return up to *n_needed* (row, evidence) pairs for batch generation."""
    candidates: list[tuple[pd.Series, dict]] = []
    for q_idx, row in df.iterrows():
        sid = source_question_id_for_row(row, q_idx)
        for ev in iter_evidence_dicts_from_row(row):
            if not isinstance(ev, dict):
                continue
            eid = ev.get("evidence_id")
            if not eid:
                eid = scoped_evidence_id(sid, ev)
                ev = dict(ev)
                ev["evidence_id"] = eid
            key = str(eid).strip()
            if key in exclude_eids:
                continue
            if not _evidence_matches_domain(ev, persona_id, want_in_domain):
                continue
            candidates.append((row, ev))

    rng.shuffle(candidates)
    return candidates[:n_needed]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate corpus supplement sessions.")
    parser.add_argument("--primary-manifest", type=Path, default=DEFAULT_PRIMARY_MANIFEST)
    parser.add_argument("--primary-output-jsonl", type=Path, default=DEFAULT_PRIMARY_JSONL)
    parser.add_argument("--supplement-manifest", type=Path, default=DEFAULT_SUPPLEMENT_MANIFEST)
    parser.add_argument("--supplement-output-jsonl", type=Path, default=DEFAULT_SUPPLEMENT_JSONL)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--persona-csv", type=Path, default=DEFAULT_PERSONA_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--persona", choices=TARGET_PERSONAS + ["all"], default="all")
    parser.add_argument(
        "--domain", choices=["in_domain", "off_topic", "both"], default="both",
        help="Which shortfall type to fill",
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--no-submit", action="store_true",
                        help="Write JSONL + manifest only; do not call OpenAI API")
    args = parser.parse_args()

    n_in = int(TOTAL_SESSIONS * IN_DOMAIN_RATIO)
    n_out = TOTAL_SESSIONS - n_in

    primary_by_persona: dict[str, list[dict]] = {pid: [] for pid in TARGET_PERSONAS}
    supplement_by_persona: dict[str, list[dict]] = {pid: [] for pid in TARGET_PERSONAS}

    p_man = args.primary_manifest.expanduser().resolve()
    p_jsonl = args.primary_output_jsonl.expanduser().resolve()
    if p_man.is_file() and p_jsonl.is_file():
        primary_records = load_valid_batch_records(p_man, p_jsonl)
        primary_by_persona = group_records_by_persona(primary_records)

    s_man = args.supplement_manifest.expanduser().resolve()
    s_jsonl = args.supplement_output_jsonl.expanduser().resolve()
    if s_man.is_file() and s_jsonl.is_file():
        supplement_records = load_valid_batch_records(s_man, s_jsonl)
        supplement_by_persona = group_records_by_persona(supplement_records)

    personas = TARGET_PERSONAS if args.persona == "all" else [args.persona]
    print("Shortfall inventory:")
    shortfalls: dict[str, dict[str, int]] = {}
    for pid in personas:
        inv = inventory_persona_pools(
            pid, primary_by_persona.get(pid, []), supplement_by_persona.get(pid, []),
            target_in_domain=n_in, target_off_topic=n_out,
        )
        shortfalls[pid] = {
            "in_domain": inv.in_domain_shortfall,
            "off_topic": inv.off_topic_shortfall,
        }
        print(f"  {pid}: in-domain shortfall={inv.in_domain_shortfall}  off-topic={inv.off_topic_shortfall}")

    if args.inventory_only:
        return

    if not args.questions.is_file():
        raise FileNotFoundError(f"Questions pickle not found: {args.questions}")

    df = pd.read_pickle(args.questions)
    persona_by_query = load_persona_map_by_query(args.persona_csv)
    rng = random.Random(args.seed)

    requests: list[dict] = []
    manifest: dict[str, dict] = {}

    for pid in personas:
        exclude_eids = _existing_evidence_for_persona(
            primary_by_persona, supplement_by_persona, pid
        )
        sf = shortfalls[pid]
        for label, need in (("in_domain", sf["in_domain"]), ("off_topic", sf["off_topic"])):
            if args.domain != "both" and args.domain != label:
                continue
            if need <= 0:
                continue
            want_in = label == "in_domain"
            pairs = _collect_shortfall_evidence(
                df, pid, want_in_domain=want_in, n_needed=need * 3,
                exclude_eids=exclude_eids, rng=rng,
            )
            print(f"  {pid} {label}: {len(pairs)} candidates for shortfall={need}")
            rows_for_batch: list[pd.Series] = []
            for row, ev in pairs[:need]:
                row_copy = row.copy()
                row_copy["evidence_list"] = []
                row_copy["memory_evidence"] = []
                row_copy["golden_memory_evidence"] = [ev]
                row_copy["golden_document_evidence"] = []
                rows_for_batch.append(row_copy)
                eid = str(ev.get("evidence_id", "")).strip()
                if eid:
                    exclude_eids.add(eid)

            if not rows_for_batch:
                continue
            batch_df = pd.DataFrame(rows_for_batch).reset_index(drop=True)
            reqs, man = build_batch_requests(
                batch_df,
                args.model,
                seed=args.seed,
                persona_by_query=persona_by_query,
                target_personas=[pid],
            )
            requests.extend(reqs)
            manifest.update(man)

    if not requests:
        print("No shortfall to fill — pools are sufficient.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.no_submit:
        import os
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        jsonl_path = args.out_dir / f"batch_input_{timestamp}.jsonl"
        manifest_path = args.out_dir / f"batch_manifest_{timestamp}.json"
        with open(jsonl_path, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote {len(requests)} requests → {jsonl_path}")
        print(f"Manifest → {manifest_path}")
    else:
        submit_batch(requests, manifest, str(args.out_dir))


if __name__ == "__main__":
    main()
