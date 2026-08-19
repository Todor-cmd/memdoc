from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import random as random_module
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import OpenAI

from session_generation.prompts import (
    DIALOGUE_STYLE_GUIDANCE,
    GENERATION_SYSTEM,
    GENERATION_SYSTEM_TEMPORAL,
    GENERATION_HUMAN_VARIANTS,
    TOPIC_CONTEXT_OFF_TOPIC,
    render_conversation_naturalism,
    render_structural_guidelines,
)
from session_generation.persona import (
    SCENARIOS,
    TURN_RANGES,
    TURN_RANGE_WEIGHTS,
    EVIDENCE_PLACEMENTS,
    TOPIC_DRIFT_LEVELS,
    TOPIC_DRIFT_WEIGHTS_BY_TURN_RANGE,
    EVIDENCE_DENSITIES,
    TRANSACTIONAL_CONSTRAINTS,
    TRANSACTIONAL_TURN_RANGE_WEIGHTS,
    TRANSACTIONAL_TOPIC_DRIFT_WEIGHTS,
    persona_dict_for_id,
    diversity_profile_for_persona,
    is_off_topic_for_persona,
)
from prepare_data.question_evidence_ids import source_question_id_for_row
from session_generation.schemas import Session
import dotenv

dotenv.load_dotenv()


def _make_strict(schema: dict) -> dict:
    """Recursively add additionalProperties: false for OpenAI strict mode."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            out[k] = _make_strict(v)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
        return out
    if isinstance(schema, list):
        return [_make_strict(item) for item in schema]
    return schema


SESSION_SCHEMA = _make_strict(Session.model_json_schema())

_DEFAULT_PERSONA_CSV = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "persona_metadata"
    / "q_2_personas.csv"
)


def _normalize_query_text(q) -> str:
    """Stable key for matching rows to persona CSV (same question text across splits)."""
    if q is None or (isinstance(q, float) and pd.isna(q)):
        raise ValueError("query is missing or NaN")
    s = str(q).strip()
    if not s:
        raise ValueError("query is empty after strip")
    return s


def load_persona_map_by_query(csv_path: os.PathLike[str] | str) -> dict[str, str]:
    """Load normalized ``query`` → ``persona`` id from a CSV (e.g. ``q_2_personas.csv``).

    Personas are assigned from the question text; ``question_idx`` order does not
    matter. Requires ``query`` and ``persona`` columns.
    """
    path = Path(csv_path)
    try:
        df = pd.read_csv(path, usecols=["query", "persona"])
    except ValueError as e:
        raise ValueError(
            f"Persona CSV must include columns 'query' and 'persona': {path}"
        ) from e
    out: dict[str, str] = {}
    for q, p in zip(df["query"], df["persona"]):
        if q is None or (isinstance(q, float) and pd.isna(q)):
            continue
        key = _normalize_query_text(q)
        pid = str(p).strip()
        if key in out and out[key] != pid:
            raise ValueError(
                f"Persona map assigns two different personas to the same query "
                f"(after strip). First bytes: {key[:120]!r}..."
            )
        out[key] = pid
    if not out:
        raise ValueError(f"No valid query → persona rows in {path}")
    return out


def _query_fingerprint(normalized_query: str) -> str:
    """Short stable id for batch ``custom_id`` (independent of dataframe row order)."""
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]


def _parse_evidence_cell(val) -> list:
    """Normalize a cell to a list (JSON string, list, ndarray, or empty)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if callable(getattr(val, "tolist", None)) and not isinstance(
        val, (dict, str, bytes)
    ):
        val = val.tolist()
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        return json.loads(s)
    return []


def _all_evidence_for_row(row: pd.Series) -> list[dict]:
    """Union memory + corpus evidence for MultiHop-style or inference-export frames."""
    out: list[dict] = []
    for col in (
        "memory_evidence",
        "evidence_list",
        "golden_memory_evidence",
        "golden_document_evidence",
    ):
        if col not in row.index:
            continue
        for ev in _parse_evidence_cell(row[col]):
            if isinstance(ev, dict):
                out.append(ev)
    return out


def _row_answer(row: pd.Series):
    """Prefer ``answer``, then ``final_answer``, then ``gold_answer``."""
    for col in ("answer", "final_answer", "gold_answer"):
        if col not in row.index:
            continue
        v = row[col]
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return v
    raise KeyError(
        "No answer column found; expected one of: answer, final_answer, gold_answer"
    )


def _sample_diversity_params(rng: random_module.Random, profile: dict | None = None) -> dict:
    """Sample diversity dimensions using per-persona profile or global defaults.

    Parameters
    ----------
    rng : random.Random
        Seeded RNG instance.
    profile : dict | None
        Per-persona diversity profile from ``PERSONA_DIVERSITY_PROFILES``.
        When None, uses global weights (backward compatible with existing runs).
    """
    # Scenario selection: use profile category weights if available
    if profile and "scenario_category_weights" in profile:
        cat_weights = profile["scenario_category_weights"]
        categories = list(cat_weights.keys())
        weights = [cat_weights[c] for c in categories]
        chosen_cat = rng.choices(categories, weights=weights, k=1)[0]
        eligible = [s for s in SCENARIOS if s["category"] == chosen_cat]
        scenario = rng.choice(eligible)
    else:
        scenario = rng.choice(SCENARIOS)

    is_transactional = scenario["category"] == "transactional"

    # Turn range weights
    tr_weights = TURN_RANGE_WEIGHTS
    if profile and "turn_range_weights" in profile and not is_transactional:
        tr_weights = profile["turn_range_weights"]

    # Topic drift weights by turn range
    td_weights_map = TOPIC_DRIFT_WEIGHTS_BY_TURN_RANGE
    if profile and "topic_drift_weights_by_turn_range" in profile and not is_transactional:
        td_weights_map = profile["topic_drift_weights_by_turn_range"]

    if is_transactional:
        turn_range = rng.choices(
            TRANSACTIONAL_CONSTRAINTS["turn_ranges"],
            weights=TRANSACTIONAL_TURN_RANGE_WEIGHTS,
            k=1,
        )[0]
        evidence_placement = rng.choice(TRANSACTIONAL_CONSTRAINTS["evidence_placements"])
        topic_drift = rng.choices(
            TRANSACTIONAL_CONSTRAINTS["topic_drifts"],
            weights=TRANSACTIONAL_TOPIC_DRIFT_WEIGHTS,
            k=1,
        )[0]
        evidence_density = rng.choice(TRANSACTIONAL_CONSTRAINTS["evidence_densities"])
    else:
        turn_range = rng.choices(TURN_RANGES, weights=tr_weights, k=1)[0]
        evidence_placement = rng.choice(list(EVIDENCE_PLACEMENTS))
        topic_drift = rng.choices(
            TOPIC_DRIFT_LEVELS,
            weights=td_weights_map[turn_range],
            k=1,
        )[0]
        evidence_density = rng.choice(list(EVIDENCE_DENSITIES))

    human_message = rng.choice(GENERATION_HUMAN_VARIANTS)

    return {
        "scenario": scenario["description"],
        "scenario_category": scenario["category"],
        "turn_range": list(turn_range),
        "evidence_placement": evidence_placement,
        "topic_drift": topic_drift,
        "evidence_density": evidence_density,
        "human_message": human_message,
    }


def format_generation_system_prompt(
    persona_character: str,
    evidence: dict,
    diversity: dict,
    *,
    is_temporal: bool,
    is_off_topic: bool = False,
) -> str:
    """Build the session-generation system prompt (same text as each batch request).

    Uses ``GENERATION_SYSTEM`` / ``GENERATION_SYSTEM_TEMPORAL`` and the same
    structural + naturalism rendering as ``build_batch_requests``.

    When ``is_off_topic`` is True, the TOPIC CONTEXT block is appended after the
    scenario to give the model framing for reconciling any scenario with
    out-of-domain evidence.
    """
    template = GENERATION_SYSTEM_TEMPORAL if is_temporal else GENERATION_SYSTEM
    structural_guidelines = render_structural_guidelines(
        turn_range=tuple(diversity["turn_range"]),
        evidence_placement=diversity["evidence_placement"],
        evidence_density=diversity["evidence_density"],
        is_temporal=is_temporal,
    )
    conversation_naturalism = render_conversation_naturalism(
        diversity["topic_drift"]
    )
    fmt_kwargs = dict(
        persona_summary=persona_character,
        dialogue_style_guidance=DIALOGUE_STYLE_GUIDANCE,
        title=evidence["title"],
        source=evidence["source"],
        fact=evidence["fact"],
        scenario=diversity["scenario"],
        conversation_naturalism=conversation_naturalism,
        structural_guidelines=structural_guidelines,
    )
    if is_temporal:
        dt = datetime.fromisoformat(evidence["published_at"])
        fmt_kwargs["published_at"] = dt.strftime("%B %d, %Y at %I:%M %p UTC")

    prompt = template.format(**fmt_kwargs)
    if is_off_topic:
        prompt += TOPIC_CONTEXT_OFF_TOPIC
    return prompt


_DEFAULT_TARGET_PERSONAS = ["persona_1", "persona_2", "persona_3"]


def build_batch_requests(
    questions_df,
    model_name,
    question_limit=None,
    seed=42,
    *,
    persona_by_query: dict[str, str],
    target_personas: list[str] | None = None,
):
    """Build Batch API request lines and a traceability manifest.

    Parameters
    ----------
    questions_df : pd.DataFrame
    model_name : str
    question_limit : int | None
        Maximum number of questions to process (all evidence items within
        each selected question are included). None means all questions.
    seed : int
        RNG seed for reproducible diversity sampling.
    persona_by_query : dict[str, str]
        Maps **normalized** ``query`` text (``str(query).strip()``) to a persona
        id such as ``persona_1``. Built from ``q_2_personas.csv`` (same question
        string as in the pickle, regardless of row order or ``question_idx``).
        Every processed row's query must appear in this map (no default persona).
        Rows with no evidence in ``memory_evidence`` / ``evidence_list`` /
        ``golden_*`` columns are skipped (e.g. ``null_query``).
    target_personas : list[str] | None
        Personas to generate sessions for per evidence item. When None, defaults
        to ``["persona_1", "persona_2", "persona_3"]`` (cross-persona design).
        Set to a single-element list to revert to the legacy single-persona mode.

    Returns
    -------
    requests : list[dict]
        One JSONL-ready dict per (evidence_item, target_persona) pair, targeting
        /v1/responses with structured output (Session schema).
    manifest : dict[str, dict]
        Maps each ``custom_id`` to ``source_question_id``, ``evidence_id``,
        ``evidence``, diversity params, ``is_temporal``, persona id and character
        snapshot, off-topic flag, and query fields for merging with batch output.

    Prints ``Total batch requests`` and per-``persona_id`` question counts
    (including ``tbd``) before returning.
    """
    if target_personas is None:
        target_personas = list(_DEFAULT_TARGET_PERSONAS)

    if question_limit is not None:
        questions_df = questions_df.head(question_limit)

    normalized_queries: list[str] = []
    for _, row in questions_df.iterrows():
        normalized_queries.append(_normalize_query_text(row["query"]))
    missing = [q for q in normalized_queries if q not in persona_by_query]
    if missing:
        sample = missing[:5]
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        preview = "; ".join(f"{s[:100]}..." if len(s) > 100 else s for s in sample)
        raise ValueError(
            "Persona map is missing entries for these query strings (after strip) "
            f"(no fallback persona): {preview}{more}"
        )

    rng = random_module.Random(seed)
    requests: list[dict] = []
    manifest: dict[str, dict] = {}
    persona_question_counts: Counter[str] = Counter()
    skipped_no_evidence = 0

    for q_idx, row in questions_df.iterrows():
        source_qid = source_question_id_for_row(row, q_idx)
        q_norm = _normalize_query_text(row["query"])
        original_persona_id = persona_by_query[q_norm]
        persona_question_counts[original_persona_id] += 1

        all_evidence = _all_evidence_for_row(row)
        if not all_evidence:
            skipped_no_evidence += 1
            continue

        is_temporal = row["question_type"] == "temporal_query"
        row_answer = _row_answer(row)
        q_fp = _query_fingerprint(q_norm)

        seen_evidence_ids: set[str] = set()
        for evidence in all_evidence:
            eid = evidence.get("evidence_id") if isinstance(evidence, dict) else None
            if not eid or not isinstance(eid, str) or not str(eid).strip():
                qprev = repr(row["query"])[:120]
                raise ValueError(
                    "Every evidence dict must include a non-empty string evidence_id "
                    "(run prepare_data/collect_reasonable_questions.py or attach ids). "
                    f"query={qprev}"
                )
            eid = str(eid).strip()
            if eid in seen_evidence_ids:
                continue
            seen_evidence_ids.add(eid)

            evidence_categories = set()
            cat = evidence.get("category")
            if cat and isinstance(cat, str):
                evidence_categories.add(cat.strip().lower())

            for target_pid in target_personas:
                target_persona = persona_dict_for_id(target_pid)
                profile = diversity_profile_for_persona(target_pid)
                off_topic = is_off_topic_for_persona(target_pid, evidence_categories)

                custom_id = f"q-{q_fp}__ev-{eid}__p-{target_pid}"

                diversity = _sample_diversity_params(rng, profile=profile)

                system_prompt = format_generation_system_prompt(
                    target_persona["character"],
                    evidence,
                    diversity,
                    is_temporal=is_temporal,
                    is_off_topic=off_topic,
                )

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model_name,
                        "input": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": diversity["human_message"]},
                        ],
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": "Session",
                                "strict": True,
                                "schema": SESSION_SCHEMA,
                            }
                        },
                    },
                }

                manifest[custom_id] = {
                    "source_question_id": source_qid,
                    "evidence_id": eid,
                    "query_fingerprint": q_fp,
                    "query": row["query"],
                    "answer": row_answer,
                    "question_type": row["question_type"],
                    "original_persona_id": original_persona_id,
                    "target_persona_id": target_pid,
                    "persona_character": target_persona["character"],
                    "evidence": evidence,
                    "diversity": diversity,
                    "is_temporal": is_temporal,
                    "is_off_topic": off_topic,
                }

                requests.append(request)

    print(f"Total batch requests: {len(requests)}")
    print(f"Target personas: {target_personas}")
    if skipped_no_evidence:
        print(
            f"Skipped {skipped_no_evidence} questions with no evidence "
            "(null_query / no hops — no evidence sessions to generate)"
        )
    print("Questions per original_persona_id (counted while building requests):")
    for pid, n in persona_question_counts.most_common():
        print(f"  {pid}: {n}")

    return requests, manifest


def submit_batch(requests, manifest, output_dir):
    """Write JSONL + manifest, upload to OpenAI, and create the batch job."""
    if not requests:
        raise ValueError(
            "No batch requests to submit. Check query text vs persona map and "
            "evidence columns (memory_evidence/evidence_list or "
            "golden_memory_evidence/golden_document_evidence)."
        )
    client = OpenAI()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    jsonl_path = os.path.join(output_dir, f"batch_input_{timestamp}.jsonl")
    with open(jsonl_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    manifest_path = os.path.join(output_dir, f"batch_manifest_{timestamp}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )

    batch_meta = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "jsonl_path": jsonl_path,
        "manifest_path": manifest_path,
        "total_requests": len(requests),
        "model": requests[0]["body"]["model"] if requests else None,
        "created_at": timestamp,
    }
    meta_path = os.path.join(output_dir, f"batch_meta_{timestamp}.json")
    with open(meta_path, "w") as f:
        json.dump(batch_meta, f, indent=2)

    print(f"Batch ID   : {batch.id}")
    print(f"Status     : {batch.status}")
    print(f"Requests   : {len(requests)}")
    print(f"JSONL      : {jsonl_path}")
    print(f"Manifest   : {manifest_path}")
    print(f"Metadata   : {meta_path}")

    return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Evidence Session Generation")
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Input evidence data path (.pkl)",
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt-5.4",
        help="LLM model name (default: gpt-5.4)",
    )
    parser.add_argument(
        "--question_limit", type=int, default=None,
        help="Maximum number of questions to process (default: all)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/batch_jobs/experiment_sessions",
        help="Output directory for batch files (default: data/batch_jobs/experiment_sessions)",
    )
    parser.add_argument(
        "--no-submit", action="store_true",
        help="Write JSONL + manifest only; do not upload or create OpenAI batch job",
    )
    parser.add_argument(
        "--persona_map",
        type=str,
        default=str(_DEFAULT_PERSONA_CSV),
        help=(
            "Required CSV with query and persona columns (default: "
            "data/persona_metadata/q_2_personas.csv). Persona is resolved by "
            "exact question text (strip whitespace), not question_idx; unknown "
            "personas and missing queries raise an error (no default persona)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    questions_df = pd.read_pickle(args.data_path)
    if questions_df is None:
        raise ValueError(f"Data file {args.data_path} does not exist")

    print(f"Total questions: {len(questions_df)}")
    print(f"Columns: {list(questions_df.columns)}")

    if not args.persona_map or not str(args.persona_map).strip():
        raise ValueError(
            "--persona_map is required: path to a CSV with query and persona "
            "columns (no default persona)."
        )
    pm_path = Path(args.persona_map).expanduser().resolve()
    if not pm_path.is_file():
        raise FileNotFoundError(
            f"Persona map not found: {pm_path}. Generate it with "
            "prepare_data/questions_to_personas.py."
        )
    persona_by_query = load_persona_map_by_query(pm_path)
    print(f"Loaded persona map: {pm_path} ({len(persona_by_query)} distinct queries)")

    requests, manifest = build_batch_requests(
        questions_df,
        args.model_name,
        args.question_limit,
        persona_by_query=persona_by_query,
    )

    if args.no_submit:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        jsonl_path = os.path.join(args.output_dir, f"batch_input_{timestamp}.jsonl")
        manifest_path = os.path.join(args.output_dir, f"batch_manifest_{timestamp}.json")
        with open(jsonl_path, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote {len(requests)} requests (no-submit) → {jsonl_path}")
        print(f"Manifest → {manifest_path}")
    else:
        submit_batch(requests, manifest, args.output_dir)
