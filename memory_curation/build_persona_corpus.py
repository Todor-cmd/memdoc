"""Build per-persona background memory corpus (500 sessions each).

Tiered assembly:
  1. Primary pool — experiment question batch (210 × personas)
  2. Supplement pool — reasonable_sessions batch (diversity-aware sampling)
  3. Optional OOD supplement batch if still short (see build_supplement_batch.py)

Default composition: 75% in-domain evidence sessions, 25% off-topic.

See docs/memory_corpus_curation.md for design rationale.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from memory_curation.batch_paths import resolve_manifest_and_jsonl  # noqa: E402
from memory_curation.diversity_sampling import diversity_summary  # noqa: E402
from memory_curation.session_pools import (  # noqa: E402
    group_records_by_persona,
    inventory_persona_pools,
    load_valid_batch_records,
    sample_tiered,
)
_DATA = REPO_ROOT / "data"
DEFAULT_PRIMARY_MANIFEST = _DATA / "batch_jobs/experiment_sessions/batch_manifest.json"
DEFAULT_PRIMARY_JSONL = _DATA / "batch_jobs/experiment_sessions/batch_output.jsonl"
DEFAULT_SUPPLEMENT_MANIFEST = _DATA / "batch_jobs/reasonable_sessions/batch_manifest_20260424_153151.json"
DEFAULT_SUPPLEMENT_JSONL = _DATA / "batch_jobs/reasonable_sessions/batch_output.jsonl"
DEFAULT_FILLER = _DATA / "custom_history/custom_history_data/5_filler_sess/data_5_filler_sess.json"
DEFAULT_OUT_DIR = _DATA / "memory_collection"

TOTAL_SESSIONS = 500
DEFAULT_IN_DOMAIN_RATIO = 0.75
DEFAULT_OFF_TOPIC_RATIO = 0.25

W_START = datetime(2023, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
W_END = datetime(2024, 2, 29, 23, 59, 59, tzinfo=timezone.utc)

DEFAULT_MEAN_DELAY_DAYS = 3.0
DEFAULT_HOUR_LO = 8
DEFAULT_HOUR_HI = 22
EPS_AFTER_PUB_SEC = 3600.0

TARGET_PERSONAS = ["persona_1", "persona_2", "persona_3"]


@dataclass
class CorpusSource:
    label: str
    sessions: list[dict]
    target_count: int


def _parse_iso_utc(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _utc_midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _format_ts(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y/%m/%d (%a) %H:%M")


def _sample_time_on_day(
    day: date,
    anchor: datetime,
    w_start: datetime,
    w_end: datetime,
    hour_lo: int,
    hour_hi: int,
    rng: random.Random,
    epsilon_sec: float = EPS_AFTER_PUB_SEC,
) -> datetime | None:
    mid = _utc_midnight(day)
    if day < w_start.date() or day > w_end.date():
        return None

    lo = float(hour_lo * 3600)
    hi = float(hour_hi * 3600 + 3599)

    if day == w_start.date():
        lo = max(lo, (w_start - mid).total_seconds())
    if day == w_end.date():
        hi = min(hi, (w_end - mid).total_seconds())

    if day == anchor.date():
        lo = max(lo, (anchor - mid).total_seconds() + epsilon_sec)

    if lo > hi:
        return None
    sec = rng.uniform(lo, hi)
    return mid + timedelta(seconds=sec)


def _uniform_filler_timestamp(
    rng: random.Random,
    w_start: datetime,
    w_end: datetime,
    hour_lo: int,
    hour_hi: int,
    max_attempts: int = 500,
) -> datetime:
    span_sec = int((w_end - w_start).total_seconds())
    for _ in range(max_attempts):
        u = rng.randint(0, max(0, span_sec))
        raw = w_start + timedelta(seconds=u)
        out = _sample_time_on_day(
            raw.date(), w_start, w_start, w_end, hour_lo, hour_hi, rng, epsilon_sec=0.0,
        )
        if out is not None and w_start <= out <= w_end:
            return out
    return w_end


def _evidence_session_timestamp(
    rec: dict,
    rng: random.Random,
    w_start: datetime,
    w_end: datetime,
    mean_delay_sec: float,
    hour_lo: int,
    hour_hi: int,
    max_attempts: int = 400,
) -> datetime:
    pub = _parse_iso_utc(rec["evidence"]["published_at"])
    t_min = max(pub, w_start)
    anchor = max(t_min, pub)

    if t_min > w_end:
        return _uniform_filler_timestamp(rng, w_start, w_end, hour_lo, hour_hi)

    max_delay = (w_end - t_min).total_seconds()
    if max_delay <= 0:
        got = _sample_time_on_day(t_min.date(), anchor, w_start, w_end, hour_lo, hour_hi, rng)
        return got or t_min

    rate = 1.0 / mean_delay_sec

    for _ in range(max_attempts):
        d = rng.expovariate(rate)
        if d > max_delay:
            continue
        raw = t_min + timedelta(seconds=d)
        day = raw.date()
        final = _sample_time_on_day(day, anchor, w_start, w_end, hour_lo, hour_hi, rng)
        if final is None:
            continue
        if final < pub:
            continue
        if w_start <= final <= w_end:
            return final

    for _ in range(max_attempts):
        d = rng.uniform(0.0, max_delay)
        raw = t_min + timedelta(seconds=d)
        final = _sample_time_on_day(raw.date(), anchor, w_start, w_end, hour_lo, hour_hi, rng)
        if final is not None and pub <= final <= w_end and final >= w_start:
            return final

    return _uniform_filler_timestamp(rng, w_start, w_end, hour_lo, hour_hi)


def _turns_from_evidence_record(rec: dict) -> list[dict]:
    raw_turns = rec["session"]["turns"]
    if raw_turns and isinstance(raw_turns[0], dict):
        return [{"role": t["role"], "content": t["content"]} for t in raw_turns]
    return [{"role": t.role, "content": t.content} for t in raw_turns]


def build_tiered_corpus_sources(
    persona_id: str,
    primary_pool: list[dict],
    supplement_pool: list[dict],
    rng: random.Random,
    *,
    total_sessions: int = TOTAL_SESSIONS,
    in_domain_ratio: float = DEFAULT_IN_DOMAIN_RATIO,
    filler_ultrachat: list[dict] | None = None,
    filler_sharegpt: list[dict] | None = None,
) -> tuple[list[CorpusSource], list[dict]]:
    """Build sources using tier-1 then tier-2 sampling; return sources + chosen records."""
    n_in_domain = int(total_sessions * in_domain_ratio)
    n_off_topic = total_sessions - n_in_domain

    filler_count = 0
    if filler_ultrachat is not None and filler_sharegpt is not None:
        filler_ratio = 1.0 - in_domain_ratio - DEFAULT_OFF_TOPIC_RATIO
        if filler_ratio > 0:
            filler_count = int(total_sessions * filler_ratio)
            n_off_topic = total_sessions - n_in_domain - filler_count

    used_ids: set[str] = set()
    used_evidence_ids: set[str] = set()
    in_samples = sample_tiered(
        persona_id, "in_domain", n_in_domain, primary_pool, supplement_pool, rng,
        used_ids=used_ids, used_evidence_ids=used_evidence_ids,
    )
    off_samples = sample_tiered(
        persona_id, "off_topic", n_off_topic, primary_pool, supplement_pool, rng,
        used_ids=used_ids, used_evidence_ids=used_evidence_ids,
    )

    sources: list[CorpusSource] = [
        CorpusSource(label="in_domain_evidence", sessions=in_samples, target_count=len(in_samples)),
        CorpusSource(label="off_topic_evidence", sessions=off_samples, target_count=len(off_samples)),
    ]

    if filler_count > 0 and filler_ultrachat is not None and filler_sharegpt is not None:
        n_ultra = filler_count // 2
        n_share = filler_count - n_ultra
        sources.append(CorpusSource(label="ultrachat", sessions=filler_ultrachat, target_count=n_ultra))
        sources.append(CorpusSource(label="sharegpt", sessions=filler_sharegpt, target_count=n_share))

    chosen_records = in_samples + off_samples
    return sources, chosen_records


def build_persona_corpus(
    persona_id: str,
    sources: list[CorpusSource],
    rng: random.Random,
    w_start: datetime,
    w_end: datetime,
    mean_delay_sec: float,
    hour_lo: int,
    hour_hi: int,
) -> list[dict]:
    """Assemble and timestamp the background corpus for one persona from sources."""
    sessions: list[dict] = []

    for source in sources:
        pool = source.sessions
        needed = source.target_count

        if not pool:
            print(f"  Warning: {source.label} pool is empty for {persona_id}, skipping.")
            continue

        if len(pool) < needed:
            print(
                f"  Warning: {persona_id} {source.label} has {len(pool)} sessions "
                f"(need {needed}); sampling with replacement for shortfall."
            )
            sample = rng.choices(pool, k=needed)
        else:
            sample = pool[:needed] if len(pool) == needed else pool

        for rec in sample:
            if source.label in ("in_domain_evidence", "off_topic_evidence"):
                ts = _evidence_session_timestamp(
                    rec, rng, w_start, w_end, mean_delay_sec, hour_lo, hour_hi
                )
                sessions.append({
                    "session_id": rec["custom_id"],
                    "source_question_id": rec.get("source_question_id"),
                    "evidence_id": rec.get("evidence_id"),
                    "session": _turns_from_evidence_record(rec),
                    "source": source.label,
                    "is_off_topic": source.label == "off_topic_evidence",
                    "_ts": ts,
                })
            else:
                ts = _uniform_filler_timestamp(rng, w_start, w_end, hour_lo, hour_hi)
                sessions.append({
                    "session_id": rec["session_id"],
                    "session": [{"role": t["role"], "content": t["content"]} for t in rec["session"]],
                    "source": source.label,
                    "is_off_topic": False,
                    "_ts": ts,
                })

    sessions.sort(key=lambda x: x["_ts"])
    for s in sessions:
        s["date"] = _format_ts(s.pop("_ts"))
    return sessions


def print_inventory(
    personas: list[str],
    primary_by_persona: dict[str, list[dict]],
    supplement_by_persona: dict[str, list[dict]],
    *,
    total_sessions: int,
    in_domain_ratio: float,
) -> None:
    n_in = int(total_sessions * in_domain_ratio)
    n_out = total_sessions - n_in
    print(f"\nPool inventory (targets: {n_in} in-domain, {n_out} off-topic per persona):")
    any_shortfall = False
    for pid in personas:
        inv = inventory_persona_pools(
            pid,
            primary_by_persona.get(pid, []),
            supplement_by_persona.get(pid, []),
            target_in_domain=n_in,
            target_off_topic=n_out,
        )
        print(f"  {pid}:")
        print(
            f"    primary:     in={inv.primary_in_domain}  out={inv.primary_off_topic}"
        )
        print(
            f"    supplement:  in={inv.supplement_in_domain}  out={inv.supplement_off_topic}"
        )
        if inv.in_domain_shortfall or inv.off_topic_shortfall:
            any_shortfall = True
            print(
                f"    SHORTFALL:   in-domain={inv.in_domain_shortfall}  "
                f"off-topic={inv.off_topic_shortfall}"
            )
        else:
            print("    OK — sufficient pools for 500-session corpus")
    if any_shortfall:
        print(
            "\n  Run memory_curation.build_supplement_batch for remaining shortfall, "
            "then re-run with --supplement-manifest pointing at the new batch."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-persona background memory corpus (tiered 500-session assembly)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--primary-manifest", type=Path, default=DEFAULT_PRIMARY_MANIFEST,
        help="Experiment sessions batch manifest",
    )
    parser.add_argument(
        "--primary-output-jsonl", type=Path, default=DEFAULT_PRIMARY_JSONL,
        help="Experiment sessions batch output JSONL",
    )
    parser.add_argument(
        "--supplement-manifest", type=Path, default=DEFAULT_SUPPLEMENT_MANIFEST,
        help="Reasonable_sessions supplement manifest (optional if primary is sufficient)",
    )
    parser.add_argument(
        "--supplement-output-jsonl", type=Path, default=DEFAULT_SUPPLEMENT_JSONL,
        help="Reasonable_sessions supplement output JSONL",
    )
    parser.add_argument("--no-supplement", action="store_true",
                        help="Use only the primary experiment batch pool")
    parser.add_argument("--inventory-only", action="store_true",
                        help="Print pool inventory and exit without building corpora")
    parser.add_argument("--filler", type=Path, default=DEFAULT_FILLER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--total-sessions", type=int, default=TOTAL_SESSIONS)
    parser.add_argument("--in-domain-ratio", type=float, default=DEFAULT_IN_DOMAIN_RATIO)
    parser.add_argument("--include-fillers", action="store_true", default=False)
    parser.add_argument("--mean-delay-days", type=float, default=DEFAULT_MEAN_DELAY_DAYS)
    parser.add_argument("--hour-start", type=int, default=DEFAULT_HOUR_LO)
    parser.add_argument("--hour-end", type=int, default=DEFAULT_HOUR_HI)
    args = parser.parse_args()

    mean_delay_sec = max(args.mean_delay_days, 1e-6) * 86400.0
    hour_lo = min(args.hour_start, args.hour_end)
    hour_hi = max(args.hour_start, args.hour_end)
    rng = random.Random(args.seed)

    print("Loading primary evidence sessions …")
    p_manifest, p_jsonl = resolve_manifest_and_jsonl(
        args.primary_manifest, args.primary_output_jsonl,
        job_dir=_DATA / "batch_jobs" / "experiment_sessions",
    )
    primary_records = load_valid_batch_records(p_manifest, p_jsonl)
    primary_by_persona = group_records_by_persona(primary_records)
    print(f"  {len(primary_records)} valid primary sessions")

    supplement_by_persona: dict[str, list[dict]] = {pid: [] for pid in TARGET_PERSONAS}
    if not args.no_supplement:
        try:
            s_manifest, s_jsonl = resolve_manifest_and_jsonl(
                args.supplement_manifest, args.supplement_output_jsonl,
                job_dir=_DATA / "batch_jobs" / "reasonable_sessions",
            )
            print("Loading supplement evidence sessions …")
            supplement_records = load_valid_batch_records(s_manifest, s_jsonl)
            supplement_by_persona = group_records_by_persona(supplement_records)
            print(f"  {len(supplement_records)} valid supplement sessions")
        except FileNotFoundError:
            print("  Supplement batch not found — primary pool only.")

    print_inventory(
        TARGET_PERSONAS,
        primary_by_persona,
        supplement_by_persona,
        total_sessions=args.total_sessions,
        in_domain_ratio=args.in_domain_ratio,
    )
    if args.inventory_only:
        return

    filler_ultrachat: list[dict] | None = None
    filler_sharegpt: list[dict] | None = None
    if args.include_fillers:
        filler_db: list[dict] = json.loads(args.filler.read_text(encoding="utf-8"))
        filler_ultrachat = [x for x in filler_db if "ultrachat" in x["session_id"]]
        filler_sharegpt = [x for x in filler_db if "sharegpt" in x["session_id"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\nCorpus config: {args.total_sessions} sessions, "
        f"in-domain={args.in_domain_ratio:.0%}"
    )

    for persona_id in TARGET_PERSONAS:
        primary_pool = primary_by_persona.get(persona_id, [])
        supplement_pool = supplement_by_persona.get(persona_id, [])
        if not primary_pool and not supplement_pool:
            print(f"\n  Skipping {persona_id}: no evidence sessions found.")
            continue

        print(f"\nBuilding corpus for {persona_id} …")
        sources, chosen = build_tiered_corpus_sources(
            persona_id,
            primary_pool,
            supplement_pool,
            rng,
            total_sessions=args.total_sessions,
            in_domain_ratio=args.in_domain_ratio,
            filler_ultrachat=filler_ultrachat,
            filler_sharegpt=filler_sharegpt,
        )

        n_in_target = int(args.total_sessions * args.in_domain_ratio)
        n_out_target = args.total_sessions - n_in_target
        in_count = sources[0].target_count if sources else 0
        out_count = sources[1].target_count if len(sources) > 1 else 0
        print(f"  tiered sample: in-domain={in_count}/{n_in_target}  off-topic={out_count}/{n_out_target}")

        if in_count < n_in_target or out_count < n_out_target:
            print(
                f"  Warning: corpus under-filled for {persona_id} "
                f"(in {in_count}/{n_in_target}, out {out_count}/{n_out_target}). "
                "Generate supplement batch for shortfall."
            )

        summary = diversity_summary(chosen)
        print(f"  diversity (chosen): scenario={dict(summary['scenario_category'])}")

        corpus = build_persona_corpus(
            persona_id, sources, rng, W_START, W_END, mean_delay_sec, hour_lo, hour_hi
        )

        source_counts: dict[str, int] = {}
        for s in corpus:
            source_counts[s["source"]] = source_counts.get(s["source"], 0) + 1
        print(f"  Final composition: {source_counts} (total={len(corpus)})")

        out_path = args.out_dir / f"{persona_id}.json"
        out_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Written → {out_path.relative_to(REPO_ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
