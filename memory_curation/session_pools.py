"""Load and tier evidence session pools for persona corpus assembly."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from session_generation.merge_batch_output import merge_batch_output_to_records
from session_generation.persona import PERSONA_DOMAINS

from memory_curation.diversity_sampling import sample_sessions_diversity_aware


def load_valid_batch_records(
    manifest_path: Path,
    output_jsonl_path: Path,
) -> list[dict]:
    """Merge batch output and drop unparseable sessions."""
    records = merge_batch_output_to_records(manifest_path, output_jsonl_path)
    failed = sum(1 for r in records if "session_parse_error" in r)
    valid = [r for r in records if "session_parse_error" not in r and r.get("session")]
    if failed:
        print(f"  Skipping {failed} records with parse errors from {manifest_path.name}.")
    return valid


def group_records_by_persona(records: list[dict]) -> dict[str, list[dict]]:
    by_persona: dict[str, list[dict]] = {}
    for rec in records:
        pid = rec.get("target_persona_id") or rec.get("persona_id", "unknown")
        by_persona.setdefault(str(pid), []).append(rec)
    return by_persona


def is_in_domain_for_persona(rec: dict, persona_id: str) -> bool:
    domains = PERSONA_DOMAINS.get(persona_id)
    if domains is None:
        return True
    cat = rec.get("evidence", {}).get("category", "")
    if isinstance(cat, str) and cat.strip().lower():
        return cat.strip().lower() in domains
    return False


def split_by_domain(records: list[dict], persona_id: str) -> tuple[list[dict], list[dict]]:
    in_domain: list[dict] = []
    off_topic: list[dict] = []
    for rec in records:
        if is_in_domain_for_persona(rec, persona_id):
            in_domain.append(rec)
        else:
            off_topic.append(rec)
    return in_domain, off_topic


@dataclass
class PoolInventory:
    persona_id: str
    primary_in_domain: int
    primary_off_topic: int
    supplement_in_domain: int
    supplement_off_topic: int
    target_in_domain: int
    target_off_topic: int

    @property
    def in_domain_shortfall(self) -> int:
        avail = self.primary_in_domain + self.supplement_in_domain
        return max(0, self.target_in_domain - avail)

    @property
    def off_topic_shortfall(self) -> int:
        avail = self.primary_off_topic + self.supplement_off_topic
        return max(0, self.target_off_topic - avail)


def inventory_persona_pools(
    persona_id: str,
    primary: list[dict],
    supplement: list[dict],
    *,
    target_in_domain: int,
    target_off_topic: int,
) -> PoolInventory:
    p_in, p_out = split_by_domain(primary, persona_id)
    s_in, s_out = split_by_domain(supplement, persona_id)
    return PoolInventory(
        persona_id=persona_id,
        primary_in_domain=len(p_in),
        primary_off_topic=len(p_out),
        supplement_in_domain=len(s_in),
        supplement_off_topic=len(s_out),
        target_in_domain=target_in_domain,
        target_off_topic=target_off_topic,
    )


def _get_evidence_id(rec: dict) -> str | None:
    return rec.get("evidence_id") or rec.get("evidence", {}).get("evidence_id")


def sample_tiered(
    persona_id: str,
    label: str,
    needed: int,
    tier1: list[dict],
    tier2: list[dict],
    rng: random.Random,
    *,
    diversity_aware_tier2: bool = True,
    used_ids: set[str] | None = None,
    used_evidence_ids: set[str] | None = None,
) -> list[dict]:
    """Fill *needed* sessions: exhaust tier1 (shuffled), then tier2.

    Deduplicates by both custom_id (session-level) and evidence_id
    (fact-level) so no evidence fact appears twice in a corpus.
    """
    if needed <= 0:
        return []

    if label == "in_domain":
        t1 = [r for r in tier1 if is_in_domain_for_persona(r, persona_id)]
        t2 = [r for r in tier2 if is_in_domain_for_persona(r, persona_id)]
    else:
        t1 = [r for r in tier1 if not is_in_domain_for_persona(r, persona_id)]
        t2 = [r for r in tier2 if not is_in_domain_for_persona(r, persona_id)]

    chosen: list[dict] = []
    if used_ids is None:
        used_ids = set()
    if used_evidence_ids is None:
        used_evidence_ids = set()

    t1_shuffled = list(t1)
    rng.shuffle(t1_shuffled)
    for rec in t1_shuffled:
        if len(chosen) >= needed:
            break
        cid = rec.get("custom_id")
        eid = _get_evidence_id(rec)
        if cid and cid in used_ids:
            continue
        if eid and eid in used_evidence_ids:
            continue
        chosen.append(rec)
        if cid:
            used_ids.add(cid)
        if eid:
            used_evidence_ids.add(eid)

    remaining = needed - len(chosen)
    if remaining <= 0:
        return chosen

    t2_avail = [
        r for r in t2
        if r.get("custom_id") not in used_ids
        and (_get_evidence_id(r) or "") not in used_evidence_ids
    ]
    if diversity_aware_tier2:
        extra = sample_sessions_diversity_aware(
            t2_avail, remaining, persona_id, rng,
            exclude_ids=used_ids, exclude_evidence_ids=used_evidence_ids,
        )
    else:
        extra = rng.sample(t2_avail, k=min(remaining, len(t2_avail))) if t2_avail else []

    for rec in extra:
        cid = rec.get("custom_id")
        eid = _get_evidence_id(rec)
        if cid and cid in used_ids:
            continue
        if eid and eid in used_evidence_ids:
            continue
        chosen.append(rec)
        if cid:
            used_ids.add(cid)
        if eid:
            used_evidence_ids.add(eid)

    if len(chosen) < needed and t2_avail:
        still = [
            r for r in t2_avail
            if r.get("custom_id") not in used_ids
            and (_get_evidence_id(r) or "") not in used_evidence_ids
        ]
        need = needed - len(chosen)
        for rec in rng.sample(still, k=min(need, len(still))):
            eid = _get_evidence_id(rec)
            if eid and eid in used_evidence_ids:
                continue
            chosen.append(rec)
            cid = rec.get("custom_id")
            if cid:
                used_ids.add(cid)
            if eid:
                used_evidence_ids.add(eid)

    return chosen[:needed]
