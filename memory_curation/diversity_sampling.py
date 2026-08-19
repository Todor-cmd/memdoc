"""Diversity-aware session sampling for persona corpus supplementation."""
from __future__ import annotations

import random
from collections import Counter

from session_generation.persona import (
    PERSONA_DIVERSITY_PROFILES,
    SCENARIOS,
    TURN_RANGES,
    TOPIC_DRIFT_LEVELS,
    diversity_profile_for_persona,
)


def _scenario_category_weights(persona_id: str) -> dict[str, float]:
    profile = diversity_profile_for_persona(persona_id) or {}
    weights = profile.get("scenario_category_weights")
    if weights:
        return dict(weights)
    cats = sorted({s["category"] for s in SCENARIOS})
    w = 1.0 / len(cats)
    return {c: w for c in cats}


def _turn_range_key(tr: list | tuple | None) -> tuple[int, int] | None:
    if not tr or len(tr) != 2:
        return None
    return (int(tr[0]), int(tr[1]))


def _turn_range_weights(persona_id: str) -> dict[tuple[int, int], float]:
    profile = diversity_profile_for_persona(persona_id) or {}
    raw = profile.get("turn_range_weights")
    if not raw or len(raw) != len(TURN_RANGES):
        w = 1.0 / len(TURN_RANGES)
        return {tr: w for tr in TURN_RANGES}
    return {TURN_RANGES[i]: float(raw[i]) for i in range(len(TURN_RANGES))}


def _topic_drift_weights(persona_id: str) -> dict[str, float]:
    """Marginal topic-drift weights averaged across turn-range buckets."""
    profile = diversity_profile_for_persona(persona_id) or {}
    td_map = profile.get("topic_drift_weights_by_turn_range")
    if not td_map:
        w = 1.0 / len(TOPIC_DRIFT_LEVELS)
        return {lvl: w for lvl in TOPIC_DRIFT_LEVELS}

    tr_weights = _turn_range_weights(persona_id)
    acc = {lvl: 0.0 for lvl in TOPIC_DRIFT_LEVELS}
    for tr, tr_w in tr_weights.items():
        bucket = td_map.get(tr)
        if not bucket or len(bucket) != len(TOPIC_DRIFT_LEVELS):
            continue
        for lvl, p in zip(TOPIC_DRIFT_LEVELS, bucket):
            acc[lvl] += tr_w * float(p)
    total = sum(acc.values())
    if total <= 0:
        w = 1.0 / len(TOPIC_DRIFT_LEVELS)
        return {lvl: w for lvl in TOPIC_DRIFT_LEVELS}
    return {lvl: acc[lvl] / total for lvl in TOPIC_DRIFT_LEVELS}


def _session_bins(rec: dict) -> tuple[str, tuple[int, int] | None, str]:
    d = rec.get("diversity") or {}
    cat = str(d.get("scenario_category") or "unknown")
    tr = _turn_range_key(d.get("turn_range"))
    drift = str(d.get("topic_drift") or "unknown")
    return cat, tr, drift


def _allocate_counts(n: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of *n* across weighted labels."""
    labels = list(weights.keys())
    raw = {lbl: n * weights[lbl] for lbl in labels}
    floored = {lbl: int(v) for lbl, v in raw.items()}
    remainder = n - sum(floored.values())
    order = sorted(labels, key=lambda lbl: raw[lbl] - floored[lbl], reverse=True)
    for lbl in order[:remainder]:
        floored[lbl] += 1
    return floored


def _get_evidence_id(rec: dict) -> str | None:
    return rec.get("evidence_id") or rec.get("evidence", {}).get("evidence_id")


def sample_sessions_diversity_aware(
    pool: list[dict],
    n: int,
    persona_id: str,
    rng: random.Random,
    *,
    exclude_ids: set[str] | None = None,
    exclude_evidence_ids: set[str] | None = None,
) -> list[dict]:
    """Sample up to *n* sessions from *pool* matching persona diversity marginals.

    Stratifies primarily by scenario category, then turn range, then topic drift.
    Deduplicates by both custom_id and evidence_id to prevent the same
    evidence fact appearing in multiple corpus sessions.
    Falls back to any remaining pool members if a bin is underfilled.
    """
    if n <= 0:
        return []
    if not pool:
        return []

    exclude_ids = exclude_ids or set()
    exclude_evidence_ids = exclude_evidence_ids or set()

    available = [
        r for r in pool
        if r.get("custom_id") not in exclude_ids
        and (_get_evidence_id(r) or "") not in exclude_evidence_ids
    ]
    if not available:
        return []

    if len(available) <= n:
        return rng.sample(available, k=len(available))

    cat_weights = _scenario_category_weights(persona_id)
    cat_targets = _allocate_counts(n, cat_weights)

    by_cat: dict[str, list[dict]] = {}
    for rec in available:
        cat, _, _ = _session_bins(rec)
        by_cat.setdefault(cat, []).append(rec)

    chosen: list[dict] = []
    chosen_ids: set[str] = set()
    chosen_eids: set[str] = set()

    def _is_eligible(r: dict) -> bool:
        cid = r.get("custom_id")
        eid = _get_evidence_id(r)
        if cid and cid in chosen_ids:
            return False
        if eid and eid in chosen_eids:
            return False
        return True

    def _mark_chosen(r: dict) -> None:
        cid = r.get("custom_id")
        eid = _get_evidence_id(r)
        if cid:
            chosen_ids.add(cid)
        if eid:
            chosen_eids.add(eid)

    for cat, target in cat_targets.items():
        cat_pool = [r for r in by_cat.get(cat, []) if _is_eligible(r)]
        if not cat_pool:
            continue
        take = min(target, len(cat_pool))
        if take <= 0:
            continue

        tr_weights = _turn_range_weights(persona_id)
        tr_label_weights = {
            f"{a}-{b}": w for (a, b), w in tr_weights.items()
        }
        tr_targets = _allocate_counts(take, tr_label_weights)
        cat_remaining = list(cat_pool)

        for tr_label, tr_n in tr_targets.items():
            if tr_n <= 0:
                continue
            parts = tr_label.split("-")
            tr_tuple = (int(parts[0]), int(parts[1])) if len(parts) == 2 else None
            tr_bin_pool = [
                r for r in cat_remaining
                if _session_bins(r)[1] == tr_tuple and _is_eligible(r)
            ]
            if not tr_bin_pool:
                tr_bin_pool = [r for r in cat_remaining if _is_eligible(r)]
            if not tr_bin_pool:
                continue

            drift_weights = _topic_drift_weights(persona_id)
            drift_targets = _allocate_counts(min(tr_n, len(tr_bin_pool)), drift_weights)

            for drift, d_n in drift_targets.items():
                if d_n <= 0:
                    continue
                drift_pool = [
                    r for r in tr_bin_pool
                    if _session_bins(r)[2] == drift and _is_eligible(r)
                ]
                if not drift_pool:
                    drift_pool = [r for r in tr_bin_pool if _is_eligible(r)]
                pick_n = min(d_n, len(drift_pool))
                if pick_n <= 0:
                    continue
                picks = rng.sample(drift_pool, k=pick_n)
                for p in picks:
                    if _is_eligible(p):
                        chosen.append(p)
                        _mark_chosen(p)
                        if p in cat_remaining:
                            cat_remaining.remove(p)

    if len(chosen) < n:
        leftover = [r for r in available if _is_eligible(r)]
        need = n - len(chosen)
        if leftover:
            picks = rng.sample(leftover, k=min(need, len(leftover)))
            for p in picks:
                if _is_eligible(p):
                    chosen.append(p)
                    _mark_chosen(p)

    return chosen[:n]


def diversity_summary(records: list[dict]) -> dict[str, Counter]:
    cats: Counter[str] = Counter()
    turns: Counter[str] = Counter()
    drifts: Counter[str] = Counter()
    for rec in records:
        cat, tr, drift = _session_bins(rec)
        cats[cat] += 1
        turns[str(tr)] += 1
        drifts[drift] += 1
    return {"scenario_category": cats, "turn_range": turns, "topic_drift": drifts}
