"""Stratified sampling of 210 questions for the MemMeetDoc experiment.

Loads ``full_reasonable.pkl`` and the persona map from ``q_2_personas.csv``,
then draws questions from the 3 eval-persona groups (persona_1, persona_2,
persona_3). Persona_4 questions are excluded because they are always
out-of-domain for every eval persona and cannot contribute to within-question
``is_in_domain`` comparisons.

By default 66 questions are drawn per eval persona (198 total), then 12
``null_query`` questions are appended at random (210 total). Null queries have
no evidence hops; they evaluate abstention under background-only memory and
map to ``original_persona = tbd``. Row order is: persona-stratified rows first,
null queries last (blocks 199–210 in ``experiment_design.csv``).

Within each persona stratum, sampling guarantees a minimum number of questions
per ``question_type`` and ``hop_count`` level (default: 5) so that these
covariates have sufficient observations for downstream analysis.

The output preserves the domain-based persona assignment as
``original_persona`` for downstream cross-persona evaluation.

Usage:
    python -m prepare_data.sample_experiment_questions
    python -m prepare_data.sample_experiment_questions --seed 42 --n-per-persona 66
    python -m prepare_data.sample_experiment_questions --n-null-query 12
    python -m prepare_data.sample_experiment_questions --min-per-covariate 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prepare_data.questions_to_personas import parse_evidence_list
from session_generation.create_batch_job import (
    _normalize_query_text,
    load_persona_map_by_query,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INPUT = _REPO_ROOT / "data" / "questions" / "full_reasonable.pkl"
_DEFAULT_PERSONA_CSV = _REPO_ROOT / "data" / "persona_metadata" / "q_2_personas.csv"
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "questions" / "experiment_210.pkl"

TARGET_PERSONAS = ["persona_1", "persona_2", "persona_3"]
DEFAULT_N_PER_PERSONA = 66
DEFAULT_N_NULL_QUERY = 12

COVARIATE_COLUMNS = ("question_type", "hop_count")


def attach_persona_column(df: pd.DataFrame, persona_by_query: dict[str, str]) -> pd.DataFrame:
    """Add ``persona`` column by matching normalized query text."""
    out = df.copy()
    personas = []
    for _, row in out.iterrows():
        q = _normalize_query_text(row["query"])
        personas.append(persona_by_query.get(q, "tbd"))
    out["persona"] = personas
    return out


def attach_hop_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``hop_count`` column: total golden evidence items per question."""
    out = df.copy()
    counts = []
    for _, row in out.iterrows():
        n_mem = len(parse_evidence_list(row.get("golden_memory_evidence")))
        n_doc = len(parse_evidence_list(row.get("golden_document_evidence")))
        counts.append(n_mem + n_doc)
    out["hop_count"] = counts
    return out


def _check_min_coverage(
    df: pd.DataFrame,
    covariate: str,
    minimum: int,
) -> dict[str, int]:
    """Return {level: count} for levels below *minimum*. Empty dict = all OK."""
    if covariate not in df.columns:
        return {}
    vc = df[covariate].value_counts()
    return {str(lvl): int(cnt) for lvl, cnt in vc.items() if cnt < minimum}


def _proportional_allocation(
    df: pd.DataFrame, total: int, personas: list[str],
) -> dict[str, int]:
    """Split *total* across personas proportionally to pool sizes.

    Remainders are distributed largest-remainder-first so the sum is exact.
    """
    pools = {pid: len(df[df["persona"] == pid]) for pid in personas}
    pool_total = sum(pools.values())
    raw = {pid: total * n / pool_total for pid, n in pools.items()}

    floored = {pid: int(v) for pid, v in raw.items()}
    remainder = total - sum(floored.values())
    fracs = sorted(personas, key=lambda p: raw[p] - floored[p], reverse=True)
    for pid in fracs[:remainder]:
        floored[pid] += 1
    return floored


def stratified_sample(
    df: pd.DataFrame,
    n_per_persona: int | None,
    seed: int,
    personas: list[str] | None = None,
    min_per_covariate: int = 0,
    total: int = 200,
) -> pd.DataFrame:
    """Draw questions from each persona group.

    When *n_per_persona* is ``None`` the *total* budget is split
    proportionally to each persona's pool size. Otherwise *n_per_persona*
    questions are drawn from each stratum (and *total* is ignored).

    When *min_per_covariate* > 0 the sampler first reserves enough questions
    from under-represented covariate levels (across all persona strata) to
    meet the floor, then fills the remaining quota with unrestricted random
    draws per persona.
    """
    if personas is None:
        personas = TARGET_PERSONAS

    if n_per_persona is not None:
        alloc = {pid: n_per_persona for pid in personas}
    else:
        alloc = _proportional_allocation(df, total, personas)

    rng_state = seed

    if min_per_covariate > 0:
        return _stratified_sample_with_coverage(
            df, alloc, rng_state, personas, min_per_covariate,
        )

    parts: list[pd.DataFrame] = []
    for pid in personas:
        n = alloc[pid]
        subset = df[df["persona"] == pid]
        available = len(subset)
        if available < n:
            raise ValueError(
                f"{pid} has only {available} questions, need {n}"
            )
        sampled = subset.sample(n=n, random_state=rng_state)
        parts.append(sampled)

    result = pd.concat(parts, axis=0, ignore_index=True)
    result = result.rename(columns={"persona": "original_persona"})
    return result


def _stratified_sample_with_coverage(
    df: pd.DataFrame,
    alloc: dict[str, int],
    seed: int,
    personas: list[str],
    minimum: int,
) -> pd.DataFrame:
    """Two-pass sampling: reserve rare covariate levels, then fill randomly.

    Pass 1 — across *all* persona strata, identify covariate levels that would
    likely fall below *minimum* in a purely random draw and pre-select enough
    questions carrying those levels.

    Pass 2 — within each persona stratum, fill up to the allocated count from
    the remaining (non-reserved) pool using unrestricted random sampling.
    """
    pool = df[df["persona"].isin(personas)].copy()
    total_sample = sum(alloc.values())
    reserved_idx: set[int] = set()

    for cov in COVARIATE_COLUMNS:
        if cov not in pool.columns:
            continue
        for level, group in pool.groupby(cov):
            if len(group) < minimum:
                reserved_idx.update(group.index.tolist())
                continue
            expected = len(group) / len(pool) * total_sample
            if expected < minimum:
                needed = minimum - int(expected)
                picks = group.sample(
                    n=min(needed, len(group)), random_state=seed,
                )
                reserved_idx.update(picks.index.tolist())

    parts: list[pd.DataFrame] = []
    for pid in personas:
        n_target = alloc[pid]
        persona_pool = pool[pool["persona"] == pid]
        persona_reserved = persona_pool[persona_pool.index.isin(reserved_idx)]
        persona_unreserved = persona_pool[~persona_pool.index.isin(reserved_idx)]

        n_already = len(persona_reserved)
        n_remaining = n_target - n_already

        if n_remaining < 0:
            picks = persona_reserved.sample(n=n_target, random_state=seed)
        elif n_remaining == 0:
            picks = persona_reserved
        else:
            available = len(persona_unreserved)
            if available < n_remaining:
                raise ValueError(
                    f"{pid}: need {n_remaining} more questions after reserving "
                    f"{n_already}, but only {available} remain"
                )
            fill = persona_unreserved.sample(n=n_remaining, random_state=seed)
            picks = pd.concat([persona_reserved, fill])

        parts.append(picks)

    result = pd.concat(parts, axis=0, ignore_index=True)
    result = result.rename(columns={"persona": "original_persona"})
    return result


def sample_null_queries(
    df: pd.DataFrame,
    n: int,
    seed: int,
    *,
    exclude_queries: set[str] | None = None,
) -> pd.DataFrame:
    """Randomly sample *n* ``null_query`` rows (appended after persona strata)."""
    if n <= 0:
        return pd.DataFrame()

    pool = df[df["question_type"].astype(str) == "null_query"]
    if exclude_queries:
        norm = pool["query"].map(_normalize_query_text)
        pool = pool[~norm.isin(exclude_queries)]

    available = len(pool)
    if available < n:
        raise ValueError(
            f"null_query pool has only {available} questions, need {n}"
        )

    sampled = pool.sample(n=n, random_state=seed).copy()
    sampled = sampled.rename(columns={"persona": "original_persona"})
    return sampled.reset_index(drop=True)


def print_covariate_coverage(df: pd.DataFrame, label: str) -> None:
    """Print distribution tables for question_type and hop_count."""
    print(f"\n── {label} ──")
    for cov in COVARIATE_COLUMNS:
        if cov not in df.columns:
            continue
        col = "original_persona" if "original_persona" in df.columns else "persona"
        vc = df[cov].value_counts().sort_index()
        print(f"  {cov}:")
        for lvl, cnt in vc.items():
            print(f"    {lvl}: {cnt}")

        cross = pd.crosstab(df[col], df[cov])
        print(f"  {cov} × persona:")
        print(cross.to_string().replace("\n", "\n    "))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified sample of 210 questions (eval personas + null_query)."
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=_DEFAULT_INPUT,
        help="Path to full_reasonable.pkl (default: data/questions/full_reasonable.pkl)",
    )
    parser.add_argument(
        "--persona-csv", type=Path, default=_DEFAULT_PERSONA_CSV,
        help="Persona CSV with query and persona columns",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=_DEFAULT_OUTPUT,
        help="Output pickle path (default: data/questions/experiment_210.pkl)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total", type=int, default=210,
                        help="Total questions to sample (split proportionally across personas)")
    parser.add_argument(
        "--n-per-persona", type=str, default=str(DEFAULT_N_PER_PERSONA),
        help="Questions per eval persona: 'auto' for proportional allocation, or an integer",
    )
    parser.add_argument(
        "--n-null-query", type=int, default=DEFAULT_N_NULL_QUERY,
        help="Additional null_query questions appended after persona strata (default: 12)",
    )
    parser.add_argument(
        "--min-per-covariate", type=int, default=5,
        help="Minimum questions per level of question_type and hop_count (default: 5)",
    )
    args = parser.parse_args()

    n_per_persona: int | None = None
    if args.n_per_persona != "auto":
        n_per_persona = int(args.n_per_persona)

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"Input not found: {inp}")
    csv_path = args.persona_csv.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Persona CSV not found: {csv_path}")

    print(f"Loading questions from {inp}")
    df = pd.read_pickle(inp)
    print(f"  {len(df)} questions loaded")

    print(f"Loading persona map from {csv_path}")
    persona_by_query = load_persona_map_by_query(csv_path)
    print(f"  {len(persona_by_query)} query→persona mappings")

    df = attach_persona_column(df, persona_by_query)
    df = attach_hop_count(df)

    all_personas = TARGET_PERSONAS + ["persona_4", "tbd"]
    counts = df["persona"].value_counts()
    print("Persona distribution in full set:")
    for pid in all_personas:
        print(f"  {pid}: {int(counts.get(pid, 0))}")

    eligible = df[df["persona"].isin(TARGET_PERSONAS)]
    print(f"\nEligible pool (eval personas only): {len(eligible)} questions")
    print_covariate_coverage(eligible, "Full pool covariate coverage")

    persona_total = args.total - args.n_null_query
    if persona_total < 0:
        raise ValueError(
            f"--total ({args.total}) must be >= --n-null-query ({args.n_null_query})"
        )

    alloc_label = (
        f"proportional (persona budget={persona_total})"
        if n_per_persona is None
        else f"{n_per_persona} per persona"
    )

    print(f"\nStratified sampling: {alloc_label}, seed={args.seed}")
    print(f"  min_per_covariate={args.min_per_covariate}")
    print(f"  n_null_query={args.n_null_query}")
    persona_sampled = stratified_sample(
        df, n_per_persona, args.seed,
        min_per_covariate=args.min_per_covariate,
        total=persona_total,
    )
    print(f"  {len(persona_sampled)} persona-stratified questions sampled")

    already = {_normalize_query_text(q) for q in persona_sampled["query"]}
    null_sampled = sample_null_queries(
        df,
        args.n_null_query,
        args.seed,
        exclude_queries=already,
    )
    if len(null_sampled):
        print(f"  {len(null_sampled)} null_query questions sampled")

    sampled = pd.concat([persona_sampled, null_sampled], axis=0, ignore_index=True)
    print(f"  {len(sampled)} questions total")

    counts_out = sampled["original_persona"].value_counts()
    print("Sampled original_persona distribution:")
    for pid in TARGET_PERSONAS + ["tbd"]:
        print(f"  {pid}: {int(counts_out.get(pid, 0))}")

    print_covariate_coverage(sampled, "Sampled covariate coverage")

    for cov in COVARIATE_COLUMNS:
        violations = _check_min_coverage(sampled, cov, args.min_per_covariate)
        if violations:
            print(f"  WARNING: {cov} levels below minimum ({args.min_per_covariate}):")
            for lvl, cnt in violations.items():
                print(f"    {lvl}: {cnt}")
        else:
            print(f"  {cov}: all levels ≥ {args.min_per_covariate} ✓")

    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_pickle(out_path)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
