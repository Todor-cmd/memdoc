# Persona 1: Tech. All the questions that require only tech domain evidence are mapped to this Persona.
# Persona 2: Business & Tech. All the questions that require exactly both business and tech domain evidence are mapped to this Persona.
# Persona 3: Sports. All the questions that require only sports domain evidence are mapped to this Persona.
# Persona 4: Collect all the remaining evidence domains. This is a Persona with an intrest in all of these domains. All the remaining questions get mapped to this Persona.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_evidence_list(val: Any) -> list[dict[str, Any]]:
    """Normalize golden_memory_evidence / golden_document_evidence to a list of dicts."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        return json.loads(s)
    return []


def evidence_domains_for_row(row: pd.Series) -> frozenset[str]:
    """Unique `category` tags from all memory and document golden evidence pieces."""
    domains: set[str] = set()
    for ev in parse_evidence_list(row["golden_memory_evidence"]) + parse_evidence_list(
        row["golden_document_evidence"]
    ):
        if isinstance(ev, dict):
            c = ev.get("category")
            if c:
                domains.add(str(c).strip().lower())
    return frozenset(domains)


def persona_for_evidence_domains(domains: frozenset[str]) -> str:
    """
    Map the set of evidence domain tags to a persona id.

    Uses MultiHop-RAG evidence `category` values: business, entertainment, health,
    science, sports, technology.
    """
    if not domains:
        return "tbd"
    if domains == frozenset({"technology"}):
        return "persona_1"
    if domains == frozenset({"business", "technology"}):
        return "persona_2"
    if domains == frozenset({"sports"}):
        return "persona_3"
    return "persona_4"


def add_persona_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `evidence_domains` (sorted tuple) and `persona` columns."""
    out = df.copy()
    dom_series = out.apply(evidence_domains_for_row, axis=1)
    out["evidence_domains"] = dom_series.apply(lambda d: tuple(sorted(d)))
    out["persona"] = dom_series.apply(persona_for_evidence_domains)
    return out


def persona_to_domains_table(enriched: pd.DataFrame) -> pd.DataFrame:
    """One row per persona: union of all evidence domain tags in questions mapped to that persona."""
    order = ["persona_1", "persona_2", "persona_3", "persona_4", "tbd"]
    rows: list[dict[str, str]] = []
    for pid in order:
        subset = enriched[enriched["persona"] == pid]
        tags: set[str] = set()
        for domains in subset["evidence_domains"]:
            tags.update(domains)
        rows.append({"persona": pid, "domains": "|".join(sorted(tags))})
    return pd.DataFrame(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Map each reasonable question to a persona from golden evidence domain tags."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=repo_root / "data" / "questions" / "full_reasonable.pkl",
        help="Pickle of reasonable questions (default: data/questions/full_reasonable.pkl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=repo_root / "data" / "persona_metadata" / "q_2_personas.csv",
        help="Output CSV with evidence_domains and persona columns (default: data/persona_metadata/q_2_personas.csv)",
    )
    args = parser.parse_args()
    inp = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"Input not found: {inp}")

    df = pd.read_pickle(inp)
    enriched = add_persona_columns(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_df = enriched.copy()
    export_df["evidence_domains"] = export_df["evidence_domains"].apply(
        lambda t: "|".join(t) if t else ""
    )
    export_df.to_csv(out, index=False)
    print(f"Wrote {len(export_df)} rows to {out}")

    persona_domains_path = out.parent / "persona_to_domains.csv"
    persona_to_domains_table(enriched).to_csv(persona_domains_path, index=False)
    print(f"Wrote persona → domain unions to {persona_domains_path}")
    counts = enriched["persona"].value_counts()
    order = ["persona_1", "persona_2", "persona_3", "persona_4", "tbd"]
    print("Questions per persona:")
    for pid in order:
        n = int(counts.get(pid, 0))
        print(f"  {pid}: {n}")
    print(f"  total: {len(enriched)}")

    p4 = enriched[enriched["persona"] == "persona_4"]
    combo_counts = p4["evidence_domains"].value_counts().sort_index()
    print("\nPersona 4 — distinct domain combinations (count of questions):")
    for domains, n in combo_counts.items():
        label = "|".join(domains) if domains else "(empty)"
        print(f"  {label}: {int(n)}")
    all_p4_tags: set[str] = set()
    for domains in p4["evidence_domains"]:
        all_p4_tags.update(domains)
    print("Persona 4 — all domain tags that appear (union):")
    print(f"  {', '.join(sorted(all_p4_tags))}")


if __name__ == "__main__":
    main()
