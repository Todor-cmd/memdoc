from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prepare_data.questions_to_personas import add_persona_columns  # noqa: E402


def load_reasonable_questions(
    path: Path | None = None,
    *,
    with_persona: bool = True,
) -> pd.DataFrame:
    """Load ``full_reasonable.pkl`` and optionally add ``persona`` / ``evidence_domains``."""
    from .paths import DEFAULT_QUESTIONS_PICKLE

    p = Path(path or DEFAULT_QUESTIONS_PICKLE).expanduser().resolve()
    df = pd.read_pickle(p)
    if with_persona:
        if "persona" not in df.columns:
            df = add_persona_columns(df)
    return df


def load_persona_background_session_ids(persona_json: Path) -> frozenset[str]:
    """``session_id`` values from a persona memory JSON array."""
    raw = json.loads(Path(persona_json).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array in {persona_json}")
    out: set[str] = set()
    for rec in raw:
        if isinstance(rec, dict) and rec.get("session_id"):
            out.add(str(rec["session_id"]))
    return frozenset(out)


def load_evidence_id_to_url(path: Path) -> dict[str, str]:
    """evidence_id -> url from JSONL (one object per line)."""
    by_id: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec: dict[str, Any] = json.loads(line)
            eid = str(rec.get("evidence_id", "")).strip()
            url = str(rec.get("url", "")).strip()
            if eid and url:
                by_id[eid] = url
    return by_id


def sort_questions_by_persona(df: pd.DataFrame) -> pd.DataFrame:
    """Stable sort: persona order persona_1..4, then ``question_idx``."""
    order = ["persona_1", "persona_2", "persona_3", "persona_4", "tbd"]

    persona_col = "original_persona" if "original_persona" in df.columns else "persona"

    def persona_key(p: Any) -> int:
        s = str(p).strip() if p is not None and not (isinstance(p, float) and pd.isna(p)) else "tbd"
        return order.index(s) if s in order else len(order)

    out = df.copy()
    out["_persona_ord"] = out[persona_col].map(persona_key)
    out = out.sort_values(by=["_persona_ord", "question_idx"], kind="stable").drop(
        columns=["_persona_ord"]
    )
    return out.reset_index(drop=True)
