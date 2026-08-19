"""URL → gold evidence fields from the experiment batch manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocEvidence:
    url: str
    title: str
    source: str
    fact: str


def load_doc_evidence_by_url(manifest_path: Path) -> dict[str, DocEvidence]:
    """Build ``url → DocEvidence`` from manifest evidence dicts (last write wins)."""
    out: dict[str, DocEvidence] = {}
    if not manifest_path.is_file():
        return out

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        return out

    for rec in manifest.values():
        if not isinstance(rec, dict):
            continue
        ev = rec.get("evidence")
        if not isinstance(ev, dict):
            continue
        url = str(ev.get("url") or "").strip()
        if not url:
            continue
        out[url] = DocEvidence(
            url=url,
            title=str(ev.get("title") or "").strip(),
            source=str(ev.get("source") or "").strip(),
            fact=str(ev.get("fact") or "").strip(),
        )
    return out


def evidence_payload(ev: DocEvidence) -> dict[str, Any]:
    return {"title": ev.title, "source": ev.source, "fact": ev.fact}
