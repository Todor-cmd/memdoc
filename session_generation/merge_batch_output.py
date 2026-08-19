"""Merge OpenAI batch output JSONL with a batch manifest into per-session records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from session_generation.schemas import Session


def _session_from_response_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (session_dict, error_info) — at most one is non-None."""
    resp = row.get("response")
    if resp is None:
        return None, {"kind": "no_response", "row_error": row.get("error")}
    body = resp.get("body")
    if not body:
        return None, {"kind": "empty_body", "status_code": resp.get("status_code")}
    out = body.get("output") or []
    if not out:
        return None, {"kind": "no_output"}
    content = out[0].get("content") or []
    if not content:
        return None, {"kind": "no_content"}
    text = content[0].get("text")
    if not text:
        return None, {"kind": "no_text"}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        return None, {"kind": "json_decode", "detail": str(e)}
    try:
        session = Session.model_validate(raw)
    except Exception as e:
        return None, {"kind": "session_validate", "detail": str(e)}
    return session.model_dump(), None


def merge_batch_output_to_records(
    manifest_path: str | Path,
    output_jsonl_path: str | Path,
) -> list[dict[str, Any]]:
    """Join manifest metadata to each batch result line by ``custom_id``.

    Each returned dict includes manifest fields plus ``session`` (parsed Session
    as a dict) and optionally ``session_parse_error`` if the response could not
    be parsed.
    """
    manifest_path = Path(manifest_path)
    output_jsonl_path = Path(output_jsonl_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest: dict[str, dict[str, Any]] = json.load(f)
    results: list[dict[str, Any]] = []
    with open(output_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            custom_id = row["custom_id"]
            if custom_id not in manifest:
                raise KeyError(
                    f"custom_id {custom_id!r} not in manifest from {manifest_path}"
                )
            meta = dict(manifest[custom_id])
            session_dict, err = _session_from_response_row(row)
            rec: dict[str, Any] = {
                "custom_id": custom_id,
                **meta,
                "session": session_dict,
            }
            if err is not None:
                rec["session_parse_error"] = err
            if row.get("error") is not None:
                rec["batch_line_error"] = row["error"]
            results.append(rec)
    return results
