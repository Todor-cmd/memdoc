"""Run Groq structured evidence-field audits over memory sessions.

Example:
    python -m session_generation.session_audit.run_audit \\
      --memory-dir data/memory_collection \\
      --manifest data/batch_jobs/experiment_sessions/batch_manifest_20260607_145849.json \\
      --out data/session_audit/evidence_field_locations.jsonl \\
      --model openai/gpt-oss-120b \\
      --limit 5
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import groq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from tqdm import tqdm

from session_generation.session_audit.join import AuditRow, build_audit_rows, load_manifest
from session_generation.session_audit.prompts import SYSTEM_PROMPT, build_user_prompt
from session_generation.session_audit.schemas import (
    EvidenceFieldLocations,
    EvidenceFieldLocationsTemporal,
    all_required_present,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MEMORY = _REPO_ROOT / "data" / "memory_collection"
_DEFAULT_MANIFEST = (
    _REPO_ROOT
    / "data"
    / "batch_jobs"
    / "experiment_sessions"
    / "batch_manifest_20260607_145849.json"
)
_DEFAULT_OUT = (
    _REPO_ROOT / "data" / "session_audit" / "evidence_field_locations.jsonl"
)


def _invoke_with_retry(llm: Any, messages: list, *, max_tries: int = 6) -> Any:
    """Exponential backoff on Groq rate-limit / transient API errors."""
    delay = 1.0
    last_exc: Optional[BaseException] = None
    for attempt in range(max_tries):
        try:
            return llm.invoke(messages)
        except groq.BadRequestError:
            raise
        except (groq.RateLimitError, groq.APIError) as exc:
            last_exc = exc
            if attempt + 1 >= max_tries:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 60.0)
    assert last_exc is not None
    raise last_exc


def _load_done_ids(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("session_id")
            if sid:
                done.add(str(sid))
    return done


def _usage_dict(raw: Any) -> Optional[dict[str, int]]:
    if raw is None:
        return None
    meta = getattr(raw, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if not usage and hasattr(raw, "usage_metadata") and raw.usage_metadata:
        um = raw.usage_metadata
        return {
            "input_tokens": int(um.get("input_tokens") or 0),
            "output_tokens": int(um.get("output_tokens") or 0),
            "total_tokens": int(um.get("total_tokens") or 0),
        }
    if not usage:
        return None
    return {
        "input_tokens": int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or 0
        ),
        "output_tokens": int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        ),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _parse_failed_generation(exc: BaseException) -> Optional[dict[str, Any]]:
    """Recover JSON from Groq tool_use_failed ``failed_generation`` payloads."""
    body = getattr(exc, "body", None)
    err = None
    if isinstance(body, dict):
        err = body.get("error")
    if err is None:
        # langchain / groq sometimes stringify the body into the message
        msg = str(exc)
        marker = "failed_generation"
        if marker not in msg:
            return None
    if isinstance(err, dict):
        raw = err.get("failed_generation")
    else:
        raw = None
        msg = str(exc)
        # message embeds a Python-repr dict; pull failed_generation string
        m = re.search(r"'failed_generation':\s*('(?:\\.|[^'])*')", msg)
        if not m:
            m = re.search(r'"failed_generation":\s*"((?:\\.|[^"])*)"', msg)
            if m:
                raw = bytes(m.group(1), "utf-8").decode("unicode_escape")
        else:
            try:
                raw = ast.literal_eval(m.group(1))
            except (SyntaxError, ValueError):
                raw = None
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _locations_from_dict(
    data: dict[str, Any],
    *,
    is_temporal: bool,
) -> EvidenceFieldLocations:
    """Build a locations model from a recovered / parsed dict."""
    payload = {
        "topic": list(data.get("topic") or []),
        "source": list(data.get("source") or []),
        "key_information": list(data.get("key_information") or []),
        "notes": str(data.get("notes") or ""),
    }
    if is_temporal:
        payload["published_at"] = list(data.get("published_at") or [])
        return EvidenceFieldLocationsTemporal(**payload)
    return EvidenceFieldLocations(**payload)


def _locations_to_dict(
    locations: EvidenceFieldLocations,
    *,
    is_temporal: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "topic": list(locations.topic),
        "source": list(locations.source),
        "key_information": list(locations.key_information),
        "notes": locations.notes,
        "all_required_present": all_required_present(
            locations, is_temporal=is_temporal
        ),
    }
    if is_temporal:
        out["published_at"] = list(
            getattr(locations, "published_at", None) or []
        )
    else:
        out["published_at"] = None
    return out


class SessionEvidenceAuditor:
    """Groq structured-output auditor for one session at a time."""

    def __init__(self, model_name: str, *, temperature: float = 0.0):
        self.model_name = model_name
        kwargs = dict(model=model_name, temperature=temperature)
        self._llm_non_temporal = ChatGroq(**kwargs).with_structured_output(
            EvidenceFieldLocations, include_raw=True
        )
        self._llm_temporal = ChatGroq(**kwargs).with_structured_output(
            EvidenceFieldLocationsTemporal, include_raw=True
        )

    def audit(self, row: AuditRow) -> dict[str, Any]:
        user_msg = build_user_prompt(
            evidence=row.evidence,
            session_turns=row.session_turns,
            is_temporal=row.is_temporal,
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
        llm = self._llm_temporal if row.is_temporal else self._llm_non_temporal

        base = {
            "session_id": row.session_id,
            "evidence_id": row.evidence_id,
            "source_question_id": row.source_question_id,
            "is_temporal": row.is_temporal,
            "persona_file": row.persona_file,
            "model": self.model_name,
        }

        try:
            out = _invoke_with_retry(llm, messages)
            parsed = out.get("parsed") if isinstance(out, dict) else None
            raw = out.get("raw") if isinstance(out, dict) else None
            if parsed is None:
                raise RuntimeError("structured output returned no parsed result")
            locs = _locations_to_dict(parsed, is_temporal=row.is_temporal)
            return {
                **base,
                **locs,
                "usage": _usage_dict(raw),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 — persist and continue
            recovered = _parse_failed_generation(exc)
            if recovered is not None:
                try:
                    model = _locations_from_dict(
                        recovered, is_temporal=row.is_temporal
                    )
                    locs = _locations_to_dict(
                        model, is_temporal=row.is_temporal
                    )
                    return {
                        **base,
                        **locs,
                        "usage": None,
                        "error": None,
                        "recovered_from_tool_use_failed": True,
                    }
                except Exception:  # noqa: BLE001
                    pass
            msg = f"{type(exc).__name__}: {exc}"
            if len(msg) > 2000:
                msg = msg[:2000] + "..."
            return {
                **base,
                "topic": [],
                "source": [],
                "key_information": [],
                "published_at": [] if row.is_temporal else None,
                "notes": "",
                "all_required_present": False,
                "usage": None,
                "error": msg,
            }


def run_audit(
    *,
    memory_dir: Path,
    manifest_path: Path,
    out_path: Path,
    model: str,
    limit: Optional[int] = None,
) -> dict[str, int]:
    load_dotenv()
    manifest = load_manifest(manifest_path)
    rows, orphans = build_audit_rows(memory_dir, manifest)

    if orphans:
        orphan_path = out_path.with_name(out_path.stem + "_orphans.txt")
        orphan_path.write_text("\n".join(orphans) + "\n", encoding="utf-8")
        print(
            f"Skipped {len(orphans)} sessions with no manifest join "
            f"(wrote {orphan_path})"
        )

    done = _load_done_ids(out_path)
    pending = [r for r in rows if r.session_id not in done]
    if limit is not None:
        pending = pending[: max(0, limit)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    auditor = SessionEvidenceAuditor(model)

    n_ok = 0
    n_fail = 0
    n_present = 0

    with out_path.open("a", encoding="utf-8") as f:
        for row in tqdm(pending, desc="Auditing sessions"):
            result = auditor.audit(row)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if result.get("error"):
                n_fail += 1
            else:
                n_ok += 1
                if result.get("all_required_present"):
                    n_present += 1

    return {
        "joined": len(rows),
        "orphans": len(orphans),
        "already_done": len(done),
        "attempted": len(pending),
        "ok": n_ok,
        "failed": n_fail,
        "all_required_present": n_present,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit evidence-field presence in memory sessions (Groq)."
    )
    parser.add_argument("--memory-dir", type=Path, default=_DEFAULT_MEMORY)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max new sessions to audit this run (resume-friendly).",
    )
    args = parser.parse_args(argv)

    if not args.memory_dir.is_dir():
        print(f"Memory dir not found: {args.memory_dir}", file=sys.stderr)
        return 1
    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    stats = run_audit(
        memory_dir=args.memory_dir,
        manifest_path=args.manifest,
        out_path=args.out,
        model=args.model,
        limit=args.limit,
    )
    print(
        "Audit complete: "
        f"joined={stats['joined']}, orphans={stats['orphans']}, "
        f"already_done={stats['already_done']}, attempted={stats['attempted']}, "
        f"ok={stats['ok']}, failed={stats['failed']}, "
        f"all_required_present={stats['all_required_present']}"
    )
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
