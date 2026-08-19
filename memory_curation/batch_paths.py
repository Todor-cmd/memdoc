"""Resolve batch manifest / output paths (latest file or explicit default)."""
from __future__ import annotations

from pathlib import Path


def resolve_manifest_and_jsonl(
    manifest: Path,
    jsonl: Path,
    *,
    job_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Return manifest + jsonl paths, falling back to newest files in *job_dir*."""
    manifest = manifest.expanduser().resolve()
    jsonl = jsonl.expanduser().resolve()

    if manifest.is_file() and jsonl.is_file():
        return manifest, jsonl

    if job_dir is None:
        job_dir = manifest.parent

    job_dir = job_dir.expanduser().resolve()
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Batch job directory not found: {job_dir}")

    manifests = sorted(job_dir.glob("batch_manifest_*.json"), reverse=True)
    jsonls = sorted(job_dir.glob("batch_output*.jsonl"), reverse=True)
    if not jsonls:
        jsonls = sorted(job_dir.glob("batch_output.jsonl"), reverse=True)

    if manifests and jsonls:
        return manifests[0], jsonls[0]

    raise FileNotFoundError(
        f"No batch manifest/jsonl found under {job_dir}. "
        "Run session_generation.create_batch_job first."
    )
