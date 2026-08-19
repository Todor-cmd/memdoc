from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUESTIONS_PICKLE = REPO_ROOT / "data" / "questions" / "full_reasonable.pkl"
DEFAULT_EXPERIMENT_QUESTIONS_PICKLE = (
    REPO_ROOT / "data" / "questions" / "experiment_210_split.pkl"
)
DEFAULT_MEMORY_DIR = REPO_ROOT / "data" / "memory_collection"
DEFAULT_DOCUMENT_CORPUS = REPO_ROOT / "data" / "document_collection" / "multihop_corpus.jsonl"
DEFAULT_EVIDENCE_ID_TO_URL = REPO_ROOT / "data" / "document_collection" / "evidence_id_to_url.jsonl"
DEFAULT_EXPERIMENT_DESIGN_CSV = REPO_ROOT / "data" / "experiment_design.csv"
DEFAULT_EXPERIMENT_OUTPUT_DIR = REPO_ROOT / "data" / "experiment_runs"
