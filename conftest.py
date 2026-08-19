import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Shared test fixtures for agent store tests
# ---------------------------------------------------------------------------


def _make_session(session_id: str, text: str, evidence_id: str = "") -> dict:
    return {
        "session_id": session_id,
        "source_question_id": "q-test",
        "evidence_id": evidence_id,
        "session": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"Response about: {text}"},
        ],
        "source": "in_domain_evidence",
        "is_off_topic": False,
        "date": "2023/10/01 (Sun) 12:00",
    }


def _make_doc(url: str, title: str, body: str) -> dict:
    return {
        "url": url,
        "title": title,
        "body": body,
        "category": "technology",
        "author": "Test Author",
        "source": "TestSource",
        "published_at": "2023-10-01T00:00:00+00:00",
    }


FIXTURE_SESSIONS = [
    _make_session("sess_tech_1", "How does GPU acceleration work for deep learning?", "ev_001"),
    _make_session("sess_tech_2", "What are the best practices for containerizing microservices?", "ev_002"),
    _make_session("sess_sports_1", "Who won the Premier League last season?", "ev_003"),
    _make_session("sess_gold", "The Verge reported on best headphone deals in December 2023", "ev_gold"),
]

FIXTURE_DOCS = [
    _make_doc("http://example.com/gpu", "GPU Deep Learning", "GPU acceleration enables faster training of neural networks using parallel compute."),
    _make_doc("http://example.com/containers", "Containerization Guide", "Docker and Kubernetes are the industry standard for deploying microservices."),
    _make_doc("http://example.com/headphones", "Best Headphone Deals", "The Verge compiled the best wireless earbuds deals for holiday shoppers in 2023."),
]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create minimal persona JSON and corpus JSONL in a temp directory."""
    memory_dir = tmp_path / "memory_collection"
    memory_dir.mkdir()
    (memory_dir / "persona_1.json").write_text(json.dumps(FIXTURE_SESSIONS))

    corpus_path = tmp_path / "corpus.jsonl"
    with corpus_path.open("w") as f:
        for doc in FIXTURE_DOCS:
            f.write(json.dumps(doc) + "\n")

    return tmp_path
