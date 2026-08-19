from __future__ import annotations

import argparse
from typing import Literal

from .base_agent import coalesce_evidence
from .golden_context_agent import GoldenContextAgent
from .prompts import SYSTEM_PROMPT, build_user_prompt, format_evidence
from .schemas import InferenceResult

EvidencePool = Literal["document", "memory"]


class PartiallyGoldenAgent(GoldenContextAgent):
    """Oracle baseline with only one gold pool in the prompt (document or memory).

    Full gold labels are still recorded in ``golden_document_evidence`` /
    ``golden_memory_evidence``; ``retrieved_evidence`` is only the pool shown
    to the model (same convention as :class:`GoldenContextAgent`).
    """

    def __init__(
        self,
        model_name: str,
        *,
        evidence_pool: EvidencePool,
        temperature: float = 0.0,
        seed: int = 42,
        test_mode: bool = False,
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            seed=seed,
            test_mode=test_mode,
        )
        self.evidence_pool = evidence_pool

    def answer_question(self, question_data: dict) -> InferenceResult:
        doc_evidence = coalesce_evidence(question_data.get("evidence_list"))
        mem_evidence = coalesce_evidence(question_data.get("memory_evidence"))
        if self.evidence_pool == "document":
            prompt_evidence = doc_evidence
        else:
            prompt_evidence = mem_evidence

        temporal = question_data.get("question_type") == "temporal_query"
        evidence_block = format_evidence(prompt_evidence, temporal=temporal)
        user_msg = build_user_prompt(question_data["query"], evidence_block)

        return self._call_llm(
            SYSTEM_PROMPT,
            user_msg,
            golden_memory_evidence=mem_evidence,
            golden_document_evidence=doc_evidence,
            retrieved_evidence=prompt_evidence,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partial golden-evidence inference (document-only or memory-only pool).",
    )
    parser.add_argument(
        "--evidence-pool",
        choices=("document", "memory"),
        required=True,
        help="Which gold pool to put in the prompt.",
    )
    parser.add_argument(
        "--model-name",
        default="llama-3.3-70b-versatile",
        help="Groq chat model id.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Smoke run: null_query + temporal_query + one other question type.",
    )
    args = parser.parse_args()
    agent = PartiallyGoldenAgent(
        model_name=args.model_name,
        evidence_pool=args.evidence_pool,
        temperature=args.temperature,
        seed=args.seed,
        test_mode=args.test_mode,
    )
    out_name = f"{args.model_name}_{args.evidence_pool}.csv"
    agent.run(
        questions_pkl_path="data/sampled_questions.pkl",
        output_csv_path=f"data/partially_golden_agent_inferences/{out_name}",
    )


if __name__ == "__main__":
    main()
