from __future__ import annotations

import argparse
from typing import Dict, List, Optional
import backoff
import groq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from .base_agent import (
    BaseAgent,
    coalesce_evidence,
    groq_giveup_bad_request,
    inference_result_from_failure,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt, format_evidence
from .schemas import AgentAnswer, InferenceResult, TokenUsage


class GoldenContextAgent(BaseAgent):
    """Golden-evidence baseline: all evidence documents in the prompt."""

    def __init__(
        self,
        model_name: str,
        *,
        temperature: float = 0.0,
        seed: int = 42,
        test_mode: bool = False,
        include_author: bool = True,
    ) -> None:
        super().__init__(test_mode=test_mode)
        self.model_name = model_name
        self.include_author = include_author
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            seed=seed,
        ).with_structured_output(AgentAnswer, include_raw=True)

    def answer_question(self, question_data: dict) -> InferenceResult:
        doc_evidence = coalesce_evidence(question_data.get("evidence_list"))
        mem_evidence = coalesce_evidence(question_data.get("memory_evidence"))
        all_evidence = doc_evidence + mem_evidence
        temporal = question_data.get("question_type") == "temporal_query"

        evidence_block = format_evidence(
            all_evidence, temporal=temporal, include_author=self.include_author,
        )
        user_msg = build_user_prompt(question_data["query"], evidence_block)

        return self._call_llm(
            SYSTEM_PROMPT,
            user_msg,
            golden_memory_evidence=mem_evidence,
            golden_document_evidence=doc_evidence,
            retrieved_evidence=all_evidence,
        )

    @backoff.on_exception(
        backoff.expo,
        (groq.RateLimitError, groq.APIError),
        max_tries=5,
        giveup=groq_giveup_bad_request,
    )
    def _call_llm(
        self,
        system: str,
        user: str,
        *,
        golden_memory_evidence: List[Dict],
        golden_document_evidence: List[Dict],
        retrieved_evidence: List[Dict],
    ) -> InferenceResult:
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        try:
            out = self.llm.invoke(messages)
            if out["parsing_error"] is not None:
                raise out["parsing_error"]
            parsed = out["parsed"]
            if parsed is None:
                raise RuntimeError("structured output returned no parsed result")
            raw = out["raw"]
            meta = getattr(raw, "usage_metadata", None)
            usage: Optional[TokenUsage] = None
            if meta:
                usage = TokenUsage(
                    input_tokens=meta.get("input_tokens"),
                    output_tokens=meta.get("output_tokens"),
                    total_tokens=meta.get("total_tokens"),
                )
            return InferenceResult(
                answer=parsed,
                usage=usage,
                golden_memory_evidence=golden_memory_evidence,
                golden_document_evidence=golden_document_evidence,
                retrieved_evidence=retrieved_evidence,
            )
        except groq.BadRequestError as exc:
            return inference_result_from_failure(
                exc,
                golden_memory_evidence=golden_memory_evidence,
                golden_document_evidence=golden_document_evidence,
                retrieved_evidence=retrieved_evidence,
            )

def main() -> None:
    parser = argparse.ArgumentParser(description="Golden-evidence baseline inference.")
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
    parser.add_argument(
        "--no-author",
        action="store_true",
        help="Exclude author field from the evidence context.",
    )
    parser.add_argument(
        "--questions-pkl",
        default="data/sampled_questions.pkl",
        help="Pickle of questions DataFrame (same schema as sampled_questions).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Destination CSV path. Default: "
            "data/golden_context_agent_inferences/<model>[_no_author].csv"
        ),
    )
    args = parser.parse_args()
    suffix = "_no_author" if args.no_author else ""
    out_csv = args.output_csv or (
        f"data/golden_context_agent_inferences/{args.model_name}{suffix}.csv"
    )
    agent = GoldenContextAgent(
        model_name=args.model_name,
        temperature=args.temperature,
        seed=args.seed,
        test_mode=args.test_mode,
        include_author=not args.no_author,
    )
    agent.run(
        questions_pkl_path=args.questions_pkl,
        output_csv_path=out_csv,
    )


if __name__ == "__main__":
    main()