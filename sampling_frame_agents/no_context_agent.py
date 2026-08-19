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
from .prompts import SYSTEM_PROMPT_NO_CONTEXT, build_question_only_user_prompt
from .schemas import AgentAnswer, InferenceResult, TokenUsage


class NoContextAgent(BaseAgent):
    """Question-only baseline: no evidence in the prompt.

    Answers use general knowledge only; compare to evidence-backed runs to
    study contamination or retrieval effects offline.
    """

    def __init__(
        self,
        model_name: str,
        *,
        temperature: float = 0.0,
        seed: int = 42,
        test_mode: bool = False,
    ) -> None:
        super().__init__(test_mode=test_mode)
        self.model_name = model_name
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            seed=seed,
        ).with_structured_output(AgentAnswer, include_raw=True)

    def answer_question(self, question_data: dict) -> InferenceResult:
        doc_evidence = coalesce_evidence(question_data.get("evidence_list"))
        mem_evidence = coalesce_evidence(question_data.get("memory_evidence"))
        user_msg = build_question_only_user_prompt(question_data["query"])
        return self._call_llm(
            SYSTEM_PROMPT_NO_CONTEXT,
            user_msg,
            golden_memory_evidence=mem_evidence,
            golden_document_evidence=doc_evidence,
            retrieved_evidence=[],
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
    parser = argparse.ArgumentParser(description="No-context baseline inference.")
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
        "--questions-pkl",
        default="data/sampled_questions.pkl",
        help="Pickle of questions DataFrame (same schema as sampled_questions).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Destination CSV path. Default: data/no_context_agent_inferences/<model>.csv",
    )
    args = parser.parse_args()
    out_csv = args.output_csv or f"data/no_context_agent_inferences/{args.model_name}.csv"
    agent = NoContextAgent(
        model_name=args.model_name,
        temperature=args.temperature,
        seed=args.seed,
        test_mode=args.test_mode,
    )
    agent.run(
        questions_pkl_path=args.questions_pkl,
        output_csv_path=out_csv,
    )


if __name__ == "__main__":
    main()
