"""Prompts for the per-chunk corrective RAG pipeline.

Includes per-chunk grading (plain text), sufficiency assessment (structured),
gap-driven dual-query rewrite (structured), and generation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-chunk grading (plain text response -- variable-length ID list)
# ---------------------------------------------------------------------------

GRADING_SYSTEM = """\
You are a retrieval evaluator. For each numbered passage below, determine \
whether it contains information useful for answering the user's question. \
A passage is relevant if it provides facts, context, or evidence that helps \
answer the question -- even partially. Be generous with partial relevance."""

GRADING_USER = """\
Question: {question}

{numbered_passages}

Which passages contain information useful for answering the question?
List relevant IDs (comma-separated), or "none":"""

# ---------------------------------------------------------------------------
# Sufficiency assessment (structured output: SufficiencyAssessment)
# ---------------------------------------------------------------------------

SUFFICIENCY_SYSTEM = """\
You are an evidence assessor. Given a question and collected evidence passages, \
determine whether the evidence is sufficient to answer the question. If \
insufficient, identify the specific knowledge gap -- what information is still \
missing. The gap should be a single concise sentence."""

SUFFICIENCY_USER = """\
Question: {question}

Evidence collected:
{context}"""

# ---------------------------------------------------------------------------
# Gap-driven dual-query rewrite (structured output: DualQueryRewrite)
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = """\
You are a query rewriter. You will be given a question and a description of \
what information is still missing. Rewrite the question into two retrieval \
queries that target this gap -- one for personal conversation memory and one \
for a document corpus."""

REWRITE_USER = """\
Original question: {question}
Knowledge gap: {knowledge_gap}"""

# ---------------------------------------------------------------------------
# Generation (structured output: AgentAnswer)
# ---------------------------------------------------------------------------

GENERATE_SYSTEM = """\
You are a precise question-answering system. You will be given a question and \
retrieved passages from conversation memory and a document corpus. Answer using \
only what is supported by the provided passages. If the passages are missing, \
insufficient, or do not support a definite answer, respond with exactly: \
Insufficient Information. Reason step-by-step, then give a short factual final \
answer (or that exact phrase)."""

GENERATE_USER = """\
PASSAGES:
{context}

QUESTION:
{question}"""
