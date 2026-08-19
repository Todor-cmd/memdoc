"""Helpers for qualitative notebooks (dist × topic inspection and gold-doc memory)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.naive_rag.gold_document_memory import evidence_id_from_session_id, url_for_session_id
from analysis.collect_evaluation_metadata import collect
from analysis.retrieval.audit_map import load_audit_map
from analysis.retrieval.doc_evidence import load_doc_evidence_by_url
from analysis.retrieval.pair_index import PairIndexCache, load_session_turns
from analysis.retrieval.strict_recall import (
    _DEFAULT_AUDIT,
    _DEFAULT_MANIFEST,
    _DEFAULT_MEMORY,
    _parse_retrieved,
    evidence_hit,
    gold_ids_for_row,
    strict_doc_hit,
    strict_mem_hit,
)
from experiment.agent_registry import AgentSpec, get_agent_spec
from experiment.design_runner import (
    filter_design_for_agent,
    load_design_matrix,
    run_design_experiment,
)
from experiment.paths import (
    DEFAULT_EXPERIMENT_DESIGN_CSV,
    DEFAULT_EXPERIMENT_QUESTIONS_PICKLE,
    REPO_ROOT,
)
from experiment.topical_relevance import label_dataframe as label_topical_relevance

SAMPLE_SEED = 42
N_EXAMPLES = 10
DIST_LEVELS = ("document_only", "memory_only", "integrated")
N_PER_DIST = 3
FEATURES_PATH = REPO_ROOT / "data" / "analysis" / "model_features_full.csv"
LABELED_RUNS_PATH = REPO_ROOT / "data" / "experiment_runs" / "labeled_runs.csv"
SEPARATE_STORE_AGENT = "agent_1"


def load_features(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or FEATURES_PATH)


def load_labeled_runs(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or LABELED_RUNS_PATH)


def confirmatory(df: pd.DataFrame) -> pd.DataFrame:
    if "question_type" not in df.columns:
        return df
    return df[df["question_type"] != "null_query"].copy()


def sample_dist_disagreement_triplets(
    features: pd.DataFrame,
    *,
    n_per_dist: int = N_PER_DIST,
    seed: int = SAMPLE_SEED,
    n: int | None = None,
) -> pd.DataFrame:
    """Sample disagreement units, **3 per evidence distribution** by default.

    Pool: ``(block_id, persona, agent)`` with ≥2 assigned dists and non-constant EM
    (source of variance for dist, with topical_relevance fixed on the unit).
    For each of document_only / memory_only / integrated, draw ``n_per_dist``
    units whose assigned cells include that dist, without replacement across
    strata.
    """
    if n is not None:
        n_per_dist = n
    df = confirmatory(features)
    records: list[dict[str, Any]] = []
    for (block_id, persona, agent), g in df.groupby(
        ["block_id", "persona", "agent"], sort=False
    ):
        if g["dist"].nunique() < 2:
            continue
        if g["correct"].nunique() < 2:
            continue
        dists = tuple(sorted(g["dist"].astype(str).unique()))
        records.append(
            {
                "block_id": int(block_id),
                "persona": persona,
                "agent": agent,
                "topical_relevance": g["topical_relevance"].iloc[0],
                "n_dist": int(len(dists)),
                "dists": ",".join(dists),
                "mean_correct": float(g["correct"].mean()),
            }
        )
    pool = pd.DataFrame.from_records(records)
    if pool.empty:
        raise ValueError("No (block_id, persona, agent) triplets with EM varying by dist")

    rng = np.random.default_rng(seed)
    remaining = pool.copy()
    picked: list[pd.DataFrame] = []
    for dist in DIST_LEVELS:
        eligible = remaining[remaining["dists"].str.contains(dist, regex=False)]
        if eligible.empty:
            continue
        k = min(n_per_dist, len(eligible))
        idx = rng.choice(eligible.index.to_numpy(), size=k, replace=False)
        take = remaining.loc[idx].copy()
        take["sampled_for_dist"] = dist
        picked.append(take)
        remaining = remaining.drop(index=idx)

    if not picked:
        raise ValueError("Dist-stratified sample was empty")
    return pd.concat(picked, ignore_index=True)


def sample_variable_questions(
    features: pd.DataFrame,
    *,
    agent: str = SEPARATE_STORE_AGENT,
    n: int = N_EXAMPLES,
    seed: int = SAMPLE_SEED,
) -> list[int]:
    """Questions whose EM on *agent* is neither always 0 nor always 1."""
    df = confirmatory(features)
    df = df[df["agent"] == agent]
    means = df.groupby("block_id")["correct"].mean()
    eligible = means[(means > 0) & (means < 1)].index.to_list()
    if not eligible:
        raise ValueError(f"No variable-difficulty questions for {agent}")
    ser = pd.Series(eligible)
    return ser.sample(n=min(n, len(ser)), random_state=seed).astype(int).tolist()


def rows_for_triplets(
    labeled: pd.DataFrame,
    triplets: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["block_id", "persona", "agent"]
    extra = [c for c in triplets.columns if c not in keys and c not in labeled.columns]
    merged = labeled.merge(triplets[keys + extra], on=keys, how="inner")
    return merged


def load_strict_resources():
    audit = load_audit_map(_DEFAULT_AUDIT)
    evidence_by_url = load_doc_evidence_by_url(_DEFAULT_MANIFEST)
    sessions = load_session_turns(_DEFAULT_MEMORY)
    pair_cache = PairIndexCache(sessions)
    return audit, evidence_by_url, pair_cache


def per_gold_hits(
    row: pd.Series,
    *,
    audit,
    evidence_by_url,
    pair_cache,
) -> pd.DataFrame:
    gold = gold_ids_for_row(row)
    docs = _parse_retrieved(row.get("retrieved_documents"))
    mems = _parse_retrieved(row.get("retrieved_memory"))
    records = []
    for g in gold:
        hit = evidence_hit(
            g,
            retrieved_docs=docs,
            retrieved_mem=mems,
            audit=audit,
            evidence_by_url=evidence_by_url,
            pair_cache=pair_cache,
        )
        records.append({"gold_id": g, "strict_hit": bool(hit)})
    return pd.DataFrame.from_records(records)


def memory_gold_identity_hit_for_session(
    session_id: str, retrieved_mem: list[dict[str, Any]], *, match: str
) -> bool:
    """Identity hit for one assigned memory-gold session (audit unit).

    ``match='session'``: a retrieved memory chunk has this ``session_id``.
    ``match='gold_document'``: a retrieved memory chunk has this session's article URL.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return False
    if match == "session":
        return any(str(m.get("session_id") or "").strip() == sid for m in retrieved_mem)
    if match == "gold_document":
        url = (url_for_session_id(sid) or "").strip()
        if not url:
            return False
        return any(str(m.get("url") or "").strip() == url for m in retrieved_mem)
    raise ValueError(f"match must be 'session' or 'gold_document', got {match!r}")


def explode_memory_gold_identity_hits(
    df: pd.DataFrame, *, match: str, col: str | None = None
) -> pd.DataFrame:
    """One row per assigned memory-gold session × experimental run.

    Same unit as ``build_audit_table``, but includes ``memory_only`` and
    ``integrated``. Document-store gold is ignored.
    """
    if col is None:
        col = "hit_gold_session" if match == "session" else "hit_gold_document"
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        sids = [g[5:] for g in gold_ids_for_row(row) if g.startswith("mem::")]
        mems = _parse_retrieved(row.get("retrieved_memory"))
        for sid in sids:
            records.append({
                "block_id": row.get("block_id"),
                "persona": row.get("persona"),
                "dist": row.get("dist"),
                "agent": row.get("agent"),
                "session_id": sid,
                "correct": row.get("correct"),
                "prediction": row.get("prediction"),
                col: memory_gold_identity_hit_for_session(sid, mems, match=match),
            })
    return pd.DataFrame.from_records(records)


def gold_doc_memory_evidence_hit(
    gold_id: str,
    *,
    retrieved_docs: list[dict[str, Any]],
    retrieved_mem: list[dict[str, Any]],
    audit,
    evidence_by_url,
    pair_cache,
) -> bool:
    """Like ``evidence_hit``, but memory gold can match via article text in memory."""
    if gold_id.startswith("doc::"):
        return strict_doc_hit(gold_id[5:], retrieved_docs, evidence_by_url)
    if gold_id.startswith("mem::"):
        sid = gold_id[5:]
        url = None
        for m in retrieved_mem:
            if str(m.get("session_id") or "").strip() == sid and m.get("url"):
                url = str(m["url"]).strip()
                break
        if not url:
            url = url_for_session_id(sid) or None
        if url and strict_doc_hit(url, retrieved_mem, evidence_by_url):
            return True
        return strict_mem_hit(sid, retrieved_mem, audit, pair_cache)
    return False


def add_gold_doc_memory_strict_recall(
    df: pd.DataFrame,
    *,
    audit=None,
    evidence_by_url=None,
    pair_cache=None,
) -> pd.DataFrame:
    if audit is None or evidence_by_url is None or pair_cache is None:
        audit, evidence_by_url, pair_cache = load_strict_resources()

    values = []
    for _, row in df.iterrows():
        gold = gold_ids_for_row(row)
        if not gold:
            values.append(float(np.nan))
            continue
        docs = _parse_retrieved(row.get("retrieved_documents"))
        mems = _parse_retrieved(row.get("retrieved_memory"))
        hits = sum(
            1
            for g in gold
            if gold_doc_memory_evidence_hit(
                g,
                retrieved_docs=docs,
                retrieved_mem=mems,
                audit=audit,
                evidence_by_url=evidence_by_url,
                pair_cache=pair_cache,
            )
        )
        values.append(hits / len(gold))
    out = df.copy()
    out["recall_full_context_strict_gold_doc"] = values
    return out


def truncate(text: Any, n: int = 500) -> str:
    s = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def format_retrieved(
    items: Any,
    *,
    gold_session_ids: set[str] | None = None,
    gold_urls: set[str] | None = None,
    limit: int = 10,
    text_chars: int = 280,
) -> str:
    parsed = _parse_retrieved(items)
    gold_session_ids = gold_session_ids or set()
    gold_urls = gold_urls or set()
    lines = []
    for i, item in enumerate(parsed[:limit], 1):
        sid = str(item.get("session_id") or "").strip()
        url = str(item.get("url") or "").strip()
        tags = []
        if sid and sid in gold_session_ids:
            tags.append("GOLD_MEM")
        if url and url in gold_urls:
            tags.append("GOLD_DOC")
        tag = f" [{' '.join(tags)}]" if tags else ""
        src = sid or url or "?"
        score = item.get("score")
        score_s = f" score={score}" if score is not None else ""
        lines.append(f"{i}.{tag} {src}{score_s}\n{truncate(item.get('text'), text_chars)}")
    return "\n\n".join(lines) if lines else "(none)"


def _md_cell(text: Any) -> str:
    s = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    return s.replace("|", "\\|").replace("\n", " ").strip()


def _md_fence(text: str) -> str:
    body = text.replace("```", "'''")
    return f"```\n{body}\n```"


def _failure_notes(prediction: str, hits: pd.DataFrame, correct: int) -> list[str]:
    pred = (prediction or "").strip()
    notes: list[str] = []
    if pred == "Insufficient Information":
        notes.append("abstained (Insufficient Information)")
    if pred == "[INFERENCE_FAILED]":
        notes.append("inference failed")
    if not hits.empty and hits["strict_hit"].all() and correct == 0:
        notes.append("strict recall full but EM fail (generation / formatting)")
    if not hits.empty and (not hits["strict_hit"].any()) and correct == 0:
        notes.append("no gold evidence in retrieved pool (retrieval miss)")
    if not hits.empty and hits["strict_hit"].any() and not hits["strict_hit"].all() and correct == 0:
        notes.append("partial strict recall")
    return notes


def _retrieved_markdown(
    items: Any,
    *,
    gold_session_ids: set[str],
    gold_urls: set[str],
    text_chars: int,
) -> str:
    parsed = _parse_retrieved(items)
    if not parsed:
        return "_None retrieved._\n"
    blocks: list[str] = []
    for i, item in enumerate(parsed, 1):
        sid = str(item.get("session_id") or "").strip()
        url = str(item.get("url") or "").strip()
        tags = []
        if sid and sid in gold_session_ids:
            tags.append("gold-mem")
        if url and url in gold_urls:
            tags.append("gold-doc")
        tag = ", ".join(tags) if tags else "distractor"
        src = sid or url or "?"
        score = item.get("score")
        score_s = f", score={score}" if score is not None else ""
        excerpt = truncate(item.get("text"), text_chars)
        blocks.append(
            f"{i}. **{tag}** `{src}`{score_s}\n\n" + _md_fence(excerpt)
        )
    return "\n\n".join(blocks) + "\n"


def _intended_gold_fact_text(session_id: str, evidence_by_url) -> str:
    """Gold fact the generated session was supposed to state (from the evidence record)."""
    eid = evidence_id_from_session_id(session_id) or ""
    url = url_for_session_id(session_id)
    ev = evidence_by_url.get(url) if url else None
    lines = [
        "Intended gold (what the session should have embedded):",
        f"evidence_id: {eid or '(unknown)'}",
    ]
    if ev is None:
        lines.append(f"url: {url or '(unknown)'}")
        lines.append("(no manifest fact for this evidence)")
        return "\n".join(lines)
    lines += [
        f"title: {ev.title}",
        f"source: {ev.source}",
        f"url: {url}",
        "",
        ev.fact or "(empty fact)",
    ]
    return "\n".join(lines)


def _missed_gold_markdown(
    missed_ids: list[str],
    *,
    evidence_by_url,
    audit,
    pair_cache: PairIndexCache,
) -> str:
    if not missed_ids:
        return "_None missed._\n"
    blocks: list[str] = []
    for gid in missed_ids:
        if gid.startswith("doc::"):
            url = gid[5:]
            ev = evidence_by_url.get(url)
            if ev is None:
                body = f"(no manifest fact for this URL)\n\n{url}"
            else:
                body = (
                    f"title: {ev.title}\n"
                    f"source: {ev.source}\n"
                    f"url: {url}\n\n"
                    f"{ev.fact or '(empty fact)'}"
                )
            blocks.append(f"1. **missed document** `{url}`\n\n" + _md_fence(body))
        elif gid.startswith("mem::"):
            sid = gid[5:]
            entry = audit.get(sid)
            pairs = pair_cache.get_pairs(sid)
            intended = _intended_gold_fact_text(sid, evidence_by_url)
            if entry is None:
                header = (
                    "Audit record missing for this session_id "
                    "(not the same as 'integrated has no memory gold')."
                )
                body = intended + "\n\n" + header
                if pairs:
                    body += "\n\nSession pairs:\n\n" + "\n\n".join(pairs)
            elif not entry.key_information_turns:
                header = (
                    "Session was audited, but the auditor marked key_information "
                    "as absent from user turns (all_required_present="
                    f"{entry.all_required_present}). Strict memory recall therefore "
                    "cannot credit this gold session."
                )
                if entry.notes:
                    header += f"\nAuditor notes: {entry.notes}"
                body = intended + "\n\n" + header
                if pairs:
                    body += "\n\nSession pairs (what was actually in memory):\n\n" + "\n\n".join(
                        f"[pair {i}]\n{p}" for i, p in enumerate(pairs)
                    )
            else:
                parts = [intended, ""]
                for t in entry.key_information_turns:
                    if 0 <= t < len(pairs):
                        parts.append(f"[pair {t}]\n{pairs[t]}")
                    else:
                        parts.append(f"[pair {t}] (out of range; {len(pairs)} pairs)")
                body = "\n\n".join(parts)
            blocks.append(f"1. **missed memory** `{sid}`\n\n" + _md_fence(body))
        else:
            blocks.append(f"1. **missed** `{gid}`\n\n" + _md_fence("(unknown gold id prefix)"))
    # number consecutively
    numbered = []
    for i, b in enumerate(blocks, 1):
        numbered.append(b.replace("1. **", f"{i}. **", 1))
    return "\n\n".join(numbered) + "\n"


def triplet_inspection_markdown(
    rows: pd.DataFrame,
    features_rows: pd.DataFrame,
    *,
    audit,
    evidence_by_url,
    pair_cache,
    case_index: int | None = None,
    n_cases: int | None = None,
    retrieved_chars: int = 450,
) -> str:
    """Readable Markdown for one (block_id, persona, agent) disagreement unit."""
    rows = rows.sort_values("dist").reset_index(drop=True)
    if rows.empty:
        return "_No rows._\n"
    r0 = rows.iloc[0]
    title = (
        f"Case {case_index}/{n_cases}: "
        if case_index is not None and n_cases is not None
        else "Case: "
    )
    lines: list[str] = [
        f"## {title}block {int(r0['block_id'])} · {r0['persona']} · {r0['agent']}",
        "",
        f"- **Topical relevance:** {r0.get('topical_relevance', '')}",
        f"- **Question type:** {r0.get('question_type', '')}",
        f"- **Hops:** {r0.get('hop_count', '')}",
        f"- **Assigned dists:** {', '.join(rows['dist'].astype(str))}",
        "",
        "### Question",
        "",
        _md_fence(str(r0["query"])),
        "",
        f"**Gold answer:** `{_md_cell(r0['gold_answer'])}`",
        "",
        "### Outcome by distribution",
        "",
        "| dist | EM | prediction | strict recall |",
        "| --- | --- | --- | --- |",
    ]

    feat = features_rows.set_index("dist") if not features_rows.empty else pd.DataFrame()
    for _, row in rows.iterrows():
        dist = row["dist"]
        rec = feat.loc[dist] if dist in feat.index else None
        if rec is not None and isinstance(rec, pd.DataFrame):
            rec = rec.iloc[0]
        strict = rec["recall_full_context_strict"] if rec is not None else ""
        try:
            st_s = f"{float(strict):.2f}" if pd.notna(strict) else ""
        except (TypeError, ValueError):
            st_s = str(strict)
        pred = _md_cell(truncate(row.get("prediction"), 70))
        lines.append(
            f"| `{dist}` | {int(row['correct'])} | {pred} | {st_s} |"
        )

    lines += ["", "### Notes for coding (fill in)", "", "- Mechanism:", "- Other:", ""]

    for _, row in rows.iterrows():
        gold = gold_ids_for_row(row)
        gold_mem = {g[5:] for g in gold if g.startswith("mem::")}
        gold_doc = {g[5:] for g in gold if g.startswith("doc::")}
        hits = per_gold_hits(
            row, audit=audit, evidence_by_url=evidence_by_url, pair_cache=pair_cache,
        )
        notes = _failure_notes(str(row.get("prediction") or ""), hits, int(row["correct"]))
        lines += [
            f"### `{row['dist']}` — EM={int(row['correct'])}",
            "",
            f"**Prediction:** `{_md_cell(row.get('prediction'))}`",
            "",
        ]
        if notes:
            lines += ["**Auto tags:** " + "; ".join(notes), ""]
        lines += ["**Strict evidence hits**", ""]
        if hits.empty:
            lines += ["_No gold IDs for this dist._", ""]
        else:
            lines += ["| gold id | strict hit |", "| --- | --- |"]
            for _, h in hits.iterrows():
                mark = "yes" if bool(h["strict_hit"]) else "no"
                lines.append(f"| `{h['gold_id']}` | {mark} |")
            lines.append("")
        missed: list[str] = []
        if not hits.empty:
            missed = [
                str(g)
                for g in hits.loc[~hits["strict_hit"].astype(bool), "gold_id"].tolist()
            ]
        lines += [
            "**Missed gold evidence**",
            "",
            _missed_gold_markdown(
                missed,
                evidence_by_url=evidence_by_url,
                audit=audit,
                pair_cache=pair_cache,
            ),
            "**Retrieved memory**",
            "",
            _retrieved_markdown(
                row.get("retrieved_memory"),
                gold_session_ids=gold_mem,
                gold_urls=gold_doc,
                text_chars=retrieved_chars,
            ),
            "**Retrieved documents**",
            "",
            _retrieved_markdown(
                row.get("retrieved_documents"),
                gold_session_ids=gold_mem,
                gold_urls=gold_doc,
                text_chars=retrieved_chars,
            ),
        ]
    return "\n".join(lines).rstrip() + "\n"


def build_agent(agent_id: str, **kwargs_override: Any):
    spec = get_agent_spec(agent_id)
    kwargs = {**spec.kwargs, **kwargs_override}
    return AgentSpec(spec.agent_module, spec.agent_class, kwargs).build()


def run_design_subset(
    agent_id: str,
    design_df: pd.DataFrame,
    output_path: Path,
    *,
    memory_gold_source: str = "session",
) -> list[dict[str, Any]]:
    """Run design rows for one agent (rebuilds stores only for personas in *design_df*)."""
    work = filter_design_for_agent(design_df, agent_id) if "agent" in design_df.columns else design_df
    if work.empty:
        return []
    extra: dict[str, Any] = {}
    if memory_gold_source != "session":
        extra["memory_gold_source"] = memory_gold_source
    agent = build_agent(agent_id, **extra)
    personas = sorted(work["persona"].astype(str).unique())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return run_design_experiment(
        work,
        agent_id,
        agent.hooks_factory,
        agent.answer_fn,
        output_path,
        eval_personas=personas,
    )


def load_merged_design(
    design_csv: Path | None = None,
    questions_pkl: Path | None = None,
) -> pd.DataFrame:
    return load_design_matrix(
        design_csv or DEFAULT_EXPERIMENT_DESIGN_CSV,
        questions_pkl or DEFAULT_EXPERIMENT_QUESTIONS_PICKLE,
    )


def label_runs(runs: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    labeled = collect(runs, questions)
    return label_topical_relevance(labeled, questions)


def compare_em(original: pd.DataFrame, rerun: pd.DataFrame) -> pd.DataFrame:
    keys = ["block_id", "persona", "dist", "agent"]
    a = original[keys + ["correct", "prediction"]].copy()
    b = rerun[keys + ["correct", "prediction"]].copy()
    if "reasoning" in rerun.columns:
        b = b.merge(
            rerun[keys + ["reasoning"]],
            on=keys,
            how="left",
        )
    merged = a.merge(b, on=keys, suffixes=("_orig", "_rerun"))
    merged["em_match"] = merged["correct_orig"] == merged["correct_rerun"]
    return merged
