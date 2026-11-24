"""Phase 15 - evaluation. Retrieval and answer quality are measured separately.

Eval file (JSONL), one object per line:
  {"question": "...", "answer": "...", "evidence": ["key phrase", ...]}
A retrieved chunk counts as relevant if it contains any evidence phrase
(case-insensitive). An empty evidence list marks an unanswerable question: the
correct behaviour is to abstain."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import prompts
from .chunker import split_sentences


def load_eval(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk_text: str, evidence: list[str], min_coverage: float = 0.5) -> bool:
    """A chunk is relevant if it contains at least `min_coverage` of the sentences of
    any gold evidence entry (a short phrase, or a whole paragraph as in QASPER).
    Sentence-level matching is needed because chunks can split a gold paragraph."""
    t = _norm(chunk_text)
    for e in evidence:
        sents = [_norm(s) for s in split_sentences(e)] or [_norm(e)]
        if sum(s in t for s in sents) / len(sents) >= min_coverage:
            return True
    return False


# ------------------------------------------------------------ retrieval metrics
def evaluate_retrieval(pipeline, qa: list[dict], ks=(1, 3, 5, 10), rerank: bool = True,
                       scoped: bool = False) -> dict:
    """Recall@k, Precision@k, MRR and mean latency over the answerable questions.

    scoped=True searches only the document a question was written about (the standard
    QASPER setting, since its questions don't name their paper); scoped=False searches
    the whole library, which is the harder multi-document setting."""
    items = [q for q in qa if q["evidence"]]
    kmax = max(ks)
    recall = {k: 0.0 for k in ks}
    precision = {k: 0.0 for k in ks}
    rr_sum, latency = 0.0, 0.0

    for q in items:
        t0 = time.perf_counter()
        sources = [q["paper"]] if scoped and q.get("paper") else None
        hits = pipeline.retriever.retrieve(q["question"], kmax, sources=sources, rerank=rerank)
        latency += time.perf_counter() - t0
        rel = [is_relevant(h.chunk.text, q["evidence"]) for h in hits]
        for k in ks:
            top = rel[:k]
            recall[k] += 1.0 if any(top) else 0.0
            precision[k] += sum(top) / k
        first = next((i for i, r in enumerate(rel, 1) if r), None)
        rr_sum += 1.0 / first if first else 0.0

    n = max(len(items), 1)
    return {
        **{f"recall@{k}": recall[k] / n for k in ks},
        **{f"precision@{k}": precision[k] / n for k in ks},
        "mrr": rr_sum / n,
        "latency_ms": 1000 * latency / n,
        "n_questions": len(items),
    }


# ------------------------------------------------------------- answer metrics
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "integer"},
        "relevance": {"type": "integer"},
        "faithfulness": {"type": "integer"},
        "completeness": {"type": "integer"},
        "abstained": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correctness", "relevance", "faithfulness", "completeness", "abstained", "reason"],
    "additionalProperties": False,
}


def evaluate_answers(pipeline, qa: list[dict], limit: int | None = None, scoped: bool = False) -> dict:
    """Generates an answer per question, then has Claude grade it against the
    reference using the retrieved context (LLM-as-judge)."""
    if not pipeline.supports_tools:
        raise RuntimeError("Answer evaluation needs Claude (set ANTHROPIC_API_KEY).")
    rows = []
    for q in (qa[:limit] if limit else qa):
        t0 = time.perf_counter()
        ans = pipeline.answer(q["question"], sources=[q["paper"]] if scoped and q.get("paper") else None)
        text = "".join(ans.stream)
        latency = time.perf_counter() - t0
        reference = q["answer"] if q["evidence"] else "NOT ANSWERABLE from the documents"
        msg = (
            f"<question>{q['question']}</question>\n<reference_answer>{reference}</reference_answer>\n"
            f"<retrieved_context>\n{prompts.format_context(ans.hits)}\n</retrieved_context>\n"
            f"<system_answer>{text}</system_answer>"
        )
        grade = pipeline.llm.complete_json(prompts.JUDGE_SYSTEM,
                                           [{"role": "user", "content": msg}], JUDGE_SCHEMA, max_tokens=4000)
        rows.append({"question": q["question"], "answerable": bool(q["evidence"]),
                     "answer": text, "latency_s": latency, **grade})

    def mean(key, subset):
        return sum(r[key] for r in subset) / len(subset) if subset else float("nan")

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    return {
        "correctness": mean("correctness", rows),
        "relevance": mean("relevance", rows),
        "faithfulness": mean("faithfulness", rows),
        "completeness": mean("completeness", rows),
        "latency_s": mean("latency_s", rows),
        "abstain_rate_on_unanswerable": (sum(r["abstained"] for r in unanswerable) / len(unanswerable)
                                         if unanswerable else float("nan")),
        "false_abstain_rate_on_answerable": (sum(r["abstained"] for r in answerable) / len(answerable)
                                             if answerable else float("nan")),
        "rows": rows,
    }
