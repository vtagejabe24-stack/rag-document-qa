"""Phase 15 - evaluation. Retrieval and answer quality are measured separately.

Eval file (JSONL), one object per line:
  {"question": "...", "answer": "...", "answers": ["...", ...], "evidence": ["...", ...]}
`answers` (optional) holds every annotator's reference answer; `answer` is the first.
A retrieved chunk counts as relevant if it covers enough of a gold evidence entry.
An empty evidence list marks an unanswerable question: the correct behaviour is to
abstain."""
from __future__ import annotations

import json
import random
import re
import string
import time
from collections import Counter
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


# ------------------------------------------------------------ statistics
def bootstrap_ci(values: list[float], n_resamples: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of per-question scores."""
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return lo, hi


# ------------------------------------------------------------ retrieval metrics
def evaluate_retrieval(pipeline, qa: list[dict], ks=(1, 3, 5, 10), rerank: bool = True,
                       scoped: bool = False, hybrid: bool = False,
                       candidates: int | None = None, ci: bool = False) -> dict:
    """Recall@k, Precision@k, MRR and mean latency over the answerable questions.

    scoped=True searches only the document a question was written about (the standard
    QASPER setting, since its questions don't name their paper); scoped=False searches
    the whole library, which is the harder multi-document setting.
    ci=True adds 95% bootstrap confidence intervals ("<metric>_ci": [low, high])."""
    items = [q for q in qa if q["evidence"]]
    kmax = max(ks)
    per_q: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
    per_q.update({f"precision@{k}": [] for k in ks})
    per_q["mrr"] = []
    latency = 0.0

    for q in items:
        t0 = time.perf_counter()
        sources = [q["paper"]] if scoped and q.get("paper") else None
        hits = pipeline.retriever.retrieve(q["question"], kmax, sources=sources, rerank=rerank,
                                           hybrid=hybrid, candidates=candidates)
        latency += time.perf_counter() - t0
        rel = [is_relevant(h.chunk.text, q["evidence"]) for h in hits]
        for k in ks:
            top = rel[:k]
            per_q[f"recall@{k}"].append(1.0 if any(top) else 0.0)
            per_q[f"precision@{k}"].append(sum(top) / k)
        first = next((i for i, r in enumerate(rel, 1) if r), None)
        per_q["mrr"].append(1.0 / first if first else 0.0)

    n = max(len(items), 1)
    out: dict = {name: sum(v) / n for name, v in per_q.items()}
    out["latency_ms"] = 1000 * latency / n
    out["n_questions"] = len(items)
    if ci:
        for name in (f"recall@{k}" for k in ks):
            out[f"{name}_ci"] = list(bootstrap_ci(per_q[name]))
        out["mrr_ci"] = list(bootstrap_ci(per_q["mrr"]))
    return out


# ------------------------------------------------------------- answer metrics
def normalize_answer(text: str) -> str:
    """SQuAD/QASPER normalisation: lower case, drop punctuation, articles and extra spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(pred), overlap / len(ref)
    return 2 * p * r / (p + r)


def is_abstention(answer: str) -> bool:
    a = answer.lower()
    return ("couldn't find" in a or "could not find" in a or "not in the documents" in a
            or ("not contain" in a and ("document" in a or "context" in a)))


_CITATION = re.compile(r"\[(\d+)\]")


def strip_citations(answer: str) -> str:
    return _CITATION.sub("", answer)


def citation_check(answer: str, n_hits: int) -> dict:
    """Do the [n] markers in an answer point at excerpts that were actually provided?"""
    nums = [int(n) for n in _CITATION.findall(answer)]
    return {"n_citations": len(nums),
            "invalid_citations": sum(1 for n in nums if not 1 <= n <= n_hits)}


def answer_f1(prediction: str, q: dict) -> float:
    """QASPER Answer-F1: max token F1 over the annotators' answers. For an
    unanswerable question, abstaining scores 1 and anything else 0 (and vice versa)."""
    if not q["evidence"]:
        return 1.0 if is_abstention(prediction) else 0.0
    if is_abstention(prediction):
        return 0.0
    refs = q.get("answers") or [q["answer"]]
    pred = strip_citations(prediction)
    return max(token_f1(pred, r) for r in refs)


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
JUDGE_KEYS = ("correctness", "relevance", "faithfulness", "completeness")


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def evaluate_answers(pipeline, qa: list[dict], limit: int | None = None, scoped: bool = False,
                     judge=None, **answer_kwargs) -> dict:
    """Generates an answer per question and scores it with QASPER Answer-F1 (free,
    works with any LLM). If `judge` (a ClaudeLLM) is given, Claude also grades each
    answer against the reference and the retrieved context (LLM-as-judge)."""
    rows = []
    for q in (qa[:limit] if limit else qa):
        t0 = time.perf_counter()
        ans = pipeline.answer(q["question"], sources=[q["paper"]] if scoped and q.get("paper") else None,
                              **answer_kwargs)
        text = "".join(ans.stream)
        latency = time.perf_counter() - t0
        row = {"question": q["question"], "answerable": bool(q["evidence"]), "answer": text,
               "reference": q["answer"], "latency_s": latency, "answer_f1": answer_f1(text, q),
               "abstained": is_abstention(text),
               "retrieved_evidence": any(is_relevant(h.chunk.text, q["evidence"]) for h in ans.hits)
               if q["evidence"] else None,
               **citation_check(text, len(ans.hits))}
        if judge is not None:
            reference = q["answer"] if q["evidence"] else "NOT ANSWERABLE from the documents"
            msg = (
                f"<question>{q['question']}</question>\n<reference_answer>{reference}</reference_answer>\n"
                f"<retrieved_context>\n{prompts.format_context(ans.hits)}\n</retrieved_context>\n"
                f"<system_answer>{text}</system_answer>"
            )
            grade = judge.complete_json(prompts.JUDGE_SYSTEM, [{"role": "user", "content": msg}],
                                        JUDGE_SCHEMA, max_tokens=4000)
            row.update({k: grade[k] for k in JUDGE_KEYS}, judge_abstained=grade["abstained"],
                       judge_reason=grade["reason"])
        rows.append(row)

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    found = [r for r in answerable if r["retrieved_evidence"]]
    with_cites = [r for r in rows if r["n_citations"]]
    f1 = [r["answer_f1"] for r in rows]
    summary = {
        "answer_f1": _mean(f1),
        "answer_f1_ci": list(bootstrap_ci(f1)),
        "answer_f1_answerable": _mean(r["answer_f1"] for r in answerable),
        # isolates generation from retrieval: F1 where the right evidence was retrieved
        "answer_f1_when_evidence_retrieved": _mean(r["answer_f1"] for r in found),
        "abstain_rate_on_unanswerable": _mean(r["abstained"] for r in unanswerable),
        "false_abstain_rate_on_answerable": _mean(r["abstained"] for r in answerable),
        "answers_with_citations": len(with_cites) / len(rows) if rows else float("nan"),
        "invalid_citation_rate": (sum(r["invalid_citations"] for r in rows)
                                  / max(1, sum(r["n_citations"] for r in rows))),
        "latency_s": _mean(r["latency_s"] for r in rows),
        "n_questions": len(rows),
    }
    if judge is not None:
        summary.update({k: _mean(r[k] for r in rows) for k in JUDGE_KEYS})
    summary["rows"] = rows
    return summary
