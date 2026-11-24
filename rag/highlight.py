"""Find the sentence(s) inside a retrieved chunk that best support an answer, so the
UI can highlight exactly where the answer came from instead of showing a wall of text."""
from __future__ import annotations

import html
import re

from .chunker import split_sentences
from .embeddings import Embedder

_CITATION = re.compile(r"\[\d+\]")


def best_sentences(embedder: Embedder, answer: str, chunk_text: str, n: int = 1,
                   min_score: float = 0.5) -> list[str]:
    """The `n` chunk sentences most similar to the answer (cosine >= min_score)."""
    sents = split_sentences(chunk_text)
    answer = _CITATION.sub("", answer).strip()
    if not sents or not answer:
        return []
    vecs = embedder.encode_documents(sents + [answer])
    scores = vecs[:-1] @ vecs[-1]
    order = scores.argsort()[::-1][:n]
    return [sents[i] for i in sorted(order) if scores[i] >= min_score]


def highlight_html(chunk_text: str, sentences: list[str], max_chars: int = 900) -> str:
    """Chunk text as HTML with the given sentences wrapped in <mark>, trimmed around the
    first highlight so it stays visible in a long chunk."""
    text = " ".join(split_sentences(chunk_text))
    if sentences:
        start = max(0, text.find(sentences[0]) - max_chars // 3)
        if start:
            start = text.find(" ", start) + 1
            text = "…" + text[start:]
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    out = html.escape(text)
    for s in sentences:
        es = html.escape(s)
        out = out.replace(es, f"<mark>{es}</mark>", 1)
    return out
