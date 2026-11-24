"""Keyword search (Okapi BM25) for hybrid retrieval.

Dense embeddings capture meaning but blur exact terms - dataset names, acronyms,
metric names, numbers. BM25 scores exact word overlap, weighted by how rare each
word is. Fusing both rankings (see retriever.py) gets the best of each."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
# Very common English words carry no signal for keyword matching.
STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers him his how i if in into is it its itself just me more
most my no nor not now of off on once only or other our ours out over own same she should so
some such than that the their theirs them then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you
your yours
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        docs = [Counter(tokenize(t)) for t in texts]
        self.doc_len = np.array([sum(d.values()) for d in docs], dtype="float32")
        self.avg_len = float(self.doc_len.mean()) if len(docs) else 0.0
        df: Counter = Counter()
        for d in docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # inverted index: term -> (doc ids, term frequencies)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, d in enumerate(docs):
            for t, tf in d.items():
                ids, tfs = postings.setdefault(t, ([], []))
                ids.append(i)
                tfs.append(tf)
        self.postings = {t: (np.array(ids), np.array(tfs, dtype="float32"))
                         for t, (ids, tfs) in postings.items()}
        self.n_docs = n

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for `query` (0 where no term matches)."""
        out = np.zeros(self.n_docs, dtype="float32")
        if not self.n_docs:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avg_len, 1e-9))
        for t in set(tokenize(query)):
            if t not in self.postings:
                continue
            ids, tf = self.postings[t]
            out[ids] += self.idf[t] * tf * (self.k1 + 1) / (tf + norm[ids])
        return out


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Merge several ranked lists of ids: each list adds 1/(k + rank) to an id's score.
    Rank-based, so the very different score scales of BM25 and cosine don't matter."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=fused.get, reverse=True)
