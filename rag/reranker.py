"""Phase 10 - cross-encoder reranking. Unlike the bi-encoder (which embeds query
and passage separately), a cross-encoder reads the (query, passage) pair together
through one transformer, so it is more accurate but too slow to run on every chunk.
We therefore rerank only the top candidates from FAISS."""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list, top_k: int) -> list:
        if not hits:
            return []
        scores = self.model.predict([(query, h.chunk.text) for h in hits])
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:top_k]


@lru_cache(maxsize=2)
def get_reranker(model_name: str) -> Reranker:
    return Reranker(model_name)
