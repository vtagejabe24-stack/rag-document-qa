"""Phase 7/10 - retrieval: query embedding -> FAISS top-N -> (optional) rerank -> top-k."""
from __future__ import annotations

from dataclasses import dataclass

from .chunker import Chunk
from .embeddings import Embedder
from .reranker import Reranker, get_reranker
from .vector_store import VectorStore


@dataclass
class Hit:
    chunk: Chunk
    score: float                       # cosine similarity from the bi-encoder
    rerank_score: float | None = None  # cross-encoder relevance logit


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore,
                 reranker_model: str | None = None, candidates: int = 20):
        self.embedder = embedder
        self.store = store
        self.reranker_model = reranker_model
        self._reranker: Reranker | None = None
        self.candidates = candidates

    @property
    def reranker(self) -> Reranker | None:
        """Loaded lazily, so the cross-encoder costs nothing unless reranking is used."""
        if self._reranker is None and self.reranker_model:
            self._reranker = get_reranker(self.reranker_model)
        return self._reranker

    def retrieve(self, query: str, k: int, sources: list[str] | None = None,
                 rerank: bool = False) -> list[Hit]:
        use_rerank = rerank and self.reranker_model is not None
        n = max(self.candidates, k) if use_rerank else k
        hits = [Hit(c, s) for c, s in self.store.search(self.embedder.encode_query(query), n, sources)]
        if use_rerank:
            hits = self.reranker.rerank(query, hits, k)
        return hits[:k]
