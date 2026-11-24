"""Phase 7/10 - retrieval: query -> dense (FAISS) or hybrid (FAISS + BM25) top-N
-> (optional) cross-encoder rerank -> top-k."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25, reciprocal_rank_fusion
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
        self._bm25: BM25 | None = None
        self._bm25_version = -1

    @property
    def reranker(self) -> Reranker | None:
        """Loaded lazily, so the cross-encoder costs nothing unless reranking is used."""
        if self._reranker is None and self.reranker_model:
            self._reranker = get_reranker(self.reranker_model)
        return self._reranker

    @property
    def bm25(self) -> BM25:
        """Keyword index, rebuilt only when the document collection changes."""
        if self._bm25 is None or self._bm25_version != self.store.version:
            self._bm25 = BM25([c.text for c in self.store.chunks])
            self._bm25_version = self.store.version
        return self._bm25

    def _hybrid(self, query: str, query_vec: np.ndarray, n: int,
                sources: list[str] | None) -> list[Hit]:
        """Top-n by reciprocal rank fusion of the dense and BM25 rankings."""
        store = self.store
        if store.vectors is None or not store.chunks:
            return []
        cos = store.vectors @ query_vec.ravel()
        kw = self.bm25.scores(query)
        if sources:
            allowed = set(sources)
            mask = np.array([c.source in allowed for c in store.chunks])
            cos = np.where(mask, cos, -np.inf)
            kw = np.where(mask, kw, 0.0)
        pool = max(n, self.candidates)      # fuse deeper lists than we return
        dense = [int(i) for i in np.argsort(-cos)[:pool] if np.isfinite(cos[i])]
        sparse = [int(i) for i in np.argsort(-kw)[:pool] if kw[i] > 0]
        fused = reciprocal_rank_fusion([dense, sparse])[:n]
        return [Hit(store.chunks[i], float(cos[i])) for i in fused]

    def retrieve(self, query: str, k: int, sources: list[str] | None = None,
                 rerank: bool = False, hybrid: bool = False,
                 candidates: int | None = None) -> list[Hit]:
        use_rerank = rerank and self.reranker_model is not None
        n = max(candidates or self.candidates, k) if use_rerank else k
        qvec = self.embedder.encode_query(query)
        if hybrid:
            hits = self._hybrid(query, qvec, n, sources)
        else:
            hits = [Hit(c, s) for c, s in self.store.search(qvec, n, sources)]
        if use_rerank:
            hits = self.reranker.rerank(query, hits, k)
        return hits[:k]
