"""Phase 2 - transformer sentence embeddings (Sentence Transformers)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

# BGE models are trained with an instruction prefix on the *query* side only.
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        self.max_tokens = int(self.model.max_seq_length)   # longer inputs get truncated
        self.query_prefix = QUERY_PREFIX.get(model_name, "")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        # normalised vectors -> inner product == cosine similarity
        vecs = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
        )
        return np.asarray(vecs, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([self.query_prefix + query], normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")


@lru_cache(maxsize=4)
def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)
