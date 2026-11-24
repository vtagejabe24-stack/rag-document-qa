"""Phase 6 - vector database. A FAISS inner-product index over normalised
embeddings, plus the chunk metadata, persisted to disk so documents are never
embedded twice (Phase 17)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self.doc_hashes: dict[str, str] = {}     # file name -> sha256, to skip unchanged files
        self.index: faiss.Index | None = None
        self.version = 0                          # bumped on every change (invalidates caches such as BM25)

    # ---------- mutation ----------
    def _rebuild(self) -> None:
        self.version += 1
        self.index = None
        if self.vectors is not None and len(self.vectors):
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)

    def add(self, chunks: list[Chunk], vectors: np.ndarray, source: str, digest: str) -> None:
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.doc_hashes[source] = digest
        if self.index is None:
            self._rebuild()
        else:
            self.index.add(vectors)
            self.version += 1

    def remove(self, source: str) -> None:
        if source not in self.doc_hashes:
            return
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else None
        del self.doc_hashes[source]
        self._rebuild()

    @property
    def next_id(self) -> int:
        return max((c.id for c in self.chunks), default=-1) + 1

    def documents(self) -> dict[str, int]:
        counts: dict[str, int] = {s: 0 for s in sorted(self.doc_hashes)}
        for c in self.chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    # ---------- search ----------
    def search(self, query_vec: np.ndarray, k: int, sources: list[str] | None = None) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity, optionally restricted to some documents."""
        if self.index is None:
            return []
        n = len(self.chunks)
        scores, ids = self.index.search(query_vec.reshape(1, -1), n if sources else min(k, n))
        allowed = set(sources) if sources else None
        results = []
        for i, s in zip(ids[0], scores[0]):
            if i == -1:
                continue
            chunk = self.chunks[i]
            if allowed is None or chunk.source in allowed:
                results.append((chunk, float(s)))
                if len(results) == k:
                    break
        return results

    # ---------- persistence ----------
    @staticmethod
    def exists(directory: Path) -> bool:
        return (directory / "meta.json").exists()

    def save(self, directory: Path, settings: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self.vectors is not None:
            np.save(directory / "vectors.npy", self.vectors)
        elif (directory / "vectors.npy").exists():
            (directory / "vectors.npy").unlink()
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)
        with open(directory / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"settings": settings, "doc_hashes": self.doc_hashes}, f, indent=2)

    @classmethod
    def load(cls, directory: Path) -> tuple["VectorStore", dict]:
        store = cls()
        with open(directory / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        with open(directory / "chunks.json", encoding="utf-8") as f:
            store.chunks = [Chunk(**c) for c in json.load(f)]
        if (directory / "vectors.npy").exists():
            store.vectors = np.load(directory / "vectors.npy")
        store.doc_hashes = meta["doc_hashes"]
        store._rebuild()
        return store, meta["settings"]
