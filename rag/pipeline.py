"""The full RAG pipeline:
  documents -> extraction -> chunking -> embeddings -> FAISS
  question -> (rewrite with history) -> retrieve -> rerank -> LLM -> answer + sources"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import config
from . import prompts
from .chunker import chunk_document
from .embeddings import get_embedder
from .generator import make_llm, resolve_backend
from .loader import iter_documents, load_file
from .retriever import Hit, Retriever
from .vector_store import VectorStore


@dataclass
class Answer:
    question: str
    search_query: str      # standalone query after rewriting (Phase 11)
    hits: list[Hit]
    stream: Iterator[str]


class RAGPipeline:
    def __init__(
        self,
        index_dir: Path | None = config.INDEX_DIR,
        docs_dir: Path = config.DOCS_DIR,
        embed_model: str = config.EMBED_MODEL,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        llm_backend: str = config.LLM_BACKEND,
    ):
        self.index_dir = index_dir
        self.docs_dir = docs_dir
        self.embedder = get_embedder(embed_model)
        # never build chunks the embedding model would silently truncate
        self.chunk_size = min(chunk_size, self.embedder.max_tokens - 2)
        self.chunk_overlap = chunk_overlap
        self.settings = {"embed_model": embed_model, "chunk_size": self.chunk_size,
                         "chunk_overlap": chunk_overlap}

        self.store = VectorStore()
        if index_dir and VectorStore.exists(index_dir):
            store, saved = VectorStore.load(index_dir)
            if saved == self.settings:          # settings changed -> re-index from scratch
                self.store = store

        self.retriever = Retriever(self.embedder, self.store, config.RERANK_MODEL, config.TOP_K_RETRIEVE)
        self.llm_backend = resolve_backend(llm_backend)
        self._llm = None

    # ------------------------------------------------------------------ LLM
    @property
    def llm(self):
        if self._llm is None:
            self._llm = make_llm(self.llm_backend, config.CLAUDE_MODEL,
                                 config.CLAUDE_EFFORT, config.LOCAL_LLM)
        return self._llm

    @property
    def llm_name(self) -> str:
        return f"Claude ({config.CLAUDE_MODEL})" if self.llm_backend == "claude" \
            else f"{config.LOCAL_LLM} (local)"

    @property
    def supports_tools(self) -> bool:
        return self.llm_backend == "claude"

    # ------------------------------------------------------------- indexing
    def _save(self) -> None:
        if self.index_dir:
            self.store.save(self.index_dir, self.settings)

    def add_file(self, path: Path) -> str:
        """Index one file. Returns 'added', 'updated', 'unchanged' or 'empty'."""
        path = Path(path)
        source = path.name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.store.doc_hashes.get(source) == digest:
            return "unchanged"
        status = "updated" if source in self.store.doc_hashes else "added"

        chunks = chunk_document(load_file(path), self.chunk_size, self.chunk_overlap,
                                self.embedder.count_tokens, start_id=self.store.next_id)
        if not chunks:
            return "empty"          # e.g. a scanned PDF with no text layer (needs OCR)
        vectors = self.embedder.encode_documents([c.text for c in chunks])
        self.store.remove(source)
        self.store.add(chunks, vectors, source, digest)
        self._save()
        return status

    def remove_document(self, source: str, delete_file: bool = False) -> None:
        self.store.remove(source)
        self._save()
        if delete_file:
            (self.docs_dir / Path(source).name).unlink(missing_ok=True)

    def sync_directory(self) -> dict[str, str]:
        """Make the index match the documents folder."""
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        report = {p.name: self.add_file(p) for p in iter_documents(self.docs_dir)}
        for source in list(self.store.doc_hashes):
            if source not in report:
                self.store.remove(source)
                report[source] = "removed"
        self._save()
        return report

    def documents(self) -> dict[str, int]:
        return self.store.documents()

    def document_text(self, source: str) -> str:
        """Full text of one document with page/section markers (for the study tools)."""
        parts = []
        for page in load_file(self.docs_dir / Path(source).name):
            marker = f"[page {page.page}]" if page.page else f"[section: {page.section}]" if page.section else ""
            parts.append(f"{marker}\n{page.text.strip()}".strip())
        return "\n\n".join(parts)

    # ------------------------------------------------------------ question answering
    def search_query(self, question: str, history: list[dict]) -> str:
        """Phase 11: rewrite a follow-up into a standalone query using the conversation."""
        if not config.ENABLE_QUERY_REWRITE or not history or not self.supports_tools:
            return question
        convo = "\n".join(f"{m['role'].upper()}: {m['content'][:1500]}" for m in history[-6:])
        msg = f"<conversation>\n{convo}\n</conversation>\n\nFollow-up question: {question}"
        rewritten = self.llm.complete(prompts.REWRITE_SYSTEM, [{"role": "user", "content": msg}],
                                      max_tokens=2000, effort="low").strip()
        return rewritten if rewritten and "\n" not in rewritten else question

    def fit_context(self, question: str, hits: list[Hit]) -> list[Hit]:
        """Keep only the best-ranked excerpts that fit the LLM's input window, so a small
        local model never has its context silently truncated mid-excerpt. (Claude's
        window is far larger than k excerpts, so it gets all of them.)"""
        budget_fn = getattr(self.llm, "context_budget", None)
        if budget_fn is None or not hits:
            return hits
        budget, kept = budget_fn(question), []
        for h in hits:
            if self.llm.count_tokens(prompts.format_context(kept + [h])) > budget:
                break
            kept.append(h)
        return kept or hits[:1]          # always pass at least the best excerpt

    @staticmethod
    def nothing_relevant(hits: list[Hit]) -> bool:
        """True when no passage is relevant enough to answer from. Small models often
        answer anyway instead of saying "not found", so this check is done before them."""
        if not hits or config.MIN_RERANK_SCORE is None or hits[0].rerank_score is None:
            return False
        return hits[0].rerank_score < config.MIN_RERANK_SCORE

    def answer(self, question: str, history: list[dict] | None = None,
               sources: list[str] | None = None, k: int = config.TOP_K_CONTEXT,
               rerank: bool | None = None, hybrid: bool | None = None) -> Answer:
        rerank = config.ENABLE_RERANKER if rerank is None else rerank
        hybrid = config.ENABLE_HYBRID if hybrid is None else hybrid
        history = list(history or []) if config.ENABLE_MEMORY else []
        if not self.store.chunks:
            return Answer(question, question, [], iter(["No documents are indexed yet - upload one first."]))

        query = self.search_query(question, history)
        hits = self.retriever.retrieve(query, k, sources=sources, rerank=rerank, hybrid=hybrid,
                                       candidates=config.RERANK_CANDIDATES)
        if self.nothing_relevant(hits):
            return Answer(question, query, [], iter([prompts.NOT_FOUND]))
        hits = self.fit_context(question, hits)

        recent = history[-6:]
        while recent and recent[0]["role"] != "user":   # the API needs a user turn first
            recent = recent[1:]
        stream = self.llm.answer_stream(question, prompts.format_context(hits), recent)
        return Answer(question, query, hits, stream)
