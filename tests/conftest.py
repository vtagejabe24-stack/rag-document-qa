"""Shared test helpers. The tests use a tiny deterministic 'embedder' instead of a real
transformer, so they run in seconds and need no model downloads."""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeEmbedder:
    """Bag-of-words hashed into 256 dimensions, L2-normalised. Texts sharing words
    get similar vectors, which is all retrieval tests need."""

    model_name = "fake"
    max_tokens = 512
    dim = 256

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_documents(self, texts, batch_size=32):
        return np.stack([self._vec(t) for t in texts]).astype("float32")

    def encode_query(self, query):
        return self._vec(query).reshape(1, -1)


class FakeLLM:
    """Echoes how many excerpts it was given; records the last call."""

    supports_tools = False
    name = "fake-llm"

    def __init__(self, budget: int | None = None):
        self.budget = budget
        self.last_context = None
        if budget is not None:          # behave like a small local model with a context window
            self.context_budget = lambda question: budget
            self.count_tokens = lambda text: len(text.split())

    def answer_stream(self, question, context, history):
        self.last_context = context
        yield f"Answer based on {context.count('] ')} excerpts [1]."


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def make_pipeline(tmp_path, monkeypatch):
    """Factory for a RAGPipeline over `files` (name -> text), using the fake models."""
    from rag import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "get_embedder", lambda name: FakeEmbedder())

    def make(files: dict[str, str], llm=None, persist: bool = True):
        docs = tmp_path / "docs"
        docs.mkdir(exist_ok=True)
        for name, text in files.items():
            (docs / name).write_text(text, encoding="utf-8")
        p = pipeline_mod.RAGPipeline(index_dir=tmp_path / "index" if persist else None, docs_dir=docs,
                                     embed_model="fake", llm_backend="local")
        p._llm = llm or FakeLLM()
        return p

    return make


def make_table_pdf(path: Path) -> None:
    """Write a one-page PDF with a ruled 3x2 table (Model/F1, BERT/0.81, GPT/0.77) and a
    sentence above it - built by hand so the tests need no PDF-writing library."""
    rows = [("Model", "F1"), ("BERT", "0.81"), ("GPT", "0.77")]
    x0, y0, cw, rh = 100, 600, 150, 30
    ops = ["BT /F1 12 Tf 100 680 Td (Results of the experiment.) Tj ET", "0.5 w"]
    for r in range(len(rows) + 1):                       # horizontal rules
        y = y0 - r * rh
        ops.append(f"{x0} {y} m {x0 + 2 * cw} {y} l S")
    for c in range(3):                                   # vertical rules
        x = x0 + c * cw
        ops.append(f"{x} {y0} m {x} {y0 - len(rows) * rh} l S")
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            ops.append(f"BT /F1 12 Tf {x0 + c * cw + 10} {y0 - r * rh - 20} Td ({cell}) Tj ET")
    stream = "\n".join(ops).encode()

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
