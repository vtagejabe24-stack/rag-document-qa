"""Phase 5 - chunking. Packs whole sentences into chunks of at most `chunk_size`
tokens, with `overlap` tokens of trailing sentences repeated at the start of the
next chunk. Chunks may cross page/section boundaries; they remember every page and
section they span so citations stay accurate."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .loader import Page

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    pages: list[int] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)

    def location(self) -> str:
        parts = []
        if self.pages:
            first, last = self.pages[0], self.pages[-1]
            parts.append(f"page {first}" if first == last else f"pages {first}-{last}")
        if self.sections:
            parts.append("§ " + " / ".join(self.sections))
        return ", ".join(parts) or "whole document"


@dataclass
class _Unit:          # one sentence (or a piece of an over-long sentence)
    text: str
    n_tokens: int
    page: int | None
    section: str | None


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in _SENT_SPLIT.split(text) if s]


def _units(page: Page, chunk_size: int, count_tokens: Callable[[str], int]) -> list[_Unit]:
    units = []
    for sent in split_sentences(page.text):
        n = count_tokens(sent)
        if n <= chunk_size:
            units.append(_Unit(sent, n, page.page, page.section))
            continue
        # sentence longer than a chunk (tables, run-on PDF text): split by words
        words = sent.split()
        step = max(1, int(len(words) * chunk_size / n * 0.9))
        for i in range(0, len(words), step):
            piece = " ".join(words[i : i + step])
            units.append(_Unit(piece, count_tokens(piece), page.page, page.section))
    return units


def _make_chunk(cid: int, units: list[_Unit], source: str) -> Chunk:
    pages = sorted({u.page for u in units if u.page is not None})
    sections = list(dict.fromkeys(u.section for u in units if u.section))
    return Chunk(cid, " ".join(u.text for u in units), source, pages, sections)


def chunk_document(
    pages: list[Page],
    chunk_size: int,
    overlap: int,
    count_tokens: Callable[[str], int],
    start_id: int = 0,
) -> list[Chunk]:
    if not pages:
        return []
    source = pages[0].source
    chunks: list[Chunk] = []
    current: list[_Unit] = []
    current_len = 0

    for unit in (u for p in pages for u in _units(p, chunk_size, count_tokens)):
        if current and current_len + unit.n_tokens > chunk_size:
            chunks.append(_make_chunk(start_id + len(chunks), current, source))
            carry, carry_len = [], 0
            for prev in reversed(current):
                if carry_len + prev.n_tokens > overlap:
                    break
                carry.insert(0, prev)
                carry_len += prev.n_tokens
            if carry_len + unit.n_tokens > chunk_size:
                carry, carry_len = [], 0
            current, current_len = carry, carry_len
        current.append(unit)
        current_len += unit.n_tokens

    if current:
        chunks.append(_make_chunk(start_id + len(chunks), current, source))
    return chunks
