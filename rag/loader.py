"""Phase 4 - text extraction. Turns PDF / DOCX / PPTX / TXT / MD files into
Page records that keep their metadata (file name, page or slide number, section)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".pptx", ".txt", ".md"}

_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


@dataclass
class Page:
    text: str
    source: str                 # file name
    page: int | None = None     # page (PDF) or slide (PPTX) number
    section: str | None = None  # heading (DOCX / Markdown)


def _load_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    pages = []
    for i, p in enumerate(PdfReader(str(path)).pages, start=1):
        text = p.extract_text() or ""
        if text.strip():
            pages.append(Page(text, path.name, page=i))
    return pages


def _load_docx(path: Path) -> list[Page]:
    import docx

    pages, section, buf = [], None, []

    def flush():
        if any(t.strip() for t in buf):
            pages.append(Page("\n".join(buf), path.name, section=section))

    for par in docx.Document(str(path)).paragraphs:
        if par.style is not None and par.style.name.lower().startswith("heading") and par.text.strip():
            flush()
            section, buf = par.text.strip(), [par.text.strip() + "."]
        else:
            buf.append(par.text)
    flush()
    return pages


def _load_pptx(path: Path) -> list[Page]:
    from pptx import Presentation

    pages = []
    for i, slide in enumerate(Presentation(str(path)).slides, start=1):
        texts = [s.text_frame.text for s in slide.shapes if s.has_text_frame and s.text_frame.text.strip()]
        if texts:
            pages.append(Page("\n".join(texts), path.name, page=i))
    return pages


def _load_text(path: Path) -> list[Page]:
    """Plain text / Markdown, split into sections at Markdown headings."""
    pages, section, buf = [], None, []

    def flush():
        if any(t.strip() for t in buf):
            pages.append(Page("\n".join(buf), path.name, section=section))

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _MD_HEADING.match(line)
        if m:
            flush()
            section, buf = m.group(1), [m.group(1) + "."]
        else:
            buf.append(line)
    flush()
    return pages


def load_file(path: str | Path) -> list[Page]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext == ".pptx":
        return _load_pptx(path)
    if ext in {".txt", ".md"}:
        return _load_text(path)
    raise ValueError(f"Unsupported file type: {ext}")


def iter_documents(directory: str | Path):
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path
