"""Phase 4 - text extraction. Turns PDF / DOCX / PPTX / TXT / MD files (and images)
into Page records that keep their metadata (file name, page or slide number, section).

PDF tables are also extracted row by row ("Table: header = value; ...") so a row's
cells stay together, and pages with no text layer (scans) are read with OCR."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

IMAGE_TYPES = {".png", ".jpg", ".jpeg"}
SUPPORTED = {".pdf", ".docx", ".pptx", ".txt", ".md"} | IMAGE_TYPES
OCR_ENABLED = os.environ.get("RAG_OCR", "1") == "1"     # OCR scanned pages and images

_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


@dataclass
class Page:
    text: str
    source: str                 # file name
    page: int | None = None     # page (PDF) or slide (PPTX) number
    section: str | None = None  # heading (DOCX / Markdown)


# ------------------------------------------------------------------ OCR
@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr_onnxruntime import RapidOCR      # ONNX models ship with the package: works offline

    return RapidOCR()


def ocr_image(image) -> str:
    """Text from a PIL image or numpy array, one detected line per output line."""
    import numpy as np

    result, _ = _ocr_engine()(np.asarray(image.convert("RGB")) if hasattr(image, "convert") else image)
    if not result:
        return ""
    # top-to-bottom, then left-to-right (box = 4 corner points)
    lines = sorted(result, key=lambda r: (round(r[0][0][1] / 10), r[0][0][0]))
    return "\n".join(r[1] for r in lines)


def _ocr_pdf_pages(path: Path, page_numbers: list[int]) -> dict[int, str]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return {n: ocr_image(pdf[n - 1].render(scale=2.5).to_pil()) for n in page_numbers}
    finally:
        pdf.close()


def _load_image(path: Path) -> list[Page]:
    from PIL import Image

    if not OCR_ENABLED:
        return []
    with Image.open(path) as img:
        text = ocr_image(img)
    return [Page(text, path.name, page=1)] if text.strip() else []


# ------------------------------------------------------------------ tables
def table_to_text(rows: list[list[str | None]]) -> list[str]:
    """One sentence per data row, pairing each cell with its column header, e.g.
    'Table: Model = BERT; F1 = 0.81.' - readable for the LLM and keeps rows intact."""
    rows = [[re.sub(r"\s+", " ", c or "").strip() for c in r] for r in rows if r and any(r)]
    if len(rows) < 2:
        return []
    header, body = rows[0], rows[1:]
    out = []
    for r in body:
        cells = [f"{h} = {c}" if h else c for h, c in zip(header, r) if c]
        if cells:
            out.append("Table: " + "; ".join(cells) + ".")
    return out


def _pdf_tables(path: Path) -> dict[int, list[str]]:
    try:
        import pdfplumber
    except ImportError:
        return {}
    tables: dict[int, list[str]] = {}
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                for t in page.extract_tables():
                    tables.setdefault(i, []).extend(table_to_text(t))
    except Exception:          # table extraction is a bonus; never fail the whole file
        return {}
    return tables


def _load_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    texts = [p.extract_text() or "" for p in PdfReader(str(path)).pages]
    tables = _pdf_tables(path)
    scanned = [i for i, t in enumerate(texts, start=1) if not t.strip()]
    ocr = _ocr_pdf_pages(path, scanned) if scanned and OCR_ENABLED else {}

    pages = []
    for i, text in enumerate(texts, start=1):
        text = text if text.strip() else ocr.get(i, "")
        if tables.get(i):
            text = text + "\n\n" + "\n".join(tables[i])
        if text.strip():
            pages.append(Page(text, path.name, page=i))
    return pages


def _load_docx(path: Path) -> list[Page]:
    import docx

    pages, section, buf = [], None, []

    def flush():
        if any(t.strip() for t in buf):
            pages.append(Page("\n".join(buf), path.name, section=section))

    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    for block in document.iter_inner_content():      # paragraphs and tables, in reading order
        if isinstance(block, Table):
            buf.extend(table_to_text([[c.text for c in row.cells] for row in block.rows]))
            continue
        par: Paragraph = block
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
    if ext in IMAGE_TYPES:
        return _load_image(path)
    raise ValueError(f"Unsupported file type: {ext}")


def iter_documents(directory: str | Path):
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path
