import pytest

from rag.loader import SUPPORTED, iter_documents, load_file, table_to_text
from tests.conftest import make_table_pdf


def test_markdown_sections(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("Intro text.\n\n# Methods\nWe used X.\n\n## Results ##\nIt worked.\n", encoding="utf-8")
    pages = load_file(f)
    assert [p.section for p in pages] == [None, "Methods", "Results"]
    assert pages[1].text.startswith("Methods.")
    assert all(p.source == "notes.md" for p in pages)


def test_plain_text(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("Just some text.", encoding="utf-8")
    assert load_file(f)[0].text.strip() == "Just some text."


def test_docx_headings_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Preface.")
    doc.add_heading("Results", level=1)
    table = doc.add_table(rows=2, cols=2)
    for row, vals in zip(table.rows, [("Model", "F1"), ("BERT", "0.81")]):
        for cell, v in zip(row.cells, vals):
            cell.text = v
    doc.add_paragraph("BERT wins.")
    path = tmp_path / "r.docx"
    doc.save(path)

    pages = load_file(path)
    assert [p.section for p in pages] == [None, "Results"]
    assert "Table: Model = BERT; F1 = 0.81." in pages[1].text
    assert pages[1].text.index("Table:") < pages[1].text.index("BERT wins")   # reading order kept


def test_pptx_slides(tmp_path):
    pptx = pytest.importorskip("pptx")
    from pptx.util import Inches

    prs = pptx.Presentation()
    for text in ["First slide", "", "Third slide"]:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        if text:
            slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1)).text_frame.text = text
    path = tmp_path / "deck.pptx"
    prs.save(path)
    pages = load_file(path)
    assert [(p.page, p.text) for p in pages] == [(1, "First slide"), (3, "Third slide")]


def test_pdf_text_and_tables(tmp_path):
    path = tmp_path / "table.pdf"
    make_table_pdf(path)
    pages = load_file(path)
    assert len(pages) == 1 and pages[0].page == 1
    assert "Results of the experiment." in pages[0].text
    assert "Table: Model = BERT; F1 = 0.81." in pages[0].text
    assert "Table: Model = GPT; F1 = 0.77." in pages[0].text


def test_table_to_text():
    rows = [["Model", "F1", None], ["BERT", "0.81", ""], [None, None, None], ["GPT", "0.77", "x"]]
    assert table_to_text(rows) == ["Table: Model = BERT; F1 = 0.81.", "Table: Model = GPT; F1 = 0.77; x."]
    assert table_to_text([["only header"]]) == []


def test_unsupported_type(tmp_path):
    f = tmp_path / "a.xyz"
    f.write_text("x")
    with pytest.raises(ValueError):
        load_file(f)


def test_iter_documents_filters_and_sorts(tmp_path):
    for name in ["b.md", "a.pdf", "c.exe", "d.TXT"]:
        (tmp_path / name).write_text("x")
    assert [p.name for p in iter_documents(tmp_path)] == ["a.pdf", "b.md", "d.TXT"]
    assert ".png" in SUPPORTED


@pytest.mark.slow
def test_ocr_reads_image_only_pdf(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1200, 200), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except OSError:
        font = ImageFont.load_default(size=48)
    ImageDraw.Draw(img).text((40, 60), "Scanned pages need OCR", fill="black", font=font)
    img.save(tmp_path / "scan.pdf")
    img.save(tmp_path / "scan.png")
    for name in ["scan.pdf", "scan.png"]:
        text = load_file(tmp_path / name)[0].text.lower()
        assert "scanned" in text and "ocr" in text
