from rag.chunker import Chunk, chunk_document, split_sentences
from rag.loader import Page


def count(text: str) -> int:
    return len(text.split())


def sentence(i: int, words: int = 10) -> str:
    return " ".join([f"w{i}"] * (words - 1)) + f" end{i}."


def test_split_sentences_collapses_whitespace():
    assert split_sentences("One.  Two!\nThree?") == ["One.", "Two!", "Three?"]
    assert split_sentences("   ") == []


def test_chunks_respect_size_limit():
    pages = [Page(" ".join(sentence(i) for i in range(30)), "doc.txt")]
    chunks = chunk_document(pages, chunk_size=50, overlap=0, count_tokens=count)
    assert len(chunks) > 1
    assert all(count(c.text) <= 50 for c in chunks)
    # every sentence ends up in some chunk, in order
    joined = " ".join(c.text for c in chunks)
    assert all(f"end{i}." in joined for i in range(30))


def test_overlap_repeats_trailing_sentences():
    pages = [Page(" ".join(sentence(i) for i in range(12)), "doc.txt")]
    chunks = chunk_document(pages, chunk_size=40, overlap=10, count_tokens=count)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = split_sentences(prev.text)[-1]
        assert nxt.text.startswith(last_sentence)


def test_no_overlap_when_zero():
    pages = [Page(" ".join(sentence(i) for i in range(12)), "doc.txt")]
    chunks = chunk_document(pages, chunk_size=40, overlap=0, count_tokens=count)
    total = sum(len(split_sentences(c.text)) for c in chunks)
    assert total == 12


def test_overlong_sentence_is_split_by_words():
    long = " ".join(f"x{i}" for i in range(300)) + "."
    chunks = chunk_document([Page(long, "d.txt")], chunk_size=64, overlap=0, count_tokens=count)
    assert len(chunks) >= 5
    assert all(count(c.text) <= 64 for c in chunks)


def test_chunks_track_pages_and_sections():
    pages = [
        Page(sentence(1, 30), "d.pdf", page=1),
        Page(sentence(2, 30), "d.pdf", page=2),
        Page(sentence(3, 30), "d.pdf", page=3, section="Methods"),
    ]
    chunks = chunk_document(pages, chunk_size=70, overlap=0, count_tokens=count, start_id=7)
    assert chunks[0].id == 7
    assert chunks[0].pages == [1, 2]
    assert chunks[-1].pages[-1] == 3 and "Methods" in chunks[-1].sections


def test_location_strings():
    assert Chunk(0, "t", "s", [2], []).location() == "page 2"
    assert Chunk(0, "t", "s", [2, 3, 4], ["Intro"]).location() == "pages 2-4, § Intro"
    assert Chunk(0, "t", "s").location() == "whole document"


def test_empty_input():
    assert chunk_document([], 100, 10, count) == []
