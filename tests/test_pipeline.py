from rag.highlight import best_sentences, highlight_html
from tests.conftest import FakeEmbedder, FakeLLM

DOCS = {
    "pets.md": "# Cats\nCats are small furry pets that purr.\n\n# Dogs\nDogs are loyal pets that bark.\n",
    "nlp.md": "# Attention\nTransformers use self attention over tokens.\n",
}


def test_sync_directory_add_unchanged_update_remove(make_pipeline):
    p = make_pipeline(DOCS)
    assert p.sync_directory() == {"nlp.md": "added", "pets.md": "added"}
    assert p.sync_directory() == {"nlp.md": "unchanged", "pets.md": "unchanged"}

    (p.docs_dir / "nlp.md").write_text("# New\nCompletely new text about parsing.\n", encoding="utf-8")
    (p.docs_dir / "pets.md").unlink()
    report = p.sync_directory()
    assert report == {"nlp.md": "updated", "pets.md": "removed"}
    assert list(p.documents()) == ["nlp.md"]


def test_index_persists_between_pipelines(make_pipeline):
    p = make_pipeline(DOCS)
    p.sync_directory()
    p2 = make_pipeline({})            # same tmp dirs, fresh object loads the saved index
    assert p2.documents() == p.documents()
    assert p2.sync_directory() == {"nlp.md": "unchanged", "pets.md": "unchanged"}


def test_answer_returns_hits_and_stream(make_pipeline):
    p = make_pipeline(DOCS)
    p.sync_directory()
    ans = p.answer("which pets bark", k=2)
    assert ans.hits[0].chunk.source == "pets.md"
    assert "excerpts" in "".join(ans.stream)
    restricted = p.answer("which pets bark", sources=["nlp.md"])
    assert all(h.chunk.source == "nlp.md" for h in restricted.hits)


def test_answer_with_empty_library(make_pipeline):
    p = make_pipeline({})
    p.sync_directory()
    ans = p.answer("anything")
    assert ans.hits == [] and "No documents" in "".join(ans.stream)


def test_low_rerank_score_answers_not_found(make_pipeline, monkeypatch):
    import config
    from rag.prompts import NOT_FOUND

    class FakeReranker:
        def __init__(self, score):
            self.score = score

        def rerank(self, query, hits, top_k):
            for h in hits:
                h.rerank_score = self.score
            return hits[:top_k]

    llm = FakeLLM()
    p = make_pipeline(DOCS, llm=llm)
    p.sync_directory()
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", -8.0)

    p.retriever._reranker = FakeReranker(-10.0)        # nothing relevant: skip the LLM
    ans = p.answer("what is the capital of france", rerank=True)
    assert ans.hits == [] and "".join(ans.stream) == NOT_FOUND and llm.last_context is None

    p.retriever._reranker = FakeReranker(2.0)          # relevant: answer as usual
    ans = p.answer("which pets bark", rerank=True)
    assert ans.hits and "excerpts" in "".join(ans.stream)


def test_fit_context_drops_excerpts_that_do_not_fit(make_pipeline):
    small = FakeLLM(budget=25)          # ~ one excerpt's worth of words
    p = make_pipeline(DOCS, llm=small)
    p.sync_directory()
    ans = p.answer("pets", k=3)
    assert len(ans.hits) == 1           # only what the model can actually read
    "".join(ans.stream)
    assert small.last_context.count("] ") == 1


def test_fit_context_keeps_all_for_large_models(make_pipeline):
    p = make_pipeline(DOCS, llm=FakeLLM())  # no context_budget -> everything is sent
    p.sync_directory()
    n_chunks = len(p.store.chunks)
    assert n_chunks >= 2
    assert len(p.answer("pets", k=n_chunks).hits) == n_chunks


def test_remove_document_deletes_file(make_pipeline):
    p = make_pipeline(DOCS)
    p.sync_directory()
    p.remove_document("pets.md", delete_file=True)
    assert not (p.docs_dir / "pets.md").exists()
    assert "pets.md" not in p.documents()


def test_highlight_best_sentence_and_escaping():
    chunk = "Cats purr. Dogs bark loudly at <night>. Fish swim."
    marks = best_sentences(FakeEmbedder(), "Dogs bark loudly [1].", chunk, min_score=0.1)
    assert marks == ["Dogs bark loudly at <night>."]
    html = highlight_html(chunk, marks)
    assert "<mark>Dogs bark loudly at &lt;night&gt;.</mark>" in html
    assert "<night>" not in html                     # user text is escaped
    assert highlight_html("x" * 2000, [], max_chars=100).endswith("…")
