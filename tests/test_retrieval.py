import numpy as np

from rag.bm25 import BM25, reciprocal_rank_fusion, tokenize
from rag.chunker import Chunk
from rag.retriever import Retriever
from rag.vector_store import VectorStore

TEXTS = {
    "a.md": ["Cats are small furry pets that purr.", "The XLM-R model scores 0.81 F1 on MLQA."],
    "b.md": ["Dogs are loyal pets that bark.", "Transformers use self attention over tokens."],
}


def build_store(embedder) -> VectorStore:
    store, cid = VectorStore(), 0
    for source, texts in TEXTS.items():
        chunks = [Chunk(cid + i, t, source) for i, t in enumerate(texts)]
        cid += len(texts)
        store.add(chunks, embedder.encode_documents(texts), source, digest=source)
    return store


# ------------------------------------------------------------------ BM25
def test_tokenize_drops_stopwords_keeps_terms():
    assert tokenize("The XLM-R model is at 0.81 F1") == ["xlm-r", "model", "0.81", "f1"]


def test_bm25_ranks_exact_term_first():
    docs = ["cats purr", "dogs bark loudly", "cats and dogs"]
    scores = BM25(docs).scores("bark")
    assert scores.argmax() == 1 and scores[0] == 0 and scores[2] == 0


def test_bm25_rare_terms_weigh_more():
    docs = ["common rare", "common", "common", "common"]
    bm = BM25(docs)
    assert bm.scores("rare")[0] > bm.scores("common")[0]


def test_bm25_empty_corpus():
    assert len(BM25([]).scores("anything")) == 0


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 9]])
    assert fused[0] == 2 or fused[0] == 3
    assert set(fused) == {1, 2, 3, 9}
    assert fused.index(2) < fused.index(1) and fused.index(2) < fused.index(9)


# ------------------------------------------------------------------ vector store
def test_store_search_and_source_filter(fake_embedder):
    store = build_store(fake_embedder)
    q = fake_embedder.encode_query("which pets bark")
    assert store.search(q, 1)[0][0].text.startswith("Dogs")
    only_a = store.search(q, 5, sources=["a.md"])
    assert {c.source for c, _ in only_a} == {"a.md"}


def test_store_remove_and_version(fake_embedder):
    store = build_store(fake_embedder)
    v = store.version
    store.remove("a.md")
    assert store.version > v
    assert {c.source for c in store.chunks} == {"b.md"}
    assert store.documents() == {"b.md": 2}
    store.remove("missing.md")          # no-op


def test_store_save_load_roundtrip(fake_embedder, tmp_path):
    store = build_store(fake_embedder)
    store.save(tmp_path, {"embed_model": "fake"})
    loaded, settings = VectorStore.load(tmp_path)
    assert settings == {"embed_model": "fake"}
    assert [c.text for c in loaded.chunks] == [c.text for c in store.chunks]
    assert np.allclose(loaded.vectors, store.vectors)
    q = fake_embedder.encode_query("self attention")
    assert loaded.search(q, 1)[0][0].text == store.search(q, 1)[0][0].text


# ------------------------------------------------------------------ retriever
def test_dense_and_hybrid_retrieval(fake_embedder):
    store = build_store(fake_embedder)
    r = Retriever(fake_embedder, store, reranker_model=None)
    for hybrid in (False, True):
        hits = r.retrieve("XLM-R F1 score", k=2, hybrid=hybrid)
        assert hits[0].chunk.text.startswith("The XLM-R")
        assert len(hits) == 2


def test_hybrid_respects_sources_and_tracks_changes(fake_embedder):
    store = build_store(fake_embedder)
    r = Retriever(fake_embedder, store, reranker_model=None)
    hits = r.retrieve("pets", k=4, hybrid=True, sources=["b.md"])
    assert hits and all(h.chunk.source == "b.md" for h in hits)

    r.retrieve("pets", k=1, hybrid=True)        # builds the BM25 index
    texts = ["Parrots are pets that talk."]
    store.add([Chunk(99, texts[0], "c.md")], fake_embedder.encode_documents(texts), "c.md", "c")
    hits = r.retrieve("parrots talk", k=1, hybrid=True)
    assert hits[0].chunk.source == "c.md"     # BM25 rebuilt after the store changed


def test_retrieve_on_empty_store(fake_embedder):
    r = Retriever(fake_embedder, VectorStore(), reranker_model=None)
    assert r.retrieve("x", 3) == [] and r.retrieve("x", 3, hybrid=True) == []
