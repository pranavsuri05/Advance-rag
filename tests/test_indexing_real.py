"""Embedding / FAISS / BM25 tests with the REAL local models over the real 15-document corpus."""
import json

import numpy as np
import pytest

from src.ingestion.indexer import KnowledgeBase
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore


def test_real_embedder_dimensions_normalisation_and_semantics(real_embedder):
    v = real_embedder.encode(["Reciprocal rank fusion merges lists", "RRF combines ranked results", "banana bread recipe"])
    assert real_embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert v.shape == (3, 384) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)
    assert float(v[0] @ v[1]) > float(v[0] @ v[2]) + 0.2          # related pair clearly closer than unrelated


def test_real_embedder_single_empty_and_blank_inputs(real_embedder):
    assert real_embedder.encode(["only one"]).shape == (1, 384)
    assert len(real_embedder.encode([])) == 0                      # empty list -> empty result, no crash
    blank = real_embedder.encode([""])
    assert blank.shape == (1, 384) and np.isfinite(blank).all()


def test_real_index_structure_and_id_consistency(real_kb):
    kb = real_kb
    ids = [c.chunk_id for c in kb.chunks]
    assert len(kb.documents) == 15 and len(ids) == len(set(ids)) >= 15
    assert kb.vector_store.index.ntotal == len(ids) == len(kb.bm25.ids) == len(kb.embeddings)
    assert kb.vector_store.ids == ids == kb.bm25.ids                # FAISS and BM25 share chunk ids and order
    assert sum(d["num_chunks"] for d in kb.documents.values()) == len(ids)
    assert {c.doc_id for c in kb.chunks} == set(kb.documents)
    assert all(c.source == kb.documents[c.doc_id]["filename"] for c in kb.chunks)
    assert kb.meta["embedding_model"] == kb.embedder.model_name and kb.meta["dimension"] == 384


def test_real_retrieval_finds_expected_documents(real_kb):
    q = "What is the default value of k in Reciprocal Rank Fusion?"
    qv = real_kb.embedder.encode([q])[0]
    vec = [real_kb.chunk_map()[c].source for c, _ in real_kb.vector_store.search(qv, 5)]
    bm = [real_kb.chunk_map()[c].source for c, _ in real_kb.bm25.search(q, 5)]
    assert vec[0] == "04_hybrid_search_and_rrf.md" and "04_hybrid_search_and_rrf.md" in bm


def test_faiss_persistence_and_reload_gives_identical_results(real_kb, settings, real_embedder):
    for name in ("faiss.index", "chunks.json", "embeddings.npy", "documents.json", "bm25_corpus.json", "index_meta.json"):
        assert (settings.index_dir / name).exists(), name
    q = real_kb.embedder.encode(["cross-encoder reranking precision"])[0]
    before = real_kb.vector_store.search(q, 5)
    reloaded = KnowledgeBase(settings, real_embedder)
    after = reloaded.vector_store.search(q, 5)
    assert [c for c, _ in before] == [c for c, _ in after]
    assert np.allclose([s for _, s in before], [s for _, s in after], atol=1e-5)
    assert reloaded.bm25.search("cross encoder", 3) == real_kb.bm25.search("cross encoder", 3)
    assert [c.chunk_id for c in reloaded.chunks] == [c.chunk_id for c in real_kb.chunks]
    import faiss
    assert faiss.read_index(str(settings.index_dir / "faiss.index")).ntotal == len(real_kb.chunks)


def test_bm25_corpus_file_matches_chunks(real_kb, settings):
    corpus = json.loads((settings.index_dir / "bm25_corpus.json").read_text())
    assert len(corpus) == len(real_kb.chunks) and all(isinstance(t, list) and t for t in corpus)


def test_empty_index_behaviour(settings, real_embedder):
    kb = KnowledgeBase(settings, real_embedder)
    assert not kb.ready and kb.status()["chunks"] == 0
    assert kb.vector_store.search(np.zeros(384, dtype="float32"), 5) == []
    assert kb.bm25.search("anything", 5) == []
    res = HybridRetriever(kb).search(["anything at all"])
    assert res.fused == [] and res.vector_results == [] and res.bm25_results == []


def test_missing_faiss_file_recovers_from_embeddings(real_kb, settings, real_embedder):
    (settings.index_dir / "faiss.index").unlink()
    kb = KnowledgeBase(settings, real_embedder)
    assert kb.ready and kb.vector_store.index.ntotal == len(real_kb.chunks)


@pytest.mark.parametrize("victim", ["embeddings.npy", "index_meta.json", "documents.json"])
def test_missing_or_corrupt_index_files_are_reported_and_rebuild_recovers(real_kb, settings, real_embedder, victim):
    (settings.index_dir / victim).unlink()
    kb = KnowledgeBase(settings, real_embedder)
    assert not kb.ready and kb.last_error and "ebuild" in kb.last_error     # no crash, clear message
    info = kb.rebuild()                                                     # documents dir still has the files
    assert info["documents"] == 15 and kb.ready and kb.vector_store.index.ntotal == info["chunks"]


def test_corrupt_chunks_json_reported_and_recovered(real_kb, settings, real_embedder):
    (settings.index_dir / "chunks.json").write_text("{not valid json")
    kb = KnowledgeBase(settings, real_embedder)
    assert not kb.ready and "unreadable" in kb.last_error
    assert kb.rebuild()["chunks"] == len(real_kb.chunks)


def test_rebuild_does_not_reembed_unchanged_text(real_kb, settings):
    calls = []
    orig = real_kb.embedder.encode
    real_kb.embedder.encode = lambda texts: (calls.append(len(texts)), orig(texts))[1] if texts else orig(texts)
    try:
        info = real_kb.rebuild()
    finally:
        real_kb.embedder.encode = orig
    assert info["chunks"] == len(real_kb.chunks) and calls == []          # every chunk's vector was reused from cache


def test_duplicate_chunk_text_across_documents_is_indexed_with_distinct_ids(settings, real_embedder, tmp_path):
    kb = KnowledgeBase(settings, real_embedder)
    shared = "This shared paragraph appears in both documents verbatim.\n\n"
    (tmp_path / "x.md").write_text(shared + "Only X has this tail about apples.")
    (tmp_path / "y.md").write_text(shared + "Only Y has this tail about oranges.")
    kb.add_file(tmp_path / "x.md"); kb.add_file(tmp_path / "y.md")
    ids = [c.chunk_id for c in kb.chunks]
    assert len(ids) == len(set(ids)) and kb.vector_store.index.ntotal == len(ids)
    assert kb.vector_store.search(kb.embedder.encode(["shared paragraph appears in both"])[0], 5)[0][1] > 0.5


def test_vector_store_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        VectorStore().build(np.zeros((3, 4), dtype="float32"), ["a", "b"])
    vs = VectorStore()
    vs.build(np.eye(3, dtype="float32"), ["a", "b", "c"])
    assert vs.search(np.array([1, 0, 0], dtype="float32"), 10)[0][0] == "a" and len(vs.search(np.array([1, 0, 0], dtype="float32"), 10)) == 3
