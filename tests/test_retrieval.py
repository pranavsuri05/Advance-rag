from src.retrieval.hybrid_retriever import HybridRetriever, rrf_fuse


def test_embeddings_normalised_and_deterministic(kb):
    a = kb.embedder.encode(["reciprocal rank fusion"])
    b = kb.embedder.encode(["reciprocal rank fusion"])
    assert (a == b).all() and abs((a ** 2).sum() - 1) < 1e-5


def test_vector_and_bm25_retrieval(kb):
    q = kb.embedder.encode(["reciprocal rank fusion constant"])[0]
    hits = kb.vector_store.search(q, 3)
    assert hits[0][0].startswith("rrf") and hits[0][1] >= hits[-1][1]
    bm = kb.bm25.search("BM25 term frequency", 3)
    assert bm[0][0].startswith("bm25")


def test_bm25_empty_cases(kb):
    assert kb.bm25.search("", 5) == []
    assert kb.bm25.search("zzzzqqq", 5) == []


def test_rrf_math_and_ranking():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60))
    assert abs(fused["b"] - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(fused["a"] - 1 / 61) < 1e-12 and abs(fused["d"] - 1 / 62) < 1e-12
    order = [c for c, _ in rrf_fuse([["a", "b", "c"], ["b", "d"]])]
    assert order[0] == "b" and set(order) == {"a", "b", "c", "d"}


def test_rrf_dedup_and_empty():
    assert rrf_fuse([]) == [] and rrf_fuse([[], []]) == []
    assert dict(rrf_fuse([["a", "a"]]))["a"] == 1 / 61


def test_hybrid_result_keeps_individual_ranks(kb):
    res = HybridRetriever(kb).search(["reciprocal rank fusion constant"])
    assert res.fused[0].rrf_score >= res.fused[-1].rrf_score
    ids = [f.chunk.chunk_id for f in res.fused]
    assert len(ids) == len(set(ids))
    top = res.fused[0]
    assert top.vector_rank is not None and top.bm25_rank is not None
    expected = 1 / (60 + top.vector_rank) + 1 / (60 + top.bm25_rank)
    assert abs(top.rrf_score - expected) < 1e-9


def test_hybrid_multi_query(kb):
    res = HybridRetriever(kb).search(["reciprocal rank fusion", "BM25 term frequency", "reciprocal rank fusion"])
    assert len(res.per_query) == 2  # duplicate query removed
    sources = {f.chunk.source for f in res.fused}
    assert {"rrf.md", "bm25.md"} <= sources


def test_hybrid_empty_query(kb):
    res = HybridRetriever(kb).search(["   "])
    assert res.fused == []
