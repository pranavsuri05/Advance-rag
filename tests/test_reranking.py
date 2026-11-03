from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from tests.conftest import FakeCrossEncoder


def candidates(kb, q):
    return HybridRetriever(kb).search([q]).fused


def test_rerank_sorts_by_score_and_keeps_metadata(kb):
    cands = candidates(kb, "bm25 term frequency inverse document")
    res = CrossEncoderReranker("fake", model=FakeCrossEncoder()).rerank(["bm25 term frequency inverse document"], cands, 2)
    assert res.reranked and res.status == "cross_encoder" and len(res.items) == 2
    scores = [i.rerank_score for i in res.items]
    assert scores == sorted(scores, reverse=True)
    top = res.items[0]
    assert top.chunk.source == "bm25.md" and top.post_rerank_rank == 1
    assert top.pre_rerank_rank is not None and top.rrf_score is not None


def test_rerank_empty(kb):
    res = CrossEncoderReranker("fake", model=FakeCrossEncoder()).rerank(["q"], [], 3)
    assert res.items == [] and not res.reranked and res.status == "empty"


def test_rerank_model_failure_is_not_reported_as_reranked(kb):
    class Broken:
        def predict(self, pairs):
            raise RuntimeError("boom")
    res = CrossEncoderReranker("fake", model=Broken()).rerank(["q"], candidates(kb, "fusion"), 2)
    assert not res.reranked and res.status == "unavailable" and "boom" in res.error
    assert all(i.rerank_score is None for i in res.items)


def test_rerank_uses_max_over_subqueries(kb):
    cands = candidates(kb, "fusion bm25 faiss")
    res = CrossEncoderReranker("fake", model=FakeCrossEncoder()).rerank(["fusion", "bm25 keyword", "faiss vector"], cands, 3)
    assert {i.chunk.source for i in res.items} == {"rrf.md", "bm25.md", "faiss.md"}
