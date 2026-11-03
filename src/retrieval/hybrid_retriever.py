"""Hybrid retrieval: FAISS dense + BM25 sparse, fused with Reciprocal Rank Fusion."""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.ingestion.chunker import Chunk
from src.ingestion.indexer import KnowledgeBase


class RetrievedChunk(BaseModel):
    chunk: Chunk
    vector_rank: int | None = None   # 1-based; None = not returned by that retriever
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    fused_rank: int | None = None
    # filled in by the reranker
    pre_rerank_rank: int | None = None
    rerank_score: float | None = None
    post_rerank_rank: int | None = None


class QueryRetrieval(BaseModel):
    query: str
    vector: list[RetrievedChunk] = Field(default_factory=list)
    bm25: list[RetrievedChunk] = Field(default_factory=list)


class HybridResult(BaseModel):
    queries: list[str]
    per_query: list[QueryRetrieval]
    vector_results: list[RetrievedChunk]   # de-duplicated across queries, best rank kept
    bm25_results: list[RetrievedChunk]
    fused: list[RetrievedChunk]            # RRF over every ranked list


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_i 1 / (k + rank_i(d)), ranks are 1-based.

    Chunks that appear in only some lists simply get fewer terms. Duplicates within one list
    count once (their best rank). Ties are broken by first appearance for determinism.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, cid in enumerate(ranking, start=1):
            if cid in seen:
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(cid, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase, rrf_k: int = 60, per_retriever: int = 20, fused_limit: int = 18):
        self.kb, self.rrf_k, self.per_retriever, self.fused_limit = kb, rrf_k, per_retriever, fused_limit

    def search(self, queries: list[str]) -> HybridResult:
        cmap = self.kb.chunk_map()
        per_query: list[QueryRetrieval] = []
        rankings: list[list[str]] = []
        best_vec: dict[str, RetrievedChunk] = {}
        best_bm: dict[str, RetrievedChunk] = {}
        queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
        for q in queries:
            qvec = self.kb.embedder.encode([q])[0]
            vec_hits = self.kb.vector_store.search(qvec, self.per_retriever)
            bm_hits = self.kb.bm25.search(q, self.per_retriever)
            v = [RetrievedChunk(chunk=cmap[c], vector_rank=r, vector_score=s) for r, (c, s) in enumerate(vec_hits, 1)]
            b = [RetrievedChunk(chunk=cmap[c], bm25_rank=r, bm25_score=s) for r, (c, s) in enumerate(bm_hits, 1)]
            per_query.append(QueryRetrieval(query=q, vector=v, bm25=b))
            rankings += [[h.chunk.chunk_id for h in v], [h.chunk.chunk_id for h in b]]
            for h in v:
                cur = best_vec.get(h.chunk.chunk_id)
                if cur is None or h.vector_rank < cur.vector_rank:
                    best_vec[h.chunk.chunk_id] = h
            for h in b:
                cur = best_bm.get(h.chunk.chunk_id)
                if cur is None or h.bm25_rank < cur.bm25_rank:
                    best_bm[h.chunk.chunk_id] = h
        fused: list[RetrievedChunk] = []
        for pos, (cid, score) in enumerate(rrf_fuse(rankings, self.rrf_k)[: self.fused_limit], start=1):
            v, b = best_vec.get(cid), best_bm.get(cid)
            fused.append(RetrievedChunk(chunk=cmap[cid], vector_rank=v.vector_rank if v else None,
                                        vector_score=v.vector_score if v else None,
                                        bm25_rank=b.bm25_rank if b else None, bm25_score=b.bm25_score if b else None,
                                        rrf_score=score, fused_rank=pos))
        sort_v = sorted(best_vec.values(), key=lambda h: (h.vector_rank, -h.vector_score))
        sort_b = sorted(best_bm.values(), key=lambda h: (h.bm25_rank, -h.bm25_score))
        return HybridResult(queries=queries, per_query=per_query, vector_results=sort_v, bm25_results=sort_b, fused=fused)
