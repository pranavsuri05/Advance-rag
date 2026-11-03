"""Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2, local)."""
from __future__ import annotations

from pydantic import BaseModel

from src.retrieval.hybrid_retriever import RetrievedChunk


class RerankResult(BaseModel):
    items: list[RetrievedChunk]
    reranked: bool            # True only if cross-encoder inference really ran
    status: str               # "cross_encoder" | "empty" | "unavailable"
    error: str | None = None


class CrossEncoderReranker:
    def __init__(self, model_name: str, model=None):
        self.model_name = model_name
        self._model = model    # injectable for tests; any object with .predict(list[(q, d)])
        self._load_error: str | None = None

    def _ensure(self):
        if self._model is None and self._load_error is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"{exc.__class__.__name__}: {exc}"
        return self._model

    def rerank(self, queries: list[str], candidates: list[RetrievedChunk], top_n: int) -> RerankResult:
        """Score every (query, chunk) pair; a chunk's score is its max over `queries` (original +
        subqueries), so each sub-question can pull in its own best evidence."""
        if not candidates:
            return RerankResult(items=[], reranked=False, status="empty")
        for pos, c in enumerate(candidates, 1):
            c.pre_rerank_rank = pos
        model = self._ensure()
        if model is None:
            kept = [c.model_copy() for c in candidates[:top_n]]
            for i, c in enumerate(kept, 1):
                c.post_rerank_rank = i
            return RerankResult(items=kept, reranked=False, status="unavailable", error=self._load_error)
        queries = [q for q in dict.fromkeys(queries) if q]
        try:
            pairs = [(q, c.chunk.text) for q in queries for c in candidates]
            raw = [float(s) for s in model.predict(pairs)]
        except Exception as exc:  # noqa: BLE001
            kept = [c.model_copy() for c in candidates[:top_n]]
            for i, c in enumerate(kept, 1):
                c.post_rerank_rank = i
            return RerankResult(items=kept, reranked=False, status="unavailable",
                                error=f"{exc.__class__.__name__}: {exc}")
        n = len(candidates)
        scored = []
        for j, c in enumerate(candidates):
            c2 = c.model_copy()
            c2.rerank_score = max(raw[qi * n + j] for qi in range(len(queries)))
            scored.append(c2)
        scored.sort(key=lambda c: (-c.rerank_score, c.pre_rerank_rank))
        for i, c in enumerate(scored, 1):
            c.post_rerank_rank = i
        return RerankResult(items=scored[:top_n], reranked=True, status="cross_encoder")
