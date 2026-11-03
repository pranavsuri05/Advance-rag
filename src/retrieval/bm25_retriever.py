"""BM25 keyword retrieval (rank-bm25, Okapi variant)."""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.utils.text_utils import tokenize


class BM25Retriever:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self.ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.ids = [c.chunk_id for c in chunks]
        corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to k (chunk_id, bm25_score) pairs with score > 0, best first."""
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]
