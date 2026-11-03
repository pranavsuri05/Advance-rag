"""FAISS inner-product index over L2-normalised vectors (= cosine similarity)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.omp_guard import import_torch_before_faiss

import_torch_before_faiss()   # must run before FAISS is used (see omp_guard)
import faiss  # noqa: E402


class VectorStore:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.ids: list[str] = []

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        if len(ids) != len(vectors):
            raise ValueError("vectors and ids must have equal length")
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.ids = list(ids)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to k (chunk_id, cosine_score) pairs, best first."""
        if self.index is None or not self.ids:
            return []
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype="float32")
        scores, idxs = self.index.search(q, min(k, len(self.ids)))
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]

    def save(self, path: Path) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(path))
