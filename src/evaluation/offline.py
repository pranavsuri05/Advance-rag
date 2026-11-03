"""Offline retrieval evaluation: runs the REAL local retrieval stack (embeddings, FAISS, BM25, RRF,
cross-encoder) over the gold dataset with NO LLM calls and NO network access.

Because there is no LLM, the raw question is used as the query (no rewriting / decomposition), and
history-dependent follow-up questions are excluded. Metrics are therefore a lower-bound view of
retrieval quality and are NOT end-to-end results.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

from src.evaluation import scoring as sc
from src.evaluation.evaluator import GoldItem
from src.ingestion.indexer import KnowledgeBase
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

VARIANTS = ("vector_only", "bm25_only", "hybrid_rrf", "hybrid_rrf_rerank")


def _sources(kb: KnowledgeBase, ids: list[str]) -> list[str]:
    cmap = kb.chunk_map()
    return [cmap[i].source for i in ids]


def run_offline_retrieval_eval(kb: KnowledgeBase, reranker: CrossEncoderReranker, items: list[GoldItem],
                               k: int = 5, per_retriever: int = 20, fused_limit: int = 18, rrf_k: int = 60) -> dict:
    hybrid = HybridRetriever(kb, rrf_k, per_retriever, fused_limit)
    rows, excluded = [], []
    for it in items:
        if not it.answerable or not it.expected_documents:
            excluded.append({"id": it.id, "reason": "unanswerable (no relevant documents to retrieve)"})
            continue
        if it.history:
            excluded.append({"id": it.id, "reason": "needs conversation history (requires LLM rewrite)"})
            continue
        q, rel = it.question, set(it.expected_documents)
        qvec = kb.embedder.encode([q])[0]
        vec_ids = [c for c, _ in kb.vector_store.search(qvec, per_retriever)]
        bm_ids = [c for c, _ in kb.bm25.search(q, per_retriever)]
        fused = hybrid.search([q]).fused
        rr = reranker.rerank([q], fused, k)
        ranked = {"vector_only": _sources(kb, vec_ids), "bm25_only": _sources(kb, bm_ids),
                  "hybrid_rrf": [f.chunk.source for f in fused],
                  "hybrid_rrf_rerank": [f.chunk.source for f in rr.items]}
        row = {"id": it.id, "category": it.category, "question": q, "expected_documents": it.expected_documents,
               "reranked": rr.reranked, "variants": {}}
        for name, docs in ranked.items():
            row["variants"][name] = {"top_docs": docs[:k], "precision_at_k": sc.precision_at_k(docs, rel, k),
                                     "recall_at_k": sc.recall_at_k(docs, rel, k), "hit_rate_at_k": sc.hit_rate_at_k(docs, rel, k),
                                     "mrr": sc.reciprocal_rank(docs[:k], rel)}
        rows.append(row)
    agg = {v: {m: sc.mean([r["variants"][v][m] for r in rows]) for m in ("precision_at_k", "recall_at_k", "hit_rate_at_k", "mrr")}
           for v in VARIANTS}
    return {"kind": "offline_retrieval", "llm_calls": 0, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "k": k, "embedding_model": kb.embedder.model_name, "num_chunks": len(kb.chunks),
            "evaluated": len(rows), "excluded": excluded, "aggregate": agg, "results": rows,
            "note": "Raw question used as query; no LLM rewrite/decomposition; document-level relevance labels."}


def save_offline(run: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"offline_retrieval_{run['timestamp'].replace(':', '-')}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    return path
