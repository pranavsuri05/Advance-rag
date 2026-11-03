"""Central RAG pipeline orchestrating all stages and returning a structured response."""
from __future__ import annotations

import time

from pydantic import BaseModel, Field

from src.config import Settings
from src.ingestion.indexer import KnowledgeBase
from src.llm.provider import LLMError, TracedLLM
from src.memory.session_manager import SessionManager
from src.rag.context_grader import ContextGrade, ContextGrader
from src.rag.corrective_rag import Attempt, CorrectiveRAG
from src.rag.generator import CitationInfo, Generator
from src.rag.query_transformer import QueryTransformer
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.retrieval.reranker import CrossEncoderReranker


class StageRecord(BaseModel):
    name: str
    status: str            # success | skipped | failed | fallback
    duration_ms: int = 0
    detail: str = ""


class RAGResponse(BaseModel):
    answer: str = ""
    abstained: bool = False
    error: str | None = None            # user-facing error (pipeline did not complete)
    citations: list[CitationInfo] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    missing_citations: bool = False
    original_query: str = ""
    rewritten_query: str = ""
    decomposed_queries: list[str] = Field(default_factory=list)
    transformation_reason: str = ""
    vector_results: list[RetrievedChunk] = Field(default_factory=list)
    bm25_results: list[RetrievedChunk] = Field(default_factory=list)
    fused_results: list[RetrievedChunk] = Field(default_factory=list)
    reranked_results: list[RetrievedChunk] = Field(default_factory=list)
    rerank_status: str = ""
    rerank_error: str | None = None
    context_grade: ContextGrade | None = None
    attempts: list[Attempt] = Field(default_factory=list)
    retried: bool = False
    retry_queries: list[list[str]] = Field(default_factory=list)
    llm_call_ids: list[str] = Field(default_factory=list)
    stages: list[StageRecord] = Field(default_factory=list)
    timing_ms: dict[str, int] = Field(default_factory=dict)
    session_id: str = ""


class RAGPipeline:
    def __init__(self, settings: Settings, kb: KnowledgeBase, llm: TracedLLM | None,
                 reranker: CrossEncoderReranker, sessions: SessionManager):
        self.s, self.kb, self.llm, self.reranker, self.sessions = settings, kb, llm, reranker, sessions
        self.retriever = HybridRetriever(kb, settings.rrf_k, settings.candidates_per_retriever, settings.fused_candidates)
        if llm is not None:
            self.transformer = QueryTransformer(llm)
            self.grader = ContextGrader(llm)
            self.corrective = CorrectiveRAG(llm, self.grader, settings.max_corrective_retries)
            self.generator = Generator(llm)

    # ----- public API -----
    def ask(self, question: str, session_id: str, remember: bool = True, history: list[dict] | None = None) -> RAGResponse:
        """Run the full pipeline. `remember=False` skips memory writes (evaluation); `history` overrides stored history."""
        question = (question or "").strip()
        r = RAGResponse(original_query=question, rewritten_query=question, session_id=session_id)
        t0 = time.perf_counter()
        if not question:
            r.error = "Please enter a question."
            return r
        if self.llm is None:
            r.error = LLMError("missing_key", "").user_message()
            return r
        if not self.kb.ready:
            r.error = "The knowledge base is empty. Index documents on the Knowledge Base page first."
            return r
        if history is None:
            history = self.sessions.window(session_id, self.s.history_window) if remember else []
        try:
            self._run(r, question, history, session_id)
        except LLMError as exc:
            r.error = exc.user_message()
            r.stages.append(StageRecord(name="llm_error", status="failed", detail=f"{exc.kind}: {exc.message[:200]}"))
            if exc.call_id:
                r.llm_call_ids.append(exc.call_id)
        r.timing_ms["total"] = int((time.perf_counter() - t0) * 1000)
        if remember:
            self._remember(r, question, session_id)
        return r

    def inspect(self, question: str, session_id: str | None = None) -> RAGResponse:
        """Run retrieval stages (transform, hybrid, rerank, grade, corrective) without generation or memory."""
        r = RAGResponse(original_query=question, rewritten_query=question, session_id=session_id or "inspector")
        if not question.strip() or self.llm is None or not self.kb.ready:
            r.error = ("Please enter a query." if not question.strip() else
                       "Configure an LLM API key first (see .env)." if self.llm is None else "Knowledge base is empty.")
            return r
        history = self.sessions.window(session_id, self.s.history_window) if session_id and self.sessions.exists(session_id) else []
        try:
            self._run(r, question.strip(), history, r.session_id, generate=False)
        except LLMError as exc:
            r.error = exc.user_message()
        return r

    # ----- internals -----
    def _stage(self, r: RAGResponse, name: str, fn, detail=lambda out: ""):
        t = time.perf_counter()
        out = fn()
        ms = int((time.perf_counter() - t) * 1000)
        r.timing_ms[name] = ms
        return out, ms

    def _run(self, r: RAGResponse, question: str, history: list[dict], sid: str, generate: bool = True) -> None:
        # 1. query transformation (rewrite + optional decomposition)
        tq, ms = self._stage(r, "query_transformation", lambda: self.transformer.transform(question, history, sid))
        r.rewritten_query, r.decomposed_queries, r.transformation_reason = tq.rewritten_query, tq.subqueries, tq.reason
        r.llm_call_ids += tq.call_ids
        status = "fallback" if tq.fell_back else "success"
        detail = ("fell back to original query: " + (tq.error or tq.reason)) if tq.fell_back else (
            f"rewritten; {len(tq.subqueries)} subqueries" if tq.rewrite_changed or tq.subqueries else "query already clear; unchanged")
        r.stages.append(StageRecord(name="Query transformation", status=status, duration_ms=ms, detail=detail))

        # 2-5. corrective loop wraps hybrid retrieval + reranking + grading
        timings = {"vector_bm25_rrf": 0, "rerank": 0}

        def retrieve(queries: list[str]):
            t = time.perf_counter()
            hybrid = self.retriever.search(queries)
            timings["vector_bm25_rrf"] += int((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            rerank = self.reranker.rerank(queries, hybrid.fused, self.s.final_top_k)
            timings["rerank"] += int((time.perf_counter() - t) * 1000)
            return hybrid, rerank

        cres, ms = self._stage(r, "corrective_retrieval",
                               lambda: self.corrective.run(question, tq.retrieval_queries, retrieve, sid))
        r.timing_ms.update(timings)
        first, final = cres.attempts[0], cres.final
        r.llm_call_ids += [a.grade.call_id for a in cres.attempts if a.grade.call_id] + cres.refine_call_ids
        r.attempts, r.retried, r.retry_queries = cres.attempts, cres.retried, cres.retry_queries
        r.vector_results, r.bm25_results, r.fused_results = (final.hybrid.vector_results, final.hybrid.bm25_results,
                                                             final.hybrid.fused)
        r.reranked_results, r.rerank_status, r.rerank_error = final.rerank.items, final.rerank.status, final.rerank.error
        r.context_grade = final.grade
        h = first.hybrid
        r.stages += [
            StageRecord(name="Vector search (FAISS)", status="success" if h.vector_results else "failed",
                        duration_ms=timings["vector_bm25_rrf"], detail=f"{len(h.vector_results)} hits"),
            StageRecord(name="BM25 keyword search", status="success" if h.bm25_results else "failed",
                        detail=f"{len(h.bm25_results)} hits"),
            StageRecord(name="Reciprocal Rank Fusion", status="success" if h.fused else "failed",
                        detail=f"{len(h.fused)} fused candidates (k={self.s.rrf_k})"),
            StageRecord(name="Cross-encoder reranking", duration_ms=timings["rerank"],
                        status="success" if final.rerank.reranked else "failed",
                        detail=(f"{len(final.rerank.items)} chunks kept" if final.rerank.reranked else
                                f"NOT performed ({final.rerank.status}): {final.rerank.error or 'no candidates'}")),
            StageRecord(name="Context grading", status="success" if first.grade.graded else "fallback",
                        detail=(f"initial: {'sufficient' if first.grade.relevant else 'insufficient'} "
                                f"(score {first.grade.score:.2f})" + ("" if first.grade.graded else " - grader unavailable"))),
        ]
        if cres.retried:
            last = final.grade
            r.stages.append(StageRecord(name="Corrective retry", status="success",
                                        detail=f"retry query: {final.queries[0]!r}; final grade: "
                                               f"{'sufficient' if last.relevant else 'insufficient'} ({last.score:.2f})"))
        else:
            reason = "initial context sufficient" if first.grade.relevant else (cres.refine_error or "retries disabled")
            r.stages.append(StageRecord(name="Corrective retry", status="skipped", detail=reason))
        if not generate:
            return

        # 6. generation or abstention
        if cres.proceed_to_generation:
            gen, ms = self._stage(r, "generation", lambda: self.generator.generate(question, final.rerank.items, history, sid))
            r.stages.append(StageRecord(name="Answer generation", status="success" if gen.parse_ok else "fallback",
                                        duration_ms=ms, detail="abstained" if gen.abstained else "answered from evidence"))
        else:
            gen, ms = self._stage(r, "generation", lambda: self.generator.abstain(
                question, final.grade.missing_information, final.grade.explanation, sid))
            r.stages.append(StageRecord(name="Answer generation", status="skipped", duration_ms=ms,
                                        detail="context insufficient after retries -> abstention message"))
        if gen.call_id:
            r.llm_call_ids.append(gen.call_id)
        if not gen.answer.strip():
            r.error = "The model returned an empty answer. Please try again."
            r.stages[-1] = StageRecord(name="Answer generation", status="failed", duration_ms=ms, detail="empty model output")
            return
        r.answer, r.abstained, r.citations = gen.answer, gen.abstained, gen.citations
        r.invalid_citations, r.missing_citations = gen.invalid_citations, gen.missing_citations
        if gen.abstained:
            vstatus, vdetail = "skipped", "abstained: no citations expected"
        elif gen.missing_citations:
            vstatus, vdetail = "failed", "answer has no valid citations"
        else:
            vstatus = "success"
            vdetail = f"{len(gen.citations)} valid, {len(gen.invalid_citations)} removed, {gen.corrected_citations} repaired"
        r.stages.append(StageRecord(name="Citation validation", status=vstatus, detail=vdetail))

    def _remember(self, r: RAGResponse, question: str, sid: str) -> None:
        self.sessions.add_message(sid, "user", question)
        if r.error:
            return
        srcs = [c.model_dump() for c in r.citations]
        self.sessions.add_message(sid, "assistant", r.answer, srcs, {
            "abstained": r.abstained, "rewritten_query": r.rewritten_query, "subqueries": r.decomposed_queries,
            "retried": r.retried, "timing_ms": r.timing_ms, "llm_call_ids": r.llm_call_ids,
            "stages": [s.model_dump() for s in r.stages], "context_grade": r.context_grade.model_dump() if r.context_grade else None,
            "retry_queries": r.retry_queries, "invalid_citations": r.invalid_citations,
            "reranked": [{"chunk_id": c.chunk.chunk_id, "source": c.chunk.source, "rerank_score": c.rerank_score}
                         for c in r.reranked_results]})
        self.sessions.add_context(sid, {"question": question, "rewritten": r.rewritten_query,
                                        "chunks": [c.chunk.chunk_id for c in r.reranked_results]})
