"""Full-pipeline tests: REAL embeddings, FAISS, BM25 and cross-encoder over the real corpus; the LLM is a scripted mock.

The mock grader/generator reason over the actual retrieved text (e.g. "sufficient iff a passage contains X"), so a corrective
recovery here is caused by genuinely different retrieval results, not by a hard-coded flag.
"""
import json
import re

import pytest

from src.config import Settings
from src.ingestion.indexer import KnowledgeBase
from src.llm.provider import LLMError, TracedLLM
from src.memory.session_manager import SessionManager
from src.observability.tracer import Tracer
from src.rag.pipeline import RAGPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from tests.conftest import ScriptedProvider

VAGUE = "how can I make my pet happier"                       # retrieves nothing about the RRF paper
REFINED = "Cormack Clarke Buettcher RRF paper"                # retrieves the chunk containing 'Cormack'
MARK = "Cormack"


@pytest.fixture(scope="module")
def kb_ro(tmp_path_factory, real_embedder):
    root = tmp_path_factory.mktemp("realkb")
    s = Settings(documents_dir=root / "docs", index_dir=root / "idx", sessions_dir=root / "sess", logs_dir=root / "logs")
    s.ensure_dirs()
    kb = KnowledgeBase(s, real_embedder)
    from pathlib import Path
    for f in sorted((Path(__file__).resolve().parent.parent / "data" / "documents").glob("*.md")):
        kb.add_file(f)
    return kb


def cites_in(prompt):
    return re.findall(r"Source: (\S+) \| Chunk: (\S+)", prompt)


def make_handler(*, rewrite=None, decompose=None, grade=None, refine=None, answer=None, abstain=None):
    """Each hook gets (prompt) -> str|Exception|None (None = default behaviour)."""
    def default_rewrite(p):
        q = p.split("Latest user question:\n")[-1].strip()
        return json.dumps({"rewritten_query": q, "needs_decomposition": False, "reason": "clear"})

    def default_grade(p):
        ok = MARK in p
        return json.dumps({"relevant": ok, "score": 0.9 if ok else 0.1, "missing_information": "" if ok else "the RRF paper authors",
                           "explanation": "found" if ok else "not found"})

    def default_refine(p):
        return json.dumps({"query": REFINED, "alternatives": [], "reason": "target missing info"})

    def default_answer(p):
        pairs = cites_in(p)[:2]
        return json.dumps({"abstain": False, "answer": "Supported claim. " + " ".join(f"[Source: {s}, Chunk: {c}]" for s, c in pairs)})

    def default_abstain(p):
        return json.dumps({"abstain": True, "answer": "The knowledge base does not contain enough information."})

    hooks = {"query-understanding": (rewrite, default_rewrite), "split a multi-part": (decompose, lambda p: json.dumps({"subqueries": ["a", "b"], "reason": "r"})),
             "evidence grader": (grade, default_grade), "improve failed retrieval": (refine, default_refine),
             "Retrieval was graded insufficient": (abstain, default_abstain), "grounded RAG assistant.": (answer, default_answer)}

    def handler(prompt, system):
        for key, (custom, default) in hooks.items():
            if key in (system or ""):
                out = custom(prompt) if custom else None
                return default(prompt) if out is None else out
        return "{}"
    return handler


@pytest.fixture
def build(kb_ro, tmp_path, real_reranker):
    def _build(handler=None, retries=1, **hooks):
        s = Settings(documents_dir=kb_ro.settings.documents_dir, index_dir=kb_ro.settings.index_dir, sessions_dir=tmp_path / "sess",
                     logs_dir=tmp_path / "logs", final_top_k=5, max_corrective_retries=retries)
        prov = ScriptedProvider(handler or make_handler(**hooks))
        llm = TracedLLM(prov, Tracer(tmp_path / "calls.jsonl"), max_retries=0, backoff_base_s=0)
        pipe = RAGPipeline(s, kb_ro, llm, real_reranker, SessionManager(s.sessions_dir))
        return pipe, prov, s
    return _build


def calls_of(prov, key):
    return [c for c in prov.calls if key in (c["system"] or "")]


# ---------------- A. query transformation through the pipeline ----------------
def test_clear_query_not_rewritten_and_single_retrieval_query(build):
    pipe, prov, _ = build()
    r = pipe.ask("What is the default value of k in Reciprocal Rank Fusion?", pipe.sessions.create())
    assert r.rewritten_query == r.original_query and r.decomposed_queries == [] and len(calls_of(prov, "split a multi-part")) == 0
    assert r.attempts[0].hybrid.queries == [r.original_query]


def test_multipart_decomposition_drives_retrieval_for_each_subquery(build):
    subs = ["How does hybrid retrieval combine vector and keyword search?", "How does cross-encoder reranking improve relevance?",
            "What is the difference between fusion and reranking?"]
    pipe, prov, _ = build(rewrite=lambda p: json.dumps({"rewritten_query": "Explain hybrid retrieval and reranking and why both", "needs_decomposition": True, "reason": "multi"}),
                          decompose=lambda p: json.dumps({"subqueries": subs, "reason": "three needs"}), grade=lambda p: json.dumps({"relevant": True, "score": 1.0, "missing_information": "", "explanation": "ok"}))
    r = pipe.ask("Explain how hybrid retrieval and reranking work together, and why both are needed.", pipe.sessions.create())
    assert r.decomposed_queries == subs and r.attempts[0].hybrid.queries[1:] == subs
    assert len(r.attempts[0].hybrid.per_query) == 4
    docs = {x.chunk.source for x in r.reranked_results}
    assert "04_hybrid_search_and_rrf.md" in docs and "05_cross_encoder_reranking.md" in docs   # each sub-need pulled its own evidence


def test_followup_resolved_with_history_and_used_for_retrieval(build):
    def rewrite(p):
        if "What is Reciprocal Rank Fusion?" in p.split("Latest user question:")[0]:
            return json.dumps({"rewritten_query": "What is the default value of the RRF constant k?", "needs_decomposition": False, "reason": "resolved"})
    pipe, prov, _ = build(rewrite=rewrite)
    sid = pipe.sessions.create()
    pipe.ask("What is Reciprocal Rank Fusion?", sid)
    r = pipe.ask("And what is its default constant?", sid)
    assert r.rewritten_query == "What is the default value of the RRF constant k?"
    assert r.attempts[0].hybrid.queries == [r.rewritten_query]
    assert "What is Reciprocal Rank Fusion?" in calls_of(prov, "query-understanding")[1]["prompt"]


@pytest.mark.parametrize("bad", ["not json at all", "{}", '{"rewritten_query": ""}', '{"rewritten_query": 123}', '["list"]', ""])
def test_malformed_or_empty_rewrite_falls_back_to_original_query(build, bad):
    pipe, _, _ = build(rewrite=lambda p: bad)
    r = pipe.ask("What is BM25?", pipe.sessions.create())
    assert r.error is None and r.rewritten_query == "What is BM25?" and r.attempts[0].hybrid.queries == ["What is BM25?"]
    assert next(s for s in r.stages if s.name == "Query transformation").status == "fallback"


@pytest.mark.parametrize("bad", ["garbage", '{"subqueries": []}', '{"subqueries": ["only one"]}', '{"subqueries": "text"}'])
def test_decomposition_failure_keeps_pipeline_running(build, bad):
    pipe, _, _ = build(rewrite=lambda p: json.dumps({"rewritten_query": "two things a and b", "needs_decomposition": True, "reason": "x"}),
                       decompose=lambda p: bad)
    r = pipe.ask("two things a and b", pipe.sessions.create())
    assert r.error is None and r.decomposed_queries == [] and r.answer


def test_decomposition_llm_error_nonfatal(build):
    pipe, _, _ = build(rewrite=lambda p: json.dumps({"rewritten_query": "q about rrf", "needs_decomposition": True, "reason": "x"}),
                       decompose=lambda p: LLMError("server", "503"))
    r = pipe.ask("q about rrf", pipe.sessions.create())
    assert r.error is None and r.decomposed_queries == [] and r.answer


# ---------------- B. hybrid retrieval ----------------
def test_ranking_stability_and_no_duplicates(kb_ro):
    h = HybridRetriever(kb_ro)
    a, b = h.search(["Reciprocal Rank Fusion constant"]), h.search(["Reciprocal Rank Fusion constant"])
    assert [(f.chunk.chunk_id, f.rrf_score) for f in a.fused] == [(f.chunk.chunk_id, f.rrf_score) for f in b.fused]
    for lst in (a.fused, a.vector_results, a.bm25_results):
        ids = [x.chunk.chunk_id for x in lst]
        assert len(ids) == len(set(ids))
    assert [f.fused_rank for f in a.fused] == list(range(1, len(a.fused) + 1))
    assert all(a.fused[i].rrf_score >= a.fused[i + 1].rrf_score for i in range(len(a.fused) - 1))


def test_hybrid_when_bm25_returns_nothing_vector_still_works(kb_ro):
    res = HybridRetriever(kb_ro).search(["zzzzqqqq xxyyzz"])
    assert res.bm25_results == [] and res.vector_results and res.fused
    assert all(f.bm25_rank is None and f.bm25_score is None and f.vector_rank for f in res.fused)
    top = res.fused[0]
    assert abs(top.rrf_score - 1 / (60 + top.vector_rank)) < 1e-12      # RRF with a single list


def test_empty_query_through_pipeline_makes_no_llm_calls(build):
    pipe, prov, _ = build()
    for q in ("", "   ", None):
        assert pipe.ask(q, pipe.sessions.create()).error
    assert prov.calls == []


# ---------------- C. real cross-encoder ----------------
def test_real_cross_encoder_reorders_and_preserves_metadata(kb_ro, real_reranker):
    q = "How does BM25 score documents using term frequency?"
    cands = HybridRetriever(kb_ro).search([q]).fused
    res = real_reranker.rerank([q], cands, 5)
    assert res.reranked and res.status == "cross_encoder" and len(res.items) == 5
    assert [i.post_rerank_rank for i in res.items] == [1, 2, 3, 4, 5]
    s = [i.rerank_score for i in res.items]
    assert s == sorted(s, reverse=True) and res.items[0].chunk.source == "03_bm25_keyword_retrieval.md"
    for it in res.items:
        orig = next(c for c in cands if c.chunk.chunk_id == it.chunk.chunk_id)
        assert it.pre_rerank_rank == orig.fused_rank and it.rrf_score == orig.rrf_score and it.chunk == orig.chunk


def test_real_cross_encoder_handles_empty_oversized_and_blank_candidates(kb_ro, real_reranker):
    assert real_reranker.rerank(["q"], [], 3).items == []
    cands = HybridRetriever(kb_ro).search(["fusion"]).fused[:3]
    cands[0] = cands[0].model_copy(update={"chunk": cands[0].chunk.model_copy(update={"text": "word " * 3000})})   # >512 tokens
    cands[1] = cands[1].model_copy(update={"chunk": cands[1].chunk.model_copy(update={"text": ""})})
    res = real_reranker.rerank(["fusion"], cands, 3)
    assert res.reranked and len(res.items) == 3 and all(i.rerank_score is not None for i in res.items)


def test_top_k_larger_than_candidates(kb_ro, real_reranker):
    cands = HybridRetriever(kb_ro).search(["fusion"]).fused[:3]
    assert len(real_reranker.rerank(["fusion"], cands, 50).items) == 3


# ---------------- D. grounded generation & citations (full pipeline) ----------------
def test_supported_answer_has_valid_multiple_citations_mapped_to_chunks_and_docs(build):
    pipe, _, _ = build()
    r = pipe.ask("What is the default value of k in Reciprocal Rank Fusion?", pipe.sessions.create())
    retrieved = {x.chunk.chunk_id: x.chunk for x in r.reranked_results}
    assert len(r.citations) == 2
    for c in r.citations:
        assert c.chunk_id in retrieved and c.source == retrieved[c.chunk_id].source and c.text == retrieved[c.chunk_id].text
        assert f"[Source: {c.source}, Chunk: {c.chunk_id}]" in r.answer


def test_citation_to_chunk_that_exists_in_kb_but_was_not_retrieved_is_rejected(build, kb_ro):
    def answer(p):
        supplied = {c for _, c in cites_in(p)}
        outsider = next(c for c in kb_ro.chunk_map() if c not in supplied)
        s, c = cites_in(p)[0]
        return json.dumps({"abstain": False, "answer": f"A [Source: {s}, Chunk: {c}] B [Source: x.md, Chunk: {outsider}]."})
    pipe, _, _ = build(answer=answer)
    r = pipe.ask("What is BM25?", pipe.sessions.create())
    assert len(r.citations) == 1 and len(r.invalid_citations) == 1 and "x.md" not in r.answer


@pytest.mark.parametrize("raw", ["[Source: a.md, Chunk: id1, id2]", "[Source: a.md Chunk: bad]", "[Source: a.md]", "see [Source: ghost_chunk_9]"])
def test_malformed_citation_markup_never_survives_unvalidated(build, raw):
    pipe, _, _ = build(answer=lambda p: json.dumps({"abstain": False, "answer": f"Claim {raw}."}))
    r = pipe.ask("What is BM25?", pipe.sessions.create())
    assert "[Source:" not in r.answer, r.answer           # nothing citation-shaped may reach the user unchecked
    assert r.citations == [] and r.missing_citations


def test_missing_citations_flagged(build):
    pipe, _, _ = build(answer=lambda p: json.dumps({"abstain": False, "answer": "Uncited claim."}))
    r = pipe.ask("What is BM25?", pipe.sessions.create())
    assert r.missing_citations and next(s for s in r.stages if s.name == "Citation validation").status == "failed"


def test_empty_llm_answer_is_an_error_not_a_blank_success(build):
    pipe, _, _ = build(answer=lambda p: "")
    sid = pipe.sessions.create()
    r = pipe.ask("What is BM25?", sid)
    assert r.error and not r.answer.strip()
    assert [m["role"] for m in pipe.sessions.messages(sid)] == ["user"]      # no blank assistant message persisted


def test_malformed_generation_json_shown_as_text_but_flagged(build):
    pipe, _, _ = build(answer=lambda p: "plain prose, no json")
    r = pipe.ask("What is BM25?", pipe.sessions.create())
    assert r.answer == "plain prose, no json" and r.missing_citations and next(s for s in r.stages if s.name == "Answer generation").status == "fallback"


def test_page_references_for_pdf_sources(tmp_path, settings, real_embedder, real_reranker):
    from tests.pdfgen import make_pdf
    (tmp_path / "paper.pdf").write_bytes(make_pdf(["Intro page about apples.", "Page two explains the RRF constant k equals 60."]))
    kb = KnowledgeBase(settings, real_embedder)
    kb.add_file(tmp_path / "paper.pdf")
    p2 = next(c for c in kb.chunks if c.page == 2)
    prov = ScriptedProvider(make_handler(grade=lambda p: json.dumps({"relevant": True, "score": 1, "missing_information": "", "explanation": "ok"}),
                                         answer=lambda p: json.dumps({"abstain": False, "answer": f"k is 60 [Source: wrong.pdf, Chunk: {p2.chunk_id}, Page: 1]."})))
    s = Settings(sessions_dir=tmp_path / "s", index_dir=settings.index_dir, documents_dir=settings.documents_dir, logs_dir=tmp_path / "l", final_top_k=5)
    pipe = RAGPipeline(s, kb, TracedLLM(prov, Tracer(tmp_path / "t.jsonl"), max_retries=0), real_reranker, SessionManager(s.sessions_dir))
    r = pipe.ask("What is the RRF constant k?", pipe.sessions.create())
    assert r.citations[0].page == 2 and r.citations[0].source == "paper.pdf"
    assert "[Source: paper.pdf, Chunk:" in r.answer and "Page: 2]" in r.answer and "wrong.pdf" not in r.answer   # filename and page repaired


# ---------------- E. corrective RAG branches ----------------
def test_E1_sufficient_first_retrieval_no_correction(build):
    pipe, prov, _ = build()
    r = pipe.ask("Who proposed the smoothing constant for rank fusion?", pipe.sessions.create())
    assert not r.retried and len(r.attempts) == 1 and r.context_grade.relevant and len(calls_of(prov, "improve failed retrieval")) == 0 and r.answer


def test_E2_E3_insufficient_then_corrective_query_then_real_recovery(build):
    pipe, prov, _ = build()
    r = pipe.ask(VAGUE, pipe.sessions.create())
    a0, a1 = r.attempts
    assert not any(MARK in i.chunk.text for i in a0.rerank.items) and not a0.grade.relevant            # first retrieval really lacks the evidence
    assert "the RRF paper authors" in calls_of(prov, "improve failed retrieval")[0]["prompt"]            # grader diagnosis fed into refinement
    assert a1.queries == [REFINED] and any(MARK in i.chunk.text for i in a1.rerank.items) and a1.grade.relevant
    assert r.retried and r.retry_queries == [[REFINED]] and not r.abstained and r.citations and r.answer
    assert any(MARK in x.chunk.text for x in r.reranked_results)                                         # final context = second retrieval
    assert {a.attempt for a in r.attempts} == {0, 1} and r.context_grade is a1.grade
    corrective = next(s for s in r.stages if s.name == "Corrective retry")
    assert corrective.status == "success" and REFINED in corrective.detail
    assert len(calls_of(prov, "Evidence rules")) == 1 and "Cormack" in calls_of(prov, "Evidence rules")[0]["prompt"]


def test_E4_second_retrieval_still_insufficient_abstains(build):
    pipe, prov, _ = build(refine=lambda p: json.dumps({"query": "what is a good outdoor activity", "alternatives": [], "reason": "r"}))
    sid = pipe.sessions.create()
    r = pipe.ask(VAGUE, sid)
    assert r.retried and len(r.attempts) == 2 and not r.context_grade.relevant and r.abstained and r.citations == []
    assert len(calls_of(prov, "Evidence rules")) == 0 and len(calls_of(prov, "Retrieval was graded insufficient")) == 1
    last = pipe.sessions.messages(sid)[-1]
    assert last["role"] == "assistant" and last["metadata"]["abstained"] is True and last["sources"] == []   # abstained turn persisted


@pytest.mark.parametrize("bad", ["garbage", '{"query": ""}', '{"query": 5}', "{}"])
def test_E5_corrective_query_generation_failure_abstains_without_second_retrieval(build, bad):
    pipe, prov, _ = build(refine=lambda p: bad)
    r = pipe.ask(VAGUE, pipe.sessions.create())
    assert len(r.attempts) == 1 and not r.retried and r.abstained and r.citations == []
    assert next(s for s in r.stages if s.name == "Corrective retry").status == "skipped"


def test_E5b_corrective_llm_error_nonfatal_abstains(build):
    pipe, _, _ = build(refine=lambda p: LLMError("server", "503"))
    r = pipe.ask(VAGUE, pipe.sessions.create())
    assert r.abstained and len(r.attempts) == 1 and r.error is None


@pytest.mark.parametrize("retries,attempts", [(0, 1), (1, 2), (2, 3)])
def test_E6_retry_limit_respected(build, retries, attempts):
    topics = iter(["gardening hobbies weekend", "mountain hiking trails", "ocean sailing boats", "pottery classes"])
    pipe, prov, _ = build(retries=retries, refine=lambda p: json.dumps({"query": next(topics), "alternatives": [], "reason": "r"}))
    r = pipe.ask(VAGUE, pipe.sessions.create())
    assert len(r.attempts) == attempts and len(calls_of(prov, "improve failed retrieval")) == attempts - 1 and r.abstained


def test_E7_empty_second_retrieval_is_insufficient_without_llm_grading(kb_ro, tmp_path):
    from src.rag.context_grader import ContextGrader
    from src.rag.corrective_rag import CorrectiveRAG
    from src.retrieval.reranker import RerankResult
    prov = ScriptedProvider(make_handler(grade=lambda p: json.dumps({"relevant": False, "score": 0, "missing_information": "m", "explanation": "e"})))
    llm = TracedLLM(prov, Tracer(tmp_path / "t.jsonl"), max_retries=0)
    h = HybridRetriever(kb_ro)
    state = {"n": 0}

    def retrieve(qs):
        state["n"] += 1
        hy = h.search(qs)
        return (hy, RerankResult(items=hy.fused[:3], reranked=True, status="cross_encoder")) if state["n"] == 1 else (hy.model_copy(update={"fused": []}), RerankResult(items=[], reranked=False, status="empty"))
    res = CorrectiveRAG(llm, ContextGrader(llm), 1).run("q", ["fusion"], retrieve)
    assert res.retried and not res.proceed_to_generation and res.final.rerank.items == [] and res.final.grade.explanation == "Retrieval returned nothing."
    assert len(calls_of(prov, "evidence grader")) == 1            # the empty second attempt was NOT sent to the LLM grader


def test_E8_duplicate_first_and_second_results_terminate_cleanly(build):
    pipe, _, _ = build(refine=lambda p: json.dumps({"query": VAGUE, "alternatives": [], "reason": "same"}))
    r = pipe.ask(VAGUE, pipe.sessions.create())
    ids = [[i.chunk.chunk_id for i in a.rerank.items] for a in r.attempts]
    assert len(r.attempts) == 2 and ids[0] == ids[1] and r.abstained


# ---------------- F. abstention ----------------
def test_F_unrelated_question_abstains_with_no_fabricated_citations(build):
    pipe, prov, _ = build(refine=lambda p: json.dumps({"query": "winner of the world cup", "alternatives": [], "reason": "r"}),
                          abstain=lambda p: json.dumps({"abstain": True, "answer": "No info. [Source: made_up.md, Chunk: made_up_001]"}))
    sid = pipe.sessions.create()
    r = pipe.ask("Who won the 2018 FIFA World Cup?", sid)
    assert r.abstained and r.citations == [] and "made_up" not in r.answer and "[Source:" not in r.answer


def test_F_generator_level_abstention_when_grader_said_sufficient(build):
    pipe, _, _ = build(answer=lambda p: json.dumps({"abstain": True, "answer": "Evidence does not cover this."}))
    r = pipe.ask("What is BM25?", pipe.sessions.create())
    assert r.abstained and r.citations == [] and not r.missing_citations


def test_F_partially_supported_answer_keeps_cited_part(build):
    pipe, _, _ = build(answer=lambda p: json.dumps({"abstain": False, "answer": "Supported part [Source: %s, Chunk: %s]. The rest is not covered by the sources." % cites_in(p)[0]}))
    r = pipe.ask("What is BM25 and what does it cost?", pipe.sessions.create())
    assert not r.abstained and len(r.citations) == 1 and "not covered" in r.answer


def test_F_inconsistent_grader_output_relevant_false_with_high_score_is_insufficient(build):
    pipe, _, _ = build(grade=lambda p: json.dumps({"relevant": False, "score": 0.95, "missing_information": "m", "explanation": "e"}),
                       retries=0)
    r = pipe.ask("What is BM25?", pipe.sessions.create())
    assert r.abstained and not r.context_grade.relevant


def test_F_empty_knowledge_base_abstention_path_makes_no_llm_call(tmp_path, settings, real_embedder, real_reranker):
    kb = KnowledgeBase(settings, real_embedder)
    prov = ScriptedProvider()
    pipe = RAGPipeline(settings, kb, TracedLLM(prov, Tracer(tmp_path / "t.jsonl")), real_reranker, SessionManager(settings.sessions_dir))
    r = pipe.ask("anything", pipe.sessions.create())
    assert "empty" in r.error.lower() and prov.calls == []
