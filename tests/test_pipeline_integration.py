import json

from tests.conftest import default_handler


def test_end_to_end_sufficient_context(make_pipeline, tracer):
    pipe, prov = make_pipeline()
    sid = pipe.sessions.create()
    r = pipe.ask("What is the constant k in reciprocal rank fusion?", sid)
    assert r.error is None and not r.abstained
    assert r.original_query and r.rewritten_query
    assert r.vector_results and r.bm25_results and r.fused_results and r.reranked_results
    assert r.rerank_status == "cross_encoder" and r.context_grade.relevant and not r.retried
    valid = {x.chunk.chunk_id for x in r.reranked_results}
    assert r.citations and all(c.chunk_id in valid for c in r.citations)
    names = {s.name: s for s in r.stages}
    assert names["Corrective retry"].status == "skipped" and names["Cross-encoder reranking"].status == "success"
    assert names["Citation validation"].status == "success"
    msgs = pipe.sessions.messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["sources"]
    traced = {x["call_type"] for x in tracer.read()}
    assert {"query_rewrite", "context_grading", "answer_generation"} <= traced
    assert set(r.llm_call_ids) <= {x["call_id"] for x in tracer.read()} and r.timing_ms["total"] >= 0


def test_insufficient_evidence_path_retries_then_abstains(make_pipeline):
    def h(prompt, system):
        if "evidence grader" in system:
            return json.dumps({"relevant": False, "score": 0.1, "missing_information": "pricing", "explanation": "no pricing info"})
        if "improve failed retrieval" in system:
            return json.dumps({"query": "pricing subscription cost", "alternatives": [], "reason": "r"})
        return default_handler(prompt, system)
    pipe, prov = make_pipeline(h)
    sid = pipe.sessions.create()
    r = pipe.ask("How much does Pinecone cost?", sid)
    assert r.retried and len(r.attempts) == 2 and r.retry_queries[0][0] == "pricing subscription cost"
    assert r.abstained and r.citations == [] and r.answer
    types = [x["system"] for x in prov.calls]
    assert not any("Evidence rules" in t for t in types)   # answer generator never saw weak context
    assert pipe.sessions.messages(sid)[1]["metadata"]["abstained"] is True


def test_followup_uses_history(make_pipeline):
    pipe, prov = make_pipeline()
    sid = pipe.sessions.create()
    pipe.ask("What is reciprocal rank fusion?", sid)
    pipe.ask("and what is its constant?", sid)
    rewrite_prompts = [c["prompt"] for c in prov.calls if "query-understanding" in (c["system"] or "")]
    assert "What is reciprocal rank fusion?" in rewrite_prompts[1]
    assert len(pipe.sessions.messages(sid)) == 4


def test_errors_handled(make_pipeline, settings):
    from src.llm.provider import LLMError
    pipe, _ = make_pipeline(lambda *_: LLMError("quota", "429"))
    sid = pipe.sessions.create()
    r = pipe.ask("anything about fusion", sid)
    assert "quota" in r.error.lower() or "429" in r.error
    assert pipe.ask("   ", sid).error
    pipe.llm = None
    assert "API key" in pipe.ask("q", sid).error


def test_empty_kb(make_pipeline):
    pipe, _ = make_pipeline()
    for d in list(pipe.kb.documents):
        pipe.kb.remove_document(d)
    assert "empty" in pipe.ask("anything", pipe.sessions.create()).error.lower()


def test_inspect_does_not_generate_or_remember(make_pipeline):
    pipe, prov = make_pipeline()
    sid = pipe.sessions.create()
    r = pipe.inspect("reciprocal rank fusion", sid)
    assert r.fused_results and r.answer == ""
    assert pipe.sessions.messages(sid) == []
    assert not any("Evidence rules" in (c["system"] or "") for c in prov.calls)
