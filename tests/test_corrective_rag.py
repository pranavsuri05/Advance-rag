import json

import pytest

from src.llm.provider import TracedLLM
from src.rag.context_grader import ContextGrader
from src.rag.corrective_rag import CorrectiveRAG
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from tests.conftest import FakeCrossEncoder, ScriptedProvider


def build(kb, tracer, grades, retries=1):
    """grades: list of relevance booleans returned by successive grader calls."""
    state = {"i": 0}

    def h(prompt, system):
        if "evidence grader" in system:
            rel = grades[min(state["i"], len(grades) - 1)]
            state["i"] += 1
            return json.dumps({"relevant": rel, "score": 0.9 if rel else 0.1, "missing_information": "" if rel else "details on X", "explanation": "e"})
        return json.dumps({"query": "better query about fusion", "alternatives": [], "reason": "r"})
    p = ScriptedProvider(h)
    llm = TracedLLM(p, tracer, max_retries=0)
    retr, rer = HybridRetriever(kb), CrossEncoderReranker("f", model=FakeCrossEncoder())

    def fn(qs):
        hy = retr.search(qs)
        return hy, rer.rerank(qs, hy.fused, 3)
    return CorrectiveRAG(llm, ContextGrader(llm), retries), fn, p


def test_relevant_context_means_no_retry(kb, tracer):
    c, fn, p = build(kb, tracer, [True])
    res = c.run("what is rrf", ["rrf constant"], fn)
    assert not res.retried and res.proceed_to_generation and len(res.attempts) == 1 and len(p.calls) == 1


def test_weak_context_triggers_retry_then_succeeds(kb, tracer):
    c, fn, p = build(kb, tracer, [False, True])
    res = c.run("what is rrf", ["rrf constant"], fn)
    assert res.retried and res.proceed_to_generation and len(res.attempts) == 2
    assert res.attempts[1].queries[0] == "better query about fusion"
    assert res.attempts[0].grade.missing_information == "details on X"
    assert [x["system"][:30] for x in p.calls].count(p.calls[0]["system"][:30]) >= 2


def test_failed_retry_abstains(kb, tracer):
    c, fn, _ = build(kb, tracer, [False, False])
    res = c.run("q", ["rrf"], fn)
    assert res.retried and not res.proceed_to_generation and len(res.attempts) == 2


def test_max_retries_respected(kb, tracer):
    for retries in (0, 1, 2):
        c, fn, p = build(kb, tracer, [False] * 10, retries)
        res = c.run("q", ["rrf"], fn)
        assert len(res.attempts) == retries + 1
        assert sum(1 for x in p.calls if "evidence grader" in x["system"]) == retries + 1


def test_malformed_grader_output_is_flagged_not_trusted(kb, tracer):
    p = ScriptedProvider(lambda *_: "garbage")
    g = ContextGrader(TracedLLM(p, tracer, max_retries=0))
    hy = HybridRetriever(kb).search(["rrf"])
    grade = g.grade("q", hy.fused[:2])
    assert grade.graded is False and grade.error


def test_no_context_is_insufficient_without_llm_call(kb, tracer):
    p = ScriptedProvider()
    grade = ContextGrader(TracedLLM(p, tracer)).grade("q", [])
    assert not grade.relevant and p.calls == []


@pytest.mark.parametrize("alts", ['"alternatives": "abc"', '"alternatives": 5', '"alternatives": {"a": 1}'])
def test_non_list_alternatives_are_ignored_not_split_into_characters(kb, tracer, alts):
    """Regression: a string 'alternatives' used to be iterated character by character into 1-char queries."""
    p = ScriptedProvider(lambda prompt, system: '{"query": "better query", ' + alts + ', "reason": "r"}')
    qs, _, err = CorrectiveRAG(TracedLLM(p, tracer, max_retries=0), None, 1)._refine("q", "prev", type("G", (), {"missing_information": "m", "explanation": "e"})(), None)
    assert qs == ["better query"] and err is None
