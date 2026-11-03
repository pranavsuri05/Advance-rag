import json

from src.llm.provider import TracedLLM
from src.rag.query_transformer import QueryTransformer
from tests.conftest import ScriptedProvider


def run(tracer, handler, q, history=None):
    p = ScriptedProvider(handler)
    return QueryTransformer(TracedLLM(p, tracer, max_retries=0)).transform(q, history), p


def test_simple_query_unchanged(tracer):
    out, p = run(tracer, lambda *_: json.dumps({"rewritten_query": "What is BM25?", "needs_decomposition": False, "reason": "clear"}), "What is BM25?")
    assert out.rewritten_query == "What is BM25?" and not out.rewrite_changed and out.subqueries == []
    assert len(p.calls) == 1 and out.retrieval_queries == ["What is BM25?"]


def test_rewritten_query(tracer):
    out, _ = run(tracer, lambda *_: json.dumps({"rewritten_query": "How does RRF handle ties?", "needs_decomposition": False, "reason": "resolved pronoun"}),
                 "how does it handle ties?", [{"role": "user", "content": "What is RRF?"}])
    assert out.rewrite_changed and out.rewritten_query == "How does RRF handle ties?"


def test_conversation_history_sent_to_llm(tracer):
    _, p = run(tracer, lambda *_: "{}", "and why?", [{"role": "user", "content": "Tell me about RRF"}])
    assert "Tell me about RRF" in p.calls[0]["prompt"]


def test_multi_part_decomposition(tracer):
    def h(prompt, system):
        if "query-understanding" in system:
            return json.dumps({"rewritten_query": "Explain hybrid retrieval and reranking and why both", "needs_decomposition": True, "reason": "multi"})
        return json.dumps({"subqueries": ["How does hybrid retrieval work?", "How does reranking work?", "Fusion vs reranking?"], "reason": "split"})
    out, p = run(tracer, h, "Explain hybrid retrieval and reranking and why both")
    assert len(out.subqueries) == 3 and len(p.calls) == 2 and len(out.call_ids) == 2
    assert out.retrieval_queries[0] == out.rewritten_query and len(out.retrieval_queries) == 4


def test_invalid_llm_response_falls_back(tracer):
    out, _ = run(tracer, lambda *_: "this is not json", "What is FAISS?")
    assert out.fell_back and out.rewritten_query == "What is FAISS?" and out.subqueries == []


def test_json_in_code_fence_is_parsed(tracer):
    out, _ = run(tracer, lambda *_: '```json\n{"rewritten_query": "X?", "needs_decomposition": false, "reason": "r"}\n```', "x")
    assert out.rewritten_query == "X?"
