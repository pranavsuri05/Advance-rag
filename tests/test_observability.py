import json

import pytest

from src.llm.provider import LLMError, TracedLLM
from src.observability.metrics import summarize
from tests.conftest import ScriptedProvider


def test_every_call_traced_with_usage_and_latency(tracer):
    llm = TracedLLM(ScriptedProvider(lambda *_: "ok"), tracer)
    r = llm.generate("prompt text", "answer_generation", system="sys", session_id="s1")
    rec = tracer.read()[0]
    assert rec["call_id"] == r.call_id and rec["call_type"] == "answer_generation" and rec["session_id"] == "s1"
    assert rec["input_tokens"] == 100 and rec["output_tokens"] == 20 and rec["total_tokens"] == 120
    assert rec["tokens_estimated"] is False and rec["latency_ms"] >= 0 and rec["status"] == "success"
    assert "prompt text" in rec["prompt"] and rec["response"] == "ok" and rec["model"] == "mock-model"


def test_missing_usage_is_labelled_estimated(tracer):
    r = TracedLLM(ScriptedProvider(lambda *_: "some output text", usage=False), tracer).generate("p" * 40, "query_rewrite")
    rec = tracer.read()[0]
    assert rec["tokens_estimated"] is True and r.tokens_estimated and rec["input_tokens"] > 0


def test_failure_logged(tracer):
    llm = TracedLLM(ScriptedProvider(lambda *_: LLMError("quota", "429 exhausted")), tracer, max_retries=2, backoff_base_s=0)
    with pytest.raises(LLMError):
        llm.generate("p", "context_grading")
    rec = tracer.read()[0]
    assert rec["status"] == "failure" and "quota" in rec["error"] and rec["attempts"] == 1  # quota is not retried


def test_transient_errors_retried_with_bound(tracer):
    p = ScriptedProvider(lambda *_: LLMError("server", "503"))
    llm = TracedLLM(p, tracer, max_retries=2, backoff_base_s=0)
    with pytest.raises(LLMError):
        llm.generate("p", "context_grading")
    assert len(p.calls) == 3 and len(tracer.read()) == 1


def test_transient_then_success(tracer):
    seq = iter([LLMError("network", "x"), "fine"])
    llm = TracedLLM(ScriptedProvider(lambda *_: next(seq)), tracer, max_retries=2, backoff_base_s=0)
    assert llm.generate("p", "query_rewrite").text == "fine"
    assert tracer.read()[0]["attempts"] == 2


def test_api_keys_never_logged(tracer, monkeypatch):
    secret = "AIza" + "SyA1234567890abcdefghijklmnopqrstuv"
    monkeypatch.setenv("GEMINI_API_KEY", "my-super-secret-key-123")
    TracedLLM(ScriptedProvider(lambda *_: f"echo {secret} my-super-secret-key-123"), tracer).generate(f"use {secret}", "query_rewrite")
    raw = tracer.path.read_text()
    assert secret not in raw and "my-super-secret-key-123" not in raw and "REDACTED" in raw


def test_metrics_summary(tracer):
    llm = TracedLLM(ScriptedProvider(lambda *_: "ok"), tracer)
    llm.generate("a", "query_rewrite"); llm.generate("b", "answer_generation")
    s = summarize(tracer.read())
    assert s["total_calls"] == 2 and s["input_tokens"] == 200 and s["calls_by_type"]["query_rewrite"] == 1 and s["failure_rate"] == 0
    assert summarize([])["total_calls"] == 0


def test_jsonl_lines_are_valid_json(tracer):
    TracedLLM(ScriptedProvider(), tracer).generate("a", "query_rewrite")
    for line in tracer.path.read_text().splitlines():
        json.loads(line)
