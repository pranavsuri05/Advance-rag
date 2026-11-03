"""Mocked tests for the Groq provider (httpx.MockTransport). No network, no API key."""
import json

import httpx
import pytest

from src.config import Settings
from src.llm.factory import build_llm
from src.llm.groq_client import GroqProvider
from src.llm.provider import LLMError, TracedLLM
from src.rag.query_transformer import QueryTransformer

KEY = "gsk_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"


def ok_body(text='{"rewritten_query": "What is RRF?", "needs_decomposition": false, "reason": "clear"}', finish="stop"):
    return {"id": "chatcmpl-1", "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 9, "total_tokens": 51}}


def provider(handler, **kw):
    return GroqProvider(KEY, "llama-3.3-70b-versatile", transport=httpx.MockTransport(handler), **kw)


def test_request_shape_and_usage_parsing():
    seen = {}

    def h(req: httpx.Request):
        seen["body"], seen["auth"], seen["url"] = json.loads(req.content), req.headers["authorization"], str(req.url)
        return httpx.Response(200, json=ok_body(), headers={"x-request-id": "req_123", "x-ratelimit-remaining-tokens": "5000",
                                                            "set-cookie": "nope"})
    raw = provider(h).complete("hello", system="sys", json_mode=True, temperature=0.0, max_output_tokens=256)
    b = seen["body"]
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions" and seen["auth"] == f"Bearer {KEY}"
    assert b["model"] == "llama-3.3-70b-versatile" and b["max_completion_tokens"] == 256 and b["temperature"] == 0.0
    assert b["response_format"] == {"type": "json_object"}
    assert b["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    assert (raw.input_tokens, raw.output_tokens) == (42, 9)
    assert raw.diagnostics["finish_reason"] == "stop" and raw.diagnostics["headers"] == {"x-request-id": "req_123", "x-ratelimit-remaining-tokens": "5000"}


def test_no_json_mode_or_cap_when_not_requested():
    seen = {}
    provider(lambda r: (seen.update(b=json.loads(r.content)), httpx.Response(200, json=ok_body("hi")))[1]).complete("p")
    assert "response_format" not in seen["b"] and "max_completion_tokens" not in seen["b"]


@pytest.mark.parametrize("status,kind", [(401, "auth"), (403, "auth"), (404, "model_not_found"), (429, "quota"), (503, "server"), (400, "bad_request")])
def test_http_errors_classified(status, kind):
    h = lambda r: httpx.Response(status, json={"error": {"message": "boom", "type": "t", "code": "c"}}, headers={"retry-after": "12"})
    with pytest.raises(LLMError) as e:
        provider(h).complete("p")
    assert e.value.kind == kind and e.value.diagnostics["http_status"] == status
    assert e.value.diagnostics["headers"]["retry-after"] == "12"
    assert KEY not in str(e.value) and KEY not in json.dumps(e.value.diagnostics)


def test_decommissioned_model_is_model_not_found():
    h = lambda r: httpx.Response(400, json={"error": {"message": "The model `x` has been decommissioned", "type": "invalid_request_error"}})
    with pytest.raises(LLMError) as e:
        provider(h).complete("p")
    assert e.value.kind == "model_not_found"


def test_timeout_and_connect_errors_are_network():
    def boom(r): raise httpx.ConnectTimeout("t")
    with pytest.raises(LLMError) as e:
        provider(boom).complete("p")
    assert e.value.kind == "network" and e.value.transient


def test_empty_response_length_hint():
    h = lambda r: httpx.Response(200, json=ok_body("", finish="length"))
    with pytest.raises(LLMError, match="LLM_MAX_OUTPUT_TOKENS"):
        provider(h).complete("p")


def test_missing_key_and_model():
    with pytest.raises(LLMError) as e:
        GroqProvider("", "m")
    assert e.value.kind == "missing_key"
    with pytest.raises(LLMError):
        GroqProvider("k", "")


def test_traced_logging_records_real_usage_model_latency_and_scrubs_key(tracer):
    llm = TracedLLM(provider(lambda r: httpx.Response(200, json=ok_body(), headers={"x-request-id": "rq9"})), tracer, max_retries=0)
    llm.generate("my question", "query_rewrite", system="s", json_mode=True)
    rec = tracer.read()[0]
    assert rec["model"] == "llama-3.3-70b-versatile" and rec["input_tokens"] == 42 and rec["output_tokens"] == 9
    assert rec["total_tokens"] == 51 and rec["tokens_estimated"] is False and rec["latency_ms"] >= 0 and rec["status"] == "success"
    assert rec["diagnostics"]["headers"]["x-request-id"] == "rq9"
    assert KEY not in tracer.path.read_text()


def test_429_logged_not_retried(tracer):
    n = {"c": 0}

    def h(r):
        n["c"] += 1
        return httpx.Response(429, json={"error": {"message": f"rate limit {KEY}", "type": "tokens"}}, headers={"retry-after": "7"})
    llm = TracedLLM(provider(h), tracer, max_retries=2, backoff_base_s=0)
    with pytest.raises(LLMError) as e:
        llm.generate("p", "context_grading")
    assert n["c"] == 1 and e.value.kind == "quota" and "quota" in e.value.user_message().lower()
    rec = tracer.read()[0]
    assert rec["status"] == "failure" and rec["diagnostics"]["headers"]["retry-after"] == "7" and KEY not in tracer.path.read_text()


def test_503_retried_with_bound(tracer):
    n = {"c": 0}

    def h(r):
        n["c"] += 1
        return httpx.Response(503, json={"error": {"message": "x"}})
    with pytest.raises(LLMError):
        TracedLLM(provider(h), tracer, max_retries=2, backoff_base_s=0).generate("p", "query_rewrite")
    assert n["c"] == 3 and len(tracer.read()) == 1


def test_missing_usage_estimated(tracer):
    body = ok_body(); body.pop("usage")
    TracedLLM(provider(lambda r: httpx.Response(200, json=body)), tracer).generate("p" * 40, "query_rewrite")
    assert tracer.read()[0]["tokens_estimated"] is True


def test_query_transformer_works_through_groq_provider(tracer):
    llm = TracedLLM(provider(lambda r: httpx.Response(200, json=ok_body())), tracer, max_retries=0)
    out = QueryTransformer(llm).transform("what is rrf?")
    assert out.rewritten_query == "What is RRF?" and not out.fell_back and len(tracer.read()) == 1


def test_factory_selection(tmp_path, tracer):
    base = dict(documents_dir=tmp_path, index_dir=tmp_path, sessions_dir=tmp_path, logs_dir=tmp_path)
    g = build_llm(Settings(llm_provider="groq", groq_api_key=KEY, groq_model="llama-3.3-70b-versatile", gemini_api_key="", **base), tracer)
    assert isinstance(g.provider, GroqProvider) and g.model == "llama-3.3-70b-versatile"
    assert build_llm(Settings(llm_provider="groq", groq_api_key="", gemini_api_key="gem-key-123456789", **base), tracer) is None
    ge = build_llm(Settings(llm_provider="gemini", groq_api_key=KEY, gemini_api_key="gem-key-123456789", gemini_model="gemini-x", **base), tracer)
    assert ge.model == "gemini-x" and type(ge.provider).__name__ == "GeminiProvider"
    with pytest.raises(LLMError):
        build_llm(Settings(llm_provider="nope", **base), tracer)


def test_settings_active_model_and_key_flag(tmp_path):
    s = Settings(llm_provider="groq", groq_api_key="", gemini_api_key="g" * 20, groq_model="m1", gemini_model="m2")
    assert not s.has_api_key and s.active_model == "m1"
    s.llm_provider = "gemini"
    assert s.has_api_key and s.active_model == "m2"


def test_env_var_scrubbed_from_traces(tracer, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "custom-groq-secret-value-xyz")
    assert "custom-groq-secret-value-xyz" not in tracer.scrub("leak custom-groq-secret-value-xyz here")
