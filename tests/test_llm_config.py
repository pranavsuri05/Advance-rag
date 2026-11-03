"""Local tests for output caps, thinking config and failure diagnostics. No network, no API key."""
import json
from types import SimpleNamespace

import pytest

from src.llm.gemini_client import GeminiProvider, _classify
from src.llm.provider import LLMError, TracedLLM
from src.observability.diagnostics import error_diagnostics, response_diagnostics, safe_headers
from tests.conftest import ScriptedProvider


def fake_provider(**kw):
    p = GeminiProvider("test-key-not-real", "gemini-test", **kw)
    captured = {}

    class Models:
        def generate_content(self, model, contents, config):
            captured.update(model=model, config=config)
            return SimpleNamespace(text="{}", usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=2, thoughts_token_count=3),
                                   candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
                                   sdk_http_response=SimpleNamespace(headers={"x-goog-request-id": "abc123", "set-cookie": "secret", "x-goog-api-key": "nope"}))
    p._client = SimpleNamespace(models=Models())
    return p, captured


def test_output_caps_by_call_type(tracer):
    prov = ScriptedProvider(lambda *_: "ok")
    llm = TracedLLM(prov, tracer, max_output_tokens=2048, max_output_tokens_short=700)
    llm.generate("p", "context_grading"); llm.generate("p", "answer_generation"); llm.generate("p", "evaluation_judging")
    assert [c["max_output_tokens"] for c in prov.calls] == [700, 2048, 700]


def test_thinking_level_and_cap_reach_sdk_config():
    p, cap = fake_provider(thinking_level="low")
    raw = p.complete("hi", max_output_tokens=300)
    cfg = cap["config"]
    assert cfg.max_output_tokens == 300 and cfg.thinking_config.thinking_level.name == "LOW"
    assert raw.output_tokens == 5 and raw.input_tokens == 5          # thinking tokens billed as output
    assert raw.diagnostics["finish_reason"] == "STOP" and raw.diagnostics["thoughts_tokens"] == 3


def test_thinking_budget_and_default_unset():
    p, cap = fake_provider(thinking_budget=128)
    p.complete("hi")
    assert cap["config"].thinking_config.thinking_budget == 128
    p2, cap2 = fake_provider()
    p2.complete("hi")
    assert cap2["config"].thinking_config is None and cap2["config"].max_output_tokens is None


def test_conflicting_or_invalid_thinking_settings_rejected():
    with pytest.raises(LLMError):
        GeminiProvider("k", "m", thinking_level="low", thinking_budget=10)
    with pytest.raises(LLMError, match="one of"):
        GeminiProvider("k", "m", thinking_level="turbo")


def test_empty_response_max_tokens_has_actionable_error():
    p, _ = fake_provider()
    p._client.models.generate_content = lambda **kw: SimpleNamespace(text=None, usage_metadata=None,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))], sdk_http_response=None)
    with pytest.raises(LLMError, match="GEMINI_MAX_OUTPUT_TOKENS") as e:
        p.complete("hi")
    assert e.value.diagnostics["finish_reason"] == "MAX_TOKENS"


def test_safe_headers_allowlist():
    h = safe_headers({"X-Goog-Request-Id": "r1", "Authorization": "Bearer x", "x-goog-api-key": "k", "Set-Cookie": "c",
                      "Retry-After": "30", "Content-Type": "json"})
    assert h == {"x-goog-request-id": "r1", "retry-after": "30"}
    assert safe_headers(None) == {}


def test_error_diagnostics_and_logging(tracer):
    exc = SimpleNamespace(code=503, status="UNAVAILABLE", message="high demand",
                          details={"error": {"details": [{"reason": "x"}]}},
                          response=SimpleNamespace(headers={"x-goog-request-id": "req-9", "x-goog-api-key": "leak"}))
    d = error_diagnostics(exc)
    assert d["http_status"] == 503 and d["status"] == "UNAVAILABLE" and d["headers"] == {"x-goog-request-id": "req-9"}

    def boom(prompt, system):
        e = LLMError("server", "503 UNAVAILABLE")
        e.diagnostics = d
        return e
    with pytest.raises(LLMError):
        TracedLLM(ScriptedProvider(boom), tracer, max_retries=0).generate("secret prompt text", "query_rewrite")
    rec = tracer.read()[0]
    assert rec["diagnostics"]["headers"]["x-goog-request-id"] == "req-9"
    assert "leak" not in tracer.path.read_text()
    assert "secret prompt text" not in json.dumps(rec["diagnostics"])   # diagnostics never contain prompt text


def test_diagnostics_scrub_api_keys(tracer):
    key = "AIza" + "SyA1234567890abcdefghijklmnopqrstuv"
    tracer.log(call_id="c", model="m", call_type="t", prompt="p", response="r", input_tokens=1, output_tokens=1, tokens_estimated=False,
               latency_ms=1, status="failure", error=None, session_id=None, diagnostics={"message": f"bad {key}"})
    assert key not in tracer.path.read_text()


def test_success_diagnostics_logged(tracer):
    p, _ = fake_provider()
    TracedLLM(p, tracer, max_retries=0).generate("hi", "query_rewrite")
    rec = tracer.read()[0]
    assert rec["diagnostics"]["headers"] == {"x-goog-request-id": "abc123"} and rec["output_tokens"] == 5 and not rec["tokens_estimated"]


def test_404_classified_as_model_not_found():
    e = _classify(SimpleNamespace(code=404, __str__=lambda s: "404"))
    assert e.kind == "model_not_found" and e.fatal and "GEMINI_MODEL" in e.user_message()
