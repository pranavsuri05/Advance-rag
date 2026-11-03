"""Provider/error-handling QA: Groq via mocked HTTP, Gemini via a fake SDK client. No network, no real keys."""
import inspect
import json
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from src.llm.gemini_client import GeminiProvider, _classify
from src.llm.groq_client import GroqProvider
from src.llm.provider import LLMError, LLMProvider, TracedLLM
from tests.conftest import ScriptedProvider

GKEY = "gsk_" + "Qq1Ww2Ee3Rr4Tt5Yy6Uu7Ii8Oo9Pp0"
AKEY = "AIza" + "SyQ1w2e3r4t5y6u7i8o9p0aSdFgHjKlZxCvB"


def err_body(msg="boom", typ="invalid_request_error"):
    return {"error": {"message": msg, "type": typ, "code": typ}}


def groq(handler):
    return GroqProvider(GKEY, "m", transport=httpx.MockTransport(handler))


def gemini_with(exc=None, response=None):
    p = GeminiProvider("test-key-not-real", "gemini-test")
    calls = {"n": 0}

    def gen(**kw):
        calls["n"] += 1
        if exc is not None:
            raise exc
        return response
    p._client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))
    p.calls = calls
    return p


def sdk_error(code, status, msg="m"):
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "message": msg, "status": status}})


# ----- interface conformance -----
def test_both_providers_implement_the_same_interface():
    for cls in (GroqProvider, GeminiProvider):
        assert issubclass(cls, LLMProvider)
        assert inspect.signature(cls.complete) == inspect.signature(LLMProvider.complete)
    for p in (groq(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]})),
              gemini_with(response=SimpleNamespace(text="x", usage_metadata=None, candidates=[], sdk_http_response=None))):
        raw = p.complete("hi", system="s", json_mode=True, temperature=0.0, max_output_tokens=10)
        assert raw.text == "x" and hasattr(raw, "input_tokens") and hasattr(raw, "output_tokens") and p.model


# ----- Groq HTTP status matrix -----
@pytest.mark.parametrize("status,kind", [(400, "bad_request"), (401, "auth"), (403, "auth"), (404, "model_not_found"), (429, "quota"),
                                         (500, "server"), (502, "server"), (503, "server"), (504, "server")])
def test_groq_status_matrix(status, kind):
    with pytest.raises(LLMError) as e:
        groq(lambda r: httpx.Response(status, json=err_body())).complete("p")
    assert e.value.kind == kind and e.value.diagnostics["http_status"] == status
    assert e.value.user_message() and "Traceback" not in e.value.user_message()


def test_groq_non_json_error_body_and_malformed_200():
    with pytest.raises(LLMError) as e:                      # Cloudflare-style HTML 502
        groq(lambda r: httpx.Response(502, text="<html>Bad gateway</html>")).complete("p")
    assert e.value.kind == "server"
    for body in (b"not json", json.dumps({"choices": []}).encode(), json.dumps({"choices": [{"message": {"content": None}}]}).encode(), b"{}"):
        with pytest.raises(LLMError) as e2:
            groq(lambda r, b=body: httpx.Response(200, content=b)).complete("p")
        assert e2.value.kind == "bad_request"


def test_groq_partial_usage_is_marked_estimated(tracer):
    body = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7}}
    TracedLLM(groq(lambda r: httpx.Response(200, json=body)), tracer, max_retries=0).generate("p", "query_rewrite")
    rec = tracer.read()[0]
    assert rec["tokens_estimated"] is True and rec["status"] == "success"


# ----- Gemini SDK error matrix -----
@pytest.mark.parametrize("code,status,kind", [(400, "INVALID_ARGUMENT", "bad_request"), (401, "UNAUTHENTICATED", "auth"), (403, "PERMISSION_DENIED", "auth"),
                                              (404, "NOT_FOUND", "model_not_found"), (429, "RESOURCE_EXHAUSTED", "quota"), (500, "INTERNAL", "server"),
                                              (502, "BAD_GATEWAY", "server"), (503, "UNAVAILABLE", "server"), (504, "DEADLINE_EXCEEDED", "server")])
def test_gemini_status_matrix(code, status, kind):
    p = gemini_with(exc=sdk_error(code, status))
    with pytest.raises(LLMError) as e:
        p.complete("p")
    assert e.value.kind == kind and e.value.diagnostics["http_status"] == code and e.value.diagnostics["status"] == status


def test_gemini_invalid_api_key_400_is_auth():
    with pytest.raises(LLMError) as e:
        gemini_with(exc=sdk_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.")).complete("p")
    assert e.value.kind == "auth"


@pytest.mark.parametrize("exc", [httpx.ReadTimeout(""), httpx.ConnectTimeout("t"), httpx.ConnectError("All connection attempts failed"),
                                 httpx.RemoteProtocolError(""), TimeoutError(), ConnectionResetError("reset")])
def test_gemini_network_exceptions_are_classified_as_network_not_unknown(exc):
    with pytest.raises(LLMError) as e:
        gemini_with(exc=exc).complete("p")
    assert e.value.kind == "network" and e.value.transient


def test_gemini_missing_usage_and_empty_response_handling(tracer):
    ok = SimpleNamespace(text="fine", usage_metadata=None, candidates=[], sdk_http_response=None)
    TracedLLM(gemini_with(response=ok), tracer, max_retries=0).generate("p", "query_rewrite")
    assert tracer.read()[0]["tokens_estimated"] is True
    for empty in (SimpleNamespace(text="", usage_metadata=None, candidates=[], sdk_http_response=None),
                  SimpleNamespace(text=None, usage_metadata=None, candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="SAFETY"))], sdk_http_response=None)):
        with pytest.raises(LLMError) as e:
            gemini_with(response=empty).complete("p")
        assert e.value.kind == "bad_request" and "empty response" in e.value.message


# ----- retry policy (both providers, via TracedLLM) -----
@pytest.mark.parametrize("make", ["groq", "gemini"])
@pytest.mark.parametrize("kind_status,expected_calls", [(("auth", 401), 1), (("quota", 429), 1), (("model_not_found", 404), 1), (("bad_request", 400), 1),
                                                         (("server", 503), 3), (("server", 504), 3)])
def test_retry_policy_bounded_and_never_for_fatal_kinds(tracer, make, kind_status, expected_calls):
    kind, status = kind_status
    if make == "groq":
        n = {"c": 0}

        def h(r):
            n["c"] += 1
            return httpx.Response(status, json=err_body())
        prov, count = groq(h), lambda: n["c"]
    else:
        prov = gemini_with(exc=sdk_error(status, "X"))
        count = lambda: prov.calls["n"]
    with pytest.raises(LLMError) as e:
        TracedLLM(prov, tracer, max_retries=2, backoff_base_s=0).generate("p", "context_grading")
    assert e.value.kind == kind and count() == expected_calls
    assert len(tracer.read()) == 1 and tracer.read()[0]["attempts"] == expected_calls     # one trace record per logical call


@pytest.mark.parametrize("retries", [0, 1, 4])
def test_retry_limit_is_exact_and_retry_disabled_means_one_request(tracer, retries):
    p = ScriptedProvider(lambda *_: LLMError("server", "503"))
    with pytest.raises(LLMError):
        TracedLLM(p, tracer, max_retries=retries, backoff_base_s=0).generate("p", "query_rewrite")
    assert len(p.calls) == 1 + retries


def test_no_retry_loop_when_provider_succeeds_after_one_transient(tracer):
    seq = iter([LLMError("network", "x"), "ok"])
    p = ScriptedProvider(lambda *_: next(seq))
    assert TracedLLM(p, tracer, max_retries=3, backoff_base_s=0).generate("p", "query_rewrite").text == "ok" and len(p.calls) == 2


# ----- redaction -----
def test_keys_redacted_in_errors_prompts_responses_and_diagnostics(tracer, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", GKEY)
    p = groq(lambda r: httpx.Response(401, json=err_body(f"Invalid API Key {GKEY}")))
    with pytest.raises(LLMError):
        TracedLLM(p, tracer, max_retries=0).generate(f"user pasted {AKEY} into the question", "query_rewrite")
    raw = tracer.path.read_text()
    assert GKEY not in raw and AKEY not in raw and raw.count("REDACTED") >= 2
    exc = sdk_error(401, "UNAUTHENTICATED", f"bad key {AKEY}")
    with pytest.raises(LLMError):
        TracedLLM(gemini_with(exc=exc), tracer, max_retries=0).generate("p", "query_rewrite")
    assert AKEY not in tracer.path.read_text()


def test_classify_is_defensive_for_unknown_exception_shapes():
    assert _classify(ValueError("weird")).kind == "unknown"
    assert _classify(SimpleNamespace(code="x", __str__=lambda s: "s")).kind in {"unknown"}
