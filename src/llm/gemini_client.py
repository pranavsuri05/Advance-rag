"""Gemini implementation of LLMProvider using the google-genai SDK."""
from __future__ import annotations

from src.config import Settings
from src.llm.provider import LLMError, LLMProvider, RawCompletion, TracedLLM
from src.observability.diagnostics import error_diagnostics, response_diagnostics
from src.observability.tracer import Tracer

THINKING_LEVELS = ("minimal", "low", "medium", "high")


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, timeout_s: int = 60,
                 thinking_level: str | None = None, thinking_budget: int | None = None):
        if not api_key:
            raise LLMError("missing_key", "GEMINI_API_KEY is not set")
        if thinking_level and thinking_budget is not None:
            raise LLMError("bad_request", "Set only one of GEMINI_THINKING_LEVEL or GEMINI_THINKING_BUDGET, not both.")
        if thinking_level and thinking_level not in THINKING_LEVELS:
            raise LLMError("bad_request", f"GEMINI_THINKING_LEVEL must be one of {', '.join(THINKING_LEVELS)}.")
        from google import genai
        from google.genai import types

        self.model = model
        self.thinking_level, self.thinking_budget = thinking_level, thinking_budget
        self._types = types
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=timeout_s * 1000,
            # The SDK retries 408/429/5xx up to 5x on its own. Disable that so TracedLLM is the ONLY retry layer
            # and every HTTP request is accounted for (important on free-tier quotas).
            retry_options=types.HttpRetryOptions(attempts=1)))

    def _thinking_config(self):
        t = self._types
        if self.thinking_level:
            return t.ThinkingConfig(thinking_level=t.ThinkingLevel[self.thinking_level.upper()])
        if self.thinking_budget is not None:
            return t.ThinkingConfig(thinking_budget=self.thinking_budget)
        return None

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 temperature: float = 0.2, max_output_tokens: int | None = None) -> RawCompletion:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system, temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
            max_output_tokens=max_output_tokens, thinking_config=self._thinking_config(),
        )
        try:
            resp = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
            err = _classify(exc)
            err.diagnostics = error_diagnostics(exc)
            raise err from exc
        text = resp.text or ""
        diag = response_diagnostics(resp)
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", None) if usage else None
        out_tok = getattr(usage, "candidates_token_count", None) if usage else None
        if usage and out_tok is not None:
            out_tok += getattr(usage, "thoughts_token_count", None) or 0  # thinking tokens are billed as output
        if not text:
            reason = diag.get("finish_reason", "unknown")
            hint = (" The output limit was hit before any text was produced (thinking tokens count toward it): raise "
                    "GEMINI_MAX_OUTPUT_TOKENS(_SHORT) or lower GEMINI_THINKING_LEVEL." if reason == "MAX_TOKENS" else "")
            raise LLMError("bad_request", f"Gemini returned an empty response (finish_reason={reason}).{hint}", diagnostics=diag)
        return RawCompletion(text=text, input_tokens=in_tok, output_tokens=out_tok, diagnostics=diag or None)


def _classify(exc: Exception) -> LLMError:
    import httpx

    code = getattr(exc, "code", None)
    msg = str(exc)[:500]
    low = msg.lower()
    if code == 404 or "not_found" in low:
        return LLMError("model_not_found", msg)
    if code == 429 or "resource_exhausted" in low or "quota" in low:
        return LLMError("quota", msg)
    if code in (401, 403) or "api key" in low or "api_key_invalid" in low or "permission" in low:
        return LLMError("auth", msg)
    if isinstance(code, int) and code >= 500:
        return LLMError("server", msg)
    if code is None and (isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError))
                         or any(w in low for w in ("connect", "timeout", "timed out", "network", "dns"))):
        return LLMError("network", msg or exc.__class__.__name__)      # by exception TYPE, not only by message text
    if code == 400:
        return LLMError("auth" if "api key" in low else "bad_request", msg)
    return LLMError("unknown", msg)


def build_llm(settings: Settings, tracer: Tracer) -> TracedLLM | None:
    """Return a traced Gemini client, or None when no API key is configured."""
    if not settings.has_api_key:
        return None
    provider = GeminiProvider(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_s,
                              settings.gemini_thinking_level, settings.gemini_thinking_budget)
    return TracedLLM(provider, tracer, max_retries=settings.llm_max_retries,
                     max_output_tokens=settings.max_output_tokens, max_output_tokens_short=settings.max_output_tokens_short)
