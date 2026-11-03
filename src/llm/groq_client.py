"""Groq provider (OpenAI-compatible chat completions API) implemented with plain httpx.

Docs: https://console.groq.com/docs/api-reference#chat  -  base URL https://api.groq.com/openai/v1
"""
from __future__ import annotations

import httpx

from src.llm.provider import LLMError, LLMProvider, RawCompletion
from src.observability.diagnostics import safe_headers

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, timeout_s: int = 60, base_url: str = GROQ_BASE_URL,
                 transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise LLMError("missing_key", "GROQ_API_KEY is not set")
        if not model:
            raise LLMError("bad_request", "GROQ_MODEL is not set")
        self.model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        # No transport-level retries: TracedLLM is the only retry layer, so every HTTP request is accounted for.
        self._client = httpx.Client(timeout=timeout_s, transport=transport or httpx.HTTPTransport(retries=0),
                                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 temperature: float = 0.2, max_output_tokens: int | None = None) -> RawCompletion:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_output_tokens:
            body["max_completion_tokens"] = max_output_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}   # prompts already instruct "Output ONLY JSON"
        try:
            resp = self._client.post(self._url, json=body)
        except httpx.TimeoutException as exc:
            raise LLMError("network", f"Request timed out: {exc.__class__.__name__}",
                           diagnostics={"exception_type": exc.__class__.__name__}) from exc
        except httpx.HTTPError as exc:
            raise LLMError("network", f"Network error: {exc.__class__.__name__}",
                           diagnostics={"exception_type": exc.__class__.__name__}) from exc
        diag = {"http_status": resp.status_code, "headers": safe_headers(resp.headers)}
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code != 200:
            err = data.get("error", {}) if isinstance(data, dict) else {}
            msg = str(err.get("message") or resp.text[:200])[:300]
            diag.update({"status": err.get("type"), "code": err.get("code"), "message": msg})
            raise LLMError(_kind(resp.status_code, msg), f"{resp.status_code} {err.get('type') or ''}: {msg}".strip(),
                           diagnostics=diag)
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        diag["finish_reason"] = choice.get("finish_reason")
        diag["response_id"] = data.get("id")
        if not text:
            hint = (" The output limit was reached before any text was produced: raise "
                    "LLM_MAX_OUTPUT_TOKENS(_SHORT)." if choice.get("finish_reason") == "length" else "")
            raise LLMError("bad_request", f"Groq returned an empty response (finish_reason={choice.get('finish_reason')}).{hint}",
                           diagnostics=diag)
        return RawCompletion(text=text, input_tokens=usage.get("prompt_tokens"),
                             output_tokens=usage.get("completion_tokens"), diagnostics=diag)


def _kind(status: int, msg: str) -> str:
    low = msg.lower()
    if status in (401, 403):
        return "auth"
    if status == 404 or "decommissioned" in low or "does not exist" in low or "model_not_found" in low:
        return "model_not_found"
    if status == 429:
        return "quota"
    if status >= 500 or status in (408, 409):
        return "server"
    if status == 400:
        return "bad_request"
    return "unknown"
