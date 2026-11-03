"""LLM abstraction: a minimal provider interface plus a traced wrapper.

Every LLM call in the app goes through `TracedLLM.generate`, which gives us uniform
retries, error classification, token accounting and JSONL tracing.
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.observability.tracer import Tracer
from src.utils.text_utils import estimate_tokens

CALL_TYPES = ("query_rewrite", "query_decomposition", "context_grading",
              "corrective_query_generation", "answer_generation", "evaluation_judging")


class LLMError(Exception):
    """Provider failure with a coarse `kind` so the UI can show a helpful message."""

    def __init__(self, kind: str, message: str, call_id: str | None = None, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}
        self.kind = kind  # missing_key | auth | quota | model_not_found | network | server | bad_request | unknown
        self.message = message
        self.call_id = call_id

    @property
    def transient(self) -> bool:
        return self.kind in {"network", "server"}

    @property
    def fatal(self) -> bool:
        """Errors retrying or falling back cannot fix; the pipeline surfaces them to the user."""
        return self.kind in {"quota", "auth", "missing_key", "model_not_found"}

    def user_message(self) -> str:
        return {
            "missing_key": "No API key configured for the selected LLM provider. Set GEMINI_API_KEY (LLM_PROVIDER=gemini) or GROQ_API_KEY (LLM_PROVIDER=groq) in your .env file and restart.",
            "auth": "The LLM API key was rejected. Check that the key for the selected provider is valid and has access to the model.",
            "quota": "LLM quota / rate limit reached (HTTP 429). Wait a minute and retry, or use a key with more quota.",
            "model_not_found": "The configured model is not available to this API key. Set GEMINI_MODEL / GROQ_MODEL in .env to a model your account can use and restart.",
            "network": "Could not reach the LLM API. Check your network connection and retry.",
            "server": "The LLM service returned a temporary error. Please retry shortly.",
        }.get(self.kind, f"The LLM call failed: {self.message}")


@dataclass
class RawCompletion:
    text: str
    input_tokens: int | None = None   # None => provider did not report usage
    output_tokens: int | None = None
    diagnostics: dict | None = None   # request ids, finish reason, thinking tokens (never prompt text)


@dataclass
class LLMResult:
    text: str
    call_id: str
    call_type: str
    input_tokens: int
    output_tokens: int
    tokens_estimated: bool
    latency_ms: int


class LLMProvider(ABC):
    model: str

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 temperature: float = 0.2, max_output_tokens: int | None = None) -> RawCompletion:
        """Single provider call. Raise LLMError on failure."""


class TracedLLM:
    """Wraps a provider with bounded retries, token accounting and JSONL tracing."""

    SHORT_CALLS = frozenset({"query_rewrite", "query_decomposition", "context_grading",
                             "corrective_query_generation", "evaluation_judging"})

    def __init__(self, provider: LLMProvider, tracer: Tracer, max_retries: int = 2, backoff_base_s: float = 1.0,
                 max_output_tokens: int | None = None, max_output_tokens_short: int | None = None):
        self.max_output_tokens, self.max_output_tokens_short = max_output_tokens, max_output_tokens_short
        self.provider = provider
        self.tracer = tracer
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s

    @property
    def model(self) -> str:
        return self.provider.model

    def generate(self, prompt: str, call_type: str, system: str | None = None, json_mode: bool = False,
                 temperature: float = 0.2, session_id: str | None = None) -> LLMResult:
        call_id = str(uuid.uuid4())
        full_prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}" if system else prompt
        start = time.perf_counter()
        attempts = 0
        while True:
            attempts += 1
            try:
                cap = self.max_output_tokens_short if call_type in self.SHORT_CALLS else self.max_output_tokens
                raw = self.provider.complete(prompt, system=system, json_mode=json_mode, temperature=temperature,
                                             max_output_tokens=cap)
                break
            except LLMError as exc:
                if exc.transient and attempts <= self.max_retries:
                    time.sleep(min(self.backoff_base_s * 2 ** (attempts - 1), 8))
                    continue
                latency = int((time.perf_counter() - start) * 1000)
                self.tracer.log(call_id=call_id, model=self.model, call_type=call_type, prompt=full_prompt,
                                response="", input_tokens=estimate_tokens(full_prompt), output_tokens=0,
                                tokens_estimated=True, latency_ms=latency, status="failure",
                                error=f"{exc.kind}: {exc.message}", session_id=session_id, attempts=attempts,
                                diagnostics=exc.diagnostics)
                exc.call_id = call_id
                raise
        latency = int((time.perf_counter() - start) * 1000)
        estimated = raw.input_tokens is None or raw.output_tokens is None
        in_tok = raw.input_tokens if raw.input_tokens is not None else estimate_tokens(full_prompt)
        out_tok = raw.output_tokens if raw.output_tokens is not None else estimate_tokens(raw.text)
        self.tracer.log(call_id=call_id, model=self.model, call_type=call_type, prompt=full_prompt,
                        response=raw.text, input_tokens=in_tok, output_tokens=out_tok, tokens_estimated=estimated,
                        latency_ms=latency, status="success", error=None, session_id=session_id, attempts=attempts,
                        diagnostics=raw.diagnostics)
        return LLMResult(raw.text, call_id, call_type, in_tok, out_tok, estimated, latency)
