"""Selects the LLM provider from configuration (LLM_PROVIDER = gemini | groq)."""
from __future__ import annotations

from src.config import Settings
from src.llm.gemini_client import build_llm as build_gemini
from src.llm.groq_client import GroqProvider
from src.llm.provider import LLMError, TracedLLM
from src.observability.tracer import Tracer

PROVIDERS = ("gemini", "groq")


def build_llm(settings: Settings, tracer: Tracer) -> TracedLLM | None:
    """Return a traced client for the selected provider, or None when its API key is missing."""
    provider = settings.llm_provider
    if provider not in PROVIDERS:
        raise LLMError("bad_request", f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)} (got '{provider}').")
    if not settings.has_api_key:
        return None
    if provider == "gemini":
        return build_gemini(settings, tracer)
    return TracedLLM(GroqProvider(settings.groq_api_key, settings.groq_model, settings.llm_timeout_s), tracer,
                     max_retries=settings.llm_max_retries, max_output_tokens=settings.max_output_tokens,
                     max_output_tokens_short=settings.max_output_tokens_short)
