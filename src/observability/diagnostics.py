"""Safe diagnostic metadata extraction for Google API calls (no keys, no prompt text)."""
from __future__ import annotations

import re
from typing import Any

# Response headers worth keeping for support tickets / capacity debugging. Everything else is dropped.
_ALLOWED = re.compile(r"^(x-goog-(?!api-key)[a-z0-9\-]+|x-groq-[a-z0-9\-]+|cf-ray|x-request-id|request-id|retry-after|date|server-timing|"
                      r"x-envoy-upstream-service-time|x-ratelimit-[a-z\-]+)$", re.I)


def safe_headers(headers: Any) -> dict[str, str]:
    """Allow-listed response headers only; credentials and cookies are never copied."""
    if not headers:
        return {}
    try:
        items = list(headers.items())
    except AttributeError:
        return {}
    return {str(k).lower(): str(v)[:200] for k, v in items if _ALLOWED.match(str(k))}


def error_diagnostics(exc: Exception) -> dict:
    """Structured, size-bounded details from a google-genai APIError (status, message, error details, headers)."""
    diag: dict = {}
    code, status = getattr(exc, "code", None), getattr(exc, "status", None)
    if code is not None:
        diag["http_status"] = code
    if status:
        diag["status"] = str(status)
    msg = getattr(exc, "message", None)
    if msg:
        diag["message"] = str(msg)[:300]
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        err = details.get("error", details)
        extra = err.get("details") if isinstance(err, dict) else None
        if extra:
            diag["error_details"] = extra if len(str(extra)) < 1500 else str(extra)[:1500]
    resp = getattr(exc, "response", None)
    headers = safe_headers(getattr(resp, "headers", None))
    if headers:
        diag["headers"] = headers
    diag["exception_type"] = exc.__class__.__name__
    return diag


def response_diagnostics(resp: Any) -> dict:
    """Diagnostics for a successful response: request-id style headers, finish reason, thinking tokens."""
    diag: dict = {}
    http = getattr(resp, "sdk_http_response", None)
    headers = safe_headers(getattr(http, "headers", None))
    if headers:
        diag["headers"] = headers
    cands = getattr(resp, "candidates", None) or []
    if cands and getattr(cands[0], "finish_reason", None) is not None:
        fr = cands[0].finish_reason
        diag["finish_reason"] = getattr(fr, "name", str(fr))
    usage = getattr(resp, "usage_metadata", None)
    thoughts = getattr(usage, "thoughts_token_count", None) if usage else None
    if thoughts is not None:
        diag["thoughts_tokens"] = thoughts
    return diag
