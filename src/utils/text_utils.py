"""Small text helpers shared across modules."""
from __future__ import annotations

import json
import re

_TOKEN_RE = re.compile(r"[^\W_]+")   # unicode letters/digits (ASCII-only would turn "café" into "caf" and drop CJK entirely)
_STOP = {"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are", "was", "were",
         "it", "its", "that", "this", "with", "as", "by", "be", "at", "from"}


def tokenize(text: str) -> list[str]:
    """Lowercase unicode alphanumeric tokens (used by BM25 and scoring)."""
    return _TOKEN_RE.findall(text.lower())


def tokenize_no_stop(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOP]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "doc"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Always labelled 'estimated' wherever it is used."""
    return max(1, len(text) // 4) if text else 0


def extract_json(text: str) -> dict | None:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    for candidate in (cleaned, _first_braces(cleaned)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _first_braces(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else None
