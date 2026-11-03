"""JSONL tracer for every LLM call. Secrets are scrubbed before anything is written."""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

_KEY_RE = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|gsk_[0-9A-Za-z]{20,})")


class Tracer:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def scrub(text: str | None) -> str | None:
        if text is None:
            return None
        text = _KEY_RE.sub("[REDACTED_API_KEY]", text)
        for var in ("GEMINI_API_KEY", "GROQ_API_KEY"):
            key = os.getenv(var, "")
            if len(key) > 8:
                text = text.replace(key, "[REDACTED_API_KEY]")
        return text

    def log(self, *, call_id: str, model: str, call_type: str, prompt: str, response: str,
            input_tokens: int, output_tokens: int, tokens_estimated: bool, latency_ms: int,
            status: str, error: str | None, session_id: str | None, attempts: int = 1,
            diagnostics: dict | None = None) -> dict:
        record = {
            "call_id": call_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "call_type": call_type,
            "session_id": session_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "tokens_estimated": tokens_estimated,
            "latency_ms": latency_ms,
            "status": status,
            "attempts": attempts,
            "error": self.scrub(error),
            "diagnostics": json.loads(self.scrub(json.dumps(diagnostics))) if diagnostics else None,
            "prompt": self.scrub(prompt),
            "response": self.scrub(response),
        }
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    _DEFAULTS = {"call_id": "", "timestamp": "", "model": "?", "call_type": "?", "status": "unknown", "attempts": 1,
                 "tokens_estimated": False, "error": None, "diagnostics": None, "prompt": "", "response": "", "session_id": None}

    @classmethod
    def _normalize(cls, rec: dict) -> dict:
        """Fill missing fields and coerce numerics so one incomplete log line cannot break metrics or the UI."""
        out = {**cls._DEFAULTS, **rec}
        for key in ("input_tokens", "output_tokens", "latency_ms", "attempts"):
            try:
                out[key] = int(out.get(key, 0) or 0)
            except (TypeError, ValueError):
                out[key] = 0
        try:
            out["total_tokens"] = int(rec.get("total_tokens", out["input_tokens"] + out["output_tokens"]))
        except (TypeError, ValueError):
            out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
        for key in ("timestamp", "model", "call_type", "status", "prompt", "response"):
            out[key] = str(out[key]) if out[key] is not None else ""
        return out

    def read(self) -> list[dict]:
        """All valid JSON-object records (malformed, partial or non-object lines are skipped)."""
        if not self.path.exists():
            return []
        out = []
        with self.path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a partially written line
                if isinstance(obj, dict):
                    out.append(self._normalize(obj))
        return out
