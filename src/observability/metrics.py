"""Aggregate metrics computed from real trace records (nothing is synthesised)."""
from __future__ import annotations

from collections import Counter


def summarize(records: list[dict]) -> dict:
    total = len(records)
    ok = [r for r in records if r.get("status") == "success"]
    failed = total - len(ok)
    lat = [r["latency_ms"] for r in ok]
    return {
        "total_calls": total,
        "successful": len(ok),
        "failed": failed,
        "failure_rate": (failed / total) if total else 0.0,
        "input_tokens": sum(r.get("input_tokens", 0) for r in records),
        "output_tokens": sum(r.get("output_tokens", 0) for r in records),
        "total_tokens": sum(r.get("total_tokens", 0) for r in records),
        "estimated_token_calls": sum(1 for r in records if r.get("tokens_estimated")),
        "avg_latency_ms": (sum(lat) / len(lat)) if lat else 0.0,
        "calls_by_type": dict(Counter(r.get("call_type", "?") for r in records)),
    }
