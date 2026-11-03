from __future__ import annotations

import streamlit as st

from src.observability.metrics import summarize
from ui.components import bar_chart, empty_state, evidence, html_table, line_chart, metric_cards


def render(svc) -> None:
    st.title("Observability")
    st.caption("Every LLM call (Groq or Gemini) is traced to logs/llm_calls.jsonl by the central client. Numbers below come straight from those records.")
    records = svc.tracer.read()
    if not records:
        empty_state("No LLM calls recorded yet. Ask a question in Chat to generate traces.")
        return
    m = summarize(records)
    metric_cards([("LLM calls", str(m["total_calls"]), f"{m['successful']} ok · {m['failed']} failed"),
                  ("Failure rate", f"{m['failure_rate']:.0%}", ""),
                  ("Input tokens", f"{m['input_tokens']:,}", ""), ("Output tokens", f"{m['output_tokens']:,}", ""),
                  ("Total tokens", f"{m['total_tokens']:,}", f"{m['estimated_token_calls']} calls estimated"),
                  ("Avg latency", f"{m['avg_latency_ms']:,.0f} ms", "successful calls")])
    c1, c2 = st.columns(2)
    with c1:
        bar_chart("Calls by type", dict(sorted(m["calls_by_type"].items(), key=lambda kv: -kv[1])))
    with c2:
        by_type: dict[str, int] = {}
        for r in records:
            by_type[r["call_type"]] = by_type.get(r["call_type"], 0) + r.get("total_tokens", 0)
        bar_chart("Tokens by call type", by_type)
    c3, c4 = st.columns(2)
    recent = records[-60:]
    with c3:
        line_chart("Latency per call (last 60)", [r["latency_ms"] for r in recent], " ms")
    with c4:
        line_chart("Tokens per call (last 60)", [r.get("total_tokens", 0) for r in recent])
    st.subheader("Recent calls")
    rows = list(reversed(records[-50:]))
    html_table(["Time (UTC)", "Type", "Status", "In", "Out", "Latency", "Tokens"],
               [[r["timestamp"][11:19], r["call_type"], r["status"], r["input_tokens"], r["output_tokens"], f"{r['latency_ms']} ms",
                 "estimated" if r.get("tokens_estimated") else "provider"] for r in rows])
    st.write("")
    for r in rows[:15]:
        with st.expander(f"{r['timestamp'][11:19]} · {r['call_type']} · {r['status']} · {r['latency_ms']} ms"):
            st.caption(f"call_id {r['call_id']} · model {r['model']} · session {r.get('session_id')} · attempts {r.get('attempts', 1)}")
            if r.get("error"):
                st.error(r["error"])
            if r.get("diagnostics"):
                st.markdown("**Diagnostics** (request ids, headers, finish reason)")
                st.json(r["diagnostics"])
            st.markdown("**Prompt**"); evidence(r["prompt"])
            st.markdown("**Response**"); evidence(r["response"] or "(empty)")
