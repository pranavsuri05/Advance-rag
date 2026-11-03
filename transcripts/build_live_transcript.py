"""Builds transcripts/sample_run.md ONLY from genuine, already-recorded live LLM interactions.

Inputs (written by real runs, never by mocks): logs/llm_calls.jsonl (every provider call: prompt, response, tokens,
latency, request ids) and data/sessions/<id>.json (persisted turns). A session is included only if every one of its
logged calls succeeded on a live provider. Nothing here calls any API.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.text_utils import extract_json  # noqa: E402

calls = [json.loads(l) for l in (ROOT / "logs" / "llm_calls.jsonl").read_text().splitlines() if l.strip()]
by_session = defaultdict(list)
for c in calls:
    if c.get("session_id"):
        by_session[c["session_id"]].append(c)


def call_table(rows):
    out = ["| # | Call type | Model | Tokens in/out | Latency | Request ID |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        rid = ((r.get("diagnostics") or {}).get("headers") or {}).get("x-request-id", "-")
        tok = f"{r['input_tokens']}/{r['output_tokens']}" + (" (est.)" if r["tokens_estimated"] else " (provider)")
        out.append(f"| {i} | {r['call_type']} | {r['model']} | {tok} | {r['latency_ms']} ms | `{rid}` |")
    return out


def turn_from_session(msg_user, msg_asst):
    m = msg_asst["metadata"]
    lines = [f"**User:** {msg_user['content']}", "", "Pipeline stages (recorded):"]
    lines += [f"- {s['name']}: **{s['status']}** - {s['detail']}" for s in m.get("stages", [])]
    lines += ["", f"- Rewritten query: `{m.get('rewritten_query')}`", f"- Subqueries: {m.get('subqueries') or 'none'}"]
    g = m.get("context_grade")
    if g:
        lines.append(f"- Context grade: relevant={g['relevant']}, score={g['score']}; explanation: {g['explanation']}")
        if g.get("missing_information"):
            lines.append(f"- Missing information (grader): {g['missing_information']}")
    lines.append(f"- Corrective retry occurred: **{m.get('retried')}**")
    lines.append("- Reranked chunks: " + ", ".join(f"`{c['chunk_id']}` (CE {c['rerank_score']:.2f})" for c in m.get("reranked", [])))
    lines += ["", f"**Assistant (live, model output):** {msg_asst['content']}", ""]
    lines.append("Citations: " + (", ".join(f"`{s['source']}` / `{s['chunk_id']}`" for s in msg_asst["sources"]) or "none (abstained)"))
    lines.append(f"Outcome: **{'abstained' if m.get('abstained') else 'answered'}**")
    return lines


out = ["# Sample transcript - recorded LIVE runs", "",
       "Every response below was produced by a real LLM call and recorded in `logs/llm_calls.jsonl` / `data/sessions/`. "
       "No mocked output appears in this file. Provider: **Groq**, model **`qwen/qwen3.8-27b`**. Retrieval "
       "(MiniLM embeddings, FAISS, BM25, RRF, cross-encoder) ran locally over the 15-document corpus. Retries were "
       "disabled and request counts capped during these runs.", "",
       "**Coverage honesty:** only what actually ran live is shown. The follow-up turn is *partial* (request cap reached "
       "before the final message) and a successful corrective retry, PDF page citations and an invalid-citation case were "
       "never observed live. See `transcripts/illustrative_mock_run.md` for a clearly labelled mock illustration.", ""]
n = 0
for sid, rows in by_session.items():
    live = [r for r in rows if r["model"].startswith("qwen")]
    if not live or any(r["status"] != "success" for r in rows):
        continue
    sess_file = ROOT / "data" / "sessions" / f"{sid}.json"
    msgs = json.loads(sess_file.read_text())["messages"] if sess_file.exists() else []
    n += 1
    out += [f"## Session {n}: `{sid}`", ""]
    pairs = [(msgs[i], msgs[i + 1]) for i in range(0, len(msgs) - 1, 2)]
    for j, (u, a) in enumerate(pairs, 1):
        out += [f"### Turn {j} (full pipeline, persisted)", ""] + turn_from_session(u, a) + [""]
    # calls not covered by a persisted turn => partial turn reconstructed from the raw call records
    consumed = sum(len(p[1]["metadata"].get("llm_call_ids", [])) for p in pairs)
    rest = rows[consumed:]
    if rest:
        out += ["### Turn (PARTIAL - run stopped by the request cap before the final message)", ""]
        for r in rest:
            d = extract_json(r["response"]) or {}
            if r["call_type"] == "query_rewrite":
                hist = r["prompt"].split("Latest user question:")[-1].strip()
                out += [f"- **User question:** {hist}", f"- Rewrite (uses conversation history): `{d.get('rewritten_query')}` - {d.get('reason')}"]
            elif r["call_type"] == "context_grading":
                out.append(f"- Context grade: relevant={d.get('relevant')}, score={d.get('score')}; missing: {d.get('missing_information')}")
            elif r["call_type"] == "corrective_query_generation":
                out.append(f"- Corrective query generated: `{d.get('query')}`")
        out += ["- Final answer/abstention message: **not generated** (request cap reached); outcome not recorded.", ""]
    out += ["LLM calls for this session:", ""] + call_table(rows) + [""]
(ROOT / "transcripts" / "sample_run.md").write_text("\n".join(out))
print("sessions included:", n)
