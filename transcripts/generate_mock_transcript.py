"""Generates transcripts/illustrative_mock_run.md with a MOCK LLM (no API key needed).

Retrieval (local embeddings, FAISS, BM25, RRF, cross-encoder) is REAL; only the Gemini responses are scripted.
Run with a real key via the app / evaluation runner to get a genuine Gemini transcript.
"""
import json, re, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bootstrap import build_services
from src.config import Settings
from src.llm.provider import LLMProvider, RawCompletion, TracedLLM
from src.observability.tracer import Tracer
from src.rag.pipeline import RAGPipeline

STATE = {"grade_calls": 0}


class Mock(LLMProvider):
    model = "MOCK (not Gemini)"

    def complete(self, prompt, system=None, json_mode=False, temperature=0.2):
        s, q = system or "", prompt
        if "query-understanding" in s:
            latest = prompt.split("Latest user question:\n")[-1].strip()
            if "and why is that needed" in latest.lower():
                return RawCompletion(json.dumps({"rewritten_query": "Why is cross-encoder reranking needed after hybrid retrieval?", "needs_decomposition": False, "reason": "Resolved follow-up using history"}))
            multi = "together" in latest
            return RawCompletion(json.dumps({"rewritten_query": latest, "needs_decomposition": multi, "reason": "multi-part" if multi else "already clear"}))
        if "split a multi-part" in s:
            return RawCompletion(json.dumps({"subqueries": ["How does hybrid retrieval combine vector and keyword search?", "How does cross-encoder reranking improve relevance?", "What is the difference between fusion and reranking?"], "reason": "three information needs"}))
        if "evidence grader" in s:
            if "Pinecone" in q or "World Cup" in q:
                return RawCompletion(json.dumps({"relevant": False, "score": 0.05, "missing_information": "Pricing / unrelated topic is not covered", "explanation": "Passages discuss retrieval concepts, nothing about the question."}))
            if "how do i make the first pass better" in q.lower() and STATE["grade_calls"] == 0:
                STATE["grade_calls"] += 1
                return RawCompletion(json.dumps({"relevant": False, "score": 0.2, "missing_information": "Specific mention of query rewriting", "explanation": "Vague wording retrieved general passages."}))
            return RawCompletion(json.dumps({"relevant": True, "score": 0.9, "missing_information": "", "explanation": "Passages state the answer."}))
        if "improve failed retrieval" in s:
            return RawCompletion(json.dumps({"query": "query rewriting and decomposition before retrieval", "alternatives": [], "reason": "target missing topic"}))
        if "Retrieval was graded insufficient" in s:
            return RawCompletion(json.dumps({"abstain": True, "answer": "The knowledge base does not contain enough information to answer this: it covers RAG techniques, not this topic."}))
        cites = re.findall(r"Source: (\S+) \| Chunk: (\S+)", q)[:2]
        tag = " ".join(f"[Source: {a}, Chunk: {b}]" for a, b in cites)
        return RawCompletion(json.dumps({"abstain": False, "answer": f"(Mock answer text) The sources state the relevant facts. {tag}"}))


svc = build_services()
if not svc.kb.ready:
    svc.kb.rebuild()
tmp = Path(tempfile.mkdtemp())
tracer = Tracer(tmp / "calls.jsonl")
llm = TracedLLM(Mock(), tracer, max_retries=0)
pipe = RAGPipeline(svc.settings, svc.kb, llm, svc.reranker, svc.sessions)
sid = svc.sessions.create()
questions = [
    ("1. Normal factual question", "What is the default value of k in Reciprocal Rank Fusion?"),
    ("2. Multi-part question with decomposition / hybrid retrieval", "Explain how hybrid retrieval and reranking work together, and why both are needed."),
    ("3. Corrective retrieval triggered", "How do I make the first pass better?"),
    ("4. Unanswerable question (abstention)", "What is the monthly price of a managed Pinecone subscription?"),
    ("5. Follow-up using conversation memory", "and why is that needed?"),
]
out = ["# Sample transcript (ILLUSTRATIVE - MOCK LLM)", "",
       "> **This is NOT a real Gemini run.** No `GEMINI_API_KEY` was available when this file was generated. "
       "Embeddings, FAISS, BM25, RRF and the cross-encoder are the **real local components** over the bundled corpus; "
       "every LLM response (rewrite, grading, answer) is scripted by `transcripts/generate_mock_transcript.py`, "
       "so answer text is placeholder and says so. Re-generate with a real key by chatting in the app.", ""]
for title, q in questions:
    r = pipe.ask(q, sid)
    out += [f"## {title}", f"**User:** {q}", ""]
    out += [f"- Rewritten query: `{r.rewritten_query}`", f"- Subqueries: {r.decomposed_queries or 'none'}"]
    out += [f"- Vector hits: {len(r.vector_results)} · BM25 hits: {len(r.bm25_results)} · RRF candidates: {len(r.fused_results)} · reranker: {r.rerank_status}"]
    for a in r.attempts:
        out.append(f"- Attempt {a.attempt}: queries={a.queries} → grade {'sufficient' if a.grade.relevant else 'insufficient'} ({a.grade.score:.2f}); top chunks: {[x.chunk.chunk_id for x in a.rerank.items[:3]]}")
    out += [f"- Corrective retry occurred: **{r.retried}**", f"- Outcome: **{'abstained' if r.abstained else 'answered'}**", "", "Stages:"]
    out += [f"  - {s.name}: {s.status} - {s.detail}" for s in r.stages]
    out += ["", f"**Assistant (mock text):** {r.answer}", ""]
    out += ["Citations: " + (", ".join(f"`{c.source}` / `{c.chunk_id}`" for c in r.citations) or "none"), ""]
Path(__file__).with_name("illustrative_mock_run.md").write_text("\n".join(out))
print("written")
