"""Technique 4: grounded generation with citations, citation validation and abstention."""
from __future__ import annotations

from pydantic import BaseModel

from src.llm.provider import LLMError, TracedLLM
from src.rag import prompts
from src.rag.context_grader import format_context
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.utils.citations import validate_citations
from src.utils.text_utils import extract_json


class CitationInfo(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    text: str


class GenerationResult(BaseModel):
    answer: str
    abstained: bool
    citations: list[CitationInfo]
    invalid_citations: list[str] = []
    corrected_citations: int = 0
    missing_citations: bool = False    # answered (not abstained) but no valid citation survived
    parse_ok: bool = True
    call_id: str | None = None


def _history(history: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history) or "(none)"


class Generator:
    def __init__(self, llm: TracedLLM):
        self.llm = llm

    def generate(self, question: str, items: list[RetrievedChunk], history: list[dict] | None = None,
                 session_id: str | None = None) -> GenerationResult:
        res = self.llm.generate(
            prompts.ANSWER_USER.format(history=_history(history or []), context=format_context(items), question=question),
            "answer_generation", system=prompts.ANSWER_SYSTEM, json_mode=True, temperature=0.1, session_id=session_id)
        data = extract_json(res.text)
        parse_ok = bool(data and isinstance(data.get("answer"), str))
        answer = data["answer"] if parse_ok else res.text.strip()
        abstain = bool(data.get("abstain")) if parse_ok else False
        index = {it.chunk.chunk_id: it for it in items}
        report = validate_citations(answer, {cid: {"source": it.chunk.source, "page": it.chunk.page} for cid, it in index.items()})
        cites = [CitationInfo(chunk_id=c.chunk_id, source=c.source, page=c.page, text=index[c.chunk_id].chunk.text)
                 for c in report.valid]
        return GenerationResult(answer=report.text.strip(), abstained=abstain, citations=cites,
                                invalid_citations=report.invalid, corrected_citations=report.corrected,
                                missing_citations=(not abstain and not cites), parse_ok=parse_ok, call_id=res.call_id)

    def abstain(self, question: str, missing: str, explanation: str, session_id: str | None = None) -> GenerationResult:
        """LLM-written abstention message based on grader notes (no evidence passages are supplied)."""
        try:
            res = self.llm.generate(prompts.ABSTAIN_USER.format(question=question, missing=missing or "unspecified",
                                                                explanation=explanation or "unspecified"),
                                    "answer_generation", system=prompts.ABSTAIN_SYSTEM, json_mode=True,
                                    temperature=0.1, session_id=session_id)
        except LLMError as exc:
            if exc.fatal:
                raise
            return GenerationResult(answer="The knowledge base does not contain enough information to answer this "
                                           "question. (Fallback notice: the explanation model call failed.)",
                                    abstained=True, citations=[], parse_ok=False)
        data = extract_json(res.text)
        ok = bool(data and isinstance(data.get("answer"), str))
        text = report_text = data["answer"] if ok else res.text.strip()
        # An abstention must not carry citations; strip any that slipped in.
        report = validate_citations(report_text, {})
        return GenerationResult(answer=report.text.strip(), abstained=True, citations=[], parse_ok=ok, call_id=res.call_id)
