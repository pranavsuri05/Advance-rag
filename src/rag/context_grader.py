"""Context grading: an LLM judges whether retrieved passages are sufficient to answer."""
from __future__ import annotations

from pydantic import BaseModel

from src.llm.provider import LLMError, TracedLLM
from src.rag import prompts
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.utils.text_utils import extract_json


class ContextGrade(BaseModel):
    relevant: bool
    score: float = 0.0
    missing_information: str = ""
    explanation: str = ""
    graded: bool = True         # False => grader failed/malformed; `relevant` is then NOT a real verdict
    call_id: str | None = None
    error: str | None = None


def format_context(items: list[RetrievedChunk], max_chars: int = 1500) -> str:
    parts = []
    for i, it in enumerate(items, 1):
        c = it.chunk
        page = f" | Page: {c.page}" if c.page else ""
        parts.append(f"[{i}] Source: {c.source} | Chunk: {c.chunk_id}{page}\n{c.text[:max_chars]}")
    return "\n\n".join(parts)


class ContextGrader:
    def __init__(self, llm: TracedLLM):
        self.llm = llm

    def grade(self, question: str, items: list[RetrievedChunk], session_id: str | None = None) -> ContextGrade:
        if not items:
            return ContextGrade(relevant=False, score=0.0, missing_information="No passages were retrieved.",
                                explanation="Retrieval returned nothing.", graded=True)
        try:
            res = self.llm.generate(prompts.CONTEXT_GRADE_USER.format(question=question, context=format_context(items)),
                                    "context_grading", system=prompts.CONTEXT_GRADE_SYSTEM, json_mode=True,
                                    temperature=0.0, session_id=session_id)
        except LLMError as exc:
            if exc.fatal:
                raise
            return ContextGrade(relevant=True, graded=False, error=exc.message,
                                explanation="Grader unavailable; proceeding to generation, which enforces evidence rules.")
        data = extract_json(res.text)
        if not data or not isinstance(data.get("relevant"), bool):
            return ContextGrade(relevant=True, graded=False, call_id=res.call_id, error="malformed grader output",
                                explanation="Grader output could not be parsed; proceeding to generation, which enforces evidence rules.")
        try:
            score = max(0.0, min(1.0, float(data.get("score", 1.0 if data["relevant"] else 0.0))))
        except (TypeError, ValueError):
            score = 1.0 if data["relevant"] else 0.0
        return ContextGrade(relevant=data["relevant"], score=score, call_id=res.call_id,
                            missing_information=str(data.get("missing_information", "") or ""),
                            explanation=str(data.get("explanation", "") or ""))
