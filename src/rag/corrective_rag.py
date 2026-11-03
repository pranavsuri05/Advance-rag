"""Technique 5: corrective retrieval loop (retrieve -> rerank -> grade -> refine query -> retry)."""
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from src.llm.provider import LLMError, TracedLLM
from src.rag import prompts
from src.rag.context_grader import ContextGrade, ContextGrader
from src.retrieval.hybrid_retriever import HybridResult
from src.retrieval.reranker import RerankResult
from src.utils.text_utils import extract_json

# retrieve_fn(queries) -> (hybrid_result, rerank_result)
RetrieveFn = Callable[[list[str]], tuple[HybridResult, RerankResult]]


class Attempt(BaseModel):
    attempt: int                # 0 = initial retrieval, 1.. = corrective retries
    queries: list[str]
    hybrid: HybridResult
    rerank: RerankResult
    grade: ContextGrade


class CorrectiveResult(BaseModel):
    attempts: list[Attempt]
    final: Attempt
    retried: bool
    proceed_to_generation: bool    # False => abstain
    refine_call_ids: list[str] = []
    refine_error: str | None = None

    @property
    def retry_queries(self) -> list[list[str]]:
        return [a.queries for a in self.attempts[1:]]


class CorrectiveRAG:
    def __init__(self, llm: TracedLLM, grader: ContextGrader, max_retries: int = 1):
        self.llm, self.grader, self.max_retries = llm, grader, max_retries

    def run(self, question: str, queries: list[str], retrieve_fn: RetrieveFn, session_id: str | None = None) -> CorrectiveResult:
        attempts: list[Attempt] = []
        refine_ids: list[str] = []
        refine_error = None
        current = queries
        for n in range(self.max_retries + 1):
            hybrid, rerank = retrieve_fn(current)
            grade = self.grader.grade(question, rerank.items, session_id)
            attempts.append(Attempt(attempt=n, queries=current, hybrid=hybrid, rerank=rerank, grade=grade))
            if grade.relevant:          # includes ungraded fallback (graded=False): never loops without a verdict
                break
            if n == self.max_retries:
                break
            new_queries, call_id, err = self._refine(question, current[0], grade, session_id)
            if call_id:
                refine_ids.append(call_id)
            if not new_queries:
                refine_error = err or "No refined query produced"
                break
            current = new_queries
        final = attempts[-1]
        return CorrectiveResult(attempts=attempts, final=final, retried=len(attempts) > 1,
                                proceed_to_generation=final.grade.relevant, refine_call_ids=refine_ids,
                                refine_error=refine_error)

    def _refine(self, question: str, previous: str, grade: ContextGrade, session_id: str | None):
        try:
            res = self.llm.generate(
                prompts.CORRECTIVE_QUERY_USER.format(question=question, previous_query=previous,
                                                     missing=grade.missing_information or "unspecified",
                                                     explanation=grade.explanation or "unspecified"),
                "corrective_query_generation", system=prompts.CORRECTIVE_QUERY_SYSTEM, json_mode=True,
                temperature=0.3, session_id=session_id)
        except LLMError as exc:
            if exc.fatal:
                raise
            return [], None, exc.message
        data = extract_json(res.text) or {}
        q = data.get("query")
        if not isinstance(q, str) or not q.strip():
            return [], res.call_id, "malformed corrective query output"
        raw_alts = data.get("alternatives")
        alts = [a.strip() for a in raw_alts if isinstance(a, str) and a.strip()][:2] if isinstance(raw_alts, list) else []
        return list(dict.fromkeys([q.strip(), *alts])), res.call_id, None
