"""Technique 1: query transformation (history-aware rewriting + decomposition)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.llm.provider import LLMError, TracedLLM
from src.rag import prompts
from src.utils.text_utils import extract_json


class TransformedQuery(BaseModel):
    original_query: str
    rewritten_query: str
    subqueries: list[str] = Field(default_factory=list)
    reason: str = ""
    rewrite_changed: bool = False
    fell_back: bool = False
    call_ids: list[str] = Field(default_factory=list)
    error: str | None = None

    @property
    def retrieval_queries(self) -> list[str]:
        """Queries sent to the retriever: the (re)written query plus any subqueries."""
        return list(dict.fromkeys([self.rewritten_query, *self.subqueries]))


def _fmt_history(history: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history) or "(none)"


class QueryTransformer:
    def __init__(self, llm: TracedLLM):
        self.llm = llm

    def transform(self, question: str, history: list[dict] | None = None, session_id: str | None = None) -> TransformedQuery:
        result = TransformedQuery(original_query=question, rewritten_query=question)
        try:
            res = self.llm.generate(
                prompts.QUERY_REWRITE_USER.format(history=_fmt_history(history or []), question=question),
                "query_rewrite", system=prompts.QUERY_REWRITE_SYSTEM, json_mode=True, temperature=0.0, session_id=session_id)
        except LLMError as exc:
            if exc.fatal:
                raise
            return result.model_copy(update={"fell_back": True, "error": exc.message, "reason": "LLM unavailable; using original query"})
        result.call_ids.append(res.call_id)
        data = extract_json(res.text)
        if not data or not isinstance(data.get("rewritten_query"), str) or not data["rewritten_query"].strip():
            return result.model_copy(update={"fell_back": True, "reason": "Malformed rewrite output; using original query"})
        result.rewritten_query = data["rewritten_query"].strip()
        result.rewrite_changed = result.rewritten_query != question.strip()
        result.reason = str(data.get("reason", ""))
        if data.get("needs_decomposition") is True:
            try:
                dres = self.llm.generate(prompts.QUERY_DECOMPOSE_USER.format(question=result.rewritten_query),
                                         "query_decomposition", system=prompts.QUERY_DECOMPOSE_SYSTEM,
                                         json_mode=True, temperature=0.0, session_id=session_id)
                result.call_ids.append(dres.call_id)
                ddata = extract_json(dres.text) or {}
                raw_subs = ddata.get("subqueries")
                subs = [s.strip() for s in raw_subs if isinstance(s, str) and s.strip()] if isinstance(raw_subs, list) else []
                if len(subs) >= 2:
                    result.subqueries = subs[:4]
                    result.reason = (result.reason + " | " + str(ddata.get("reason", ""))).strip(" |")
            except LLMError as exc:
                if exc.fatal:
                    raise
                result.error = exc.message
        return result
