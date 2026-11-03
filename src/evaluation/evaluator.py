"""Runs the gold dataset through the real pipeline and aggregates metrics."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from src.evaluation import scoring as sc
from src.llm.provider import LLMError, TracedLLM
from src.rag import prompts
from src.rag.context_grader import format_context
from src.rag.pipeline import RAGPipeline
from src.utils.text_utils import extract_json


class GoldItem(BaseModel):
    id: str
    category: str = "factual"
    question: str
    reference_answer: str = ""
    expected_documents: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    answerable: bool = True
    should_abstain: bool = False
    history: list[dict] = Field(default_factory=list)   # prior turns for follow-up questions


def load_dataset(path: Path) -> list[GoldItem]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list) or not all(isinstance(r, dict) for r in raw):
        raise ValueError("Dataset must be a JSON list of question objects.")
    return [GoldItem(**r) for r in raw]


def validate_dataset(items: list[GoldItem], available_docs: set[str] | None = None) -> list[str]:
    """Human-readable problems in a gold dataset (empty list = valid). `available_docs` = filenames in the corpus."""
    problems, seen = [], set()
    for it in items:
        if it.id in seen:
            problems.append(f"{it.id}: duplicate id")
        seen.add(it.id)
        if not it.question.strip():
            problems.append(f"{it.id}: empty question")
        if it.answerable and not it.expected_documents:
            problems.append(f"{it.id}: answerable but has no expected_documents")
        if not it.answerable and it.expected_documents:
            problems.append(f"{it.id}: marked unanswerable but lists expected_documents")
        if it.answerable and not it.reference_answer.strip():
            problems.append(f"{it.id}: answerable but has no reference_answer")
        for d in it.expected_documents:
            if available_docs is not None and d not in available_docs:
                problems.append(f"{it.id}: expected document '{d}' is not in the corpus")
    return problems


class Evaluator:
    def __init__(self, pipeline: RAGPipeline, llm: TracedLLM | None, use_judge: bool = True):
        self.pipeline, self.llm, self.use_judge = pipeline, llm, use_judge

    def run(self, items: list[GoldItem], progress: Callable[[int, int, str], None] | None = None) -> dict:
        results = []
        for i, item in enumerate(items):
            if progress: progress(i, len(items), item.question)
            results.append(self.evaluate_item(item))
        if progress: progress(len(items), len(items), "done")
        return {"run_id": uuid.uuid4().hex[:8], "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": self.llm.model if self.llm else None, "k": self.pipeline.s.final_top_k,
                "judge_enabled": self.use_judge, "aggregate": aggregate(results), "results": results}

    def evaluate_item(self, item: GoldItem) -> dict:
        k = self.pipeline.s.final_top_k
        sid = f"eval-{uuid.uuid4().hex[:8]}"
        t = time.perf_counter()
        r = self.pipeline.ask(item.question, sid, remember=False, history=item.history or [])
        elapsed = int((time.perf_counter() - t) * 1000)
        ranked_docs = [x.chunk.source for x in r.reranked_results]
        rel = set(item.expected_documents)
        retrieved_ids = {x.chunk.chunk_id for x in r.reranked_results}
        cited = [c.chunk_id for c in r.citations]
        res = {"id": item.id, "category": item.category, "question": item.question,
               "reference_answer": item.reference_answer, "answer": r.answer, "error": r.error,
               "answerable": item.answerable, "abstained": r.abstained, "expected_documents": item.expected_documents,
               "retrieved": [{"chunk_id": x.chunk.chunk_id, "source": x.chunk.source, "rerank_score": x.rerank_score,
                              "text": x.chunk.text} for x in r.reranked_results],
               "rewritten_query": r.rewritten_query, "subqueries": r.decomposed_queries, "retried": r.retried,
               "stages": [s.model_dump() for s in r.stages], "llm_call_ids": r.llm_call_ids,
               "citations": cited, "invalid_citations": r.invalid_citations, "latency_ms": elapsed,
               "abstention_correct": None if r.error else sc.abstention_correct(r.abstained, item.answerable)}
        if r.error:
            res["metrics"] = {}
            return res
        m: dict = {"abstention_correct": float(res["abstention_correct"])}
        if item.answerable:
            f1 = sc.token_f1(r.answer, item.reference_answer)
            kw = sc.keyword_recall(r.answer, item.expected_keywords)
            m.update({"exact_match": sc.exact_match(r.answer, item.reference_answer), "token_f1": f1,
                      "keyword_recall": kw, "heuristic_correct": float(sc.heuristic_correct(f1, kw) and not r.abstained),
                      "precision_at_k": sc.precision_at_k(ranked_docs, rel, k),
                      "recall_at_k": sc.recall_at_k(ranked_docs, rel, k),
                      "hit_rate_at_k": sc.hit_rate_at_k(ranked_docs, rel, k),
                      "mrr": sc.reciprocal_rank(ranked_docs, rel)})
        m["citation_validity"] = sc.citation_validity(cited, retrieved_ids) if cited or not r.invalid_citations else 0.0
        if r.invalid_citations:   # citations the model emitted that were removed count as invalid
            total = len(cited) + len(r.invalid_citations)
            m["citation_validity"] = len(cited) / total
        res["metrics"] = m
        res["judge"] = self._judge(item, r) if self.use_judge and self.llm else {"available": False, "reason": "judge disabled"}
        return res

    def _judge(self, item: GoldItem, r) -> dict:
        try:
            res = self.llm.generate(prompts.JUDGE_USER.format(question=item.question, reference=item.reference_answer or "(none; question is unanswerable)",
                                                              answer=r.answer, context=format_context(r.reranked_results, 1200)),
                                    "evaluation_judging", system=prompts.JUDGE_SYSTEM, json_mode=True, temperature=0.0)
        except LLMError as exc:
            return {"available": False, "reason": f"{exc.kind}: {exc.message[:150]}"}
        d = extract_json(res.text)
        try:
            return {"available": True, "correctness": float(d["correctness"]), "faithfulness": float(d["faithfulness"]),
                    "unsupported_claims": d.get("unsupported_claims", []), "explanation": d.get("explanation", ""),
                    "call_id": res.call_id}
        except (TypeError, KeyError, ValueError):
            return {"available": False, "reason": "judge returned malformed output"}


def aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("error")]
    ans = [r for r in ok if r["answerable"]]
    unans = [r for r in ok if not r["answerable"]]
    judged = [r for r in ok if r.get("judge", {}).get("available")]

    def avg(rows, key):
        return sc.mean([r["metrics"].get(key) for r in rows])

    return {
        "total_questions": len(results), "evaluated": len(ok), "errors": len(results) - len(ok),
        "heuristic_correctness": avg(ans, "heuristic_correct"), "token_f1": avg(ans, "token_f1"),
        "exact_match": avg(ans, "exact_match"), "keyword_recall": avg(ans, "keyword_recall"),
        "precision_at_k": avg(ans, "precision_at_k"), "recall_at_k": avg(ans, "recall_at_k"),
        "hit_rate_at_k": avg(ans, "hit_rate_at_k"), "mrr": avg(ans, "mrr"),
        "citation_validity": avg(ok, "citation_validity"),
        "abstention_accuracy_unanswerable": avg(unans, "abstention_correct"),
        "false_abstention_rate": (sum(1 for r in ans if r["abstained"]) / len(ans)) if ans else None,
        "judge_correctness": sc.mean([r["judge"]["correctness"] for r in judged if r["answerable"]]),
        "judge_faithfulness": sc.mean([r["judge"]["faithfulness"] for r in judged]),
        "judged_questions": len(judged),
        "judge_unavailable": len(ok) - len(judged),
    }


def save_results(run: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{run['timestamp'].replace(':', '-')}_{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    return path
