import json

import pytest

from src.config import ROOT
from src.evaluation import scoring as sc
from src.evaluation.evaluator import Evaluator, GoldItem, aggregate, load_dataset, save_results


def test_gold_dataset_parses():
    items = load_dataset(ROOT / "evaluation" / "dataset.json")
    assert 15 <= len(items) <= 20
    assert any(not i.answerable for i in items) and any(i.history for i in items)
    assert all(i.expected_documents for i in items if i.answerable)
    import os
    for i in items:
        for d in i.expected_documents:
            assert os.path.exists(ROOT / "data" / "documents" / d), d


def test_exact_match_and_f1():
    assert sc.exact_match("The answer, is 60!", "the answer is 60") == 1.0
    assert sc.exact_match("a", "b") == 0.0
    assert sc.token_f1("a b c d", "a b x y") == pytest.approx(0.5)
    assert sc.token_f1("", "a") == 0.0 and sc.token_f1("x", "y") == 0.0


def test_keyword_recall_and_heuristic():
    assert sc.keyword_recall("k is 60 by default", ["60", "default"]) == 1.0
    assert sc.keyword_recall("nothing", ["60"]) == 0.0 and sc.keyword_recall("x", []) is None
    assert sc.heuristic_correct(0.0, 0.7) and not sc.heuristic_correct(0.9, 0.2) and sc.heuristic_correct(0.5, None)


def test_retrieval_metrics():
    ranked = ["x", "a", "y", "b"]
    rel = {"a", "b"}
    assert sc.precision_at_k(ranked, rel, 4) == 0.5 and sc.precision_at_k(ranked, rel, 2) == 0.5
    assert sc.recall_at_k(ranked, rel, 2) == 0.5 and sc.recall_at_k(ranked, rel, 4) == 1.0
    assert sc.hit_rate_at_k(ranked, rel, 1) == 0.0 and sc.hit_rate_at_k(ranked, rel, 2) == 1.0
    assert sc.reciprocal_rank(ranked, rel) == 0.5 and sc.reciprocal_rank(["z"], rel) == 0.0
    assert sc.precision_at_k([], rel, 3) == 0.0 and sc.recall_at_k(ranked, set(), 3) == 0.0


def test_citation_validity_and_abstention():
    assert sc.citation_validity(["a", "b"], {"a"}) == 0.5
    assert sc.citation_validity([], {"a"}) is None
    assert sc.abstention_correct(True, False) and sc.abstention_correct(False, True)
    assert not sc.abstention_correct(True, True) and not sc.abstention_correct(False, False)


def test_aggregate_and_serialization(tmp_path):
    results = [
        {"answerable": True, "abstained": False, "error": None, "metrics": {"heuristic_correct": 1.0, "mrr": 1.0, "token_f1": 0.5, "citation_validity": 1.0}, "judge": {"available": True, "correctness": 1.0, "faithfulness": 0.8}},
        {"answerable": False, "abstained": True, "error": None, "metrics": {"abstention_correct": 1.0, "citation_validity": None}, "judge": {"available": False, "reason": "quota"}},
        {"answerable": True, "abstained": False, "error": "quota", "metrics": {}},
    ]
    a = aggregate(results)
    assert a["evaluated"] == 2 and a["errors"] == 1 and a["mrr"] == 1.0
    assert a["abstention_accuracy_unanswerable"] == 1.0 and a["judge_faithfulness"] == 0.8 and a["judge_unavailable"] == 1
    path = save_results({"run_id": "r1", "timestamp": "2026-01-01T00:00:00", "results": results, "aggregate": a}, tmp_path)
    assert json.loads(path.read_text())["run_id"] == "r1"


def test_evaluator_runs_with_mock_pipeline(make_pipeline):
    pipe, _ = make_pipeline()
    items = [GoldItem(id="a", question="What is the constant k in reciprocal rank fusion?", reference_answer="RRF sums reciprocal ranks",
                      expected_documents=["rrf.md"], expected_keywords=["reciprocal"]),
             GoldItem(id="b", question="unanswerable thing", answerable=False, should_abstain=True)]
    # unanswerable item: default mock handler answers, so abstention_correct must be False (no fabricated pass)
    run = Evaluator(pipe, pipe.llm, use_judge=False).run(items)
    r0, r1 = run["results"]
    assert r0["metrics"]["hit_rate_at_k"] == 1.0 and r0["metrics"]["citation_validity"] == 1.0 and r0["metrics"]["heuristic_correct"] == 1.0
    assert r1["abstention_correct"] is False
    assert run["aggregate"]["judge_unavailable"] == 2 and run["aggregate"]["judge_faithfulness"] is None


def test_judge_unavailable_is_reported_not_faked(make_pipeline):
    from src.llm.provider import LLMError
    from tests.conftest import default_handler

    def h(prompt, system):
        return LLMError("quota", "429") if "impartial evaluation judge" in system else default_handler(prompt, system)
    pipe, _ = make_pipeline(h)
    item = GoldItem(id="a", question="reciprocal rank fusion constant", reference_answer="x", expected_documents=["rrf.md"])
    r = Evaluator(pipe, pipe.llm).evaluate_item(item)
    assert r["judge"]["available"] is False and "quota" in r["judge"]["reason"]


def test_offline_retrieval_eval_makes_no_llm_calls(kb):
    from src.evaluation.offline import run_offline_retrieval_eval, save_offline
    from src.retrieval.reranker import CrossEncoderReranker
    from tests.conftest import FakeCrossEncoder
    items = [GoldItem(id="a", question="reciprocal rank fusion constant k", expected_documents=["rrf.md"]),
             GoldItem(id="b", question="bm25 term frequency", expected_documents=["bm25.md"]),
             GoldItem(id="c", question="unanswerable", answerable=False),
             GoldItem(id="d", question="and why?", expected_documents=["rrf.md"], history=[{"role": "user", "content": "x"}])]
    run = run_offline_retrieval_eval(kb, CrossEncoderReranker("fake", model=FakeCrossEncoder()), items, k=3)
    assert run["llm_calls"] == 0 and run["evaluated"] == 2 and {e["id"] for e in run["excluded"]} == {"c", "d"}
    assert set(run["aggregate"]) == {"vector_only", "bm25_only", "hybrid_rrf", "hybrid_rrf_rerank"}
    assert run["aggregate"]["hybrid_rrf"]["hit_rate_at_k"] == 1.0 and run["results"][0]["reranked"] is True
    assert save_offline(run, kb.settings.logs_dir / "evals").exists()
