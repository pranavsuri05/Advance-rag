"""Evaluation + observability QA (local only)."""
import importlib.util
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import ROOT
from src.evaluation import scoring as sc
from src.evaluation.evaluator import GoldItem, load_dataset, save_results, validate_dataset
from src.evaluation.offline import run_offline_retrieval_eval, save_offline
from src.llm.provider import TracedLLM
from src.observability.metrics import summarize
from src.observability.tracer import Tracer
from src.retrieval.reranker import CrossEncoderReranker
from tests.conftest import FakeCrossEncoder, ScriptedProvider


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "evaluation" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- dataset ----------
def test_real_dataset_is_valid_against_the_real_corpus():
    items = load_dataset(ROOT / "evaluation" / "dataset.json")
    corpus = {p.name for p in (ROOT / "data" / "documents").glob("*.md")}
    assert validate_dataset(items, corpus) == []
    assert len(items) == 18 and sum(not i.answerable for i in items) == 3
    assert {"factual", "multi_part", "multi_hop", "cross_document", "followup", "unanswerable"} <= {i.category for i in items}


def test_validate_dataset_flags_each_problem():
    items = [GoldItem(id="a", question="q", reference_answer="r", expected_documents=["x.md"]),
             GoldItem(id="a", question=" ", answerable=True),
             GoldItem(id="u", question="q", answerable=False, expected_documents=["x.md"]),
             GoldItem(id="m", question="q", reference_answer="r", expected_documents=["missing.md"])]
    probs = "\n".join(validate_dataset(items, {"x.md"}))
    for needle in ("duplicate id", "empty question", "no expected_documents", "no reference_answer", "unanswerable but lists", "'missing.md' is not in the corpus"):
        assert needle in probs, needle


def test_load_dataset_rejects_malformed_files(tmp_path):
    for content in ("not json", '{"id": "x"}', '[{"id": "x"}]', '[{"question": "q"}]'):
        p = tmp_path / "d.json"
        p.write_text(content)
        with pytest.raises(ValueError):
            load_dataset(p)
    with pytest.raises(OSError):
        load_dataset(tmp_path / "missing.json")


# ---------- scoring edge cases ----------
def test_scoring_edge_cases():
    assert sc.precision_at_k(["a"], {"a"}, 0) == 0.0 and sc.hit_rate_at_k([], {"a"}, 5) == 0.0 and sc.reciprocal_rank([], {"a"}) == 0.0
    assert sc.recall_at_k(["a", "a", "a"], {"a", "b"}, 3) == 0.5                      # duplicates do not inflate recall
    assert sc.token_f1("a a a", "a") == pytest.approx(0.5) and sc.keyword_recall("x", ["A B"]) == 0.0
    assert sc.mean([None, None]) is None and sc.mean([1.0, None, 0.0]) == 0.5
    assert sc.citation_validity(["a", "a"], {"a"}) == 1.0


# ---------- offline evaluation: zero LLM calls ----------
def test_offline_eval_has_no_llm_in_its_signature_and_makes_zero_calls(kb, monkeypatch):
    assert "llm" not in inspect.signature(run_offline_retrieval_eval).parameters
    called = []
    monkeypatch.setattr(TracedLLM, "generate", lambda *a, **k: called.append(1))
    monkeypatch.setattr(ScriptedProvider, "complete", lambda *a, **k: called.append(1))
    items = [GoldItem(id="a", question="reciprocal rank fusion constant", reference_answer="r", expected_documents=["rrf.md"])]
    run = run_offline_retrieval_eval(kb, CrossEncoderReranker("fake", model=FakeCrossEncoder()), items, k=3)
    assert called == [] and run["llm_calls"] == 0 and run["evaluated"] == 1


def test_offline_eval_invalid_ground_truth_and_empty_retrieval_do_not_crash(kb, settings, real_embedder=None):
    items = [GoldItem(id="ghost", question="reciprocal rank fusion", reference_answer="r", expected_documents=["does_not_exist.md"])]
    run = run_offline_retrieval_eval(kb, CrossEncoderReranker("fake", model=FakeCrossEncoder()), items, k=3)
    for v in run["aggregate"].values():
        assert v["hit_rate_at_k"] == 0.0 and v["recall_at_k"] == 0.0 and v["mrr"] == 0.0 and v["precision_at_k"] == 0.0


def test_offline_eval_empty_index_and_no_evaluable_items(settings):
    from src.ingestion.indexer import KnowledgeBase
    from tests.conftest import HashEmbedder
    empty = KnowledgeBase(settings, HashEmbedder())
    items = [GoldItem(id="a", question="anything", reference_answer="r", expected_documents=["rrf.md"])]
    run = run_offline_retrieval_eval(empty, CrossEncoderReranker("fake", model=FakeCrossEncoder()), items, k=3)
    assert run["evaluated"] == 1 and run["aggregate"]["hybrid_rrf"]["mrr"] == 0.0
    none = run_offline_retrieval_eval(empty, CrossEncoderReranker("fake", model=FakeCrossEncoder()), [GoldItem(id="u", question="q", answerable=False)])
    assert none["evaluated"] == 0 and none["aggregate"]["vector_only"]["mrr"] is None


def test_report_generation_roundtrip(kb, tmp_path):
    items = [GoldItem(id="a", question="bm25 term frequency", reference_answer="r", expected_documents=["bm25.md"])]
    run = run_offline_retrieval_eval(kb, CrossEncoderReranker("fake", model=FakeCrossEncoder()), items, k=3)
    path = save_offline(run, tmp_path / "out")
    assert json.loads(path.read_text()) == run and path.name.startswith("offline_retrieval_")
    p2 = save_results({"run_id": "r1", "timestamp": "2026-01-01T00:00:00", "aggregate": {}, "results": []}, tmp_path / "out")
    assert json.loads(p2.read_text())["run_id"] == "r1"


# ---------- CLI paths ----------
def test_offline_cli_runs_and_prints_table_with_zero_llm_calls(kb, settings, capsys, monkeypatch):
    mod = load_script("run_offline_eval")
    monkeypatch.setattr(mod, "build_services", lambda: SimpleNamespace(kb=kb, reranker=CrossEncoderReranker("fake", model=FakeCrossEncoder()), settings=settings))
    monkeypatch.setattr(TracedLLM, "generate", lambda *a, **k: pytest.fail("LLM called from the offline CLI"))
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "0 LLM calls" in out and "hybrid_rrf_rerank" in out and "Saved:" in out
    assert list((settings.logs_dir / "evals").glob("offline_retrieval_*.json"))


def test_live_eval_cli_without_key_exits_2_and_makes_no_calls(kb, settings, capsys, monkeypatch):
    mod = load_script("run_evaluation")
    monkeypatch.setattr(mod, "build_services", lambda: SimpleNamespace(llm=None, kb=kb, settings=settings, pipeline=None))
    monkeypatch.setattr("sys.argv", ["run_evaluation.py"])
    assert mod.main() == 2 and "not set" in capsys.readouterr().err


def test_live_eval_cli_with_mock_pipeline_limit_and_no_judge(make_pipeline, settings, kb, capsys, monkeypatch):
    mod = load_script("run_evaluation")
    pipe, prov = make_pipeline()
    monkeypatch.setattr(mod, "build_services", lambda: SimpleNamespace(llm=pipe.llm, kb=kb, settings=settings, pipeline=pipe))
    monkeypatch.setattr("sys.argv", ["run_evaluation.py", "--limit", "2", "--no-judge"])
    assert mod.main() == 0
    assert "Aggregate metrics" in capsys.readouterr().out
    assert len(list((settings.logs_dir / "evals").glob("eval_*.json"))) == 1
    assert not any("impartial evaluation judge" in (c["system"] or "") for c in prov.calls)      # --no-judge honoured


# ---------- observability ----------
def test_tracer_tolerates_malformed_partial_and_incomplete_records(tmp_path):
    t = Tracer(tmp_path / "l.jsonl")
    t.path.write_text('\n'.join(['{"call_id": "ok", "timestamp": "2026-01-01T00:00:00+00:00", "model": "m", "call_type": "query_rewrite", "status": "success", "input_tokens": 3, "output_tokens": 4, "latency_ms": 10}',
                                 '{"call_id": "partial"', "[]", "123", "null", '"text"', "", '{"status": "failure"}', '{"input_tokens": "NaN", "latency_ms": null}',
                                 '{"call_id": "emoji", "prompt": "line1\\nline2 日本語 🚀", "status": "success"}']))
    recs = t.read()
    assert [r["call_id"] for r in recs] == ["ok", "", "", "emoji"]
    s = summarize(recs)                                              # must not raise KeyError on incomplete records
    assert s["total_calls"] == 4 and s["successful"] == 2 and s["failed"] == 2 and s["input_tokens"] == 3
    assert recs[3]["prompt"].startswith("line1\nline2 日本語")
    assert all({"latency_ms", "call_type", "timestamp", "input_tokens", "output_tokens", "total_tokens", "status", "model"} <= set(r) for r in recs)


def test_missing_diagnostics_fields_and_large_log_performance(tmp_path):
    t = Tracer(tmp_path / "big.jsonl")
    rec = {"call_id": "x", "timestamp": "2026-01-01T00:00:00+00:00", "model": "m", "call_type": "context_grading", "status": "success",
           "input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "latency_ms": 5, "tokens_estimated": False, "diagnostics": None}
    t.path.write_text("\n".join(json.dumps(rec) for _ in range(20000)))
    start = time.perf_counter()
    s = summarize(t.read())
    assert time.perf_counter() - start < 10 and s["total_calls"] == 20000 and s["total_tokens"] == 20000 * 120 and s["avg_latency_ms"] == 5


def test_trace_records_capture_request_id_finish_reason_model_latency(tracer):
    class P(ScriptedProvider):
        def complete(self, *a, **k):
            raw = super().complete(*a, **k)
            raw.diagnostics = {"headers": {"x-request-id": "req_abc", "retry-after": "3"}, "finish_reason": "stop"}
            return raw
    TracedLLM(P(lambda *_: "ok"), tracer, max_retries=0).generate("p", "answer_generation", session_id="s9")
    r = tracer.read()[0]
    assert r["diagnostics"]["headers"]["x-request-id"] == "req_abc" and r["diagnostics"]["finish_reason"] == "stop"
    assert r["model"] == "mock-model" and r["latency_ms"] >= 0 and r["session_id"] == "s9" and r["tokens_estimated"] is False
