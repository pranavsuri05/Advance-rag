import json

import pytest

from src.llm.provider import LLMError, TracedLLM
from src.rag.generator import Generator
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.citations import validate_citations
from tests.conftest import ScriptedProvider


def items(kb):
    return HybridRetriever(kb).search(["reciprocal rank fusion"]).fused[:3]


def gen(tracer, out):
    return Generator(TracedLLM(ScriptedProvider(lambda *_: out), tracer, max_retries=0))


def test_answer_with_valid_citation(kb, tracer):
    it = items(kb)
    c = it[0].chunk
    out = json.dumps({"abstain": False, "answer": f"k is 60. [Source: {c.source}, Chunk: {c.chunk_id}]"})
    r = gen(tracer, out).generate("q", it)
    assert r.citations[0].chunk_id == c.chunk_id and r.citations[0].text == c.text
    assert not r.abstained and not r.missing_citations


def test_invalid_citation_removed(kb, tracer):
    it = items(kb)
    out = json.dumps({"abstain": False, "answer": f"Claim. [Source: fake.md, Chunk: fake_chunk_999] Real. [Source: {it[0].chunk.source}, Chunk: {it[0].chunk.chunk_id}]"})
    r = gen(tracer, out).generate("q", it)
    assert len(r.invalid_citations) == 1 and "fake_chunk_999" not in r.answer and len(r.citations) == 1


def test_wrong_filename_is_repaired():
    rep = validate_citations("x [Source: wrong.md, Chunk: a_chunk_001].", {"a_chunk_001": {"source": "a.md", "page": None}})
    assert rep.corrected == 1 and "[Source: a.md, Chunk: a_chunk_001]" in rep.text


def test_page_included_for_pdf():
    rep = validate_citations("x [Source: a.pdf, Chunk: a_chunk_001].", {"a_chunk_001": {"source": "a.pdf", "page": 4}})
    assert "Page: 4" in rep.text and rep.valid[0].page == 4


def test_missing_citation_flagged(kb, tracer):
    r = gen(tracer, json.dumps({"abstain": False, "answer": "No citations here."})).generate("q", items(kb))
    assert r.missing_citations and r.citations == []


def test_model_abstention(kb, tracer):
    r = gen(tracer, json.dumps({"abstain": True, "answer": "Not enough information."})).generate("q", items(kb))
    assert r.abstained and not r.missing_citations


def test_invalid_json_response_handled(kb, tracer):
    r = gen(tracer, "Plain prose answer without json").generate("q", items(kb))
    assert not r.parse_ok and r.answer == "Plain prose answer without json"


def test_abstain_message_is_llm_written_and_uncited(tracer):
    out = json.dumps({"abstain": True, "answer": "Nothing about pricing. [Source: x.md, Chunk: y]"})
    r = gen(tracer, out).abstain("price?", "pricing", "none")
    assert r.abstained and r.citations == []


def test_quota_error_propagates(tracer):
    g = Generator(TracedLLM(ScriptedProvider(lambda *_: LLMError("quota", "429")), tracer, max_retries=0))
    with pytest.raises(LLMError) as e:
        g.abstain("q", "m", "e")
    assert e.value.kind == "quota"


def test_validator_removes_malformed_citation_markup_but_keeps_valid_ones():
    idx = {"a_chunk_000": {"source": "a.md", "page": None}}
    text = "Good [Source: a.md, Chunk: a_chunk_000]. Bad [Source: a.md, Chunk: a_chunk_000, b_chunk_001] and [Source: ghost] and [ Source: a.md]."
    rep = validate_citations(text, idx)
    assert rep.text.count("[Source:") == 1 and "[Source: a.md, Chunk: a_chunk_000]" in rep.text
    assert len(rep.invalid) == 3 and [c.chunk_id for c in rep.valid] == ["a_chunk_000"]


def test_validator_keeps_canonical_labels_stable_when_run_twice():
    idx = {"a_chunk_000": {"source": "a.md", "page": 3}}
    once = validate_citations("x [Source: a.md, Chunk: a_chunk_000, Page: 3].", idx)
    twice = validate_citations(once.text, idx)
    assert once.text == twice.text and twice.invalid == [] and len(twice.valid) == 1
