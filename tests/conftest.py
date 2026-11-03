import hashlib
import json
import os
import socket
import sys
from pathlib import Path

# Local models are cached on disk; forbid any Hugging Face network access during tests.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings  # noqa: E402
from src.ingestion.indexer import KnowledgeBase  # noqa: E402
from src.llm.provider import LLMProvider, RawCompletion, TracedLLM  # noqa: E402
from src.memory.session_manager import SessionManager  # noqa: E402
from src.observability.tracer import Tracer  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402
from src.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from src.utils.text_utils import tokenize  # noqa: E402


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download)."""
    model_name = "test/hash-embedder"
    dim = 256

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class FakeCrossEncoder:
    """Scores a pair by query-word overlap, so reranking is real but deterministic."""
    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return [len(set(tokenize(q)) & set(tokenize(d))) / (1 + len(set(tokenize(q)))) for q, d in pairs]


class ScriptedProvider(LLMProvider):
    """Mock provider. `handler(prompt, system) -> str | Exception`; records every call."""
    model = "mock-model"

    def __init__(self, handler=None, usage=True):
        self.handler, self.usage, self.calls = handler or default_handler, usage, []

    def complete(self, prompt, system=None, json_mode=False, temperature=0.2, max_output_tokens=None):
        self.calls.append({"prompt": prompt, "system": system, "max_output_tokens": max_output_tokens})
        out = self.handler(prompt, system)
        if isinstance(out, Exception):
            raise out
        return RawCompletion(out, 100 if self.usage else None, 20 if self.usage else None)


def default_handler(prompt, system):
    s = system or ""
    if "query-understanding" in s:
        return json.dumps({"rewritten_query": "reciprocal rank fusion", "needs_decomposition": False, "reason": "clear"})
    if "evidence grader" in s:
        return json.dumps({"relevant": True, "score": 0.9, "missing_information": "", "explanation": "ok"})
    if "answer-generation component of a grounded" in s and "Retrieval was graded insufficient" not in s:
        import re
        m = re.search(r"Chunk: (\S+)", prompt)
        cid = m.group(1) if m else "x"
        src = re.search(r"Source: (\S+) \|", prompt).group(1)
        return json.dumps({"abstain": False, "answer": f"RRF sums reciprocal ranks. [Source: {src}, Chunk: {cid}]"})
    if "Retrieval was graded insufficient" in s:
        return json.dumps({"abstain": True, "answer": "The knowledge base does not contain this."})
    return "{}"


DOCS = {
    "rrf.md": "# RRF\n\nReciprocal rank fusion sums 1 over k plus rank across ranked lists. The constant k is 60 by default.\n",
    "bm25.md": "# BM25\n\nBM25 is a keyword ranking function using term frequency and inverse document frequency.\n",
    "faiss.md": "# FAISS\n\nFAISS performs vector similarity search over dense embeddings using inner product.\n",
}


@pytest.fixture
def settings(tmp_path):
    s = Settings(gemini_api_key="test-key-not-real", documents_dir=tmp_path / "docs", index_dir=tmp_path / "idx",
                 sessions_dir=tmp_path / "sess", logs_dir=tmp_path / "logs", chunk_size=300, chunk_overlap=50,
                 final_top_k=3)
    s.ensure_dirs()
    return s


@pytest.fixture
def kb(settings, tmp_path):
    kb = KnowledgeBase(settings, HashEmbedder())
    src = tmp_path / "src"
    src.mkdir()
    for name, text in DOCS.items():
        (src / name).write_text(text)
        kb.add_file(src / name)
    return kb


@pytest.fixture
def tracer(settings):
    return Tracer(settings.llm_log_path)


@pytest.fixture
def make_pipeline(settings, kb, tracer):
    def _make(handler=None, retries=1):
        settings.max_corrective_retries = retries
        provider = ScriptedProvider(handler)
        llm = TracedLLM(provider, tracer, max_retries=0, backoff_base_s=0)
        pipe = RAGPipeline(settings, kb, llm, CrossEncoderReranker("fake", model=FakeCrossEncoder()),
                           SessionManager(settings.sessions_dir))
        return pipe, provider
    return _make


NETWORK_ATTEMPTS: list = []


@pytest.fixture(autouse=True)
def _no_network_no_real_keys(monkeypatch):
    """Safety net for the whole suite: no real API keys in the environment, and any non-loopback socket
    connection attempt (i.e. a live LLM call) fails the test immediately."""
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(var, "")
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if self.family != socket.AF_UNIX and host not in ("127.0.0.1", "::1", "localhost"):
            NETWORK_ATTEMPTS.append(address)
            raise AssertionError(f"network access attempted in tests: {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    yield
    assert not NETWORK_ATTEMPTS, f"network attempts: {NETWORK_ATTEMPTS}"


@pytest.fixture(scope="session")
def real_embedder():
    """The real local all-MiniLM-L6-v2 model (loaded once per test session)."""
    from src.config import EMBEDDING_MODEL
    from src.ingestion.indexer import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(EMBEDDING_MODEL)


@pytest.fixture(scope="session")
def real_reranker():
    from src.config import RERANKER_MODEL
    return CrossEncoderReranker(RERANKER_MODEL)


@pytest.fixture
def corpus_dir():
    return Path(__file__).resolve().parent.parent / "data" / "documents"


@pytest.fixture
def real_kb(settings, real_embedder, corpus_dir, tmp_path):
    """KnowledgeBase over the 15 real sample documents with the REAL embedding model (copied to a temp dir)."""
    from src.ingestion.indexer import KnowledgeBase
    kb = KnowledgeBase(settings, real_embedder)
    for f in sorted(corpus_dir.glob("*.md")):
        kb.add_file(f)
    return kb
