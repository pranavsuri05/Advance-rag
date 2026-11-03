"""Service wiring shared by the Streamlit app and the CLI evaluation runner (no Streamlit imports)."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings
from src.ingestion.indexer import KnowledgeBase, SentenceTransformerEmbedder
from src.llm.factory import build_llm
from src.llm.provider import LLMError, TracedLLM
from src.memory.session_manager import SessionManager
from src.observability.tracer import Tracer
from src.rag.pipeline import RAGPipeline
from src.retrieval.reranker import CrossEncoderReranker


@dataclass
class Services:
    settings: Settings
    kb: KnowledgeBase
    tracer: Tracer
    llm: TracedLLM | None
    llm_error: str | None
    sessions: SessionManager
    reranker: CrossEncoderReranker
    pipeline: RAGPipeline


def build_services(settings: Settings | None = None) -> Services:
    s = settings or Settings()
    s.ensure_dirs()
    tracer = Tracer(s.llm_log_path)
    llm, llm_error = None, None
    try:
        llm = build_llm(s, tracer)
        if llm is None:
            llm_error = LLMError("missing_key", "").user_message()
    except LLMError as exc:
        llm_error = exc.user_message()
    kb = KnowledgeBase(s, SentenceTransformerEmbedder(s.embedding_model))
    sessions = SessionManager(s.sessions_dir)
    reranker = CrossEncoderReranker(s.reranker_model)
    pipeline = RAGPipeline(s, kb, llm, reranker, sessions)
    return Services(s, kb, tracer, llm, llm_error, sessions, reranker, pipeline)
