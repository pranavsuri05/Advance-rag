"""Helpers for Streamlit AppTest: build Services backed by tmp dirs, a hash embedder, a fake cross-encoder and a mock LLM."""
from __future__ import annotations

from pathlib import Path

from src.bootstrap import Services
from src.config import Settings
from src.ingestion.indexer import KnowledgeBase
from src.llm.provider import LLMError, TracedLLM
from src.memory.session_manager import SessionManager
from src.observability.tracer import Tracer
from src.rag.pipeline import RAGPipeline
from src.retrieval.reranker import CrossEncoderReranker
from tests.conftest import DOCS, FakeCrossEncoder, HashEmbedder, ScriptedProvider

SVC: Services | None = None


def make_services(tmp: Path, handler=None, llm: bool = True, docs: bool = True, **settings_kw) -> Services:
    s = Settings(documents_dir=tmp / "docs", index_dir=tmp / "idx", sessions_dir=tmp / "sess", logs_dir=tmp / "logs",
                 chunk_size=300, chunk_overlap=50, final_top_k=3, llm_provider="groq", groq_api_key="x" * 20, **settings_kw)
    s.ensure_dirs()
    kb = KnowledgeBase(s, HashEmbedder())
    if docs:
        src = tmp / "src"; src.mkdir(exist_ok=True)
        for name, text in DOCS.items():
            (src / name).write_text(text)
            kb.add_file(src / name)
    tracer = Tracer(s.llm_log_path)
    provider = ScriptedProvider(handler)
    client = TracedLLM(provider, tracer, max_retries=0, backoff_base_s=0) if llm else None
    reranker = CrossEncoderReranker("fake", model=FakeCrossEncoder())
    sessions = SessionManager(s.sessions_dir)
    svc = Services(s, kb, tracer, client, None if llm else LLMError("missing_key", "").user_message(), sessions, reranker,
                   RAGPipeline(s, kb, client, reranker, sessions))
    svc.provider = provider            # exposed for assertions
    return svc


def page_script(name: str):
    """Executed by AppTest.from_function: renders one page exactly like app.py does (theme CSS + shared session id)."""
    import streamlit as st
    from tests import appharness as H
    from ui.pages import chat, evaluation, knowledge_base, memory, observability, retrieval_inspector
    from ui.styles import build_css
    svc = H.SVC
    if "session_id" not in st.session_state:
        st.session_state.session_id = svc.sessions.create()
    st.markdown(build_css(st.session_state.get("theme_mode", "Auto")), unsafe_allow_html=True)
    {"chat": chat, "kb": knowledge_base, "inspector": retrieval_inspector, "evaluation": evaluation,
     "observability": observability, "memory": memory}[name].render(svc)
