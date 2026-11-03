"""Advanced RAG Studio - Streamlit entrypoint (UI only; all logic lives in src/)."""
from __future__ import annotations

import streamlit as st

from src.bootstrap import build_services
from ui.pages import chat, evaluation, knowledge_base, memory, observability, retrieval_inspector
from ui.styles import build_css
from ui.theme import MODES

st.set_page_config(page_title="Advanced RAG Studio", page_icon="◆", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Loading models and indexes...")
def get_services():
    """Shared, read-mostly services (models, indexes). Per-user state stays in st.session_state."""
    svc = build_services()
    if not svc.kb.ready and any(svc.settings.documents_dir.iterdir()):
        svc.kb.rebuild()      # first launch: index the bundled sample corpus
    return svc


try:
    svc = get_services()
except ValueError as exc:      # e.g. a malformed numeric value in .env: show a clear message instead of a stack trace
    st.error(f"Configuration error: {exc}")
    st.stop()

# ---- session identity (per browser session) ----
if "session_id" not in st.session_state:
    st.session_state.session_id = svc.sessions.create()

# ---- theme: persisted in the URL so it survives reloads and reruns ----
if "theme_mode" not in st.session_state:
    q = st.query_params.get("theme", "Auto")
    st.session_state.theme_mode = q if q in MODES else "Auto"


def _theme_changed():
    st.query_params["theme"] = st.session_state.theme_mode


with st.sidebar:
    st.markdown('<div class="brand">ADVANCED <span>RAG</span> STUDIO</div>'
                '<div class="tagline">Intelligent Document Retrieval &amp; Grounded AI</div>', unsafe_allow_html=True)
    st.write("")

st.markdown(build_css(st.session_state.theme_mode), unsafe_allow_html=True)

pages = [
    st.Page(lambda: chat.render(svc), title="Chat", url_path="chat", default=True),
    st.Page(lambda: knowledge_base.render(svc), title="Knowledge Base", url_path="knowledge-base"),
    st.Page(lambda: retrieval_inspector.render(svc), title="Retrieval Inspector", url_path="retrieval-inspector"),
    st.Page(lambda: evaluation.render(svc), title="Evaluation", url_path="evaluation"),
    st.Page(lambda: observability.render(svc), title="Observability", url_path="observability"),
    st.Page(lambda: memory.render(svc), title="Session Memory", url_path="session-memory"),
]
nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.divider()
    st.radio("Appearance", MODES, key="theme_mode", horizontal=True, on_change=_theme_changed,
             help="Auto follows your operating-system light/dark preference.")
    st.caption(f"Session `{st.session_state.session_id}`")
    st.caption(f"{svc.settings.llm_provider.title()}: `{svc.settings.active_model}`")

nav.run()
