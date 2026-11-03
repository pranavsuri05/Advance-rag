from __future__ import annotations

import streamlit as st

from ui.components import citation_panel, esc, metric_cards, pill, stage_list

SAMPLES = [
    "What is the default value of k in Reciprocal Rank Fusion?",
    "Explain how hybrid retrieval and reranking work together, and why both are needed.",
    "Why can a similarity threshold alone not prevent hallucinations?",
    "What is the monthly price of a managed Pinecone subscription?",
]


def _trace(meta: dict) -> None:
    with st.expander("Retrieval & Reasoning Trace"):
        stage_list(meta.get("stages", []))
        st.markdown(f"**Rewritten query:** {esc(meta.get('rewritten_query', ''))}", unsafe_allow_html=True)
        if meta.get("subqueries"):
            st.markdown("**Subqueries:** " + " · ".join(esc(q) for q in meta["subqueries"]), unsafe_allow_html=True)
        g = meta.get("context_grade")
        if g:
            st.markdown(f"**Final context grade:** {'sufficient' if g['relevant'] else 'insufficient'} ({g['score']:.2f}) - {esc(g['explanation'])}",
                        unsafe_allow_html=True)
            if g.get("missing_information"):
                st.markdown(f"**Missing:** {esc(g['missing_information'])}", unsafe_allow_html=True)
        if meta.get("retry_queries"):
            st.markdown("**Corrective retry queries:** " + " · ".join(esc(" | ".join(q)) for q in meta["retry_queries"]), unsafe_allow_html=True)
        if meta.get("invalid_citations"):
            st.warning(f"{len(meta['invalid_citations'])} invalid citation(s) were removed from the answer.")
        t = meta.get("timing_ms", {})
        if t:
            st.caption("Timing (ms): " + ", ".join(f"{k} {v}" for k, v in t.items()))


def render(svc) -> None:
    sid = st.session_state.session_id
    st.markdown('<div class="hero"><h1>Advanced RAG Studio</h1><p>Intelligent Document Retrieval &amp; Grounded AI</p></div>', unsafe_allow_html=True)
    status = svc.kb.status()
    metric_cards([("Documents", str(status["documents"]), "indexed"), ("Chunks", str(status["chunks"]), "in FAISS + BM25"),
                  ("LLM", "Ready" if svc.llm else "No API key", svc.settings.active_model),
                  ("Session", sid[:8], "full id in sidebar")])
    if svc.llm is None:
        st.error(svc.llm_error)
    if not svc.kb.ready:
        st.warning("The knowledge base is empty. Open **Knowledge Base** to index the sample corpus or upload documents.")

    messages = svc.sessions.messages(sid)
    if not messages:
        st.caption("Try a sample question")
        cols = st.columns(2)
        for i, q in enumerate(SAMPLES):
            if cols[i % 2].button(q, key=f"sample{i}", use_container_width=True):
                st.session_state.pending = q
                st.rerun()
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                meta = m.get("metadata", {})
                if meta.get("abstained"):
                    st.markdown(pill("Abstained: insufficient evidence", "warn"), unsafe_allow_html=True)
                if meta.get("retried"):
                    st.markdown(pill("Corrective retry performed", "accent"), unsafe_allow_html=True)
                citation_panel(m.get("sources", []), key=m["timestamp"])
                _trace(meta)

    prompt = st.chat_input("Ask about RAG, retrieval, evaluation...", disabled=svc.llm is None)
    question = prompt or st.session_state.pop("pending", None)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Transforming query, retrieving, reranking, grading..."):
                resp = svc.pipeline.ask(question, sid)
        if resp.error:
            st.session_state.last_error = resp.error
        else:
            st.session_state.pop("last_error", None)
        st.rerun()
    if st.session_state.get("last_error"):
        st.error(st.session_state.last_error)
    if messages and st.button("Clear chat"):
        svc.sessions.reset(sid)
        st.session_state.pop("last_error", None)
        st.rerun()
