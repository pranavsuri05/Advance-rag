from __future__ import annotations

import streamlit as st

from ui.components import empty_state, html_table, metric_cards


def render(svc) -> None:
    sid = st.session_state.session_id
    st.title("Session Memory")
    st.caption("Conversation state is stored per session on disk and a bounded window is used for follow-up query rewriting.")
    data = svc.sessions.load(sid)
    metric_cards([("Session ID", sid, ""), ("Messages", str(len(data["messages"])), f"window: last {svc.settings.history_window}"),
                  ("Created", data["created_at"].replace("T", " ")[:19], "UTC"), ("Context turns", str(len(data.get("context_history", []))), "")])
    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button("Clear session"):
        svc.sessions.reset(sid)
        st.rerun()
    if c2.button("New session"):
        st.session_state.session_id = svc.sessions.create()
        st.rerun()
    tabs = st.tabs(["Transcript", "Context history", "Window sent to LLM", "State dump"])
    with tabs[0]:
        if not data["messages"]:
            empty_state("No messages yet.")
        for m in data["messages"]:
            with st.chat_message(m["role"]):
                st.caption(m["timestamp"])
                st.markdown(m["content"])
                if m["sources"]:
                    st.caption("Sources: " + ", ".join(f"{s['source']}:{s['chunk_id']}" for s in m["sources"]))
        st.download_button("Export transcript (Markdown)", svc.sessions.export_markdown(sid), file_name=f"session_{sid}.md")
    with tabs[1]:
        ctx = data.get("context_history", [])
        html_table(["Time", "Question", "Rewritten", "Chunks used"], [[c["timestamp"][11:19], c["question"][:60], c["rewritten"][:60], ", ".join(c["chunks"])] for c in ctx]) if ctx else empty_state("No context yet.")
    with tabs[2]:
        st.json(svc.sessions.window(sid, svc.settings.history_window))
    with tabs[3]:
        st.json({"session_state": {k: (v if isinstance(v, (str, int, float, bool)) else type(v).__name__) for k, v in st.session_state.items()},
                 "session_file": data})
