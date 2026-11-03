from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.ingestion.loader import SUPPORTED_EXTENSIONS, IngestionError
from ui.components import empty_state, html_table, metric_cards


def _flash(kind: str, message: str) -> None:
    """Queue a message that survives the st.rerun() that follows a mutation (otherwise it would vanish instantly)."""
    st.session_state.setdefault("kb_flash", []).append((kind, message))


def render(svc) -> None:
    kb = svc.kb
    st.title("Knowledge Base")
    for kind, message in st.session_state.pop("kb_flash", []):
        getattr(st, kind)(message)
    st.caption("Upload documents, inspect the index, and rebuild it. FAISS, BM25 and metadata are always persisted together.")
    s = kb.status()
    metric_cards([("Documents", str(s["documents"]), "indexed"), ("Chunks", str(s["chunks"]),
                  f"{svc.settings.chunk_size} chars / {svc.settings.chunk_overlap} overlap"),
                  ("Index", "Ready" if s["ready"] else "Empty", s["embedding_model"].split("/")[-1]),
                  ("Last indexed", (s["last_indexed"] or "never").replace("T", " ")[:19], "UTC"),
                  ("File types", "PDF · TXT · MD", "")])
    if s["error"]:
        st.warning(s["error"])

    up = st.file_uploader("Upload documents", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS], accept_multiple_files=True)
    if up and st.button("Index uploaded files", type="primary"):
        bar = st.progress(0.0, text="Starting")
        for n, f in enumerate(up):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / Path(f.name).name
                path.write_bytes(f.getvalue())
                try:
                    kb.add_file(path, progress=lambda p, msg, n=n: bar.progress((n + p) / len(up), text=f"{f.name}: {msg}"))
                    _flash("success", f"Indexed {f.name}")
                except IngestionError as exc:
                    _flash("error", f"{f.name}: {exc}")
        bar.empty()
        st.rerun()

    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Rebuild index", help="Re-read every file in data/documents; re-embeds only text not seen before."):
        bar = st.progress(0.0, text="Rebuilding")
        res = kb.rebuild(progress=lambda p, msg: bar.progress(min(p, 1.0), text=msg))
        bar.empty()
        for e in res["errors"]:
            _flash("warning", e)
        _flash("success", f"Index rebuilt: {res['documents']} documents, {res['chunks']} chunks")
        st.rerun()
    if c2.button("Reload from disk"):
        st.cache_resource.clear()
        st.rerun()

    st.subheader("Documents")
    query = st.text_input("Search document names", placeholder="e.g. rrf, chunking")
    docs = [d for d in kb.documents.values() if query.lower() in d["filename"].lower()]
    if not docs:
        empty_state("No documents indexed yet." if not kb.documents else "No documents match your search.")
        return
    html_table(["Document", "Type", "Size", "Pages", "Chunks", "Indexed at"],
               [[d["filename"], d["file_type"], f"{d['size_bytes'] / 1024:.1f} KB", d["pages"], d["num_chunks"], d["indexed_at"].replace("T", " ")[:19]]
                for d in sorted(docs, key=lambda d: d["filename"])])
    st.write("")
    with st.expander("Document details / remove"):
        pick = st.selectbox("Document", [d["doc_id"] for d in docs], format_func=lambda i: kb.documents[i]["filename"])
        chunks = [c for c in kb.chunks if c.doc_id == pick]
        st.caption(f"{len(chunks)} chunks · hash {kb.documents[pick]['content_hash'][:12]}")
        for c in chunks[:3]:
            st.markdown(f"`{c.chunk_id}`" + (f" · *{c.section}*" if c.section else ""))
            st.code(c.text[:400], language=None)
        if st.button("Delete document"):
            kb.remove_document(pick)
            _flash("success", "Removed; FAISS, BM25 and metadata rebuilt consistently.")
            st.rerun()
