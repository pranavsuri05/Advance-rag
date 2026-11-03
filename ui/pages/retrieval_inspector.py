from __future__ import annotations

import streamlit as st

from ui.components import chunk_rows, empty_state, esc, evidence, html_table, stage_list

HEAD = ["#", "Chunk", "Source", "Vec rank", "Vec cos", "BM25 rank", "BM25", "RRF", "Pre-rerank", "CE score", "Post-rerank"]


def _section(title: str, items, cols: list[int]) -> None:
    st.markdown(f"#### {title}")
    if not items:
        st.caption("No results at this stage.")
        return
    rows = chunk_rows(items)
    html_table([HEAD[i] for i in cols], [[r[i] for i in cols] for r in rows])


def render(svc) -> None:
    st.title("Retrieval Inspector")
    st.caption("Runs the retrieval stages for real - query transformation, FAISS, BM25, RRF, cross-encoder, grading, corrective retry - without generating an answer.")
    q = st.text_input("Query", value="Explain how hybrid retrieval and reranking work together, and why both are needed.")
    if st.button("Inspect retrieval", type="primary"):
        with st.spinner("Running retrieval pipeline..."):
            st.session_state.inspection = svc.pipeline.inspect(q, st.session_state.session_id)
    r = st.session_state.get("inspection")
    if r is None:
        empty_state("Enter a query and press Inspect retrieval.")
        return
    _show(r)


def _show(r) -> None:
    if r.error:
        st.error(r.error)
        return
    st.markdown("#### 1-2. Query transformation")
    st.markdown(f"**Original:** {esc(r.original_query)}<br>**Rewritten:** {esc(r.rewritten_query)}"
                f"<br>**Reason:** {esc(r.transformation_reason or 'n/a')}", unsafe_allow_html=True)
    if r.decomposed_queries:
        st.markdown("**Decomposed subqueries**")
        for sq in r.decomposed_queries:
            st.markdown(f"- {sq}")
    else:
        st.caption("No decomposition (single information need).")
    _section("3. FAISS vector results", r.vector_results, [0, 1, 2, 3, 4])
    _section("4. BM25 results", r.bm25_results, [0, 1, 2, 5, 6])
    _section("5. RRF fused results", r.fused_results, [0, 1, 2, 3, 5, 7])
    st.markdown("#### 6. Cross-encoder reranking")
    if r.rerank_status != "cross_encoder":
        st.error(f"Cross-encoder reranking did NOT run ({r.rerank_status}): {r.rerank_error}")
    _section("", r.reranked_results, [0, 1, 2, 8, 9, 10])
    st.markdown("#### 7. Corrective RAG history")
    for a in r.attempts:
        g = a.grade
        label = "Initial retrieval" if a.attempt == 0 else f"Retry {a.attempt}"
        with st.expander(f"{label} - {'sufficient' if g.relevant else 'insufficient'} (score {g.score:.2f})", expanded=a.attempt == len(r.attempts) - 1):
            st.markdown(f"**Queries:** {esc(' | '.join(a.queries))}", unsafe_allow_html=True)
            st.markdown(f"**Grader explanation:** {esc(g.explanation)}", unsafe_allow_html=True)
            if g.missing_information:
                st.markdown(f"**Missing information:** {esc(g.missing_information)}", unsafe_allow_html=True)
            if not g.graded:
                st.warning("Grader was unavailable or returned malformed output; verdict is not a real grade.")
            html_table(["Chunk", "Source", "CE score"], [[x.chunk.chunk_id, x.chunk.source, f"{x.rerank_score:.2f}" if x.rerank_score is not None else "-"] for x in a.rerank.items])
    if not r.retried:
        st.caption("No corrective retry: initial context was judged sufficient.")
    st.markdown("#### 8. Final context for generation")
    for it in r.reranked_results:
        with st.expander(f"{it.chunk.source} · {it.chunk.chunk_id}" + (f" · {it.chunk.section}" if it.chunk.section else "")):
            evidence(it.chunk.text)
    st.markdown("#### Stage log")
    stage_list([s.model_dump() for s in r.stages])
