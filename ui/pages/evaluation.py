from __future__ import annotations

import json

import streamlit as st

from src.evaluation.evaluator import Evaluator, load_dataset, save_results, validate_dataset
from src.evaluation.offline import run_offline_retrieval_eval, save_offline
from ui.components import empty_state, html_table, metric_cards

DEFS = {
    "Answer correctness (heuristic)": "Transparent rule: keyword recall >= 0.6 (or token F1 >= 0.4 if no keywords) and the system did not abstain. NOT an objective measure of truth.",
    "Answer correctness (LLM judge)": "The LLM judge compares the answer with the reference answer (0 / 0.5 / 1). Separate from the heuristic; shown only when the judge call succeeded.",
    "Hit@K / MRR / Precision@K / Recall@K": "Computed on the K chunks given to the generator, using document-level relevance labels from the gold dataset.",
    "Faithfulness (LLM judge)": "Fraction of answer claims supported by the retrieved context, as judged by the LLM.",
    "Citation validity": "Share of cited chunk ids that exist among the retrieved chunks (invalid ones removed by the validator count against it).",
    "Abstention accuracy": "On unanswerable questions, the share where the system abstained. False-abstention rate is the share of answerable questions it refused.",
}


def _fmt(v, pct=False):
    return "n/a" if v is None else (f"{v:.0%}" if pct else f"{v:.2f}")


def render(svc) -> None:
    st.title("Evaluation")
    st.caption("Runs the gold dataset through the real pipeline with the configured LLM provider. Scores are computed, never entered by hand.")
    ds_path = svc.settings.eval_dataset
    try:
        items = load_dataset(ds_path)
    except (OSError, ValueError) as exc:
        st.error(f"Could not load dataset: {exc}")
        return
    corpus = {p.name for p in svc.settings.documents_dir.iterdir()} if svc.settings.documents_dir.exists() else set()
    problems = validate_dataset(items, corpus)
    if problems:
        st.warning("Dataset issues: " + "; ".join(problems[:5]) + (" ..." if len(problems) > 5 else ""))
    _offline_section(svc, items)
    st.subheader("End-to-end evaluation (calls the LLM)")
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown(f"**Dataset:** `{ds_path.name}` · {len(items)} questions")
    limit = c2.number_input("Questions", 1, len(items), len(items))
    judge = c3.checkbox("LLM judge", value=True, help="Adds ~1 extra LLM call per question (faithfulness + correctness).")
    if st.button("Run evaluation", type="primary", disabled=svc.llm is None or not svc.kb.ready):
        bar = st.progress(0.0, text="Starting")
        run = Evaluator(svc.pipeline, svc.llm, use_judge=judge).run(
            items[: int(limit)], progress=lambda i, n, q: bar.progress(i / n, text=f"{min(i + 1, n)}/{n}: {q[:70]}"))
        bar.empty()
        path = save_results(run, svc.settings.logs_dir / "evals")
        st.session_state.eval_run = run
        st.success(f"Saved {path.name}")
    if svc.llm is None:
        st.error(svc.llm_error)
    saved = sorted((svc.settings.logs_dir / "evals").glob("eval_*.json")) if (svc.settings.logs_dir / "evals").exists() else []
    if "eval_run" not in st.session_state and saved:
        st.session_state.eval_run = json.loads(saved[-1].read_text())
    run = st.session_state.get("eval_run")
    if not run:
        empty_state("No evaluation run yet.")
        return
    a = run["aggregate"]
    st.caption(f"Run {run['run_id']} · {run['timestamp']} · model {run.get('model')} · K={run['k']}")
    metric_cards([
        ("Questions", f"{a['evaluated']}/{a['total_questions']}", f"{a['errors']} errored"),
        ("Correctness (heuristic)", _fmt(a["heuristic_correctness"], True), f"F1 {_fmt(a['token_f1'])}"),
        ("Correctness (judge)", _fmt(a["judge_correctness"], True), f"{a['judged_questions']} judged"),
        ("Hit@K", _fmt(a["hit_rate_at_k"], True), f"P@K {_fmt(a['precision_at_k'])} · R@K {_fmt(a['recall_at_k'])}"),
        ("MRR", _fmt(a["mrr"]), ""),
        ("Faithfulness (judge)", _fmt(a["judge_faithfulness"], True), f"{a['judge_unavailable']} judge unavailable"),
        ("Citation validity", _fmt(a["citation_validity"], True), ""),
        ("Abstention accuracy", _fmt(a["abstention_accuracy_unanswerable"], True), f"false abstain {_fmt(a['false_abstention_rate'], True)}"),
    ])
    with st.expander("Metric definitions"):
        for k, v in DEFS.items():
            st.markdown(f"**{k}** - {v}")
    results = run["results"]
    st.subheader("Per-question results")
    html_table(["ID", "Category", "Question", "Abstained", "Heuristic", "Hit@K", "Citations", "Judge corr.", "Faith."],
               [[r["id"], r["category"], r["question"][:70], "yes" if r["abstained"] else "no",
                 "ERR" if r.get("error") else _fmt(r["metrics"].get("heuristic_correct")), _fmt(r["metrics"].get("hit_rate_at_k")),
                 _fmt(r["metrics"].get("citation_validity")),
                 _fmt(r.get("judge", {}).get("correctness")) if r.get("judge", {}).get("available") else "n/a",
                 _fmt(r.get("judge", {}).get("faithfulness")) if r.get("judge", {}).get("available") else "n/a"] for r in results])
    st.subheader("Failure analysis")
    wrong = [r for r in results if not r.get("error") and r["answerable"] and r["metrics"].get("heuristic_correct") == 0.0]
    cite = [r for r in results if r.get("invalid_citations") or (r.get("metrics", {}).get("citation_validity") not in (None, 1.0))]
    abst = [r for r in results if r.get("abstention_correct") is False]
    errs = [r for r in results if r.get("error")]
    t1, t2, t3, t4 = st.tabs([f"Incorrect ({len(wrong)})", f"Citation failures ({len(cite)})", f"Abstention failures ({len(abst)})", f"Errors ({len(errs)})"])
    for tab, rows in zip((t1, t2, t3, t4), (wrong, cite, abst, errs)):
        with tab:
            if not rows:
                st.caption("None.")
            for r in rows:
                _detail(r)
    st.subheader("Inspect any question")
    pick = st.selectbox("Question", results, format_func=lambda r: f"{r['id']} · {r['question'][:80]}")
    _detail(pick, expanded=True)
    st.download_button("Download results JSON", json.dumps(run, indent=2), file_name=f"eval_{run['run_id']}.json", mime="application/json")


def _detail(r: dict, expanded: bool = False) -> None:
    with st.expander(f"{r['id']} · {r['question'][:90]}", expanded=expanded):
        if r.get("error"):
            st.error(r["error"])
        st.markdown(f"**Reference:** {r['reference_answer'] or '(unanswerable - should abstain)'}")
        st.markdown(f"**Generated:** {r['answer']}")
        st.caption(f"Rewritten: {r['rewritten_query']} · retried: {r['retried']} · expected docs: {', '.join(r['expected_documents']) or '-'}")
        j = r.get("judge", {})
        if j.get("available"):
            st.markdown(f"**Judge:** correctness {j['correctness']:.2f}, faithfulness {j['faithfulness']:.2f} - {j.get('explanation', '')}")
        elif j:
            st.caption(f"Judge unavailable: {j.get('reason')}")
        st.markdown("**Retrieved**")
        for c in r["retrieved"]:
            st.markdown(f"`{c['chunk_id']}` · {c['source']} · CE {c['rerank_score'] if c['rerank_score'] is None else round(c['rerank_score'], 2)}")
        with st.expander("Pipeline trace"):
            html_table(["Stage", "Status", "Detail"], [[s["name"], s["status"], s["detail"]] for s in r["stages"]])
            if r["invalid_citations"]:
                st.warning("Invalid citations: " + "; ".join(r["invalid_citations"]))


def _offline_section(svc, items) -> None:
    """Retrieval-only evaluation using local models: makes zero LLM/API calls."""
    st.subheader("Offline retrieval evaluation")
    st.caption("Local models only - zero LLM calls. Raw question as query (no rewrite/decomposition); answerable, history-free questions; document-level labels. Not an end-to-end result.")
    if st.button("Run offline retrieval evaluation", disabled=not svc.kb.ready):
        with st.spinner("Running local retrieval over the gold dataset..."):
            run = run_offline_retrieval_eval(svc.kb, svc.reranker, items, k=svc.settings.final_top_k)
        save_offline(run, svc.settings.logs_dir / "evals")
        st.session_state.offline_run = run
    run = st.session_state.get("offline_run")
    if not run:
        saved = sorted((svc.settings.logs_dir / "evals").glob("offline_retrieval_*.json")) if (svc.settings.logs_dir / "evals").exists() else []
        run = json.loads(saved[-1].read_text()) if saved else None
    if not run:
        empty_state("No offline evaluation yet.")
        return
    names = {"vector_only": "FAISS vector only", "bm25_only": "BM25 only", "hybrid_rrf": "Hybrid (RRF)", "hybrid_rrf_rerank": "Hybrid + cross-encoder"}
    html_table(["Retrieval variant", f"Precision@{run['k']}", f"Recall@{run['k']}", f"Hit@{run['k']}", "MRR"],
               [[names[v], f"{m['precision_at_k']:.3f}", f"{m['recall_at_k']:.3f}", f"{m['hit_rate_at_k']:.3f}", f"{m['mrr']:.3f}"] for v, m in run["aggregate"].items()])
    st.caption(f"{run['evaluated']} questions · {run['num_chunks']} chunks · {run['timestamp']} · excluded: " + ", ".join(f"{e['id']} ({e['reason'].split(' (')[0]})" for e in run["excluded"]))
