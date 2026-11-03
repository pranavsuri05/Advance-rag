"""Automated Streamlit tests (streamlit.testing AppTest) for all six pages. Mock LLM only; no browser, no network."""
import json

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.llm.provider import LLMError
from tests import appharness as H
from tests.conftest import default_handler
from tests.pdfgen import make_pdf


def open_page(name, svc, session_state=None, timeout=30):
    H.SVC = svc
    at = AppTest.from_function(H.page_script, args=(name,), default_timeout=timeout)
    for k, v in (session_state or {}).items():
        at.session_state[k] = v
    return at.run()


def texts(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [h.value for h in at.subheader] + [t.value for t in at.title] + [h.value for h in at.header])


def btn(at, label):
    return next(b for b in at.button if b.label == label)


def no_exc(at):
    assert not at.exception, [e.value for e in at.exception]


@pytest.fixture
def svc(tmp_path):
    return H.make_services(tmp_path)


# =============================== CHAT ===============================
def test_chat_empty_state_shows_status_samples_and_no_errors(svc):
    at = open_page("chat", svc)
    no_exc(at)
    t = texts(at)
    assert "Advanced RAG Studio" in t and "Documents" in t.title() or "DOCUMENTS" in t.upper()
    assert len(at.button) >= 4 and at.chat_input[0].disabled is False and len(at.chat_message) == 0 and not at.error


def test_chat_sample_question_runs_pipeline_and_renders_answer_citations_and_trace(svc):
    at = open_page("chat", svc)
    at.button[0].click().run()
    no_exc(at)
    roles = [m.name for m in at.chat_message]
    assert roles == ["user", "assistant"]
    assert "RRF sums reciprocal ranks" in at.chat_message[1].markdown[0].value
    labels = [e.label for e in at.expander]
    assert any("Retrieval & Reasoning Trace" in l for l in labels) and any(".md" in l and "chunk" in l.lower() for l in labels)   # citation panel
    sid = at.session_state["session_id"]
    msgs = svc.sessions.messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["sources"] and msgs[1]["metadata"]["stages"]


def test_chat_typed_followup_uses_session_memory(svc):
    at = open_page("chat", svc)
    at.chat_input[0].set_value("What is reciprocal rank fusion?").run()
    at.chat_input[0].set_value("and its constant?").run()
    no_exc(at)
    assert len(at.chat_message) == 4
    rewrite_prompts = [c["prompt"] for c in svc.provider.calls if "query-understanding" in (c["system"] or "")]
    assert len(rewrite_prompts) == 2 and "What is reciprocal rank fusion?" in rewrite_prompts[1]


def test_chat_abstention_and_corrective_pills_render(tmp_path):
    def h(prompt, system):
        if "evidence grader" in (system or ""):
            return json.dumps({"relevant": False, "score": 0.0, "missing_information": "m", "explanation": "e"})
        if "improve failed retrieval" in (system or ""):
            return json.dumps({"query": "other words", "alternatives": [], "reason": "r"})
        return default_handler(prompt, system)
    svc = H.make_services(tmp_path, handler=h)
    at = open_page("chat", svc)
    at.chat_input[0].set_value("something unrelated").run()
    no_exc(at)
    t = texts(at)
    assert "Abstained" in t and "Corrective retry performed" in t and not at.error
    assert svc.sessions.messages(at.session_state["session_id"])[-1]["metadata"]["abstained"] is True


def test_chat_missing_api_key_disables_input_and_shows_clear_error(tmp_path):
    svc = H.make_services(tmp_path, llm=False)
    at = open_page("chat", svc)
    no_exc(at)
    assert at.chat_input[0].disabled is True and any("API key" in e.value for e in at.error)
    at.button[0].click().run()                               # clicking a sample must not crash either
    no_exc(at)
    assert any("API key" in e.value for e in at.error) and len(svc.provider.calls) == 0


@pytest.mark.parametrize("kind,needle", [("quota", "quota"), ("auth", "key was rejected"), ("server", "temporary error"), ("network", "Could not reach"),
                                         ("model_not_found", "model is not available")])
def test_chat_llm_errors_shown_as_friendly_messages_without_stack_traces(tmp_path, kind, needle):
    svc = H.make_services(tmp_path, handler=lambda p, s: LLMError(kind, "raw provider text 429 stack"))
    at = open_page("chat", svc)
    at.chat_input[0].set_value("anything about rrf").run()
    no_exc(at)
    assert any(needle in e.value for e in at.error), [e.value for e in at.error]
    assert not any("Traceback" in e.value for e in at.error)


def test_chat_empty_knowledge_base_warning_and_error(tmp_path):
    svc = H.make_services(tmp_path, docs=False)
    at = open_page("chat", svc)
    assert any("knowledge base is empty" in w.value.lower() for w in at.warning)
    at.chat_input[0].set_value("hello").run()
    assert any("empty" in e.value.lower() for e in at.error) and svc.provider.calls == []


def test_chat_clear_button_resets_conversation(svc):
    at = open_page("chat", svc)
    at.chat_input[0].set_value("what is bm25").run()
    assert len(at.chat_message) == 2
    btn(at, "Clear chat").click().run()
    assert len(at.chat_message) == 0 and svc.sessions.messages(at.session_state["session_id"]) == []


def test_chat_survives_corrupt_session_file(svc):
    sid = svc.sessions.create()
    (svc.settings.sessions_dir / f"{sid}.json").write_text("{corrupt")
    at = open_page("chat", svc, {"session_id": sid})
    no_exc(at)


# ============================ KNOWLEDGE BASE ============================
def test_kb_page_lists_documents_counts_and_status(svc):
    at = open_page("kb", svc)
    no_exc(at)
    t = texts(at)
    assert all(n in t for n in ("rrf.md", "bm25.md", "faiss.md")) and "Ready" in t and "PDF" in t
    assert "3" in t and str(len(svc.kb.chunks)) in t


def test_kb_search_filter_and_no_match_empty_state(svc):
    at = open_page("kb", svc)
    at.text_input[0].set_value("bm25").run()
    t = texts(at)
    assert "bm25.md" in t and "faiss.md" not in t
    at.text_input[0].set_value("zzzz").run()
    assert "No documents match" in texts(at)


def test_kb_empty_state_when_no_documents(tmp_path):
    at = open_page("kb", H.make_services(tmp_path, docs=False))
    no_exc(at)
    assert "No documents indexed yet" in texts(at) and "Empty" in texts(at)


def test_kb_upload_valid_pdf_indexes_it_with_page_metadata(svc):
    at = open_page("kb", svc)
    at.file_uploader[0].set_value(("paper.pdf", make_pdf(["Page one about apples.", "Page two about pears."]), "application/pdf")).run()
    btn(at, "Index uploaded files").click().run()
    no_exc(at)
    assert "paper" in svc.kb.documents and {c.page for c in svc.kb.chunks if c.doc_id == "paper"} == {1, 2}
    assert svc.kb.vector_store.index.ntotal == len(svc.kb.chunks) == len(svc.kb.bm25.ids)


@pytest.mark.parametrize("name,content,needle", [("broken.pdf", b"not a pdf", "PDF"), ("empty.txt", b"", "empty"), ("blank.md", b"   \n ", "empty")])
def test_kb_upload_validation_errors_are_shown_and_index_untouched(svc, name, content, needle):
    before = len(svc.kb.chunks)
    at = open_page("kb", svc)
    at.file_uploader[0].set_value((name, content, "application/octet-stream")).run()
    btn(at, "Index uploaded files").click().run()
    no_exc(at)
    assert any(needle.lower() in e.value.lower() for e in at.error), [e.value for e in at.error]
    assert len(svc.kb.chunks) == before and name.rsplit(".", 1)[0] not in svc.kb.documents


def test_kb_unsupported_file_type_is_blocked_by_the_uploader_itself(svc):
    before = len(svc.kb.chunks)
    at = open_page("kb", svc)
    assert at.file_uploader[0].allowed_type and "docx" not in [t.lstrip(".") for t in at.file_uploader[0].allowed_type]
    at.file_uploader[0].set_value(("notes.docx", b"x", "application/vnd.openxmlformats")).run()
    # AppTest surfaces Streamlit's own uploader validation as an exception element; a real browser simply refuses the file.
    assert [e.value for e in at.exception] and all("Invalid file extension" in e.value for e in at.exception)
    assert len(svc.kb.chunks) == before and not any(m.value.startswith("Indexed") for m in at.success)


def test_kb_duplicate_upload_rejected(svc):
    at = open_page("kb", svc)
    at.file_uploader[0].set_value(("copy.md", next(iter(H.DOCS.values())).encode(), "text/markdown")).run()
    btn(at, "Index uploaded files").click().run()
    assert any("Duplicate" in e.value for e in at.error) and "copy" not in svc.kb.documents


def test_kb_delete_document_keeps_indexes_consistent(svc):
    at = open_page("kb", svc)
    at.selectbox[0].select("bm25")
    btn(at, "Delete document").click().run()
    no_exc(at)
    assert "bm25" not in svc.kb.documents and all(c.doc_id != "bm25" for c in svc.kb.chunks)
    assert svc.kb.vector_store.index.ntotal == len(svc.kb.chunks) == len(svc.kb.bm25.ids)
    assert not (svc.settings.documents_dir / "bm25.md").exists()


def test_kb_rebuild_button_rebuilds_consistently(svc):
    n = len(svc.kb.chunks)
    at = open_page("kb", svc)
    btn(at, "Rebuild index").click().run()
    no_exc(at)
    assert len(svc.kb.chunks) == n and svc.kb.vector_store.index.ntotal == n


# ========================== RETRIEVAL INSPECTOR ==========================
def test_inspector_empty_state_then_full_stage_sections_with_real_scores(svc):
    at = open_page("inspector", svc)
    no_exc(at)
    assert "Enter a query" in texts(at)
    at.text_input[0].set_value("reciprocal rank fusion constant").run()
    btn(at, "Inspect retrieval").click().run()
    no_exc(at)
    t = texts(at)
    for section in ("Query transformation", "FAISS vector results", "BM25 results", "RRF fused results", "Cross-encoder reranking",
                    "Corrective RAG history", "Final context for generation", "Stage log"):
        assert section in t, section
    assert "rrf.md" in t and "Initial retrieval" in " ".join(e.label for e in at.expander)
    assert not any("Evidence rules" in (c["system"] or "") for c in svc.provider.calls)        # inspection never generates an answer


def test_inspector_empty_query_and_missing_key_errors(svc, tmp_path):
    at = open_page("inspector", svc)
    at.text_input[0].set_value("").run()
    btn(at, "Inspect retrieval").click().run()
    assert any("enter a query" in e.value.lower() for e in at.error)
    at2 = open_page("inspector", H.make_services(tmp_path / "nokey", llm=False))
    btn(at2, "Inspect retrieval").click().run()
    no_exc(at2)
    assert any("API key" in e.value for e in at2.error)


def test_inspector_quota_error_is_friendly(tmp_path):
    svc = H.make_services(tmp_path, handler=lambda p, s: LLMError("quota", "429"))
    at = open_page("inspector", svc)
    btn(at, "Inspect retrieval").click().run()
    no_exc(at)
    assert any("quota" in e.value.lower() for e in at.error)


# ================================ EVALUATION ================================
def test_evaluation_page_offline_run_makes_no_llm_calls_and_shows_variants(svc):
    at = open_page("evaluation", svc)
    no_exc(at)
    assert "Offline retrieval evaluation" in texts(at) and "No offline evaluation yet" in texts(at)
    btn(at, "Run offline retrieval evaluation").click().run()
    no_exc(at)
    t = texts(at)
    assert "FAISS vector only" in t and "Hybrid + cross-encoder" in t and "excluded" in t
    assert svc.provider.calls == [] and list((svc.settings.logs_dir / "evals").glob("offline_retrieval_*.json"))


def test_evaluation_page_end_to_end_run_with_mock_llm(svc):
    at = open_page("evaluation", svc)
    at.number_input[0].set_value(2)
    at.checkbox[0].uncheck()                                       # judge off
    btn(at, "Run evaluation").click().run(timeout=60)
    no_exc(at)
    t = texts(at)
    assert "Per-question results" in t and "Failure analysis" in t and "Abstention accuracy" in t.title() or "Abstention accuracy" in t
    assert list((svc.settings.logs_dir / "evals").glob("eval_*.json"))
    assert not any("impartial evaluation judge" in (c["system"] or "") for c in svc.provider.calls)


def test_evaluation_page_without_key_disables_run_and_explains(tmp_path):
    at = open_page("evaluation", H.make_services(tmp_path, llm=False))
    no_exc(at)
    assert btn(at, "Run evaluation").disabled is True and any("API key" in e.value for e in at.error)


def test_evaluation_page_invalid_dataset_shows_error_not_traceback(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a list"}')
    at = open_page("evaluation", H.make_services(tmp_path, eval_dataset=bad))
    no_exc(at)
    assert any("Could not load dataset" in e.value for e in at.error)


# ================================ OBSERVABILITY ================================
def test_observability_empty_state(svc):
    at = open_page("observability", svc)
    no_exc(at)
    assert "No LLM calls recorded yet" in texts(at)


def test_observability_shows_real_calls_failures_diagnostics_and_survives_bad_log_lines(svc):
    svc.llm.generate("p", "query_rewrite", session_id="s1")
    svc.llm.generate("p2", "answer_generation", session_id="s1")
    svc.provider.handler = lambda p, s: LLMError("quota", "429 exhausted")
    with pytest.raises(LLMError):
        svc.llm.generate("p3", "context_grading")
    with svc.tracer.path.open("a") as fh:
        fh.write('{"partial": \n[]\n{"status": "failure"}\n')       # malformed / incomplete lines
    at = open_page("observability", svc)
    no_exc(at)
    t = texts(at)
    assert "LLM calls" in t and "query_rewrite" in t and "context_grading" in t and "Calls by type" in t and "Recent calls" in t
    assert any("quota" in e.value for e in at.error)               # error detail shown inside the call expander
    assert "provider" in t                                         # token source column (provider vs estimated)


# ================================ SESSION MEMORY ================================
def test_memory_page_empty_then_transcript_context_export_clear_and_new_session(svc):
    sid = svc.sessions.create()
    at = open_page("memory", svc, {"session_id": sid})
    no_exc(at)
    assert sid in texts(at) and "No messages yet" in texts(at)
    svc.pipeline.ask("what is bm25", sid)
    at = open_page("memory", svc, {"session_id": sid})
    assert len(at.chat_message) == 2 and "what is bm25" in texts(at)
    assert len(at.tabs) == 4 and at.json                           # window + state dump rendered
    btn(at, "Clear session").click().run()
    assert svc.sessions.messages(sid) == [] and "No messages yet" in texts(at)
    btn(at, "New session").click().run()
    assert at.session_state["session_id"] != sid


def test_memory_pages_do_not_leak_between_sessions(svc):
    a, b = svc.sessions.create(), svc.sessions.create()
    svc.sessions.add_message(a, "user", "secret-of-A")
    at = open_page("memory", svc, {"session_id": b})
    assert "secret-of-A" not in texts(at) + " ".join(str(j.value) for j in at.json)


def test_memory_page_survives_corrupt_session(svc):
    sid = svc.sessions.create()
    (svc.settings.sessions_dir / f"{sid}.json").write_text("garbage")
    no_exc(open_page("memory", svc, {"session_id": sid}))


# =============================== APP SHELL & THEMES ===============================
@pytest.fixture
def app(tmp_path, monkeypatch):
    svc = H.make_services(tmp_path)
    H.SVC = svc
    import src.bootstrap as B
    monkeypatch.setattr(B, "build_services", lambda settings=None: svc)
    st.cache_resource.clear()
    from pathlib import Path
    app_path = str(Path(__file__).resolve().parent.parent / "app.py")

    def make(query=None):
        at = AppTest.from_file(app_path, default_timeout=60)
        if query:
            at.query_params["theme"] = query
        return at.run()
    make.svc = svc
    return make


def css(at):
    return next(m.value for m in at.markdown if "<style>" in m.value)


def test_app_default_page_boots_with_theme_selector_and_session(app):
    at = app()
    no_exc(at)
    assert [r.label for r in at.radio] == ["Appearance"] and at.radio[0].options == ["Auto", "Dark", "Light"] and at.radio[0].value == "Auto"
    assert at.session_state["session_id"] and "Groq" in " ".join(c.value for c in at.caption)


def test_app_registers_all_six_pages(app, monkeypatch):
    seen = {}
    orig = st.navigation

    def spy(pages, **kw):
        seen["pages"] = [(p.title, p.url_path) for p in pages]
        return orig(pages, **kw)
    monkeypatch.setattr(st, "navigation", spy)
    app()
    assert [t for t, _ in seen["pages"]] == ["Chat", "Knowledge Base", "Retrieval Inspector", "Evaluation", "Observability", "Session Memory"]
    assert len({u for _, u in seen["pages"]}) == 6


def test_theme_auto_uses_prefers_color_scheme_media_query_with_light_base(app):
    c = css(app())
    assert "@media (prefers-color-scheme: dark)" in c and "--bg:#F6F7FB" in c and "--bg:#0B0D12" in c and "color-scheme:light dark" in c


def test_theme_dark_is_explicit_override_without_media_query(app):
    c = css(app("Dark"))
    assert "--bg:#0B0D12" in c and "--text:#F1F3F9" in c and "color-scheme:dark" in c and "prefers-color-scheme" not in c and "#F6F7FB" not in c


def test_theme_light_is_explicit_override_without_media_query(app):
    c = css(app("Light"))
    assert "--bg:#F6F7FB" in c and "--accent:#7258D8" in c and "color-scheme:light;" in c and "prefers-color-scheme" not in c and "#0B0D12" not in c


def test_theme_switch_persists_in_state_and_query_params_across_reruns(app):
    at = app()
    at.radio[0].set_value("Dark").run()
    assert at.session_state["theme_mode"] == "Dark" and at.query_params["theme"] == "Dark" and "prefers-color-scheme" not in css(at)
    at.run()                                                        # an unrelated rerun keeps the choice
    assert at.session_state["theme_mode"] == "Dark" and "--bg:#0B0D12" in css(at)
    at.radio[0].set_value("Light").run()
    assert "--bg:#F6F7FB" in css(at) and at.query_params["theme"] == "Light"


def test_invalid_theme_query_param_falls_back_to_auto(app):
    at = app("<script>alert(1)</script>")
    no_exc(at)
    assert at.session_state["theme_mode"] == "Auto" and "<script>" not in css(at)


def test_css_uses_variables_and_has_responsive_rules(app):
    c = css(app())
    assert "@media (max-width: 720px)" in c and "var(--accent)" in c and "var(--bg)" in c
    import re
    base = c.split("[data-testid", 1)[1]
    assert not re.findall(r"(?<![-\w])#[0-9A-Fa-f]{6}\b", base), "page CSS must use theme variables, not hardcoded colours"


def test_evaluation_page_warns_about_dataset_problems(tmp_path):
    bad = tmp_path / "ds.json"
    bad.write_text(json.dumps([{"id": "a", "question": "q", "reference_answer": "r", "expected_documents": ["ghost.md"]}]))
    at = open_page("evaluation", H.make_services(tmp_path, eval_dataset=bad))
    no_exc(at)
    assert any("ghost.md" in w.value for w in at.warning)
