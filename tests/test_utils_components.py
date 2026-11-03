import pytest
from streamlit.testing.v1 import AppTest

from src.utils.text_utils import estimate_tokens, extract_json, slugify, tokenize, tokenize_no_stop


@pytest.mark.parametrize("raw,expected", [('{"a": 1}', {"a": 1}), ('```json\n{"a": 1}\n```', {"a": 1}), ('Sure! Here: {"a": {"b": [1, 2]}} done', {"a": {"b": [1, 2]}}),
                                          ('```\n{"a": "x"}\n```', {"a": "x"})])
def test_extract_json_accepts_fences_and_surrounding_prose(raw, expected):
    assert extract_json(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "no json here", "[1, 2]", '{"a": ', "{'single': 'quotes'}", "null", "123"])
def test_extract_json_rejects_non_objects_and_garbage(raw):
    assert extract_json(raw) is None


def test_text_helpers():
    assert tokenize("Hello, World! 3.14 café") == ["hello", "world", "3", "14", "café"]
    assert tokenize_no_stop("the cat is on a mat") == ["cat", "mat"]
    assert slugify("My File (v2).MD") == "my-file-v2-md" and slugify("###") == "doc"
    assert estimate_tokens("") == 0 and estimate_tokens("abcd" * 10) == 10 and estimate_tokens("a") == 1


def _render_components():
    import streamlit as st
    from ui import components as C
    C.html_table(["<b>H</b>"], [["<script>alert(1)</script>", "a & b"]])
    C.metric_cards([("<i>label</i>", "<u>v</u>", "<s>sub</s>")])
    C.evidence("<img src=x onerror=alert(1)>")
    C.empty_state("<script>x</script>")
    C.bar_chart("<b>t</b>", {"<k>": 3.0})
    C.stage_list([{"name": "<n>", "status": "success", "detail": "<d>", "duration_ms": 5}])
    st.markdown("done")


def test_components_escape_untrusted_html_everywhere():
    at = AppTest.from_function(_render_components, default_timeout=20).run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    for raw in ("<script>", "<img", "<b>H</b>", "<i>label</i>", "<k>", "<n>", "<d>"):
        assert raw not in html, raw
    assert "&lt;script&gt;" in html and "&lt;img src=x onerror=alert(1)&gt;" in html and "a &amp; b" in html


def test_charts_handle_empty_and_degenerate_data():
    def render():
        from ui import components as C
        C.bar_chart("t", {})
        C.line_chart("t", [])
        C.line_chart("t", [0.0])
        C.line_chart("t", [5.0, 5.0, 5.0])
        C.bar_chart("t", {"a": 0.0})
    at = AppTest.from_function(render, default_timeout=20).run()
    assert not at.exception


def test_tokenizer_and_bm25_handle_non_ascii_text(settings):
    from src.ingestion.indexer import KnowledgeBase
    from tests.conftest import HashEmbedder
    assert tokenize("Résumé naïve Über_straße 日本語 text") == ["résumé", "naïve", "über", "straße", "日本語", "text"]
    kb = KnowledgeBase(settings, HashEmbedder())
    for name, text in {"fr.md": "Le café français sert un croissant délicieux.", "de.md": "Die Straßenbahn fährt durch Köln.", "en.md": "Plain english about databases."}.items():
        (settings.documents_dir.parent / name).write_text(text, encoding="utf-8")
        kb.add_file(settings.documents_dir.parent / name)
    top = lambda q: kb.chunk_map()[kb.bm25.search(q, 1)[0][0]].source
    assert top("café croissant") == "fr.md" and top("Straßenbahn Köln") == "de.md"      # previously: "caf" / "stra" fragments


def test_exact_match_and_f1_do_not_collapse_non_ascii_answers():
    from src.evaluation import scoring as sc
    assert sc.exact_match("日本語", "中文") == 0.0 and sc.exact_match("Café!", "café") == 1.0
    assert sc.token_f1("日本語 テキスト", "日本語 データ") == pytest.approx(0.5) and sc.normalize("A_b-C") == "a b c"
