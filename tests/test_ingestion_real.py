"""Ingestion tests against genuine PDFs and edge-case files (real pypdf extraction, real chunker, real KnowledgeBase)."""
import re

import pytest

from src.ingestion.chunker import Chunk, chunk_document, doc_id_for
from src.ingestion.indexer import KnowledgeBase
from src.ingestion.loader import IngestionError, load_file
from src.utils.citations import validate_citations
from tests.conftest import HashEmbedder
from tests.pdfgen import make_pdf

P1 = "Reciprocal Rank Fusion merges ranked lists.\nThe constant k defaults to 60 in the original paper."
P2 = "BM25 scores documents with term frequency.\nSaturation parameter k1 controls repeated terms."
P3 = "Cross-encoders read query and passage together.\nZebra-unique-token appears only on page three."


def norm(t):
    return re.sub(r"\s+", " ", t).strip()


def write_pdf(tmp_path, name, pages):
    p = tmp_path / name
    p.write_bytes(make_pdf(pages))
    return p


def test_genuine_pdf_text_extraction_matches_source(tmp_path):
    doc = load_file(write_pdf(tmp_path, "one.pdf", [P1]))
    assert doc.file_type == "pdf" and len(doc.pages) == 1 and doc.pages[0].page == 1
    assert norm(doc.pages[0].text) == norm(P1)


def test_multipage_pdf_page_numbers_preserved_through_chunks(tmp_path):
    doc = load_file(write_pdf(tmp_path, "multi.pdf", [P1, P2, P3]))
    assert [p.page for p in doc.pages] == [1, 2, 3]
    assert [norm(p.text) for p in doc.pages] == [norm(P1), norm(P2), norm(P3)]
    chunks = chunk_document(doc, 120, 20)
    assert {c.page for c in chunks} == {1, 2, 3}
    for c in chunks:                                  # every chunk's page number matches where its text came from
        assert ("Zebra" in c.text) <= (c.page == 3)
        assert ("saturation" in c.text.lower()) <= (c.page == 2)
    assert all(c.source == "multi.pdf" and c.doc_id == "multi" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_pdf_chunks_support_page_citations(tmp_path):
    doc = load_file(write_pdf(tmp_path, "cite.pdf", [P1, P2]))
    chunks = chunk_document(doc, 200, 20)
    index = {c.chunk_id: {"source": c.source, "page": c.page} for c in chunks}
    page2 = next(c for c in chunks if c.page == 2)
    rep = validate_citations(f"Claim [Source: cite.pdf, Chunk: {page2.chunk_id}].", index)
    assert rep.valid[0].page == 2 and "Page: 2" in rep.text


def test_multi_pdf_ingestion_into_knowledge_base(settings, tmp_path):
    kb = KnowledgeBase(settings, HashEmbedder())
    a = write_pdf(tmp_path, "alpha.pdf", [P1, P2])
    b = write_pdf(tmp_path, "beta.pdf", [P3])
    ia, ib = kb.add_file(a), kb.add_file(b)
    assert (ia["pages"], ib["pages"]) == (2, 1) and ia["file_type"] == "pdf"
    assert set(kb.documents) == {"alpha", "beta"}
    assert {c.source for c in kb.chunks} == {"alpha.pdf", "beta.pdf"}
    assert kb.vector_store.index.ntotal == len(kb.chunks) == len(kb.bm25.ids)
    hits = kb.bm25.search("zebra unique token", 3)
    assert kb.chunk_map()[hits[0][0]].source == "beta.pdf" and kb.chunk_map()[hits[0][0]].page == 1


def test_pdf_latin1_special_characters(tmp_path):
    doc = load_file(write_pdf(tmp_path, "accents.pdf", ["Café résumé naïve über (parentheses) back\\slash"]))
    assert "Café" in doc.pages[0].text and "über" in doc.pages[0].text
    assert "(parentheses)" in doc.pages[0].text


def test_uppercase_extension_and_pdf_with_blank_page_between(tmp_path):
    doc = load_file(write_pdf(tmp_path, "UP.PDF", [P1, "", P2]))
    assert [p.page for p in doc.pages] == [1, 3]       # blank page skipped, real page numbers kept


def test_blank_and_corrupt_and_truncated_pdfs(tmp_path):
    (tmp_path / "blank.pdf").write_bytes(make_pdf([""]))
    with pytest.raises(IngestionError, match="No extractable text"):
        load_file(tmp_path / "blank.pdf")
    (tmp_path / "junk.pdf").write_bytes(b"this is not a pdf at all")
    (tmp_path / "trunc.pdf").write_bytes(make_pdf([P1])[:60])
    (tmp_path / "zero.pdf").write_bytes(b"")
    for name in ("junk.pdf", "trunc.pdf", "zero.pdf"):
        with pytest.raises(IngestionError):
            load_file(tmp_path / name)


@pytest.mark.parametrize("name", ["a.docx", "a.csv", "a.exe", "noext", "a.pdf.exe", "a.PNG"])
def test_unsupported_formats(tmp_path, name):
    (tmp_path / name).write_bytes(b"content")
    with pytest.raises(IngestionError, match="Unsupported"):
        load_file(tmp_path / name)


def test_missing_file_and_directory_and_unreadable_input(tmp_path):
    with pytest.raises(IngestionError):
        load_file(tmp_path / "nope.md")
    d = tmp_path / "folder.md"
    d.mkdir()
    with pytest.raises(IngestionError):       # a directory named like a document must not crash with a raw OSError
        load_file(d)


def test_unicode_and_special_characters_roundtrip(tmp_path):
    text = "# Überschrift\n\nCafé – naïve 日本語のテキスト 🚀 emoji. Quotes “curly” and symbols <>&%$#@!\n"
    p = tmp_path / "uni.md"
    p.write_text(text, encoding="utf-8")
    chunks = chunk_document(load_file(p), 800, 120)
    assert len(chunks) == 1 and "日本語" in chunks[0].text and "🚀" in chunks[0].text and chunks[0].section == "Überschrift"


def test_invalid_utf8_bytes_do_not_crash(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_bytes(b"valid text \xff\xfe broken bytes but still readable")
    assert "valid text" in load_file(p).pages[0].text


def test_very_short_documents(tmp_path, settings):
    p = tmp_path / "tiny.txt"
    p.write_text("Hi")
    chunks = chunk_document(load_file(p), 800, 120)
    assert len(chunks) == 1 and chunks[0].text == "Hi"
    kb = KnowledgeBase(settings, HashEmbedder())
    assert kb.add_file(p)["num_chunks"] == 1


def test_filename_edge_cases_produce_valid_unique_ids(tmp_path):
    assert doc_id_for("My Report (final).v2.md") == "my-report-final-v2"
    assert doc_id_for("日本語.md") == "doc"            # no ASCII letters: safe fallback, never empty
    assert doc_id_for("!!!.txt") == "doc"


def test_doc_id_collision_between_different_files_is_rejected(settings, tmp_path):
    kb = KnowledgeBase(settings, HashEmbedder())
    (tmp_path / "a b.txt").write_text("first document about alpha")
    (tmp_path / "a-b.txt").write_text("second different document about beta")
    kb.add_file(tmp_path / "a b.txt")
    with pytest.raises(IngestionError, match="already indexed"):
        kb.add_file(tmp_path / "a-b.txt")
    assert len(kb.documents) == 1


def test_reingestion_after_delete_and_same_name_changed_content(settings, tmp_path):
    kb = KnowledgeBase(settings, HashEmbedder())
    f = tmp_path / "doc.md"
    f.write_text("# One\n\nOriginal content about fusion.")
    first_ids = [c.chunk_id for c in (kb.add_file(f), kb.chunks)[1]]
    f.write_text("# One\n\nCHANGED content about fusion.")
    with pytest.raises(IngestionError, match="already indexed"):   # same name, different content: explicit, not silent
        kb.add_file(f)
    assert kb.remove_document("doc") and not kb.ready
    kb.add_file(f)                                                   # re-ingestion after delete works
    assert kb.ready and "CHANGED" in kb.chunks[0].text and [c.chunk_id for c in kb.chunks] == first_ids


def test_real_corpus_chunking_invariants(corpus_dir):
    size, overlap = 800, 120
    for path in sorted(corpus_dir.glob("*.md")):
        doc = load_file(path)
        chunks = chunk_document(doc, size, overlap)
        assert chunks and all(c.text.strip() and len(c.text) <= size for c in chunks), path.name
        src = norm(doc.pages[0].text)
        for a, b in zip(chunks, chunks[1:]):                       # consecutive chunks share overlap text
            tail = norm(a.text)[-40:]
            assert tail.split()[-1] in norm(b.text) or norm(b.text)[:40] in norm(a.text), (path.name, a.chunk_id)
        assert all(norm(c.text) in src for c in chunks)            # every chunk is verbatim source text
        covered = " ".join(norm(c.text) for c in chunks)
        assert all(w in covered for w in src.split()[::25])        # nothing dropped between chunks


def test_malformed_chunk_metadata_rejected():
    with pytest.raises(Exception):
        Chunk(chunk_id="x")           # missing required fields must fail validation, not create a half-empty chunk
