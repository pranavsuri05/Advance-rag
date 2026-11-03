import pytest

from src.ingestion.chunker import chunk_document, split_text
from src.ingestion.indexer import KnowledgeBase
from src.ingestion.loader import IngestionError, load_file
from tests.conftest import HashEmbedder


def test_load_txt_and_markdown(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.md").write_text("# T\n\ntext")
    assert load_file(tmp_path / "a.txt").pages[0].text == "hello world"
    assert load_file(tmp_path / "b.md").file_type == "md"


def test_empty_and_unsupported(tmp_path):
    (tmp_path / "e.txt").write_text("   ")
    (tmp_path / "x.docx").write_text("hi")
    with pytest.raises(IngestionError, match="empty"):
        load_file(tmp_path / "e.txt")
    with pytest.raises(IngestionError, match="Unsupported"):
        load_file(tmp_path / "x.docx")


def test_corrupted_pdf(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 this is not a real pdf")
    with pytest.raises(IngestionError):
        load_file(tmp_path / "bad.pdf")


def test_pdf_without_text_rejected(tmp_path):
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(200, 200)
    with open(tmp_path / "blank.pdf", "wb") as fh:
        w.write(fh)
    with pytest.raises(IngestionError, match="No extractable text"):
        load_file(tmp_path / "blank.pdf")


def test_chunk_size_overlap_and_no_empty():
    text = " ".join(f"word{i}" for i in range(400))
    parts = split_text(text, 200, 40)
    assert len(parts) > 3
    assert all(p[1].strip() for p in parts)
    assert all(len(p[1]) <= 200 for p in parts)
    for (_, a), (_, b) in zip(parts, parts[1:]):
        assert set(a.split()[-3:]) & set(b.split()[:10])  # overlap shares words


def test_invalid_chunk_params():
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)


def test_metadata_and_sections(tmp_path):
    p = tmp_path / "My_Doc.md"
    p.write_text("# Intro\n\n" + "alpha " * 100 + "\n\n## Details\n\n" + "beta " * 100)
    chunks = chunk_document(load_file(p), 250, 30)
    assert chunks[0].chunk_id == "my-doc_chunk_000"
    assert chunks[0].source == "My_Doc.md" and chunks[0].section == "Intro"
    assert any(c.section == "Details" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_duplicate_and_consistency(kb, settings, tmp_path):
    n = len(kb.chunks)
    dup = tmp_path / "copy.md"
    dup.write_text((settings.documents_dir / "rrf.md").read_text())
    with pytest.raises(IngestionError, match="Duplicate"):
        kb.add_file(dup)
    assert len(kb.chunks) == n
    assert kb.vector_store.index.ntotal == len(kb.chunks) == len(kb.bm25.ids) == len(kb.embeddings)


def test_remove_rebuild_and_persistence(kb, settings):
    assert kb.remove_document("bm25")
    assert all(c.doc_id != "bm25" for c in kb.chunks)
    assert kb.vector_store.index.ntotal == len(kb.chunks) == len(kb.bm25.ids)
    reloaded = KnowledgeBase(settings, HashEmbedder())
    assert len(reloaded.chunks) == len(kb.chunks) and reloaded.ready
    info = reloaded.rebuild()
    assert info["documents"] == 2 and reloaded.vector_store.index.ntotal == info["chunks"]


def test_embedding_model_mismatch_detected(kb, settings):
    class Other(HashEmbedder):
        model_name = "other/model"
    k2 = KnowledgeBase(settings, Other())
    assert not k2.ready and "Rebuild" in k2.last_error
