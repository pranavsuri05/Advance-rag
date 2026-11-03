"""Configurable character-window chunking with overlap and Markdown section tracking."""
from __future__ import annotations

import re

from pydantic import BaseModel

from src.ingestion.loader import LoadedDocument
from src.utils.text_utils import slugify

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    source: str            # source filename
    page: int | None = None
    chunk_index: int
    text: str
    section: str | None = None


def doc_id_for(filename: str) -> str:
    return slugify(filename.rsplit(".", 1)[0])


def split_text(text: str, size: int = 800, overlap: int = 120) -> list[tuple[int, str]]:
    """Split into windows of ~`size` chars, preferring paragraph / sentence / word boundaries.

    Returns (start_offset, chunk_text) pairs. Consecutive chunks share ~`overlap` characters.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk_size must be > 0 and 0 <= overlap < chunk_size")
    text = text.strip()
    out: list[tuple[int, str]] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            floor = int(size * 0.5)
            for sep in ("\n\n", ". ", "\n", " "):
                cut = window.rfind(sep)
                if cut >= floor:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            out.append((start, piece))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_document(doc: LoadedDocument, size: int = 800, overlap: int = 120) -> list[Chunk]:
    doc_id = doc_id_for(doc.filename)
    doc_name = doc.filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    chunks: list[Chunk] = []
    for page in doc.pages:
        headings = [(m.start(), m.group(2).strip()) for m in _HEADING_RE.finditer(page.text)]
        for start, piece in split_text(page.text, size, overlap):
            section = None
            for pos, title in headings:
                if pos <= start:
                    section = title
            idx = len(chunks)
            chunks.append(Chunk(chunk_id=f"{doc_id}_chunk_{idx:03d}", doc_id=doc_id, doc_name=doc_name,
                                source=doc.filename, page=page.page, chunk_index=idx, text=piece, section=section))
    return chunks
