"""File loading and text extraction for PDF, TXT and Markdown."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".markdown")


class IngestionError(Exception):
    """Raised when a file cannot be turned into indexable text (message is user-safe)."""


@dataclass
class Page:
    text: str
    page: int | None = None


@dataclass
class LoadedDocument:
    filename: str
    pages: list[Page]
    size_bytes: int
    content_hash: str
    file_type: str


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_file(path: Path) -> LoadedDocument:
    """Extract text from a file; raises IngestionError for unsupported/empty/corrupt input."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type '{ext or 'none'}'. Supported: PDF, TXT, MD.")
    if not path.exists():
        raise IngestionError(f"File not found: {path.name}")
    try:
        data = path.read_bytes()
    except OSError as exc:   # directory, permission denied, vanished file...
        raise IngestionError(f"'{path.name}' could not be read ({exc.__class__.__name__}).") from exc
    if not data.strip():
        raise IngestionError(f"'{path.name}' is empty.")
    if ext == ".pdf":
        pages = _load_pdf(path)
    else:
        pages = [Page(text=data.decode("utf-8", errors="replace"))]
    pages = [p for p in pages if p.text.strip()]
    if not pages:
        raise IngestionError(f"No extractable text found in '{path.name}' (scanned or empty document?).")
    return LoadedDocument(path.name, pages, len(data), file_hash(data), ext.lstrip("."))


def _load_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(str(path))
        return [Page(text=(pg.extract_text() or ""), page=i + 1) for i, pg in enumerate(reader.pages)]
    except (PyPdfError, ValueError, OSError, KeyError, TypeError) as exc:
        raise IngestionError(f"'{path.name}' could not be read as a PDF (corrupted or encrypted).") from exc
