"""Knowledge base: owns documents, chunks, embeddings, the FAISS index and the BM25 corpus.

On-disk layout (data/indexes):
  documents.json   registry (doc_id -> metadata, content hash, chunk count, indexed_at)
  chunks.json      all chunk records, in index order
  embeddings.npy   normalised float32 matrix aligned with chunks.json
  faiss.index      FAISS IndexFlatIP over embeddings.npy
  index_meta.json  embedding model id, dim, chunk settings, last indexed time
  bm25_corpus.json tokenised corpus aligned with chunks.json
All of these are written together in `_persist`, so they cannot drift apart.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from src.config import Settings
from src.ingestion.chunker import Chunk, chunk_document, doc_id_for
from src.ingestion.loader import IngestionError, load_file
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import VectorStore
from src.utils.text_utils import tokenize


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder; the model is loaded lazily once."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        return np.asarray(vecs, dtype="float32")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KnowledgeBase:
    def __init__(self, settings: Settings, embedder: Embedder):
        self.settings = settings
        self.embedder = embedder
        settings.ensure_dirs()
        self.documents: dict[str, dict] = {}
        self.chunks: list[Chunk] = []
        self.embeddings = np.zeros((0, 0), dtype="float32")
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever()
        self.meta: dict = {}
        self.last_error: str | None = None
        self._load()

    # ---------- persistence ----------
    def _p(self, name: str) -> Path:
        return self.settings.index_dir / name

    def _load(self) -> None:
        try:
            if not self._p("chunks.json").exists():
                return
            meta = json.loads(self._p("index_meta.json").read_text())
            if meta.get("embedding_model") != self.embedder.model_name:
                self.last_error = (f"Index was built with '{meta.get('embedding_model')}' but the configured model is "
                                   f"'{self.embedder.model_name}'. Rebuild the index.")
                return
            self.documents = json.loads(self._p("documents.json").read_text())
            self.chunks = [Chunk(**c) for c in json.loads(self._p("chunks.json").read_text())]
            self.embeddings = np.load(self._p("embeddings.npy"))
            self.meta = meta
            self._rebuild_search_structures()
        except (OSError, ValueError, KeyError) as exc:
            self.documents, self.chunks, self.meta = {}, [], {}
            self.last_error = f"Stored index is unreadable ({exc.__class__.__name__}); rebuild the index."

    def _persist(self) -> None:
        d = self.settings.index_dir
        d.mkdir(parents=True, exist_ok=True)
        self.meta = {"embedding_model": self.embedder.model_name,
                     "dimension": int(self.embeddings.shape[1]) if self.embeddings.size else 0,
                     "chunk_size": self.settings.chunk_size, "chunk_overlap": self.settings.chunk_overlap,
                     "num_chunks": len(self.chunks), "num_documents": len(self.documents), "last_indexed": _now()}
        self._p("documents.json").write_text(json.dumps(self.documents, indent=2))
        self._p("chunks.json").write_text(json.dumps([c.model_dump() for c in self.chunks]))
        np.save(self._p("embeddings.npy"), self.embeddings)
        self.vector_store.save(self._p("faiss.index"))
        self._p("bm25_corpus.json").write_text(json.dumps([tokenize(c.text) for c in self.chunks]))
        self._p("index_meta.json").write_text(json.dumps(self.meta, indent=2))

    def _rebuild_search_structures(self) -> None:
        self.vector_store = VectorStore()
        if len(self.chunks):
            self.vector_store.build(self.embeddings, [c.chunk_id for c in self.chunks])
        self.bm25 = BM25Retriever()
        self.bm25.build(self.chunks)

    # ---------- queries ----------
    @property
    def ready(self) -> bool:
        return len(self.chunks) > 0

    def chunk_map(self) -> dict[str, Chunk]:
        return {c.chunk_id: c for c in self.chunks}

    def status(self) -> dict:
        return {"documents": len(self.documents), "chunks": len(self.chunks), "ready": self.ready,
                "last_indexed": self.meta.get("last_indexed"), "embedding_model": self.embedder.model_name,
                "error": self.last_error}

    # ---------- mutation ----------
    def add_file(self, path: Path, progress=None) -> dict:
        """Copy a file into the documents dir (if needed), chunk, embed and index it.

        Raises IngestionError on bad input, including duplicates. Nothing is persisted on failure.
        """
        path = Path(path)
        loaded = load_file(path)
        for info in self.documents.values():
            if info["content_hash"] == loaded.content_hash:
                raise IngestionError(f"Duplicate: '{loaded.filename}' has identical content to '{info['filename']}'.")
        doc_id = doc_id_for(loaded.filename)
        if doc_id in self.documents:
            raise IngestionError(f"A different document named '{loaded.filename}' is already indexed. "
                                 "Rename the file or delete the existing one first.")
        chunks = chunk_document(loaded, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise IngestionError(f"'{loaded.filename}' produced no chunks.")
        if progress: progress(0.3, f"Embedding {len(chunks)} chunks")
        try:
            vecs = self.embedder.encode([c.text for c in chunks])
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"Embedding failed for '{loaded.filename}': {exc}") from exc
        dest = self.settings.documents_dir / loaded.filename
        if dest.resolve() != path.resolve():
            shutil.copyfile(path, dest)
        self.chunks.extend(chunks)
        self.embeddings = vecs if not self.embeddings.size else np.vstack([self.embeddings, vecs])
        self.documents[doc_id] = {"doc_id": doc_id, "filename": loaded.filename, "size_bytes": loaded.size_bytes,
                                  "content_hash": loaded.content_hash, "num_chunks": len(chunks),
                                  "pages": len(loaded.pages), "file_type": loaded.file_type, "indexed_at": _now()}
        if progress: progress(0.8, "Building FAISS + BM25 indexes")
        self._rebuild_search_structures()
        self._persist()
        self.last_error = None
        if progress: progress(1.0, "Done")
        return self.documents[doc_id]

    def remove_document(self, doc_id: str) -> bool:
        info = self.documents.pop(doc_id, None)
        if info is None:
            return False
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.embeddings = self.embeddings[keep] if keep else np.zeros((0, 0), dtype="float32")
        (self.settings.documents_dir / info["filename"]).unlink(missing_ok=True)
        self._rebuild_search_structures()
        self._persist()
        return True

    def rebuild(self, progress=None) -> dict:
        """Re-read every file in the documents dir, re-chunk, and re-embed only text not seen before."""
        cache = {c.text: self.embeddings[i] for i, c in enumerate(self.chunks)} if self.embeddings.size else {}
        docs, chunks, errors = {}, [], []
        files = sorted(p for p in self.settings.documents_dir.iterdir() if p.is_file() and not p.name.startswith("."))
        seen_hashes: set[str] = set()
        for n, path in enumerate(files):
            if progress: progress(n / max(len(files), 1) * 0.7, f"Reading {path.name}")
            try:
                loaded = load_file(path)
            except IngestionError as exc:
                errors.append(str(exc)); continue
            if loaded.content_hash in seen_hashes:
                errors.append(f"Skipped duplicate content: {path.name}"); continue
            seen_hashes.add(loaded.content_hash)
            doc_chunks = chunk_document(loaded, self.settings.chunk_size, self.settings.chunk_overlap)
            if not doc_chunks:
                continue
            doc_id = doc_id_for(path.name)
            docs[doc_id] = {"doc_id": doc_id, "filename": path.name, "size_bytes": loaded.size_bytes,
                            "content_hash": loaded.content_hash, "num_chunks": len(doc_chunks),
                            "pages": len(loaded.pages), "file_type": loaded.file_type,
                            "indexed_at": self.documents.get(doc_id, {}).get("indexed_at", _now())}
            chunks.extend(doc_chunks)
        vecs = np.zeros((len(chunks), 0), dtype="float32")
        if chunks:
            missing = [i for i, c in enumerate(chunks) if c.text not in cache]
            if progress: progress(0.8, f"Embedding {len(missing)} new / {len(chunks)} total chunks")
            new = self.embedder.encode([chunks[i].text for i in missing]) if missing else None
            dim = new.shape[1] if new is not None else len(next(iter(cache.values())))
            vecs = np.zeros((len(chunks), dim), dtype="float32")
            for i, c in enumerate(chunks):
                if c.text in cache:
                    vecs[i] = cache[c.text]
            for j, i in enumerate(missing):
                vecs[i] = new[j]
        self.documents, self.chunks, self.embeddings = docs, chunks, (vecs if chunks else np.zeros((0, 0), dtype="float32"))
        self._rebuild_search_structures()
        self._persist()
        self.last_error = "; ".join(errors) if errors else None
        if progress: progress(1.0, "Done")
        return {"documents": len(docs), "chunks": len(chunks), "errors": errors}
