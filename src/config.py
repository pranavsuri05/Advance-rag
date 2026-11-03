"""Central configuration. Values come from environment variables / .env with safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _int_env(*names: str, default: int | None = None, minimum: int | None = None) -> int | None:
    """Integer from the first set env var among `names`, with a clear error for bad values (instead of a bare traceback)."""
    for name in names:
        raw = os.getenv(name, "").strip()
        if raw:
            try:
                value = int(raw)
            except ValueError:
                raise ValueError(f"{name} must be an integer (got '{raw}').") from None
            if minimum is not None and value < minimum:
                raise ValueError(f"{name} must be >= {minimum} (got {value}).")
            return value
    return default


@dataclass
class Settings:
    """Runtime settings. Override any field in tests by passing keyword arguments."""

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").strip().lower() or "groq")
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip() or "qwen/qwen3.8-27b")
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash")
    # Gemini generation controls. Thinking tokens count toward output tokens, so cap both deliberately.
    # Set at most one of thinking level (Gemini 3 style) or thinking budget (2.5 style); unset = model default.
    gemini_thinking_level: str | None = field(default_factory=lambda: (os.getenv("GEMINI_THINKING_LEVEL", "").strip().lower() or None))
    gemini_thinking_budget: int | None = field(default_factory=lambda: _int_env("GEMINI_THINKING_BUDGET", minimum=0))
    max_output_tokens: int = field(default_factory=lambda: _int_env("LLM_MAX_OUTPUT_TOKENS", "GEMINI_MAX_OUTPUT_TOKENS", default=2048, minimum=1))          # answers
    max_output_tokens_short: int = field(default_factory=lambda: _int_env("LLM_MAX_OUTPUT_TOKENS_SHORT", "GEMINI_MAX_OUTPUT_TOKENS_SHORT", default=1024, minimum=1))  # JSON utility calls
    embedding_model: str = EMBEDDING_MODEL
    reranker_model: str = RERANKER_MODEL

    documents_dir: Path = ROOT / "data" / "documents"
    index_dir: Path = ROOT / "data" / "indexes"
    sessions_dir: Path = ROOT / "data" / "sessions"
    logs_dir: Path = ROOT / "logs"
    eval_dataset: Path = ROOT / "evaluation" / "dataset.json"

    chunk_size: int = 800
    chunk_overlap: int = 120

    rrf_k: int = 60
    candidates_per_retriever: int = 20  # FAISS / BM25 top-N before fusion
    fused_candidates: int = 18          # fused candidates sent to the reranker
    final_top_k: int = 5                # chunks passed to generation
    max_corrective_retries: int = 1
    history_window: int = 6             # messages sent to the LLM for follow-ups

    llm_timeout_s: int = 60
    llm_max_retries: int = field(default_factory=lambda: _int_env("LLM_MAX_RETRIES", default=2, minimum=0))  # bounded backoff for transient (5xx/network) errors only

    @property
    def has_api_key(self) -> bool:
        """True when the SELECTED provider has a key configured."""
        return bool(self.groq_api_key if self.llm_provider == "groq" else self.gemini_api_key)

    @property
    def active_model(self) -> str:
        return self.groq_model if self.llm_provider == "groq" else self.gemini_model

    @property
    def llm_log_path(self) -> Path:
        return self.logs_dir / "llm_calls.jsonl"

    def ensure_dirs(self) -> None:
        for d in (self.documents_dir, self.index_dir, self.sessions_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
