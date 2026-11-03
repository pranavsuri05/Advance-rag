"""Deterministic scoring functions (no API access needed; unit-tested)."""
from __future__ import annotations

import re
from collections import Counter

from src.utils.text_utils import tokenize


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", text.lower()).split())    # unicode-aware: non-ASCII answers must not collapse to ""


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize(pred) == normalize(ref) else 0.0


def token_f1(pred: str, ref: str) -> float:
    p, r = tokenize(normalize(pred)), tokenize(normalize(ref))
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def keyword_recall(pred: str, keywords: list[str]) -> float | None:
    """Fraction of expected keywords (case-insensitive substring) present in the answer."""
    if not keywords:
        return None
    low = normalize(pred)
    return sum(1 for k in keywords if normalize(k) in low) / len(keywords)


def precision_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    top = ranked_docs[:k]
    return sum(1 for d in top if d in relevant) / len(top) if top else 0.0


def recall_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked_docs[:k]) & relevant) / len(relevant) if relevant else 0.0


def hit_rate_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked_docs[:k]) & relevant else 0.0


def reciprocal_rank(ranked_docs: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked_docs, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def citation_validity(cited_ids: list[str], retrieved_ids: set[str]) -> float | None:
    """Fraction of cited chunk ids that exist among retrieved chunks; None if nothing was cited."""
    return (sum(1 for c in cited_ids if c in retrieved_ids) / len(cited_ids)) if cited_ids else None


def abstention_correct(abstained: bool, answerable: bool) -> bool:
    return abstained != answerable


def heuristic_correct(f1: float, kw: float | None) -> bool:
    """Transparent rule: keyword recall >= 0.6 when keywords exist, otherwise token F1 >= 0.4."""
    return kw >= 0.6 if kw is not None else f1 >= 0.4


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
