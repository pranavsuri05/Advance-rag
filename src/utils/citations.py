"""Citation parsing and validation against the chunks that were actually retrieved."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Anything that still looks like a citation after strict parsing (e.g. "[Source: a.md, Chunk: id1, id2]") is unverifiable.
LEFTOVER_RE = re.compile(r"\[\s*Source:[^\]]*\]", re.IGNORECASE)
CITATION_RE = re.compile(
    r"\[Source:\s*(?P<file>[^,\]]+?)\s*,\s*Chunk:\s*(?P<chunk>[^\],\s]+)(?:\s*,\s*Page:\s*(?P<page>\d+))?\s*\]"
)


@dataclass
class Citation:
    chunk_id: str
    source: str
    page: int | None = None

    def label(self) -> str:
        page = f", Page: {self.page}" if self.page else ""
        return f"[Source: {self.source}, Chunk: {self.chunk_id}{page}]"


@dataclass
class CitationReport:
    text: str                                  # answer with citations canonicalised / invalid ones removed
    valid: list[Citation] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)   # raw invalid citation strings
    corrected: int = 0                         # citations whose filename/page was repaired


def validate_citations(answer: str, chunk_index: dict[str, dict]) -> CitationReport:
    """Check every citation against `chunk_index` (chunk_id -> {source, page}).

    - unknown chunk id  -> citation is removed and recorded as invalid
    - known chunk id but wrong filename/page -> repaired to the true metadata
    """
    report = CitationReport(text=answer)
    seen: set[str] = set()

    def repl(m: re.Match) -> str:
        chunk_id = m.group("chunk")
        meta = chunk_index.get(chunk_id)
        if meta is None:
            report.invalid.append(m.group(0))
            return ""
        cit = Citation(chunk_id=chunk_id, source=meta["source"], page=meta.get("page"))
        if cit.source != m.group("file").strip() or (m.group("page") and int(m.group("page")) != cit.page):
            report.corrected += 1
        if chunk_id not in seen:
            seen.add(chunk_id)
            report.valid.append(cit)
        return cit.label()

    def handle(m: re.Match) -> str:
        strict = CITATION_RE.fullmatch(m.group(0))
        if strict:
            return repl(strict)                  # well-formed: validate against the retrieved chunks
        report.invalid.append(m.group(0))        # citation-shaped but malformed: unverifiable, so removed
        return ""

    text = LEFTOVER_RE.sub(handle, answer)
    report.text = re.sub(r"[ \t]+([.,;])", r"\1", text)
    return report
