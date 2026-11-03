"""Reusable UI building blocks. All colour comes from CSS variables."""
from __future__ import annotations

import html
from typing import Iterable

import streamlit as st


def esc(x) -> str:
    return html.escape(str(x))


def metric_cards(items: list[tuple[str, str, str]]) -> None:
    """items: (label, value, sub)"""
    body = "".join(f'<div class="card"><div class="label">{esc(l)}</div><div class="value">{esc(v)}</div>'
                   f'<div class="sub">{esc(s)}</div></div>' for l, v, s in items)
    st.markdown(f'<div class="cards">{body}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    return f'<span class="pill {kind}">{esc(text)}</span>'


def empty_state(message: str) -> None:
    st.markdown(f'<div class="empty">{esc(message)}</div>', unsafe_allow_html=True)


def html_table(headers: list[str], rows: Iterable[Iterable]) -> None:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    st.markdown(f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>',
                unsafe_allow_html=True)


def bar_chart(title: str, data: dict[str, float], fmt: str = "{:,.0f}") -> None:
    if not data:
        empty_state("No data yet")
        return
    mx = max(data.values()) or 1
    rows = "".join(f'<div class="bar-row"><div>{esc(k)}</div><div class="bar-track"><div class="bar-fill" '
                   f'style="width:{v / mx * 100:.1f}%"></div></div><div class="v">{fmt.format(v)}</div></div>'
                   for k, v in data.items())
    st.markdown(f'<div class="chart-box"><h4>{esc(title)}</h4>{rows}</div>', unsafe_allow_html=True)


def line_chart(title: str, values: list[float], unit: str = "") -> None:
    """Inline SVG line chart; strokes use CSS variables so it follows the theme."""
    if len(values) < 1:
        empty_state("No data yet")
        return
    w, h, pad = 600, 140, 8
    mx, mn = max(values), 0
    span = (mx - mn) or 1
    pts = [(pad + (i / max(len(values) - 1, 1)) * (w - 2 * pad), h - pad - (v - mn) / span * (h - 2 * pad)) for i, v in enumerate(values)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="var(--accent)"/>' for x, y in pts)
    st.markdown(f'<div class="chart-box"><h4>{esc(title)} · max {mx:,.0f}{esc(unit)}</h4>'
                f'<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="none" role="img">'
                f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="var(--border)"/>'
                f'<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="{path}"/>{dots}</svg></div>',
                unsafe_allow_html=True)


def stage_list(stages: list[dict]) -> None:
    rows = "".join(f'<div class="stage"><span class="dot {esc(s["status"])}"></span><span class="name">{esc(s["name"])}</span>'
                   f'<span class="detail">{esc(s.get("detail", ""))}</span>'
                   f'<span class="ms">{(str(s["duration_ms"]) + " ms") if s.get("duration_ms") else ""}</span></div>' for s in stages)
    st.markdown(f'<div>{rows}</div>', unsafe_allow_html=True)


def evidence(text: str) -> None:
    st.markdown(f'<div class="evidence">{esc(text)}</div>', unsafe_allow_html=True)


def citation_panel(citations: list[dict], key: str) -> None:
    if not citations:
        return
    st.caption("Sources")
    for c in citations:
        page = f" · page {c['page']}" if c.get("page") else ""
        with st.expander(f"{c['source']} · {c['chunk_id']}{page}"):
            evidence(c["text"])


def chunk_rows(items) -> list[list]:
    """Rows for retrieval tables from RetrievedChunk objects."""
    f = lambda v, n=3: "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else v)
    return [[i + 1, it.chunk.chunk_id, it.chunk.source, f(it.vector_rank), f(it.vector_score), f(it.bm25_rank),
             f(it.bm25_score), f(it.rrf_score, 5), f(it.pre_rerank_rank), f(it.rerank_score, 2), f(it.post_rerank_rank)]
            for i, it in enumerate(items)]
