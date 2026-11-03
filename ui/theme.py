"""Design tokens + theme CSS. Dark / Light are explicit overrides; Auto follows prefers-color-scheme."""
from __future__ import annotations

DARK = {
    "bg": "#0B0D12", "sidebar": "#10131A", "card": "#151923", "elevated": "#1B2030", "border": "#282E3B",
    "text": "#F1F3F9", "muted": "#929BAD", "accent": "#9B8AFB", "accent-soft": "rgba(155,138,251,.14)",
    "on-accent": "#0B0D12", "success": "#38B982", "warning": "#E5B45B", "error": "#E66D78",
}
LIGHT = {
    "bg": "#F6F7FB", "sidebar": "#FFFFFF", "card": "#FFFFFF", "elevated": "#EEF1F7", "border": "#E2E6EF",
    "text": "#182033", "muted": "#68738A", "accent": "#7258D8", "accent-soft": "rgba(114,88,216,.10)",
    "on-accent": "#FFFFFF", "success": "#168A62", "warning": "#A87520", "error": "#C44B5A",
}
MODES = ("Auto", "Dark", "Light")


def _vars(tokens: dict) -> str:
    return "".join(f"--{k}:{v};" for k, v in tokens.items())


def token_css(mode: str) -> str:
    """CSS variable definitions for the chosen mode."""
    if mode == "Dark":
        return f":root{{{_vars(DARK)}color-scheme:dark;}}"
    if mode == "Light":
        return f":root{{{_vars(LIGHT)}color-scheme:light;}}"
    return (f":root{{{_vars(LIGHT)}color-scheme:light dark;}}"
            f"@media (prefers-color-scheme: dark){{:root{{{_vars(DARK)}}}}}")
