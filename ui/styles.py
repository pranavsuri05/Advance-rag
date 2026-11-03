"""Global CSS. Only uses theme variables (no hardcoded colours)."""
from ui.theme import token_css

BASE_CSS = """
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg) !important; color: var(--text); }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: var(--sidebar) !important; border-right: 1px solid var(--border); }
.block-container { padding-top: 2rem; padding-bottom: 5rem; max-width: 1180px; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p { color: var(--text); }
.stApp h1, .stApp h2, .stApp h3 { letter-spacing: -0.01em; font-weight: 600; }
[data-testid="stCaptionContainer"], .stApp small, .muted { color: var(--muted) !important; }
a { color: var(--accent) !important; }
hr { border-color: var(--border) !important; }

/* sidebar navigation */
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] { border-radius: 8px; color: var(--muted) !important; }
[data-testid="stSidebarNavLink"]:hover { background: var(--elevated) !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: var(--accent-soft) !important; color: var(--accent) !important; }
[data-testid="stSidebarNavLink"] span { color: inherit !important; }

/* buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stBaseButton-secondary"] {
  background: var(--elevated); color: var(--text); border: 1px solid var(--border); border-radius: 8px; font-weight: 500; transition: border-color .15s, background .15s; }
.stButton > button:hover { border-color: var(--accent); color: var(--text); background: var(--elevated); }
[data-testid="stBaseButton-primary"] { background: var(--accent) !important; color: var(--on-accent) !important; border: 1px solid var(--accent) !important; border-radius: 8px; font-weight: 600; }
[data-testid="stBaseButton-primary"] p { color: var(--on-accent) !important; }
[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); }
.stButton > button:disabled { opacity: .5; }

/* inputs */
input, textarea, [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div, [data-testid="stChatInput"], [data-testid="stChatInput"] > div {
  background: var(--card) !important; color: var(--text) !important; border-color: var(--border) !important; border-radius: 10px; }
[data-testid="stChatInput"] textarea { color: var(--text) !important; }
[data-baseweb="popover"] *, [data-baseweb="menu"] { background: var(--card) !important; color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--card); border: 1px dashed var(--border); border-radius: 12px; }
[data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { color: var(--text) !important; }
[data-baseweb="radio"] div, [data-baseweb="checkbox"] span { color: var(--text) !important; }

/* tabs / expanders / chat */
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
[data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; }
[data-baseweb="tab-highlight"] { background: var(--accent) !important; }
[data-baseweb="tab-border"] { background: transparent !important; }
[data-testid="stExpander"] { background: var(--card); border: 1px solid var(--border) !important; border-radius: 12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--text) !important; }
[data-testid="stExpander"] details { border: none; }
[data-testid="stChatMessage"] { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.1rem; }
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] { background: var(--accent-soft) !important; color: var(--accent) !important; }
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: var(--bg) !important; }
[data-testid="stAlert"] { background: var(--elevated) !important; border: 1px solid var(--border); border-radius: 10px; }
[data-testid="stAlert"] * { color: var(--text) !important; }
code, pre, [data-testid="stCode"] { background: var(--elevated) !important; color: var(--text) !important; border-radius: 8px; }
[data-testid="stProgress"] > div > div > div { background: var(--accent) !important; }
[data-testid="stProgress"] > div > div { background: var(--elevated) !important; }
[data-testid="stSpinner"] * { color: var(--muted) !important; }
[data-testid="stTooltipContent"], [data-baseweb="tooltip"] * { background: var(--elevated) !important; color: var(--text) !important; }

/* custom components */
.brand { font-size: 1.05rem; font-weight: 700; letter-spacing: .06em; color: var(--text); }
.brand span { color: var(--accent); }
.tagline { color: var(--muted); font-size: .8rem; margin-top: 2px; }
.hero h1 { margin: 0 0 .15rem 0; font-size: 1.8rem; }
.hero p { color: var(--muted); margin: 0 0 1rem 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: .5rem 0 1.2rem; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.card .label { color: var(--muted); font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; }
.card .value { font-size: 1.5rem; font-weight: 600; margin-top: 4px; color: var(--text); }
.card .sub { color: var(--muted); font-size: .75rem; margin-top: 2px; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .74rem; border: 1px solid var(--border); color: var(--muted); background: var(--elevated); }
.pill.ok { color: var(--success); border-color: var(--success); } .pill.warn { color: var(--warning); border-color: var(--warning); }
.pill.err { color: var(--error); border-color: var(--error); } .pill.accent { color: var(--accent); border-color: var(--accent); }
.stage { display: flex; gap: 10px; align-items: baseline; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: .88rem; }
.stage:last-child { border-bottom: none; }
.stage .dot { width: 9px; height: 9px; border-radius: 50%; flex: none; background: var(--muted); }
.stage .dot.success { background: var(--success); } .stage .dot.failed { background: var(--error); }
.stage .dot.fallback { background: var(--warning); } .stage .dot.skipped { background: var(--muted); opacity: .6; }
.stage .name { font-weight: 600; min-width: 190px; color: var(--text); } .stage .detail { color: var(--muted); flex: 1; } .stage .ms { color: var(--muted); font-variant-numeric: tabular-nums; }
.tbl-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; background: var(--card); max-height: 420px; overflow-y: auto; }
table.tbl { width: 100%; border-collapse: collapse; font-size: .83rem; }
table.tbl th { text-align: left; color: var(--muted); font-weight: 500; padding: 9px 12px; border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--card); white-space: nowrap; }
table.tbl td { padding: 8px 12px; border-bottom: 1px solid var(--border); color: var(--text); vertical-align: top; }
table.tbl tr:last-child td { border-bottom: none; } table.tbl tr:hover td { background: var(--elevated); }
.bar-row { display: grid; grid-template-columns: 170px 1fr 60px; gap: 10px; align-items: center; margin: 7px 0; font-size: .82rem; }
.bar-track { background: var(--elevated); border-radius: 6px; height: 10px; overflow: hidden; } .bar-fill { background: var(--accent); height: 100%; border-radius: 6px; }
.bar-row .v { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
.chart-box { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.chart-box h4 { margin: 0 0 8px 0; font-size: .85rem; color: var(--muted); font-weight: 500; }
.evidence { background: var(--elevated); border-left: 3px solid var(--accent); border-radius: 6px; padding: 10px 12px; font-size: .84rem; white-space: pre-wrap; color: var(--text); }
.empty { text-align: center; color: var(--muted); padding: 2.5rem 1rem; border: 1px dashed var(--border); border-radius: 12px; }
@media (max-width: 720px) { .block-container { padding-left: 1rem; padding-right: 1rem; } .bar-row { grid-template-columns: 110px 1fr 48px; } .stage { flex-wrap: wrap; } .stage .name { min-width: 100%; } }
"""


def build_css(mode: str) -> str:
    return f"<style>{token_css(mode)}{BASE_CSS}</style>"
