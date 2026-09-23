"""STAT-TWIN design system — single source of truth for tokens + CSS."""
from __future__ import annotations

import streamlit as st

# ── Tokens ──────────────────────────────────────────────────────────────────
BG = "#0A0E17"
BG_ELEVATED = "#0D1117"
CARD = "#111827"
CARD_HOVER = "#151C2C"
BORDER = "#1F2937"
BORDER_SOFT = "#1A2233"
SIDEBAR = "#0D1117"

ACCENT = "#3B82F6"
ACCENT_SOFT = "rgba(59,130,246,0.14)"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
VIOLET = "#8B5CF6"

TEXT = "#F9FAFB"
MUTED = "#9CA3AF"
FAINT = "#6B7280"

MONO = "'JetBrains Mono','Fira Code','Cascadia Code',Consolas,monospace"
SANS = "'Inter','Segoe UI',system-ui,-apple-system,sans-serif"

SHADOW = "0 4px 6px -1px rgba(0,0,0,0.45), 0 2px 4px -2px rgba(0,0,0,0.35)"
SHADOW_LG = "0 12px 28px -8px rgba(0,0,0,0.55)"

STATE_COLORS = {
    "HEALTHY": SUCCESS,
    "WATCH": WARNING,
    "DEGRADING": "#F97316",
    "CRITICAL": DANGER,
    "FAILURE-LIKELY": "#991B1B",
}

STATE_GLOW = {
    "HEALTHY": "rgba(16,185,129,0.45)",
    "WATCH": "rgba(245,158,11,0.45)",
    "DEGRADING": "rgba(249,115,22,0.5)",
    "CRITICAL": "rgba(239,68,68,0.55)",
    "FAILURE-LIKELY": "rgba(153,27,27,0.55)",
}

PROVENANCE_COLORS = {
    "OBSERVED": ACCENT,
    "PREDICTED": WARNING,
    "SIMULATED": VIOLET,
}

PROVENANCE_DASH = {
    "OBSERVED": "solid",
    "PREDICTED": "dash",
    "SIMULATED": "dot",
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=CARD,
    plot_bgcolor=BG,
    font=dict(color=TEXT, family=MONO, size=12),
    margin=dict(l=48, r=24, t=48, b=40),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showline=False),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED)),
    hoverlabel=dict(
        bgcolor=CARD,
        bordercolor=BORDER,
        font=dict(color=TEXT, family=MONO, size=12),
    ),
    hovermode="x unified",
)


def inject_css() -> None:
    """Inject the full design-system stylesheet.

    Must run on *every* script run: Streamlit rebuilds the DOM on each
    rerun (page switch, widget change), so a once-per-session guard would
    strip the theme after the first interaction.
    """
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {{
  --bg: {BG};
  --bg-el: {BG_ELEVATED};
  --card: {CARD};
  --card-h: {CARD_HOVER};
  --border: {BORDER};
  --border-soft: {BORDER_SOFT};
  --sidebar: {SIDEBAR};
  --accent: {ACCENT};
  --success: {SUCCESS};
  --warning: {WARNING};
  --danger: {DANGER};
  --violet: {VIOLET};
  --text: {TEXT};
  --muted: {MUTED};
  --faint: {FAINT};
  --mono: {MONO};
  --sans: {SANS};
  --shadow: {SHADOW};
  --shadow-lg: {SHADOW_LG};
}}

/* ── Global shell ─────────────────────────────────────────────────────── */
.stApp {{
  background:
    radial-gradient(1200px 520px at 88% -8%, rgba(59,130,246,0.07), transparent 58%),
    radial-gradient(900px 480px at -8% 108%, rgba(16,185,129,0.045), transparent 55%),
    radial-gradient(700px 400px at 50% 50%, rgba(139,92,246,0.025), transparent 70%),
    var(--bg);
  color: var(--text);
  font-family: var(--sans);
}}
html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
  font-family: var(--sans);
}}
.block-container {{
  padding-top: 1.15rem; padding-bottom: 3rem; max-width: 1440px;
}}
h1,h2,h3,h4,h5 {{ color: var(--text) !important; font-family: var(--sans);
  letter-spacing: -0.015em; }}
h1 {{ font-weight: 800 !important; font-size: 1.55rem !important; }}
h2 {{ font-weight: 700 !important; }}
h3 {{ font-weight: 650 !important; font-size: 1.05rem !important; }}
p, li, label, .stMarkdown, .stCaption {{ color: var(--muted); }}
b, strong {{ color: var(--text); }}
code, pre, .stCode {{ font-family: var(--mono) !important; }}
hr {{ border-color: var(--border) !important; opacity: 1; margin: 0.85rem 0; }}
[data-testid="stVerticalBlockBorderWrapper"] {{
  border-color: var(--border) !important; background: var(--card) !important;
  border-radius: 12px !important;
}}
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: #263244; border-radius: 6px;
  border: 2px solid var(--bg); }}
::-webkit-scrollbar-thumb:hover {{ background: #334155; }}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, var(--sidebar) 0%, #0A0E17 100%) !important;
  border-right: 1px solid var(--border) !important;
}}
section[data-testid="stSidebar"]::before {{
  content: ""; position: absolute; top: 0; right: 0; width: 2px; height: 100%;
  background: linear-gradient(180deg, var(--accent), var(--success));
  opacity: 0.55; pointer-events: none; z-index: 1;
}}
section[data-testid="stSidebar"] .block-container {{
  padding-top: 1.4rem; padding-left: 1.1rem; padding-right: 0.9rem;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color: var(--text) !important; }}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {{ color: var(--muted) !important; }}
section[data-testid="stSidebar"] hr {{ border-color: var(--border) !important; }}

.side-logo {{
  font-size: 1.4rem; font-weight: 800; letter-spacing: 4px; line-height: 1.1;
  background: linear-gradient(90deg, #93C5FD 0%, #3B82F6 50%, #10B981 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.side-tagline {{
  color: var(--muted); font-size: 0.66rem; text-transform: uppercase;
  letter-spacing: 1.8px; margin: 2px 0 4px;
}}
.side-section {{
  color: var(--faint); font-size: 0.64rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 2px; margin: 16px 0 6px;
}}
.side-status {{
  display: flex; align-items: center; gap: 8px;
  background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 9px 12px; margin: 8px 0 2px;
  font-size: 0.73rem; color: var(--muted);
  box-shadow: var(--shadow);
}}
.dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
.dot-ok {{ background: var(--success); box-shadow: 0 0 8px rgba(16,185,129,0.85);
  animation: pulse 2s ease-in-out infinite; }}
.dot-warn {{ background: var(--warning); box-shadow: 0 0 8px rgba(245,158,11,0.75); }}
.side-footer {{
  color: var(--faint); font-size: 0.66rem; letter-spacing: 0.5px;
  border-top: 1px solid var(--border); padding-top: 12px; margin-top: 14px;
  font-family: var(--mono); line-height: 1.55;
}}
.side-active {{ color: var(--text) !important; font-weight: 650; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.45; }} }}

/* Sidebar widgets */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
  background-color: var(--card) !important; border-color: var(--border) !important;
  color: var(--text) !important; border-radius: 8px !important;
}}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
  color: var(--muted) !important; font-size: 0.76rem !important; letter-spacing: 0.3px;
}}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {{
  background-color: var(--card); border-color: var(--border); color: var(--text);
  border-radius: 8px;
}}
div[data-testid="stSelectbox"] label p,
div[data-testid="stMultiSelect"] label p {{ color: var(--muted) !important; }}
div[data-testid="stWidgetLabel"] p {{ color: var(--muted) !important;
  font-size: 0.78rem !important; }}

/* Radio nav — pill rows */
div[data-testid="stRadio"] > div {{ gap: 3px !important; }}
div[data-testid="stRadio"] label {{
  color: var(--muted) !important; font-size: 0.86rem !important; font-weight: 500;
  padding: 9px 12px !important; border-radius: 9px;
  border: 1px solid transparent !important;
  transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
  background: transparent !important;
}}
div[data-testid="stRadio"] label:hover {{
  background: rgba(17,24,39,0.85) !important; color: var(--text) !important;
  border-color: var(--border) !important;
}}
div[data-testid="stRadio"] label:has(input:checked) {{
  background: linear-gradient(90deg, rgba(59,130,246,0.22), rgba(59,130,246,0.04)) !important;
  color: var(--text) !important; font-weight: 650 !important;
  border-color: rgba(59,130,246,0.55) !important;
  box-shadow: inset 3px 0 0 var(--accent);
}}

/* ── Hero ─────────────────────────────────────────────────────────────── */
.st-hero {{
  background:
    linear-gradient(135deg, rgba(17,24,39,0.95) 0%, rgba(10,14,23,0.55) 100%);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 14px;
  padding: 18px 24px; margin-bottom: 1rem;
  box-shadow: var(--shadow-lg);
  position: relative; overflow: hidden;
}}
.st-hero::after {{
  content: ""; position: absolute; top: -40%; right: -10%; width: 280px; height: 280px;
  background: radial-gradient(circle, rgba(59,130,246,0.12), transparent 65%);
  pointer-events: none;
}}
.st-hero-row {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px; flex-wrap: wrap; position: relative; z-index: 1;
}}
.st-hero-title {{
  font-size: 1.55rem; font-weight: 800; letter-spacing: 4.5px; line-height: 1.1;
  background: linear-gradient(90deg, #93C5FD 0%, #3B82F6 48%, #10B981 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.st-hero-sub {{
  color: var(--muted); font-size: 0.7rem; text-transform: uppercase;
  letter-spacing: 2.4px; margin-top: 5px;
}}
.st-hero-meta {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.st-pill {{
  display: inline-block; padding: 5px 13px; border-radius: 999px;
  font-size: 0.66rem; font-weight: 700; letter-spacing: 1.1px;
  text-transform: uppercase; font-family: var(--mono);
  border: 1px solid transparent;
}}
.st-pill-accent {{ background: rgba(59,130,246,0.14); color: #93C5FD;
  border-color: rgba(59,130,246,0.45); }}
.st-pill-success {{ background: rgba(16,185,129,0.12); color: #6EE7B7;
  border-color: rgba(16,185,129,0.4); }}
.st-pill-warn {{ background: rgba(245,158,11,0.12); color: #FCD34D;
  border-color: rgba(245,158,11,0.4); }}
.st-pill-violet {{ background: rgba(139,92,246,0.14); color: #C4B5FD;
  border-color: rgba(139,92,246,0.45); }}

/* ── Page title (views) ───────────────────────────────────────────────── */
.st-page-title {{
  font-size: 1.35rem; font-weight: 750; color: var(--text);
  margin: 0 0 0.15rem 0; letter-spacing: -0.01em;
}}
.st-page-sub {{
  color: var(--muted); font-size: 0.8rem; margin-bottom: 0.75rem;
  font-family: var(--mono); letter-spacing: 0.2px;
}}

/* ── Metrics / KPI ────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
  background: linear-gradient(160deg, var(--card) 0%, #0F1522 100%) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 15px 18px !important;
  box-shadow: var(--shadow);
  transition: border-color 0.15s ease, transform 0.12s ease;
}}
[data-testid="stMetric"]:hover {{ border-color: #2B3A4F !important; transform: translateY(-1px); }}
[data-testid="stMetric"] label {{
  color: var(--muted) !important; font-size: 0.7rem !important;
  text-transform: uppercase; letter-spacing: 1.1px; font-weight: 650;
}}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
  color: var(--text) !important; font-family: var(--mono) !important; font-weight: 700 !important;
}}
[data-testid="stMetric"] [data-testid="stMetricDelta"] span {{ color: var(--muted) !important; }}

/* ── Tabs ─────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  gap: 6px; background: transparent; padding: 4px 0;
  border-bottom: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
  background-color: var(--card); color: var(--muted);
  border: 1px solid var(--border); border-radius: 9px 9px 0 0;
  padding: 9px 16px; font-weight: 600; font-size: 0.84rem;
  transition: color 0.12s ease, border-color 0.12s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--text); border-color: #2B3A4F; }}
.stTabs [aria-selected="true"] {{
  background: linear-gradient(180deg, rgba(59,130,246,0.2), var(--card)) !important;
  color: var(--text) !important;
  border-bottom: 2px solid var(--accent) !important;
}}

/* ── Buttons / inputs ─────────────────────────────────────────────────── */
input, textarea {{ color: var(--text) !important; background-color: var(--card) !important; }}
div[data-baseweb="input"] {{ background-color: var(--card) !important;
  border-color: var(--border) !important; border-radius: 8px; }}
.stButton > button, div[data-baseweb="base-button-above"] > button {{
  background: linear-gradient(180deg, #1D4ED8, #1E40AF) !important;
  color: #FFF !important; border: 1px solid #2563EB !important;
  border-radius: 9px !important; font-weight: 650 !important;
  letter-spacing: 0.3px; transition: filter 0.15s ease, transform 0.1s ease;
  box-shadow: var(--shadow);
}}
.stButton > button:hover {{ filter: brightness(1.16); border-color: #3B82F6 !important; }}
.stButton > button:active {{ transform: translateY(1px); }}
.stSlider [data-testid="stTickBar"] {{ background-color: transparent; }}
.stSlider div[role="slider"] {{
  background-color: var(--accent) !important; border: 2px solid #DBEAFE !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.28);
}}

/* ── Alerts ───────────────────────────────────────────────────────────── */
div[data-testid="stAlert"], div[data-testid="stException"] {{
  border-radius: 11px !important; border-width: 1px !important;
  background: var(--card) !important; color: var(--text) !important;
  box-shadow: var(--shadow);
}}
div[data-testid="stAlert"] > div {{ color: var(--text) !important; }}

/* ── Expander / table ─────────────────────────────────────────────────── */
[data-testid="stExpander"], details {{
  background: var(--card) !important; border-color: var(--border) !important;
  border-radius: 11px !important;
}}
details summary, [data-testid="stExpander"] summary {{
  color: var(--text) !important; font-weight: 600 !important;
}}
[data-testid="stDataFrame"] {{ border-color: var(--border) !important;
  border-radius: 11px !important; overflow: hidden; }}
[data-testid="stStyledTable"] thead tr th {{
  background-color: var(--card) !important; color: var(--muted) !important;
  text-transform: uppercase; letter-spacing: 0.8px; font-size: 0.7rem;
}}
[data-testid="stStyledTable"] tbody tr {{ background-color: var(--bg) !important; }}
[data-testid="stStyledTable"] tbody tr:hover {{ background-color: var(--card) !important; }}
[data-testid="stStyledTable"] tbody td {{
  color: var(--text) !important; font-family: var(--mono); font-size: 0.8rem;
}}

/* ── Plotly in cards ──────────────────────────────────────────────────── */
.stPlotlyChart, [data-testid="stPlotlyChart"] {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 4px 2px 0;
  box-shadow: var(--shadow);
}}
.js-plotly-plot .plotly .modebar {{ opacity: 0.3; transition: opacity 0.15s; }}
.js-plotly-plot .plotly:hover .modebar {{ opacity: 1; }}

/* ── Custom HTML card helpers ─────────────────────────────────────────── */
.st-kpi {{
  background: linear-gradient(160deg, var(--card) 0%, #0F1522 100%);
  border: 1px solid var(--border); border-radius: 12px;
  padding: 15px 18px; box-shadow: var(--shadow);
  transition: border-color 0.15s ease, transform 0.12s ease;
  height: 100%;
}}
.st-kpi:hover {{ border-color: #2B3A4F; transform: translateY(-1px); }}
.st-kpi-label {{
  font-size: 0.68rem; color: var(--muted); text-transform: uppercase;
  letter-spacing: 1.2px; margin-bottom: 7px; font-weight: 650;
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
}}
.st-kpi-value {{
  font-size: 1.55rem; font-weight: 750; color: var(--text);
  font-family: var(--mono); line-height: 1.15;
}}
.st-kpi-delta {{
  font-size: 0.72rem; color: var(--faint); margin-top: 6px; font-family: var(--mono);
}}
.st-badge {{
  display: inline-block; padding: 2px 9px; border-radius: 999px;
  font-size: 0.66rem; font-weight: 700; letter-spacing: 0.8px;
  text-transform: uppercase; font-family: var(--mono);
  border: 1px solid transparent; line-height: 1.45;
}}
.st-card {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px 18px; margin-bottom: 12px;
  box-shadow: var(--shadow);
}}
.st-section-title {{
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
  margin: 0.4rem 0 0.15rem;
}}
.st-section-title h3 {{ margin: 0; }}
.st-section-sub {{ color: var(--muted); font-size: 0.8rem; font-family: var(--mono); }}

/* ── Live status bar ──────────────────────────────────────────────────── */
.st-live-bar {{
  display: flex; align-items: center; justify-content: space-between; gap: 14px;
  flex-wrap: wrap;
  background: linear-gradient(90deg, var(--card) 0%, #0F1522 100%);
  border: 1px solid var(--border); border-radius: 11px;
  padding: 8px 14px; margin-bottom: 12px;
  box-shadow: var(--shadow);
}}
.st-live-label {{
  display: inline-flex; align-items: center; gap: 8px;
  color: var(--danger); font-size: 0.7rem; font-weight: 800;
  letter-spacing: 1.6px; font-family: var(--mono);
}}
.st-live-dot {{
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--danger); display: inline-block;
  box-shadow: 0 0 8px var(--danger);
  animation: stattwin-blink 1.2s ease-in-out infinite;
}}
.st-live-meta {{
  color: var(--muted); font-family: var(--mono); font-size: 0.7rem;
}}
.st-live-feed {{
  color: var(--muted); font-size: 0.68rem; opacity: 0.85;
}}
@keyframes stattwin-blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.15; }} }}

/* ── Data freshness indicator ─────────────────────────────────────────── */
.st-freshness {{
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.68rem; color: var(--muted);
  font-family: var(--mono); letter-spacing: 0.3px;
}}
.st-freshness-dot {{
  width: 6px; height: 6px; border-radius: 50%; display: inline-block;
  animation: pulse 2s infinite;
}}
.st-freshness-ok {{ background: var(--success); }}
.st-freshness-stale {{ background: var(--warning); animation: none; }}

/* ── Tooltips ─────────────────────────────────────────────────────────── */
.st-tooltip-wrapper {{
  position: relative; display: inline-block; cursor: help;
}}
.st-tooltip-wrapper .st-tooltip-text {{
  visibility: hidden; opacity: 0;
  position: absolute; bottom: 125%; left: 50%; transform: translateX(-50%);
  background: var(--card); color: var(--text); font-size: 0.72rem;
  padding: 6px 10px; border-radius: 6px; white-space: nowrap;
  border: 1px solid var(--border); box-shadow: var(--shadow);
  z-index: 999; transition: opacity 0.15s ease;
  pointer-events: none; font-weight: 500;
}}
.st-tooltip-wrapper .st-tooltip-text::after {{
  content: ""; position: absolute; top: 100%; left: 50%;
  margin-left: -5px; border-width: 5px;
  border-style: solid; border-color: var(--card) transparent transparent transparent;
}}
.st-tooltip-wrapper:hover .st-tooltip-text {{ visibility: visible; opacity: 1; }}

/* ── Provenance badge variants ────────────────────────────────────────── */
.prov-observed {{ border: 1px solid rgba(59,130,246,0.6); }}
.prov-predicted {{ border: 1px dashed rgba(245,158,11,0.6); }}
.prov-simulated {{ border: 1px dotted rgba(139,92,246,0.6); }}

/* ── Skeleton loader ──────────────────────────────────────────────────── */
.st-skeleton {{
  background: linear-gradient(90deg, var(--card) 25%, #1a2236 50%, var(--card) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 8px; height: 60px;
}}
@keyframes shimmer {{
  0% {{ background-position: -200% 0; }}
  100% {{ background-position: 200% 0; }}
}}
.st-alert-strip {{
  border-left: 3px solid; padding-left: 12px; margin-bottom: 8px;
}}
.st-compare-arrow {{
  color: var(--muted); font-size: 1.1rem; display: inline-block;
  margin: 0 6px; vertical-align: middle;
}}

/* ── Reduced motion ───────────────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }}
  .dot-ok, .dot-warn, .st-live-dot {{ box-shadow: none !important; }}
}}
</style>
"""
