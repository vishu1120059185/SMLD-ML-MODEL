"""STAT-TWIN multipage Streamlit dashboard — industrial dark control-room UI."""
from __future__ import annotations

import importlib
import json
import re
import sys
import traceback
from pathlib import Path

import streamlit as st

# ── Ensure src/ is on sys.path so `stattwin` is importable ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # stat-twin/
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── Page config must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="STAT-TWIN",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — industrial dark theme
#   bg #0A0E17 · card #111827 · border #1F2937 · sidebar #0D1117
#   accent #3B82F6 · success #10B981 · warning #F59E0B · danger #EF4444
#   text #F9FAFB · secondary #9CA3AF
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg: #0A0E17;
    --card: #111827;
    --border: #1F2937;
    --sidebar: #0D1117;
    --accent: #3B82F6;
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --text: #F9FAFB;
    --muted: #9CA3AF;
    --shadow: 0 4px 6px -1px rgba(0,0,0,0.45), 0 2px 4px -2px rgba(0,0,0,0.35);
}

/* ── Global ─────────────────────────────────────────────────────────────── */
.stApp {
    background: radial-gradient(1200px 600px at 80% -10%, rgba(59,130,246,0.06), transparent 60%),
                radial-gradient(900px 500px at -10% 110%, rgba(16,185,129,0.04), transparent 55%),
                var(--bg);
    color: var(--text);
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1400px; }

h1, h2, h3, h4 {
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.01em;
}
h1 { font-weight: 800 !important; }
h2 { font-weight: 700 !important; }
h3 { font-weight: 650 !important; font-size: 1.12rem !important; }
p, li, label, .stMarkdown { color: var(--muted); }
caption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
code, pre, .stCode {
    font-family: 'JetBrains Mono', 'Cascadia Code', Consolas, monospace !important;
}
b, strong { color: var(--text); }
hr { border-color: var(--border) !important; opacity: 1; }
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important;
    background: var(--card) !important;
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #263244; border-radius: 6px; border: 2px solid var(--bg); }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--sidebar) 0%, #0A0E17 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"]::before {
    content: "";
    position: absolute; top: 0; right: 0; width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent) 0%, var(--success) 100%);
    opacity: 0.65; pointer-events: none; z-index: 1;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.6rem; padding-left: 1.2rem; padding-right: 1rem;
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: var(--text) !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label { color: var(--muted) !important; }

.side-logo {
    font-size: 1.35rem; font-weight: 800; letter-spacing: 3px; line-height: 1.1;
    background: linear-gradient(90deg, #60A5FA 0%, #3B82F6 55%, #10B981 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    margin-bottom: 2px;
}
.side-tagline {
    color: var(--muted); font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 1.6px; margin-bottom: 4px;
}
.side-section {
    color: #6B7280; font-size: 0.66rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.8px; margin: 14px 0 6px 0;
}
.side-status {
    display: flex; align-items: center; gap: 8px;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 12px; margin: 6px 0 2px 0;
    font-size: 0.74rem; color: var(--muted);
}
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dot-ok { background: var(--success); box-shadow: 0 0 8px rgba(16,185,129,0.8); }
.dot-warn { background: var(--warning); box-shadow: 0 0 8px rgba(245,158,11,0.7); }
.side-footer {
    color: #6B7280; font-size: 0.68rem; letter-spacing: 0.6px;
    border-top: 1px solid var(--border); padding-top: 10px; margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
}
.side-active { color: var(--text) !important; font-weight: 600; }

/* ── Sidebar widgets ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: var(--card) !important;
    border-color: var(--border) !important; color: var(--text) !important;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.4px;
}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {
    background-color: var(--card); border-color: var(--border); color: var(--text);
}
div[data-testid="stSelectbox"] label p,
div[data-testid="stMultiSelect"] label p { color: var(--muted) !important; }

/* Radio navigation */
div[data-testid="stRadio"] > div { gap: 4px !important; }
div[data-testid="stRadio"] label {
    color: var(--muted) !important; font-size: 0.88rem !important; font-weight: 500;
    padding: 9px 12px; border-radius: 8px; border-left: 2px solid transparent;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
    background: transparent !important;
}
div[data-testid="stRadio"] label:hover {
    background: rgba(17,24,39,0.9) !important;
    color: var(--text) !important;
}
div[data-testid="stRadio"] label:has(input:checked),
div[data-testid="stRadio"] [data-checked="true"] {
    background: linear-gradient(90deg, rgba(59,130,246,0.20), rgba(59,130,246,0.03)) !important;
    color: var(--text) !important; border-left: 2px solid var(--accent) !important;
    font-weight: 650 !important;
}

/* ── Hero header ────────────────────────────────────────────────────────── */
.st-hero {
    background: linear-gradient(135deg, rgba(17,24,39,0.92) 0%, rgba(10,14,23,0.6) 100%);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 12px;
    padding: 18px 24px; margin-bottom: 1.2rem;
    box-shadow: var(--shadow);
}
.st-hero-row {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; flex-wrap: wrap;
}
.st-hero-title {
    font-size: 1.7rem; font-weight: 800; letter-spacing: 4px; line-height: 1.1;
    background: linear-gradient(90deg, #93C5FD 0%, #3B82F6 50%, #10B981 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.st-hero-sub {
    color: var(--muted); font-size: 0.74rem; text-transform: uppercase;
    letter-spacing: 2.2px; margin-top: 5px;
}
.st-hero-meta { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.st-pill {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
    border: 1px solid transparent;
}
.st-pill-accent {
    background: rgba(59,130,246,0.14); color: #93C5FD;
    border-color: rgba(59,130,246,0.45);
}
.st-pill-success {
    background: rgba(16,185,129,0.12); color: #6EE7B7;
    border-color: rgba(16,185,129,0.4);
}
.st-pill-warn {
    background: rgba(245,158,11,0.12); color: #FCD34D;
    border-color: rgba(245,158,11,0.4);
}

/* ── Metric / KPI cards ─────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    box-shadow: var(--shadow);
}
[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-weight: 700 !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] span { color: var(--muted) !important; }

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: transparent; padding: 4px;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background-color: var(--card); color: var(--muted);
    border: 1px solid var(--border); border-radius: 8px 8px 0 0;
    padding: 9px 18px; font-weight: 600; font-size: 0.85rem;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text); border-color: #2B3A4F; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(59,130,246,0.18), var(--card)) !important;
    color: var(--text) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
input, textarea {
    color: var(--text) !important;
    background-color: var(--card) !important;
}
div[data-baseweb="input"] {
    background-color: var(--card) !important;
    border-color: var(--border) !important;
}
div[data-baseweb="base-button-above"] > button,
.stButton > button {
    background: linear-gradient(180deg, #1D4ED8, #1E40AF) !important;
    color: #FFFFFF !important; border: 1px solid #2563EB !important;
    border-radius: 8px !important; font-weight: 600 !important;
    letter-spacing: 0.4px; transition: filter 0.15s ease, transform 0.1s ease;
    box-shadow: var(--shadow);
}
.stButton > button:hover {
    filter: brightness(1.15);
    border-color: #3B82F6 !important;
    color: #FFF !important;
}
.stButton > button:active { transform: translateY(1px); }

/* Sliders */
.stSlider [data-testid="stTickBar"] { background-color: transparent; }
.stSlider div[role="slider"] {
    background-color: var(--accent) !important; border: 2px solid #DBEAFE !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.25);
}

/* ── Alerts / info boxes ────────────────────────────────────────────────── */
div[data-testid="stAlert"], div[data-testid="stException"] {
    border-radius: 10px !important; border-width: 1px !important;
    background: var(--card) !important; color: var(--text) !important;
    box-shadow: var(--shadow);
}
div[data-testid="stAlert"] > div { color: var(--text) !important; }
[data-testid="stNotification"] { background: var(--card); border-color: var(--border); }

/* ── Expander / popover ─────────────────────────────────────────────────── */
details {
    border: 1px solid var(--border) !important;
    background: var(--card) !important;
    border-radius: 10px !important;
}
details summary { color: var(--text) !important; font-weight: 600 !important; }
[data-testid="stExpander"] {
    background: var(--card) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: var(--text) !important; }

/* ── Dataframe / table ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-color: var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stStyledTable"] thead tr th {
    background-color: var(--card) !important;
    color: var(--muted) !important;
    text-transform: uppercase; letter-spacing: 0.8px; font-size: 0.72rem;
}
[data-testid="stStyledTable"] tbody tr { background-color: var(--bg) !important; }
[data-testid="stStyledTable"] tbody tr:hover { background-color: var(--card) !important; }
[data-testid="stStyledTable"] tbody td { color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }

/* ── Plotly charts sit inside cards ─────────────────────────────────────── */
.stPlotlyChart, [data-testid="stPlotlyChart"] {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 6px 4px 2px 2px;
    box-shadow: var(--shadow);
}
.js-plotly-plot .plotly .modebar { opacity: 0.35; transition: opacity 0.15s; }
.js-plotly-plot .plotly:hover .modebar { opacity: 1; }

/* ── Divider used by section headers ────────────────────────────────────── */
section[data-testid="stSidebar"] hr { border-color: var(--border) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
RESULTS_DIR = _PROJECT_ROOT / "results"

PAGES = {
    "⚙️  Overview": "views/1_overview",
    "📡  Sensor Monitoring": "views/2_sensors",
    "🏥  Statistical Health": "views/3_health",
    "🔮  Failure Forecast": "views/4_forecast",
    "🔍  Explainability": "views/5_explain",
    "🔧  What-If Simulator": "views/6_whatif",
    "📊  Model Comparison": "views/7_comparison",
}

st.sidebar.markdown(
    """
    <div class="side-logo">STAT-TWIN</div>
    <div class="side-tagline">Digital-Twin Health Intelligence</div>
    """,
    unsafe_allow_html=True,
)

# ── Machine selector (persisted across pages) ───────────────────────────────
available_machines: list[str] = []
meta_file = RESULTS_DIR / "meta.json"
if meta_file.exists():
    try:
        meta = json.loads(meta_file.read_text())
        available_machines = meta.get("machines", [])
    except Exception:
        pass

if not available_machines and RESULTS_DIR.is_dir():
    # Fallback: scan results/ for machine sub-directories (skip stage dirs e0_…, global)
    available_machines = sorted(
        d.name
        for d in RESULTS_DIR.iterdir()
        if d.is_dir() and d.name != "global" and not re.match(r"^e\d+_", d.name)
    )

if not available_machines:
    available_machines = ["MACHINE-001"]

st.sidebar.markdown('<div class="side-section">Active Machine</div>', unsafe_allow_html=True)
selected_machine = st.sidebar.selectbox(
    "Machine",
    available_machines,
    index=0,
    key="selected_machine",
    label_visibility="collapsed",
)

# ── Status indicator ────────────────────────────────────────────────────────
if RESULTS_DIR.is_dir():
    n_stage = len([d for d in RESULTS_DIR.iterdir() if d.is_dir()]) if RESULTS_DIR.is_dir() else 0
    if n_stage:
        status_html = (
            f'<div class="side-status"><span class="dot dot-ok"></span>'
            f'<span>Artifacts linked · <b style="color:#F9FAFB;">{n_stage}</b>'
            f' result sets</span></div>'
        )
    else:
        status_html = (
            '<div class="side-status"><span class="dot dot-warn"></span>'
            '<span>results/ empty — run pipeline</span></div>'
        )
else:
    status_html = (
        '<div class="side-status"><span class="dot dot-warn"></span>'
        '<span>results/ not found</span></div>'
    )
st.sidebar.markdown(status_html, unsafe_allow_html=True)

st.sidebar.markdown(
    f'<div class="side-section">Navigation · <span class="side-active">'
    f'{len(PAGES)} pages</span></div>',
    unsafe_allow_html=True,
)

# ── Page selector ───────────────────────────────────────────────────────────
selected_page = st.sidebar.radio(
    "Navigation",
    list(PAGES.keys()),
    label_visibility="collapsed",
    key="nav_page",
)

st.sidebar.markdown(
    f"""
    <div class="side-footer">
        © 2026 STAT-TWIN · v0.1<br/>
        Theme: Dark Industrial<br/>
        Machine: <span style="color:#93C5FD;">{selected_machine}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Hero header (main area) ─────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="st-hero">
        <div class="st-hero-row">
            <div>
                <div class="st-hero-title">STAT-TWIN</div>
                <div class="st-hero-sub">Predictive Maintenance · Statistical
                    Health · Failure Forecast</div>
            </div>
            <div class="st-hero-meta">
                <span class="st-pill st-pill-accent">{selected_page.split('  ')[-1]}</span>
                <span class="st-pill st-pill-success">● {selected_machine}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Dispatch to page module (absolute imports: stattwin.dashboard.views.*) ───
page_module_path = PAGES[selected_page]
mod_name = f"stattwin.dashboard.{page_module_path.replace('/', '.')}"

try:
    page_mod = importlib.import_module(mod_name)
except ModuleNotFoundError as exc:
    st.error(
        f"Could not import page module `{mod_name}`. "
        f"Ensure `src/` is on PYTHONPATH and the file exists. ({exc})"
    )
    with st.expander("Traceback"):
        st.code(traceback.format_exc())
else:
    try:
        if not hasattr(page_mod, "render"):
            st.error(f"Page module `{mod_name}` has no `render()` function.")
        else:
            page_mod.render()
    except Exception as exc:
        st.error(f"Error rendering page **{selected_page}**: `{exc}`")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
