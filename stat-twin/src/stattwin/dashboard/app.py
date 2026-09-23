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

from stattwin.dashboard.components.theme import inject_css  # noqa: E402

inject_css()

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

PAGE_ICONS = {
    "⚙️  Overview": "01",
    "📡  Sensor Monitoring": "02",
    "🏥  Statistical Health": "03",
    "🔮  Failure Forecast": "04",
    "🔍  Explainability": "05",
    "🔧  What-If Simulator": "06",
    "📊  Model Comparison": "07",
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
    n_stage = len([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
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
    f'{len(PAGES)} views</span></div>',
    unsafe_allow_html=True,
)

# ── Page selector (only navigation surface) ─────────────────────────────────
selected_page = st.sidebar.radio(
    "Navigation",
    list(PAGES.keys()),
    label_visibility="collapsed",
    key="nav_page",
)

n_results = (
    len([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
    if RESULTS_DIR.is_dir()
    else 0
)
st.sidebar.markdown(
    f"""
    <div class="side-footer">
        <span style="color:#93C5FD;">STAT-TWIN</span> · v0.1.0<br/>
        Theme · Dark Industrial<br/>
        Machine · <span style="color:#93C5FD;">{selected_machine}</span><br/>
        Results · <span style="color:#6EE7B7;">{n_results}</span> sets
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Hero header (main area) ─────────────────────────────────────────────────
page_label = selected_page.split("  ")[-1]
page_code = PAGE_ICONS.get(selected_page, "00")
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
                <span class="st-pill st-pill-accent">{page_code} · {page_label}</span>
                <span class="st-pill st-pill-success">● {selected_machine}</span>
                <span class="st-pill st-pill-violet">LIVE 2s</span>
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
