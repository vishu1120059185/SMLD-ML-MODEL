"""STAT-TWIN multipage Streamlit dashboard."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# ── Ensure project root is on sys.path so `stattwin` is importable ──────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Page config must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="STAT-TWIN",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark industrial CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .stApp { background-color: #0E1117; }
    section[data-testid="stSidebar"] { background-color: #161B22; }
    .stMarkdown { color: #C9D1D9; }

    /* Remove default padding */
    .block-container { padding-top: 1.5rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161B22; color: #8B949E;
        border-radius: 4px 4px 0 0; padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262D !important; color: #C9D1D9 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #161B22; border: 1px solid #21262D;
        border-radius: 8px; padding: 12px 16px;
    }
    [data-testid="stMetric"] label { color: #8B949E !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #C9D1D9 !important; }

    /* Selectbox / sliders */
    .stSelectbox > div > div { background-color: #161B22; color: #C9D1D9; }
    .stMultiSelect > div > div { background-color: #161B22; color: #C9D1D9; }

    /* Horizontal rule */
    hr { border-color: #21262D; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────
RESULTS_DIR = _PROJECT_ROOT / "results"

PAGES = {
    "Overview": "pages/1_overview",
    "Sensor Monitoring": "pages/2_sensors",
    "Statistical Health": "pages/3_health",
    "Failure Forecast": "pages/4_forecast",
    "Explainability": "pages/5_explain",
    "What-If Simulator": "pages/6_whatif",
    "Model Comparison": "pages/7_comparison",
}

st.sidebar.markdown(
    "<h2 style='color:#C9D1D9;margin-bottom:0;'>⚙ STAT-TWIN</h2>"
    "<p style='color:#8B949E;font-size:0.78rem;margin-top:2px;'>Digital-Twin Health Intelligence</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

# ── Machine selector (persisted across pages) ───────────────────────────────
available_machines: list[str] = []
meta_file = RESULTS_DIR / "meta.json"
if meta_file.exists():
    import json
    try:
        meta = json.loads(meta_file.read_text())
        available_machines = meta.get("machines", [])
    except Exception:
        pass

if not available_machines:
    # Fallback: scan results/ for machine sub-directories
    if RESULTS_DIR.is_dir():
        available_machines = sorted(
            d.name for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name != "global"
        )

if not available_machines:
    available_machines = ["MACHINE-001"]

selected_machine = st.sidebar.selectbox(
    "Machine",
    available_machines,
    index=0,
    key="selected_machine",
)
st.sidebar.caption(f"Active: **{selected_machine}**")
st.sidebar.markdown("---")

# ── Page selector ───────────────────────────────────────────────────────────
selected_page = st.sidebar.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption("© STAT-TWIN v0.1  ·  Dark Industrial Theme")

# ── Prime package hierarchy so relative imports work in page modules ────────
import types
import importlib

_PKG_DIR = Path(__file__).resolve().parent

def _ensure_package(name: str, pkg_path: Path) -> types.ModuleType:
    """Register a package in sys.modules if not already present."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [str(pkg_path)]
    mod.__package__ = name
    mod.__file__ = str(pkg_path / "__init__.py")
    sys.modules[name] = mod
    return mod

# Ensure the chain: stattwin → stattwin.dashboard → components, pages
_ensure_package("stattwin", _PROJECT_ROOT / "src" / "stattwin")
_ensure_package("stattwin.dashboard", _PKG_DIR)
_ensure_package("stattwin.dashboard.components", _PKG_DIR / "components")
_ensure_package("stattwin.dashboard.pages", _PKG_DIR / "pages")

# ── Dispatch to page module ─────────────────────────────────────────────────
page_module_path = PAGES[selected_page]
try:
    page_file = _PKG_DIR / f"{page_module_path}.py"
    mod_name = f"stattwin.dashboard.{page_module_path.replace('/', '.')}"
    spec = importlib.util.spec_from_file_location(mod_name, str(page_file))
    page_mod = importlib.util.module_from_spec(spec)
    page_mod.__package__ = "stattwin.dashboard.pages"
    sys.modules[mod_name] = page_mod
    spec.loader.exec_module(page_mod)
    page_mod.render()  # type: ignore[attr-defined]
except FileNotFoundError:
    st.error(f"Page file not found: `{page_module_path}.py`. Make sure the file exists.")
except Exception as exc:
    st.error(f"Error loading page **{selected_page}**: `{exc}`")
