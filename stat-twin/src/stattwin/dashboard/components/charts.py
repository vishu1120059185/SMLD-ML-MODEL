"""Shared Plotly chart components with dark industrial theme.

Enhanced with responsive sizing, unified hover, consistent theming,
and provenance badge support for chart titles.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

# ── Colour palette (matches app.py industrial dark theme) ──────────────────
BG = "#0A0E17"
PAPER = "#111827"
GRID = "#1F2937"
TEXT = "#F9FAFB"
ACCENT = "#3B82F6"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"

STATE_COLORS = {
    "HEALTHY": SUCCESS,
    "WATCH": WARNING,
    "DEGRADING": "#F97316",
    "CRITICAL": DANGER,
    "FAILURE-LIKELY": "#991B1B",
}

PROVENANCE_COLORS = {
    "OBSERVED": ACCENT,
    "PREDICTED": WARNING,
    "SIMULATED": "#8B5CF6",
}

PROVENANCE_DASH = {
    "OBSERVED": "solid",
    "PREDICTED": "dot",
    "SIMULATED": "dash",
}


def _layout(title: str = "", height: int = 400) -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="JetBrains Mono, Fira Code, monospace"),
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=50, r=30, t=45, b=40),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=PAPER,
            bordercolor=GRID,
            font=dict(color=TEXT, family="JetBrains Mono, Fira Code, monospace", size=12),
        ),
    )


def timeline_chart(
    timestamps: Sequence,
    values: Sequence,
    *,
    title: str = "Timeline",
    y_label: str = "",
    bands: list[dict] | None = None,
    extra_traces: list[go.Scatter] | None = None,
    state_changes: list[dict] | None = None,
    provenance: str | None = None,
) -> go.Figure:
    """Line chart with optional rolling bands and state-change shading."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(timestamps), y=list(values),
        mode="lines", name="Value",
        line=dict(color=ACCENT, width=1.5),
    ))

    if bands:
        for band in bands:
            fig.add_trace(go.Scatter(
                x=list(timestamps), y=list(band.get("upper", [])),
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=list(timestamps), y=list(band.get("lower", [])),
                fill="tonexty", mode="lines", line=dict(width=0),
                fillcolor=band.get("fill", "rgba(59,130,246,0.12)"),
                name=band.get("label", "Band"),
            ))

    if extra_traces:
        for t in extra_traces:
            fig.add_trace(t)

    if state_changes:
        colors = {
            "HEALTHY": "rgba(16,185,129,0.08)",
            "WATCH": "rgba(245,158,11,0.08)",
            "DEGRADING": "rgba(249,115,22,0.10)",
            "CRITICAL": "rgba(239,68,68,0.12)",
            "FAILURE-LIKELY": "rgba(153,27,27,0.14)",
        }
        for sc in state_changes:
            fig.add_vrect(
                x0=sc["start"], x1=sc["end"],
                fillcolor=colors.get(sc["state"], "rgba(0,0,0,0)"),
                layer="below", line_width=0,
                annotation_text=sc["state"],
                annotation_position="top left",
                annotation_font_size=9,
            )

    if provenance:
        prov_color = PROVENANCE_COLORS.get(provenance, TEXT)
        title = f"{title}  <span style='color:{prov_color};font-size:11px;'>[{provenance}]</span>"

    layout = _layout(title, height=380)
    layout["yaxis"]["title"] = y_label
    fig.update_layout(**layout)
    return fig


def gauge_chart(value: float, title: str = "SHI", max_val: float = 1.0) -> go.Figure:
    """Semi-circular gauge with color-coded thresholds."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix="", font=dict(size=28)),
        gauge=dict(
            axis=dict(range=[0, max_val], tickwidth=1),
            bar=dict(color=ACCENT),
            bgcolor=PAPER,
            borderwidth=0,
            steps=[
                dict(range=[0, 0.25], color="#10B981"),
                dict(range=[0.25, 0.50], color="#F59E0B"),
                dict(range=[0.50, 0.75], color="#F97316"),
                dict(range=[0.75, 1.0], color="#EF4444"),
            ],
        ),
        title=dict(text=title, font=dict(size=14)),
    ))
    fig.update_layout(
        paper_bgcolor=PAPER,
        font=dict(color=TEXT, family="JetBrains Mono, Fira Code, monospace"),
        height=260, margin=dict(l=30, r=30, t=40, b=10),
    )
    return fig


def heatmap_chart(
    z: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str = "Heatmap",
    colorscale: str = "RdYlGn_r",
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z, x=list(x_labels), y=list(y_labels),
        colorscale=colorscale, reversescale=False,
        hoverongaps=False,
    ))
    layout = _layout(title, height=400)
    fig.update_layout(**layout)
    return fig


def bar_chart(
    categories: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "Bar Chart",
    y_label: str = "",
    color: str | None = None,
    horizontal: bool = False,
) -> go.Figure:
    if horizontal:
        fig = go.Figure(go.Bar(
            y=list(categories), x=list(values),
            orientation="h",
            marker_color=color or ACCENT,
        ))
    else:
        fig = go.Figure(go.Bar(
            x=list(categories), y=list(values),
            marker_color=color or ACCENT,
        ))
    layout = _layout(title, height=350)
    layout["yaxis"]["title"] = y_label
    fig.update_layout(**layout)
    return fig


def multi_line_chart(
    traces_data: list[dict],
    *,
    title: str = "Multi-Line",
    y_label: str = "",
) -> go.Figure:
    """Generic multi-trace line chart. Each dict has keys: x, y, name, dash?, color?"""
    fig = go.Figure()
    for td in traces_data:
        fig.add_trace(go.Scatter(
            x=td["x"], y=td["y"],
            mode="lines", name=td["name"],
            line=dict(
                color=td.get("color", ACCENT),
                dash=td.get("dash", "solid"),
                width=1.8,
            ),
        ))
    layout = _layout(title, height=380)
    layout["yaxis"]["title"] = y_label
    fig.update_layout(**layout)
    return fig


def donut_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "Donut",
    colors: Sequence[str] | None = None,
) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=list(labels), values=list(values),
        hole=0.55,
        marker=dict(colors=list(colors) if colors else None),
        textfont=dict(size=11),
    ))
    fig.update_layout(
        paper_bgcolor=PAPER,
        font=dict(color=TEXT, family="JetBrains Mono, Fira Code, monospace"),
        title=dict(text=title, font=dict(size=14)),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
    )
    return fig


def sparkline(values: Sequence[float], *, height: int = 80, color: str = ACCENT) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=list(values), mode="lines",
        line=dict(color=color, width=1.2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.10)",
    ))
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=BG,
        height=height,
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def forest_plot(
    labels: list[str],
    means: list[float],
    ci_lo: list[float],
    ci_hi: list[float],
    *,
    title: str = "Effect Size Forest Plot",
    zero_line: float = 0.0,
) -> go.Figure:
    """Forest plot for ablation effect sizes with confidence intervals."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=means, y=labels,
        error_x=dict(
            type="data",
            symmetric=False,
            array=[hi - m for m, hi in zip(means, ci_hi)],
            arrayminus=[m - lo for m, lo in zip(means, ci_lo)],
        ),
        mode="markers",
        marker=dict(color=ACCENT, size=10),
        name="Effect size",
    ))
    fig.add_vline(x=zero_line, line_dash="dash", line_color=GRID, line_width=1)
    layout = _layout(title, height=max(200, len(labels) * 40 + 100))
    layout["xaxis"]["title"] = "Effect size (95% CI)"
    fig.update_layout(**layout)
    return fig
