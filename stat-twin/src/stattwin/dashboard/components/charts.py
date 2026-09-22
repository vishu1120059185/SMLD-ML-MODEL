"""Shared Plotly chart components with dark industrial theme."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from typing import Sequence

# ── Colour palette ──────────────────────────────────────────────────────────
BG = "#0E1117"
PAPER = "#161B22"
GRID = "#21262D"
TEXT = "#C9D1D9"
ACCENT = "#4C8BF5"

STATE_COLORS = {
    "HEALTHY": "#2E9E6B",
    "WATCH": "#E0B93B",
    "DEGRADING": "#E8862F",
    "CRITICAL": "#D64545",
    "FAILURE-LIKELY": "#8E1B3A",
}

PROVENANCE_COLORS = {
    "OBSERVED": "#4C8BF5",
    "PREDICTED": "#F5A623",
    "SIMULATED": "#9B6BFF",
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
                fillcolor=band.get("fill", "rgba(76,139,245,0.12)"),
                name=band.get("label", "Band"),
            ))

    if extra_traces:
        for t in extra_traces:
            fig.add_trace(t)

    if state_changes:
        colors = {
            "HEALTHY": "rgba(46,158,107,0.08)",
            "WATCH": "rgba(224,185,59,0.08)",
            "DEGRADING": "rgba(232,134,47,0.10)",
            "CRITICAL": "rgba(214,69,69,0.12)",
            "FAILURE-LIKELY": "rgba(142,27,58,0.14)",
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

    layout = _layout(title, height=380)
    layout["yaxis"]["title"] = y_label
    fig.update_layout(**layout)
    return fig


def gauge_chart(value: float, title: str = "SHI", max_val: float = 1.0) -> go.Figure:
    """Semi-circular gauge."""
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
                dict(range=[0, 0.25], color="#2E9E6B"),
                dict(range=[0.25, 0.50], color="#E0B93B"),
                dict(range=[0.50, 0.75], color="#E8862F"),
                dict(range=[0.75, 1.0], color="#D64545"),
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
        fill="tozeroy", fillcolor=f"rgba(76,139,245,0.10)",
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
