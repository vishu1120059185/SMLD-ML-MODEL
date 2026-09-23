"""Shared Plotly chart components with dark industrial theme.

Enhanced with responsive sizing, unified hover, consistent theming,
and provenance badge support for chart titles.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from stattwin.dashboard.components.theme import (
    ACCENT,
    BG,
    BORDER,
    CARD,
    MONO,
    PLOTLY_LAYOUT,
    PROVENANCE_COLORS,
    PROVENANCE_DASH,
    STATE_COLORS,
    TEXT,
)

# Back-compat aliases
PAPER = CARD
GRID = BORDER
MUTED_OR = "#9CA3AF"

__all__ = [
    "STATE_COLORS",
    "PROVENANCE_COLORS",
    "PROVENANCE_DASH",
    "timeline_chart",
    "gauge_chart",
    "heatmap_chart",
    "bar_chart",
    "multi_line_chart",
    "donut_chart",
    "sparkline",
    "forest_plot",
]


def _layout(title: str = "", height: int = 400) -> dict:
    layout = copy.deepcopy(PLOTLY_LAYOUT)
    layout["title"] = dict(
        text=title,
        font=dict(size=14, color=TEXT, family=MONO),
        x=0.01,
        xanchor="left",
    )
    layout["height"] = height
    return layout


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
    color: str | None = None,
) -> go.Figure:
    """Line chart with optional rolling bands and state-change shading."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(timestamps), y=list(values),
        mode="lines", name="Value",
        line=dict(color=color or ACCENT, width=1.75, shape="spline", smoothing=0.4),
        fill="tozeroy" if color is None and not bands else None,
        fillcolor="rgba(59,130,246,0.06)" if color is None and not bands else None,
    ))

    if bands:
        for band in bands:
            fig.add_trace(go.Scatter(
                x=list(timestamps), y=list(band.get("upper", [])),
                mode="lines", line=dict(width=0), showlegend=False,
                hoverinfo="skip",
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
        fills = {
            "HEALTHY": "rgba(16,185,129,0.08)",
            "WATCH": "rgba(245,158,11,0.09)",
            "DEGRADING": "rgba(249,115,22,0.11)",
            "CRITICAL": "rgba(239,68,68,0.13)",
            "FAILURE-LIKELY": "rgba(153,27,27,0.15)",
        }
        for sc in state_changes:
            fig.add_vrect(
                x0=sc["start"], x1=sc["end"],
                fillcolor=fills.get(sc["state"], "rgba(0,0,0,0)"),
                layer="below", line_width=0,
                annotation_text=sc["state"],
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color=TEXT,
            )

    if provenance:
        prov_color = PROVENANCE_COLORS.get(provenance, TEXT)
        title = f"{title}  <span style='color:{prov_color};font-size:11px;'>[{provenance}]</span>"

    layout = _layout(title, height=380)
    if y_label:
        layout["yaxis"]["title"] = dict(text=y_label, font=dict(size=11, color=TEXT, family=MONO))
    fig.update_layout(**layout)
    return fig


def gauge_chart(value: float, title: str = "SHI", max_val: float = 1.0) -> go.Figure:
    """Semi-circular gauge with color-coded thresholds."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta" if max_val else "gauge+number",
        value=value,
        number=dict(suffix="", font=dict(size=26, color=TEXT, family=MONO)),
        gauge=dict(
            axis=dict(range=[0, max_val], tickwidth=1, tickcolor="#374151"),
            bar=dict(color=ACCENT, thickness=0.55),
            bgcolor=CARD,
            borderwidth=0,
            steps=[
                dict(range=[0, 0.25], color="#0F3D2C"),
                dict(range=[0.25, 0.50], color="#3D3210"),
                dict(range=[0.50, 0.75], color="#3D2410"),
                dict(range=[0.75, 1.0], color="#3D1218"),
            ],
            threshold=dict(
                line=dict(color=TEXT, width=3),
                thickness=0.75,
                value=min(value, max_val),
            ),
        ),
        title=dict(text=title, font=dict(size=13, color=MUTED_OR, family=MONO)),
    ))
    fig.update_layout(
        paper_bgcolor=CARD,
        font=dict(color=TEXT, family=MONO),
        height=250,
        margin=dict(l=28, r=28, t=42, b=8),
        showlegend=False,
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
        colorbar=dict(tickfont=dict(family=MONO, size=10), outlinewidth=0),
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
            marker_line=dict(width=0),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ))
        layout = _layout(title, height=max(300, 42 * len(list(categories))))
        if y_label:
            layout["xaxis"]["title"] = dict(
                text=y_label, font=dict(size=11, color=TEXT, family=MONO)
            )
    else:
        fig = go.Figure(go.Bar(
            x=list(categories), y=list(values),
            marker_color=color or ACCENT,
            marker_line=dict(width=0),
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        ))
        layout = _layout(title, height=350)
        if y_label:
            layout["yaxis"]["title"] = dict(
                text=y_label, font=dict(size=11, color=TEXT, family=MONO)
            )
    fig.update_layout(bargap=0.35, **layout)
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
                width=2.0,
            ),
        ))
    layout = _layout(title, height=380)
    if y_label:
        layout["yaxis"]["title"] = dict(text=y_label, font=dict(size=11, color=TEXT, family=MONO))
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
        hole=0.58,
        marker=dict(colors=list(colors) if colors else None, line=dict(color=CARD, width=2)),
        textfont=dict(size=11, family=MONO),
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=CARD,
        font=dict(color=TEXT, family=MONO),
        title=dict(text=title, font=dict(size=14, color=TEXT, family=MONO)),
        height=320,
        margin=dict(l=20, r=20, t=44, b=20),
        showlegend=True,
        legend=dict(font=dict(size=11, color=MUTED_OR)),
    )
    return fig


def sparkline(values: Sequence[float], *, height: int = 78, color: str = ACCENT) -> go.Figure:
    arr = np.asarray(list(values), dtype=float)
    fig = go.Figure(go.Scatter(
        y=arr.tolist(), mode="lines",
        line=dict(color=color, width=1.4),
        fill="tozeroy",
        fillcolor=f"{color}1A",
        hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=BG,
        height=height,
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(visible=False),
        yaxis=dict(
            visible=False,
            range=[float(arr.min()) - 1e-9, float(arr.max()) + 1e-9] if arr.size else None,
        ),
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
            array=[hi - m for m, hi in zip(means, ci_hi, strict=False)],
            arrayminus=[m - lo for m, lo in zip(means, ci_lo, strict=False)],
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
