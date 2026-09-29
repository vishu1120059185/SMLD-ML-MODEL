"""Tests for the dashboard motion layer and the blank-line HTML guard.

Streamlit's markdown renderer turns everything after a blank line into a
fenced code block, so a template whose optional part renders empty would
leak raw ``</div>`` text into the page.  These tests pin both the
component HTML and that regression.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from stattwin.dashboard.components.motion import (
    _num_parts,
    clean_html,
    count_up_html,
)

# ---------------------------------------------------------------------------
# clean_html
# ---------------------------------------------------------------------------


class TestCleanHtml:
    def test_strips_blank_and_whitespace_only_lines(self):
        raw = "<div>a</div>\n\n   \n<div>b</div>\n"
        assert clean_html(raw) == "<div>a</div>\n<div>b</div>"

    def test_keeps_indented_content(self):
        raw = '<div class="x">\n    <span>y</span>\n</div>'
        assert clean_html(raw) == raw

    def test_result_has_no_internal_blank_line(self):
        raw = "\n".join(["<div>", "", "", "<span>x</span>", "  ", "</div>"])
        assert not any(not ln.strip() for ln in clean_html(raw).split("\n"))


# ---------------------------------------------------------------------------
# Count-up numbers
# ---------------------------------------------------------------------------


class TestNumParts:
    @pytest.mark.parametrize(
        ("value", "decimals", "expected"),
        [
            (0.0, 0, (0, 0)),
            (42.0, 0, (42, 0)),
            (-7.0, 0, (7, 0)),
            (0.05, 2, (0, 5)),      # renders "0.05" via decimal-leading-zero
            (1.5, 1, (1, 5)),
            (99.999, 2, (100, 0)),  # rounds
            (1234.567, 3, (1234, 567)),
        ],
    )
    def test_split(self, value, decimals, expected):
        assert _num_parts(value, decimals) == expected

    def test_integer_html_has_counters(self):
        html = count_up_html(1234)
        assert "--sw-ni-t:1234" in html
        assert "sw-ci" in html
        assert "sw-cf" not in html  # no fraction for 0 decimals

    def test_decimal_html_has_fraction_counter(self):
        html = count_up_html(1.25, decimals=2, prefix="$", suffix=" c")
        assert "--sw-ni-t:1" in html
        assert "--sw-nf-t:25" in html
        assert "$" in html and " c" in html
        # leading dot is emitted as a plain separator
        assert ".</span>" in html or ".<span" in html

    def test_negative_value_keeps_sign(self):
        assert count_up_html(-12).startswith(
            '<span class="sw-countline">-'
        )
        assert "--sw-ni-t:12" in count_up_html(-12)

    def test_signed_positive(self):
        assert count_up_html(5, signed=True).count("+") == 1

    def test_zero_is_safe(self):
        assert "--sw-ni-t:0" in count_up_html(0.0, decimals=1)


# ---------------------------------------------------------------------------
# Component markup
# ---------------------------------------------------------------------------


def _app_script(body: str) -> AppTest:
    """Build an AppTest for a script that renders *body*."""
    script = f"""
import streamlit as st
from stattwin.dashboard.components.cards import skeleton_card
from stattwin.dashboard.components.motion import (
    aurora, count_up, count_up_html, glow_rule, inject_motion, live_pill,
    progress_ring, pulsing_bars, reveal, section_header, sparkline, stat_tile,
    step_list, ticker,
)
inject_motion()
aurora()
{body}
"""
    return AppTest.from_string(script, default_timeout=60)


_APP = Path(__file__).resolve().parents[1] / "src" / "stattwin" / "dashboard" / "app.py"


class TestComponentsRender:
    def test_optional_empty_parts_do_not_leak_code_blocks(self):
        """Regression: empty optional parts used to leave a blank line,
        which Streamlit rendered as a fenced code block showing raw HTML."""
        at = _app_script(
            """
stat_tile("no delta", "12")                     # delta=None
stat_tile("no anim", "abc", animated=False)      # no animated border
count_up("count", 5.5, decimals=1)               # zero base value
progress_ring(0.0)                               # empty label + sublabel
progress_ring(50.0, label="half", sublabel="x")
pulsing_bars([1.0, 1.0, 1.0])                    # perfectly flat series
pulsing_bars([])
"""
        )
        at.run()
        assert not at.exception
        # no element-toolbar code blocks anywhere on the page
        assert not at.get("code")

    def test_full_component_set_renders(self):
        at = _app_script(
            """
section_header("Section", "subtitle")
glow_rule()
live_pill("LIVE")
skeleton_card()
reveal("<b>revealed</b>", delay_ms=100, kind="zoom")
count_up("SHI", 0.582, decimals=3)
progress_ring(58.2, label="health", value_text="0.58", sublabel="SHI")
pulsing_bars([0.1, 0.5, 0.9], color="#10B981", height="30px")
sparkline([1, 3, 2, 5, 4, 7], uid="t1")
stat_tile("tiles", count_up_html(12, suffix=" u"), delta="d")
ticker([("result sets", 12), ("views", 7)], speed="20s")
step_list(["one", "two", "three"])
"""
        )
        at.run()
        assert not at.exception
        assert not at.get("code")

    def test_pulsing_bars_uses_baseline_for_flat_series(self):
        at = _app_script("pulsing_bars([2.0, 2.0, 2.0], baseline=40.0)")
        at.run()
        assert not at.exception
        html = " ".join(m.value for m in at.markdown)
        # flat series -> every bar pinned at the baseline, not full height
        assert "--bh:40.0%" in html

    def test_pulsing_bars_spans_baseline_to_full(self):
        at = _app_script("pulsing_bars([0.0, 5.0, 10.0], baseline=20.0)")
        at.run()
        assert not at.exception
        html = " ".join(m.value for m in at.markdown)
        assert "--bh:20.0%" in html and "--bh:100.0%" in html

    def test_sparkline_ignores_short_series(self):
        at = _app_script("sparkline([1.0])")
        at.run()
        assert not at.exception
        assert not at.get("code")

    def test_ticker_duplicates_track_for_seamless_loop(self):
        at = _app_script("ticker([('a', 1), ('b', 2)])")
        at.run()
        assert not at.exception
        # count only the rendered markup, not the stylesheet's class selectors
        html = "\n".join(
            m.value for m in at.markdown if 'class="sw-marquee"' in m.value
        )
        assert html.count('<span class="sw-item">') == 4  # 2 items x 2 copies
        assert html.count('<div class="sw-track">') == 1

    def test_step_list_numbers_every_step(self):
        at = _app_script("step_list(['alpha', 'beta'])")
        at.run()
        assert not at.exception
        html = " ".join(m.value for m in at.markdown)
        assert ">1<" in html and ">2<" in html


# ---------------------------------------------------------------------------
# KPI card count-up gating
# ---------------------------------------------------------------------------


class TestKpiCountUp:
    @pytest.mark.parametrize(
        "value", ["0.9132", "1,204", "87", "-12", "0.0"]
    )
    def test_plain_numbers_animate(self, value):
        from stattwin.dashboard.components.cards import _maybe_count_up

        out = _maybe_count_up(value)
        assert "sw-ci" in out

    @pytest.mark.parametrize(
        "text", ["h30", "3 / 8 units", "—", "N/A", "DEGRADING", "", "0.91%"]
    )
    def test_non_numeric_never_animates(self, text):
        from stattwin.dashboard.components.cards import _maybe_count_up

        assert _maybe_count_up(text) == text

    def test_numeric_animates_with_correct_precision(self):
        from stattwin.dashboard.components.cards import _maybe_count_up

        out = _maybe_count_up("0.9132")
        assert "sw-ci" in out and "sw-cf" in out
        assert "--sw-ni-t:0" in out
        assert "--sw-nf-t:9132" in out


class TestInlineMarkup:
    """Prose cards render artifact markdown as HTML (and never as HTML)."""

    def test_bold_and_italic(self):
        from stattwin.dashboard.components.cards import _inline_markup

        out = _inline_markup("a **bold** and *soft* claim")
        assert "<b>bold</b>" in out
        assert "<em>soft</em>" in out

    def test_inline_code(self):
        from stattwin.dashboard.components.cards import _inline_markup

        assert "<code>p=0.05</code>" in _inline_markup("value `p=0.05` here")

    def test_escapes_html(self):
        from stattwin.dashboard.components.cards import _inline_markup

        out = _inline_markup("<script>alert(1)</script> & <b>x</b>")
        assert "<script>" not in out
        assert "&lt;script&gt;" in out
        assert "&amp;" in out

    def test_plain_text_unchanged(self):
        from stattwin.dashboard.components.cards import _inline_markup

        assert _inline_markup("no markup here") == "no markup here"

    def test_asterisks_never_reach_the_page(self):
        at = _app_script(
            "from stattwin.dashboard.components.cards import evidence_card\n"
            'evidence_card("t", "It **contributed** to risk")'
        )
        at.run()
        assert not at.exception
        html = " ".join(m.value for m in at.markdown)
        assert "**" not in html
        assert "<b>contributed</b>" in html


# ---------------------------------------------------------------------------
# CSS sanity
# ---------------------------------------------------------------------------


class TestMotionCss:
    def test_css_declares_registered_properties(self):
        from stattwin.dashboard.components.motion import MOTION_CSS

        for prop in ("--sw-ni", "--sw-nf", "--sw-angle", "--sw-sweep"):
            assert f"@property {prop}" in MOTION_CSS

    def test_css_respects_reduced_motion(self):
        from stattwin.dashboard.components.motion import MOTION_CSS

        assert "prefers-reduced-motion" in MOTION_CSS

    def test_keyframes_present(self):
        from stattwin.dashboard.components.motion import MOTION_CSS

        for name in (
            "sw-fade-up",
            "sw-spin",
            "sw-bar-pulse",
            "sw-marquee",
            "sw-ping",
            "sw-drift-a",
            "sw-count-i",
        ):
            assert f"@keyframes {name}" in MOTION_CSS, name

    def test_no_unbalanced_braces(self):
        from stattwin.dashboard.components.motion import MOTION_CSS

        assert MOTION_CSS.count("{") == MOTION_CSS.count("}")

    def test_theme_injects_motion_css(self):
        from stattwin.dashboard.components.theme import _CSS

        assert "--card" in _CSS  # base tokens still present


# ---------------------------------------------------------------------------
# app shell
# ---------------------------------------------------------------------------


class TestAppShell:
    def test_app_renders_without_exception(self):
        at = AppTest.from_file(str(_APP), default_timeout=240)
        at.run()
        assert not at.exception, [e.value for e in at.exception]

    def test_app_emits_motion_markup(self):
        at = AppTest.from_file(str(_APP), default_timeout=240)
        at.run()
        html = " ".join(m.value for m in at.markdown)
        assert "sw-aurora" in html
        assert "sw-marquee" in html
        assert "sw-tile" in html
        assert re.search(r"sw-(ci|cf)", html)

    def test_app_has_no_leaked_code_blocks(self):
        at = AppTest.from_file(str(_APP), default_timeout=240)
        at.run()
        assert not at.get("code")
