"""STAT-TWIN motion layer — CSS-only animation primitives for Streamlit.

Why CSS-only
------------
Streamlit re-creates its DOM on every rerun (page switch, widget change),
so JavaScript that mutates the DOM is wiped.  Everything here is therefore
plain HTML + CSS keyframes, which Streamlit renders server-side and
survives reruns.  Animations replay on each rerun, which is exactly the
"slides" feel we want.

Design references
-----------------
* 21st.dev — aurora/spotlight backdrops, animated conic-gradient borders,
  pulsing bar stacks, count-up metric tiles, bento grids.
* motion.dev / motion-slides — staggered entrance reveals, blur-in text,
  grow-in rules, marquee ribbons, easing ``cubic-bezier(.16,1,.3,1)``.

Components
----------
``reveal``          staggered fade-up/blur-in wrapper
``count_up``        animated numeric KPI (CSS counters, decimals safe)
``progress_ring``   conic-gradient ring with sweeping gradient
``pulsing_bars``    21st.dev-style animated bar stack
``stat_tile``       bento-style metric tile
``ticker``          infinite marquee ribbon
``step_list``       motion-slides style reveal-on-step list
``aurora``          animated ambient background layer
"""
from __future__ import annotations

import math
from typing import Any

import streamlit as st

__all__ = [
    "MOTION_CSS",
    "inject_motion",
    "aurora",
    "reveal",
    "count_up",
    "count_up_html",
    "progress_ring",
    "pulsing_bars",
    "stat_tile",
    "ticker",
    "step_list",
    "glow_rule",
    "live_pill",
]

# ── Easing + timing tokens ───────────────────────────────────────────────────
EASE = "cubic-bezier(.16,1,.3,1)"      # motion-slides signature ease-out
EASE_IO = "cubic-bezier(.65,0,.35,1)"
DUR_REVEAL = "0.62s"
DUR_FAST = "0.28s"
DUR_SPIN = "6s"

MOTION_CSS = """
<style>
/* ══════════════════════════════════════════════════════════════════════════
   STAT-TWIN MOTION SYSTEM
   CSS-only animation layer (Streamlit wipes JS DOM mutations on rerun)
   ══════════════════════════════════════════════════════════════════════════ */

/* Registered custom properties so they can be *animated* */
@property --sw-ni { syntax: "<integer>"; initial-value: 0; inherits: false; }
@property --sw-nf { syntax: "<integer>"; initial-value: 0; inherits: false; }
@property --sw-angle { syntax: "<angle>"; initial-value: 0deg; inherits: false; }
@property --sw-sweep { syntax: "<percentage>"; initial-value: 0%; inherits: false; }
@property --sw-hue { syntax: "<number>"; initial-value: 210; inherits: false; }

/* ── Ambient aurora backdrop ─────────────────────────────────────────────── */
.sw-aurora {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  overflow: hidden; contain: strict;
}
.sw-aurora .sw-orb {
  position: absolute; border-radius: 50%; filter: blur(90px);
  opacity: 0.5; will-change: transform;
}
.sw-aurora .sw-o1 {
  width: 46vw; height: 46vw; top: -14vw; left: -6vw;
  background: radial-gradient(circle at 40% 40%, rgba(59,130,246,.34), transparent 68%);
  animation: sw-drift-a 26s var(--ease-io) infinite;
}
.sw-aurora .sw-o2 {
  width: 40vw; height: 40vw; bottom: -14vw; right: -8vw;
  background: radial-gradient(circle at 60% 60%, rgba(16,185,129,.24), transparent 68%);
  animation: sw-drift-b 31s var(--ease-io) infinite;
}
.sw-aurora .sw-o3 {
  width: 34vw; height: 34vw; top: 42%; left: 46%;
  background: radial-gradient(circle at 50% 50%, rgba(139,92,246,.20), transparent 66%);
  animation: sw-drift-c 37s var(--ease-io) infinite;
}
.sw-aurora .sw-grid {
  position: absolute; inset: -2px;
  background-image:
    linear-gradient(rgba(148,163,184,.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148,163,184,.055) 1px, transparent 1px);
  background-size: 46px 46px;
  mask-image: radial-gradient(ellipse 90% 70% at 50% 0%, #000 20%, transparent 78%);
  -webkit-mask-image: radial-gradient(ellipse 90% 70% at 50% 0%, #000 20%, transparent 78%);
  animation: sw-grid-breathe 14s ease-in-out infinite;
}
.sw-aurora .sw-sweep {
  position: absolute; inset: 0;
  background: linear-gradient(105deg, transparent 42%, rgba(96,165,250,.05) 50%, transparent 58%);
  animation: sw-sweep-x 11s ease-in-out infinite;
}
@keyframes sw-drift-a {
  0%,100% { transform: translate3d(0,0,0) scale(1); }
  33%     { transform: translate3d(7vw,5vh,0) scale(1.10); }
  66%     { transform: translate3d(-3vw,8vh,0) scale(.94); }
}
@keyframes sw-drift-b {
  0%,100% { transform: translate3d(0,0,0) scale(1); }
  40%     { transform: translate3d(-8vw,-6vh,0) scale(1.14); }
  70%     { transform: translate3d(3vw,-3vh,0) scale(.92); }
}
@keyframes sw-drift-c {
  0%,100% { transform: translate3d(-50%,-50%,0) scale(1); opacity:.75; }
  50%     { transform: translate3d(-58%,-42%,0) scale(1.22); opacity:1; }
}
@keyframes sw-grid-breathe {
  0%,100% { opacity:.5; }
  50%     { opacity:.9; }
}
@keyframes sw-sweep-x {
  0%   { transform: translateX(-40%); }
  100% { transform: translateX(40%); }
}

/* Keep app content above the aurora */
.stApp [data-testid="stAppViewContainer"],
.stApp section[data-testid="stSidebar"],
.stApp header { position: relative; z-index: 1; }

/* ── Scroll progress rail (scroll-driven, degrades to static) ───────────── */
.sw-scrollrail {
  position: fixed; top: 0; left: 0; right: 0; height: 2px; z-index: 99;
  background: linear-gradient(90deg, #3B82F6, #10B981, #8B5CF6, #3B82F6);
  background-size: 300% 100%;
  transform-origin: 0 50%;
  animation: sw-rail-slide 3.4s linear infinite,
             sw-rail-grow linear both;
  animation-timeline: auto, scroll(root block);
  animation-range: normal, normal;
  pointer-events: none;
}
@keyframes sw-rail-slide { to { background-position: 300% 0; } }
@keyframes sw-rail-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

/* ── Entrance reveals ────────────────────────────────────────────────────── */
.sw-reveal {
  animation: sw-fade-up var(--dur, .62s) var(--ease, cubic-bezier(.16,1,.3,1)) both;
  animation-delay: var(--d, 0ms);
  will-change: opacity, transform, filter;
}
@keyframes sw-fade-up {
  from { opacity: 0; transform: translateY(16px) scale(.985); filter: blur(7px); }
  to   { opacity: 1; transform: none; filter: blur(0); }
}
.sw-reveal-left {
  animation: sw-fade-left var(--dur, .62s) var(--ease, cubic-bezier(.16,1,.3,1)) both;
  animation-delay: var(--d, 0ms);
}
@keyframes sw-fade-left {
  from { opacity: 0; transform: translateX(-22px); filter: blur(6px); }
  to   { opacity: 1; transform: none; filter: blur(0); }
}
.sw-reveal-zoom {
  animation: sw-zoom-in var(--dur, .62s) var(--ease, cubic-bezier(.16,1,.3,1)) both;
  animation-delay: var(--d, 0ms);
}
@keyframes sw-zoom-in {
  from { opacity: 0; transform: scale(.94); filter: blur(8px); }
  to   { opacity: 1; transform: none; filter: blur(0); }
}
/* Auto-stagger direct children of a single markdown block */
.sw-stagger > * { animation: sw-fade-up .58s var(--ease) both; }
.sw-stagger > *:nth-child(1)  { animation-delay: 0ms; }
.sw-stagger > *:nth-child(2)  { animation-delay: 60ms; }
.sw-stagger > *:nth-child(3)  { animation-delay: 120ms; }
.sw-stagger > *:nth-child(4)  { animation-delay: 180ms; }
.sw-stagger > *:nth-child(5)  { animation-delay: 240ms; }
.sw-stagger > *:nth-child(6)  { animation-delay: 300ms; }
.sw-stagger > *:nth-child(7)  { animation-delay: 360ms; }
.sw-stagger > *:nth-child(8)  { animation-delay: 420ms; }
.sw-stagger > *:nth-child(9)  { animation-delay: 480ms; }
.sw-stagger > *:nth-child(10) { animation-delay: 540ms; }
.sw-stagger > *:nth-child(11) { animation-delay: 600ms; }
.sw-stagger > *:nth-child(12) { animation-delay: 660ms; }

/* ── Animated gradient text ──────────────────────────────────────────────── */
.sw-shine {
  background: linear-gradient(100deg, #93C5FD 0%, #3B82F6 22%, #34D399 46%,
              #A78BFA 68%, #93C5FD 100%);
  background-size: 260% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: sw-shine 7s linear infinite;
}
@keyframes sw-shine { to { background-position: 260% 0; } }

/* ── Animated conic-gradient border (21st.dev) ──────────────────────────── */
.sw-animborder { position: relative; isolation: isolate; }
.sw-animborder::before {
  content: ""; position: absolute; inset: -1px; border-radius: inherit;
  padding: 1px; pointer-events: none; z-index: 2;
  background: conic-gradient(from var(--sw-angle, 0deg),
    rgba(59,130,246,0) 0%, rgba(59,130,246,.85) 12%, rgba(16,185,129,.9) 26%,
    rgba(139,92,246,.85) 40%, rgba(59,130,246,0) 58%, rgba(59,130,246,0) 100%);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude;
  animation: sw-spin var(--spin, 6s) linear infinite;
}
@keyframes sw-spin { to { --sw-angle: 360deg; } }
.sw-animborder:hover::before { animation-duration: 2.2s; }

/* ── Spotlight hover (pure CSS centred glow) ────────────────────────────── */
.sw-spot {
  position: relative; overflow: hidden;
  transition: transform var(--dur-fast) var(--ease),
              box-shadow var(--dur-fast) var(--ease);
}
.sw-spot::after {
  content: ""; position: absolute; inset: -1px; border-radius: inherit;
  pointer-events: none; opacity: 0; transition: opacity var(--dur-fast) ease;
  background: radial-gradient(420px circle at 50% 0%, rgba(96,165,250,.16), transparent 62%);
}
.sw-spot:hover { transform: translateY(-3px); box-shadow: 0 18px 40px -18px rgba(37,99,235,.55); }
.sw-spot:hover::after { opacity: 1; }

/* ── Count-up numbers (CSS counters; static fallback = target) ──────────── */
.sw-ci {
  --sw-ni: var(--sw-ni-t, 0);
  counter-reset: sw-i var(--sw-ni);
  animation: sw-count-i var(--dur, 1.15s) var(--ease) both;
}
.sw-ci::after { content: counter(sw-i); }
.sw-cf {
  --sw-nf: var(--sw-nf-t, 0);
  counter-reset: sw-f var(--sw-nf);
  animation: sw-count-f var(--dur, 1.15s) var(--ease) both;
}
.sw-cf::after { content: counter(sw-f, decimal-leading-zero); }
@keyframes sw-count-i { from { --sw-ni: 0; } }
@keyframes sw-count-f { from { --sw-nf: 0; } }
.sw-countline { display: inline-flex; align-items: baseline; gap: 1px; }

/* ── Progress ring (conic sweep + glow) ──────────────────────────────────── */
.sw-ring {
  --sw-sweep: var(--pct, 0%);
  position: relative; width: var(--size, 116px); height: var(--size, 116px);
  border-radius: 50%; display: grid; place-items: center;
  background:
    conic-gradient(from -90deg, var(--ring-c, #3B82F6) var(--sw-sweep),
                  rgba(148,163,184,.13) var(--sw-sweep) 100%);
  animation: sw-ring-in 1.05s var(--ease) both, sw-ring-sweep 9s linear infinite;
  filter: drop-shadow(0 0 12px rgba(59,130,246,.32));
}
.sw-ring::before {
  content: ""; position: absolute; inset: 9px; border-radius: 50%;
  background: #0E1420; border: 1px solid rgba(148,163,184,.10);
}
.sw-ring > * { position: relative; z-index: 1; }
@keyframes sw-ring-in { from { --sw-sweep: 0%; } }
@keyframes sw-ring-sweep {
  0%,100% { filter: drop-shadow(0 0 12px rgba(59,130,246,.30)) hue-rotate(0deg); }
  50%     { filter: drop-shadow(0 0 18px rgba(16,185,129,.38)) hue-rotate(24deg); }
}

/* ── Pulsing bars (21st.dev PulsingBars) ────────────────────────────────── */
.sw-bars { display: flex; align-items: flex-end; gap: 5px; height: var(--h, 44px); }
.sw-bars .sw-bar {
  flex: 1 1 0; min-width: 4px; border-radius: 3px 3px 1px 1px;
  background: linear-gradient(180deg, var(--bar-c, #3B82F6), rgba(59,130,246,.22));
  height: var(--bh, 40%);
  animation: sw-bar-pulse 1.5s ease-in-out infinite;
  animation-delay: var(--bd, 0ms);
  transform-origin: 50% 100%;
}
@keyframes sw-bar-pulse {
  0%,100% { transform: scaleY(1); opacity: .62; filter: saturate(1); }
  50%     { transform: scaleY(1.16); opacity: 1; filter: saturate(1.35) brightness(1.15); }
}

/* ── Marquee ribbon (motion-slides) ─────────────────────────────────────── */
.sw-marquee {
  position: relative; overflow: hidden; border-radius: 9px;
  border: 1px solid rgba(148,163,184,.14);
  background: linear-gradient(90deg, rgba(17,24,39,.94), rgba(14,20,32,.72));
  padding: 7px 0;
  mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent);
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent);
}
.sw-marquee .sw-track {
  display: inline-flex; gap: 34px; white-space: nowrap;
  animation: sw-marquee var(--speed, 26s) linear infinite;
  will-change: transform;
}
.sw-marquee:hover .sw-track { animation-play-state: paused; }
.sw-marquee .sw-item {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: 'JetBrains Mono', monospace; font-size: .7rem;
  color: #9CA3AF; letter-spacing: .4px;
}
.sw-marquee .sw-item b { color: #E5E7EB; font-weight: 700; }
.sw-marquee .sw-sep { color: rgba(148,163,184,.35); }
@keyframes sw-marquee { from { transform: translateX(0); } to { transform: translateX(-50%); } }

/* ── Section header with grow-in rule (motion-slides) ───────────────────── */
.sw-hd {
  position: relative; padding-left: 15px; margin: 1.05rem 0 .5rem;
  animation: sw-fade-left .55s var(--ease) both; animation-delay: var(--d, 0ms);
}
.sw-hd::before {
  content: ""; position: absolute; left: 0; top: .18em; bottom: .18em; width: 3px;
  border-radius: 3px;
  background: linear-gradient(180deg, #3B82F6, #10B981 60%, rgba(139,92,246,.85));
  transform-origin: 50% 0; animation: sw-grow-y .68s var(--ease) both;
  animation-delay: calc(var(--d, 0ms) + 90ms);
}
.sw-hd .sw-hd-title {
  font-size: 1.02rem; font-weight: 720; color: #F9FAFB; letter-spacing: -.01em;
  line-height: 1.25;
}
.sw-hd .sw-hd-sub {
  color: #9CA3AF; font-size: .76rem; margin-top: 2px;
  font-family: 'JetBrains Mono', monospace; letter-spacing: .2px;
}
@keyframes sw-grow-y { from { transform: scaleY(0); } to { transform: scaleY(1); } }
.sw-hr {
  height: 1px; border: 0; margin: .85rem 0 1rem;
  background: linear-gradient(90deg, rgba(59,130,246,.55), rgba(16,185,129,.28), transparent 78%);
  transform-origin: 0 50%; animation: sw-grow-x .8s var(--ease) both;
}
@keyframes sw-grow-x {
  from { transform: scaleX(0); opacity: 0; }
  to   { transform: scaleX(1); opacity: 1; }
}

/* ── Live pill ───────────────────────────────────────────────────────────── */
.sw-live {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 4px 12px; border-radius: 999px;
  font-family: 'JetBrains Mono', monospace; font-size: .64rem; font-weight: 700;
  letter-spacing: 1.1px; text-transform: uppercase;
  color: #6EE7B7; border: 1px solid rgba(16,185,129,.42);
  background: rgba(16,185,129,.10);
}
.sw-live .sw-live-dot {
  width: 7px; height: 7px; border-radius: 50%; background: #10B981;
  position: relative;
}
.sw-live .sw-live-dot::after {
  content: ""; position: absolute; inset: -4px; border-radius: 50%;
  border: 1.5px solid rgba(16,185,129,.75);
  animation: sw-ping 1.7s var(--ease) infinite;
}
@keyframes sw-ping {
  0%   { transform: scale(.55); opacity: .95; }
  80%  { transform: scale(2.1); opacity: 0; }
  100% { transform: scale(2.1); opacity: 0; }
}

/* ── Bento stat tile ─────────────────────────────────────────────────────── */
.sw-tile {
  position: relative; overflow: hidden; border-radius: 13px;
  padding: 14px 16px; height: 100%;
  background: linear-gradient(158deg, rgba(23,31,48,.94) 0%, rgba(13,19,32,.96) 100%);
  border: 1px solid rgba(148,163,184,.13);
  box-shadow: 0 10px 26px -18px rgba(0,0,0,.9);
  transition: transform var(--dur-fast) var(--ease), border-color var(--dur-fast) ease,
              box-shadow var(--dur-fast) ease;
  animation: sw-fade-up .6s var(--ease) both; animation-delay: var(--d, 0ms);
}
.sw-tile:hover {
  transform: translateY(-4px) scale(1.012);
  border-color: rgba(96,165,250,.42);
  box-shadow: 0 22px 44px -22px rgba(37,99,235,.6);
}
.sw-tile .sw-tile-label {
  font-size: .64rem; color: #9CA3AF; text-transform: uppercase;
  letter-spacing: 1.25px; font-weight: 700; margin-bottom: 7px;
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
}
.sw-tile .sw-tile-value {
  font-family: 'JetBrains Mono', monospace; font-weight: 750;
  font-size: var(--fs, 1.5rem); line-height: 1.1; color: #F9FAFB;
}
.sw-tile .sw-tile-delta {
  font-family: 'JetBrains Mono', monospace; font-size: .7rem;
  color: #6B7280; margin-top: 5px;
}
.sw-tile .sw-tile-glow {
  position: absolute; right: -30px; top: -30px; width: 120px; height: 120px;
  border-radius: 50%; pointer-events: none;
  background: radial-gradient(circle, var(--tile-c, rgba(59,130,246,.34)), transparent 68%);
  opacity: .55; animation: sw-tile-glow 4.4s ease-in-out infinite;
}
@keyframes sw-tile-glow {
  0%,100% { transform: scale(1); opacity: .45; }
  50%     { transform: scale(1.28); opacity: .8; }
}

/* ── Sparkline (inline SVG, animated draw) ───────────────────────────────── */
.sw-spark { display: block; overflow: visible; }
.sw-spark path.sw-spark-line {
  fill: none; stroke: var(--sp-c, #3B82F6); stroke-width: 1.8;
  stroke-linecap: round; stroke-linejoin: round;
  stroke-dasharray: var(--sp-len, 400); stroke-dashoffset: var(--sp-len, 400);
  animation: sw-draw 1.5s var(--ease) both; animation-delay: var(--d, 120ms);
  filter: drop-shadow(0 0 5px rgba(59,130,246,.5));
}
@keyframes sw-draw { to { stroke-dashoffset: 0; } }
.sw-spark path.sw-spark-area {
  fill: url(#swSparkGrad); opacity: 0;
  animation: sw-fade-in 1.1s var(--ease) both; animation-delay: .34s;
}
@keyframes sw-fade-in { to { opacity: 1; } }
.sw-spark circle.sw-spark-head {
  fill: var(--sp-c, #3B82F6);
  animation: sw-head-pop .7s var(--ease) both; animation-delay: .9s;
}
@keyframes sw-head-pop { from { r: 0; opacity: 0; } to { r: 2.6; opacity: 1; } }

/* ── Step list (motion-slides reveal) ───────────────────────────────────── */
.sw-steps { display: flex; flex-direction: column; gap: 9px; }
.sw-steps .sw-step {
  display: flex; align-items: flex-start; gap: 11px;
  padding: 10px 13px; border-radius: 10px;
  background: rgba(17,24,39,.72); border: 1px solid rgba(148,163,184,.11);
  animation: sw-fade-left .55s var(--ease) both; animation-delay: var(--d, 0ms);
  transition: border-color var(--dur-fast) ease, background var(--dur-fast) ease;
}
.sw-steps .sw-step:hover { border-color: rgba(96,165,250,.36); background: rgba(23,31,48,.9); }
.sw-steps .sw-step .sw-step-idx {
  flex: 0 0 auto; width: 22px; height: 22px; border-radius: 7px;
  display: grid; place-items: center;
  font-family: 'JetBrains Mono', monospace; font-size: .66rem; font-weight: 700;
  color: #DBEAFE; background: rgba(59,130,246,.18);
  border: 1px solid rgba(59,130,246,.35);
}
.sw-steps .sw-step .sw-step-text { color: #D1D5DB; font-size: .82rem; line-height: 1.45; }

/* ── Skeleton shimmer ────────────────────────────────────────────────────── */
.sw-skel {
  height: var(--h, 58px); border-radius: 11px;
  background: linear-gradient(100deg, rgba(17,24,39,.9) 22%, rgba(35,45,66,.95) 42%,
              rgba(17,24,39,.9) 62%);
  background-size: 260% 100%;
  animation: sw-shimmer 1.35s linear infinite;
  border: 1px solid rgba(148,163,184,.08);
}
@keyframes sw-shimmer { from { background-position: 160% 0; } to { background-position: -60% 0; } }

/* ── Widget polish: animated tab indicator + focus rings ─────────────────── */
.stTabs [data-baseweb="tab-highlight"] {
  background: linear-gradient(90deg, #3B82F6, #10B981) !important;
  animation: sw-tab-glow 2.6s ease-in-out infinite;
}
@keyframes sw-tab-glow {
  0%,100% { box-shadow: 0 0 0 rgba(59,130,246,0); }
  50%     { box-shadow: 0 0 14px rgba(59,130,246,.85); }
}
.stTabs [data-baseweb="tab"] { animation: sw-fade-up .5s var(--ease) both; }
.stTabs [data-baseweb="tab"]:nth-child(1) { animation-delay: 0ms; }
.stTabs [data-baseweb="tab"]:nth-child(2) { animation-delay: 55ms; }
.stTabs [data-baseweb="tab"]:nth-child(3) { animation-delay: 110ms; }
.stTabs [data-baseweb="tab"]:nth-child(4) { animation-delay: 165ms; }
.stTabs [data-baseweb="tab"]:nth-child(5) { animation-delay: 220ms; }
.stTabs [data-baseweb="tab"]:nth-child(6) { animation-delay: 275ms; }
.stTabs [data-baseweb="tab"]:nth-child(7) { animation-delay: 330ms; }
.stTabs [data-baseweb="tab"]:nth-child(8) { animation-delay: 385ms; }

.stButton > button, div[data-baseweb="base-button-above"] > button {
  position: relative; overflow: hidden;
  animation: sw-fade-up .5s var(--ease) both;
}
.stButton > button::after,
div[data-baseweb="base-button-above"] > button::after {
  content: ""; position: absolute; inset: 0;
  background: linear-gradient(105deg, transparent 38%, rgba(255,255,255,.26) 50%, transparent 62%);
  transform: translateX(-120%);
  transition: transform .62s var(--ease);
}
.stButton > button:hover::after,
div[data-baseweb="base-button-above"] > button:hover::after { transform: translateX(120%); }

:focus-visible { outline: 2px solid rgba(96,165,250,.85) !important; outline-offset: 2px; }

/* Sidebar nav rows slide in with stagger */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
  animation: sw-fade-left .5s var(--ease) both;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-child(1) {
  animation-delay: 40ms;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-child(2) {
  animation-delay: 90ms;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-child(3) {
  animation-delay: 140ms;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-child(4) {
  animation-delay: 190ms;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-child(5) {
  animation-delay: 240ms;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-child(6) {
  animation-delay: 290ms;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-child(7) {
  animation-delay: 340ms;
}

/* Cards + charts get the entrance + spotlight treatment */
.st-kpi, .st-card, [data-testid="stPlotlyChart"], [data-testid="stMetric"] {
  animation: sw-fade-up .58s var(--ease) both;
}

/* ── Reduced motion ──────────────────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
  .sw-aurora, .sw-aurora .sw-orb, .sw-aurora .sw-grid, .sw-aurora .sw-sweep,
  .sw-animborder::before, .sw-marquee .sw-track, .sw-bars .sw-bar,
  .sw-ring, .sw-tile .sw-tile-glow, .sw-live .sw-live-dot::after,
  .sw-skel, .sw-scrollrail, .sw-spark path, .sw-spark circle {
    animation: none !important;
  }
  .sw-ci, .sw-cf, .sw-reveal, .sw-reveal-left, .sw-reveal-zoom,
  .sw-stagger > *, .sw-hd, .sw-hd::before, .sw-hr, .sw-steps .sw-step,
  .st-kpi, .st-card, [data-testid="stPlotlyChart"], [data-testid="stMetric"] {
    animation-duration: .01ms !important; animation-delay: 0ms !important;
  }
  .sw-spark path.sw-spark-line { stroke-dashoffset: 0 !important; }
  .sw-spark path.sw-spark-area, .sw-spark circle.sw-spark-head { opacity: 1 !important; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Injectors
# ---------------------------------------------------------------------------

def inject_motion() -> None:
    """Inject the motion stylesheet.

    Runs on every script run (Streamlit rebuilds the DOM each rerun).
    """
    st.markdown(MOTION_CSS, unsafe_allow_html=True)


def aurora() -> None:
    """Render the animated ambient background + scroll progress rail."""
    st.markdown(
        """
        <div class="sw-aurora" aria-hidden="true">
          <div class="sw-grid"></div>
          <div class="sw-o1 sw-orb"></div>
          <div class="sw-o2 sw-orb"></div>
          <div class="sw-o3 sw-orb"></div>
          <div class="sw-sweep"></div>
        </div>
        <div class="sw-scrollrail" aria-hidden="true"></div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Reveal helpers
# ---------------------------------------------------------------------------

def reveal(
    html: str,
    *,
    delay_ms: int = 0,
    kind: str = "up",
    duration: str = DUR_REVEAL,
) -> None:
    """Staggered entrance reveal for arbitrary HTML.

    Parameters
    ----------
    html:
        Inner HTML to animate.
    delay_ms:
        Animation delay in milliseconds.
    kind:
        ``"up"`` (fade-up + blur), ``"left"``, or ``"zoom"``.
    duration:
        CSS duration string.
    """
    cls = {
        "up": "sw-reveal",
        "left": "sw-reveal-left",
        "zoom": "sw-reveal-zoom",
    }.get(kind, "sw-reveal")
    st.markdown(
        f'<div class="{cls}" style="--d:{delay_ms}ms;--dur:{duration};">'
        f"{html}</div>",
        unsafe_allow_html=True,
    )


def glow_rule() -> None:
    """Grow-in gradient rule used under section headers."""
    st.markdown('<hr class="sw-hr">', unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "", *, delay_ms: int = 0) -> None:
    """Motion-slides style section header with animated accent rule."""
    sub = f'<div class="sw-hd-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="sw-hd" style="--d:{delay_ms}ms;">'
        f'<div class="sw-hd-title">{title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def live_pill(text: str = "LIVE") -> None:
    """Pulsing live indicator pill with ping ring."""
    st.markdown(
        f'<span class="sw-live"><span class="sw-live-dot"></span>{text}</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def _num_parts(value: float, decimals: int) -> tuple[int, int]:
    """Split *value* into (integer part, zero-padded fraction) counters."""
    scaled = int(round(abs(float(value)) * (10**decimals)))
    int_part, frac_part = divmod(scaled, 10**decimals)
    return int_part, frac_part


def count_up_html(
    value: float,
    *,
    decimals: int = 0,
    prefix: str = "",
    suffix: str = "",
    duration: str = "1.15s",
    signed: bool = False,
) -> str:
    """Return the HTML for an animated count-up number (no ``st.markdown``).

    The static (animation-disabled) fallback renders the final value, so
    the number is always correct even without keyframe support.
    """
    sign = "-" if float(value) < 0 else ("+" if signed and float(value) > 0 else "")
    int_part, frac_part = _num_parts(value, decimals)
    frac_html = (
        f'.<span class="sw-cf" style="--sw-nf-t:{frac_part};--dur:{duration};"></span>'
        if decimals > 0
        else ""
    )
    return (
        f'<span class="sw-countline">'
        f"{sign}{prefix}"
        f'<span class="sw-ci" style="--sw-ni-t:{int_part};--dur:{duration};"></span>'
        f"{frac_html}{suffix}</span>"
    )


def count_up(
    label: str,
    value: float,
    *,
    decimals: int = 0,
    prefix: str = "",
    suffix: str = "",
    delta: str | None = None,
    color: str = "#3B82F6",
    delay_ms: int = 0,
    size: str = "1.5rem",
) -> None:
    """Animated KPI tile with count-up value (21st.dev metric-tile style)."""
    delta_html = f'<div class="sw-tile-delta">{delta}</div>' if delta else ""
    value_html = count_up_html(
        value,
        decimals=decimals,
        prefix=prefix,
        suffix=suffix,
    )
    st.markdown(
        f"""
        <div class="sw-tile sw-spot sw-animborder"
             style="--d:{delay_ms}ms;--tile-c:{color}59;--fs:{size};">
          <div class="sw-tile-glow" style="--tile-c:{color}59;"></div>
          <div class="sw-tile-label">{label}</div>
          <div class="sw-tile-value">{value_html}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_tile(
    label: str,
    value_html: str,
    *,
    delta: str | None = None,
    color: str = "#3B82F6",
    delay_ms: int = 0,
    size: str = "1.5rem",
    animated: bool = True,
) -> None:
    """Bento tile with pre-built (possibly animated) value HTML."""
    border = " sw-animborder" if animated else ""
    glow = (
        f'<div class="sw-tile-glow" style="--tile-c:{color}59;"></div>'
        if animated
        else ""
    )
    delta_html = f'<div class="sw-tile-delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="sw-tile sw-spot{border}"
             style="--d:{delay_ms}ms;--tile-c:{color}59;--fs:{size};">
          {glow}
          <div class="sw-tile-label">{label}</div>
          <div class="sw-tile-value">{value_html}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Progress visuals
# ---------------------------------------------------------------------------

def progress_ring(
    pct: float,
    *,
    label: str = "",
    value_text: str | None = None,
    color: str = "#3B82F6",
    size: str = "116px",
    sublabel: str = "",
) -> None:
    """Conic-gradient progress ring with sweeping highlight."""
    clamped = max(0.0, min(float(pct), 100.0))
    centre = value_text if value_text is not None else f"{clamped:.0f}%"
    sub = (
        '<div style="color:#6B7280;font-size:.62rem;letter-spacing:.8px;'
        f'margin-top:2px;">{sublabel}</div>'
        if sublabel
        else ""
    )
    label_html = (
        '<div style="color:#9CA3AF;font-size:.72rem;letter-spacing:.9px;'
        f'text-transform:uppercase;font-weight:650;">{label}</div>'
        if label
        else ""
    )
    st.markdown(
        f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:7px;
                    animation:sw-fade-up .6s var(--ease) both;">
          <div class="sw-ring" style="--pct:{clamped:.2f}%;--ring-c:{color};
                                     --size:{size};">
            <div style="text-align:center;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:1.28rem;
                          font-weight:750;color:#F9FAFB;line-height:1.1;">{centre}</div>
              {sub}
            </div>
          </div>
          {label_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pulsing_bars(
    values: list[float],
    *,
    color: str = "#3B82F6",
    height: str = "44px",
    max_period_ms: int = 900,
) -> None:
    """Animated bar stack (21st.dev ``PulsingBars``).

    Parameters
    ----------
    values:
        Bar magnitudes; normalised to the maximum internally.
    """
    if not values:
        return
    top = max(abs(float(v)) for v in values) or 1.0
    bars = []
    for i, v in enumerate(values):
        h = 100.0 * abs(float(v)) / top
        delay = int(max_period_ms * (i % 7) / 7)
        bars.append(
            f'<div class="sw-bar" style="--bh:{h:.1f}%;--bd:{delay}ms;'
            f'--bar-c:{color};"></div>'
        )
    st.markdown(
        f'<div class="sw-bars" style="--h:{height};">{"".join(bars)}</div>',
        unsafe_allow_html=True,
    )


def sparkline(
    values: list[float],
    *,
    color: str = "#3B82F6",
    width: int = 220,
    height: int = 48,
    uid: str = "sp",
    delay_ms: int = 120,
) -> None:
    """Inline SVG sparkline with animated stroke draw + area fill."""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    step = width / (len(vals) - 1)
    pts = [
        (round(i * step, 2), round(height - 3 - ((v - lo) / span) * (height - 8), 2))
        for i, v in enumerate(vals)
    ]
    line = " ".join(f"{x},{y}" for x, y in pts)
    area = f"{pts[0][0]},{height} {line} {pts[-1][0]},{height}"
    hx, hy = pts[-1]
    approx_len = int(sum(
        math.dist((pts[i][0], pts[i][1]), (pts[i + 1][0], pts[i + 1][1]))
        for i in range(len(pts) - 1)
    ) + 1)
    st.markdown(
        f"""
        <svg class="sw-spark" viewBox="0 0 {width} {height}" width="100%"
             height="{height}" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <linearGradient id="{uid}Grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="{color}" stop-opacity="0.34"/>
              <stop offset="100%" stop-color="{color}" stop-opacity="0"/>
            </linearGradient>
          </defs>
          <path class="sw-spark-area" d="M {area} Z" fill="url(#{uid}Grad)"/>
          <path class="sw-spark-line" d="M {line}"
                style="--sp-c:{color};--sp-len:{approx_len};--d:{delay_ms}ms;"/>
          <circle class="sw-spark-head" cx="{hx}" cy="{hy}" r="2.6"
                  style="--sp-c:{color};"/>
        </svg>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Ribbons + steps
# ---------------------------------------------------------------------------

def ticker(items: list[tuple[str, Any]], *, speed: str = "26s") -> None:
    """Infinite marquee ribbon; each item is a ``(label, value)`` pair.

    The track is duplicated so the ``translateX(-50%)`` loop is seamless.
    """
    if not items:
        return
    cells = []
    for label, value in items:
        cells.append(
            f'<span class="sw-item"><b>{value}</b><span>{label}</span></span>'
            f'<span class="sw-sep">◆</span>'
        )
    track = "".join(cells)
    st.markdown(
        f'<div class="sw-marquee" style="--speed:{speed};">'
        f'<div class="sw-track">{track}{track}</div></div>',
        unsafe_allow_html=True,
    )


def step_list(steps: list[str], *, start_delay_ms: int = 0, stagger_ms: int = 90) -> None:
    """Motion-slides style reveal-on-step list."""
    rows = []
    for i, text in enumerate(steps, start=1):
        rows.append(
            f'<div class="sw-step" style="--d:{start_delay_ms + (i - 1) * stagger_ms}ms;">'
            f'<div class="sw-step-idx">{i}</div>'
            f'<div class="sw-step-text">{text}</div></div>'
        )
    st.markdown(f'<div class="sw-steps">{"".join(rows)}</div>', unsafe_allow_html=True)
