"""Composed HUD: a top-left status card and a bottom metrics bar.

main builds a HudState each frame and hands it here; the dashboard renders it
with the theme primitives so the whole overlay shares one professional look.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config
from . import charts, theme

_MARGIN = 16


@dataclass
class MetricCard:
    label: str
    history: list          # per-frame values (may contain None)
    current: float | None
    value_range: tuple[float, float] = (0.0, 1.0)


@dataclass
class HudState:
    """Everything the dashboard needs for one frame."""

    title: str                                   # state name, e.g. "ATTEMPT"
    accent: tuple[int, int, int]
    timer: str | None = None                     # e.g. "12.4s"
    reached: int = 0
    used: int = 0
    total: int = 0
    show_progress: bool = False
    has_reference: bool = False
    avg_balance: float | None = None             # session average, for the results card
    controls: list[str] = field(default_factory=list)
    metrics: list[MetricCard] = field(default_factory=list)


def draw_status_card(frame: np.ndarray, hud: HudState) -> None:
    x, y, w = _MARGIN, _MARGIN, 320
    pad = 14
    has_progress = hud.show_progress and hud.total > 0
    h = 96 if has_progress else 58

    theme.panel(frame, x, y, w, h, radius=14, color=theme.BG, alpha=0.82)

    # Row 1: state pill (left) + big timer (right). REF badge sits just left of
    # the timer so the two never collide.
    theme.pill(frame, x + pad, y + pad, hud.title, hud.accent, scale=0.5)
    timer_x = x + w - pad
    if hud.timer:
        tw, _ = theme.text_size(hud.timer, 1.0, 2)
        timer_x = x + w - pad - tw
        theme.text(frame, hud.timer, (timer_x, y + pad + 24), 1.0, theme.TEXT, 2)
    if hud.has_reference:
        theme.pill(frame, timer_x - 52, y + pad - 1, "REF", theme.VIOLET, scale=0.4)

    # Row 2: hold progress.
    if has_progress:
        by = y + 58
        theme.text(frame, "PROGRESS", (x + pad, by - 6), 0.42, theme.TEXT_MUTED, 1)
        val = f"{hud.reached}/{hud.total}"
        vw, _ = theme.text_size(val, 0.5, 1)
        theme.text(frame, f"used {hud.used}", (x + w - pad - vw - 70, by - 6),
                   0.42, theme.TEXT_DIM, 1)
        theme.text(frame, val, (x + w - pad - vw, by - 6), 0.5, theme.TEXT, 1)
        frac = hud.reached / hud.total if hud.total else 0.0
        theme.progress_bar(frame, x + pad, by, w - 2 * pad, 8, frac, hud.accent)


def draw_bottom_bar(frame: np.ndarray, hud: HudState) -> None:
    """Bottom metrics bar. Shown once an attempt is running (it has metrics)."""
    if not hud.metrics:
        return
    fh, fw = frame.shape[:2]
    h = config.CHART_PANEL_HEIGHT
    x, y = _MARGIN, fh - h - _MARGIN
    w = fw - 2 * _MARGIN
    theme.panel(frame, x, y, w, h, radius=14, color=theme.BG, alpha=0.82)

    pad = 16
    inner_top = y + pad
    inner_h = h - 2 * pad

    # Controls occupy the right; the metric cards are centered in the space that
    # remains, so a single chart sits in the middle instead of hugging the left.
    controls_w = _controls_width(hud.controls)
    card_w = 460
    n = len(hud.metrics)
    block_w = n * card_w + (n - 1) * pad
    avail_left = x + pad
    avail_right = x + w - pad - (controls_w + 2 * pad if controls_w else 0)
    cx = avail_left + max(0, (avail_right - avail_left - block_w) // 2)
    for card in hud.metrics:
        _draw_metric_card(frame, card, cx, inner_top, card_w, inner_h)
        cx += card_w + pad

    _draw_controls(frame, hud.controls, x + w - pad, inner_top, inner_h)


def draw_top_banner(frame: np.ndarray, title: str, subtitle: str,
                    accent=theme.VIOLET) -> None:
    """A centered banner at the top, for full-screen modes like clip review.

    Sits top-center so it never collides with the status card (top-left) or the
    metrics bar (bottom) baked into a recorded frame.
    """
    fw = frame.shape[1]
    tw, _ = theme.text_size(title, 0.7, 2)
    sw, _ = theme.text_size(subtitle, 0.55, 1)
    pad = 20
    w = max(tw, sw) + 2 * pad
    h = 78
    x, y = (fw - w) // 2, 16
    theme.panel(frame, x, y, w, h, radius=14, color=theme.BG, alpha=0.85)
    theme.panel(frame, x, y, w, 5, radius=3, color=accent, alpha=0.95, border=None)
    _center_text(frame, title, x + w // 2, y + 34, 0.7, accent, 2)
    _center_text(frame, subtitle, x + w // 2, y + 62, 0.55, theme.TEXT_MUTED, 1)


def draw_hint(frame: np.ndarray, controls: list[str]) -> None:
    """A slim centered control hint for states before the bottom bar appears."""
    if not controls:
        return
    fh, fw = frame.shape[:2]
    label = "      ".join(controls)
    scale = 0.62
    tw, th = theme.text_size(label, scale, 1)
    pad = 22
    w, h = tw + 2 * pad, th + 24
    x, y = (fw - w) // 2, fh - h - _MARGIN
    theme.panel(frame, x, y, w, h, radius=h // 2, color=theme.BG, alpha=0.82)
    theme.text(frame, label, (x + pad, y + h - 16), scale, theme.TEXT_MUTED, 1)


def _center_text(frame, s, cx, y, scale, color, thickness=1):
    tw, _ = theme.text_size(s, scale, thickness)
    theme.text(frame, s, (cx - tw // 2, y), scale, color, thickness)


def _balance_rating(avg: float) -> str:
    if avg >= 0.80:
        return "SOLID"
    if avg >= 0.65:
        return "STEADY"
    if avg >= 0.45:
        return "SHAKY"
    return "OFF BALANCE"


def draw_results_card(frame: np.ndarray, hud: HudState) -> None:
    """Classy centered end-of-attempt summary: average balance in big type."""
    fh, fw = frame.shape[:2]
    cw, ch = 480, 320
    x = (fw - cw) // 2
    y = (fh - ch) // 2 - 30
    cx = x + cw // 2

    theme.panel(frame, x, y, cw, ch, radius=20, color=theme.BG, alpha=0.92)
    # Accent header strip.
    theme.panel(frame, x, y, cw, 6, radius=3, color=hud.accent, alpha=0.95, border=None)

    _center_text(frame, "SESSION SUMMARY", cx, y + 42, 0.6, theme.TEXT_MUTED, 1)

    avg = hud.avg_balance
    color = charts.gauge_color(avg) if avg is not None else theme.TEXT_DIM
    _center_text(frame, "AVERAGE BALANCE", cx, y + 88, 0.5, theme.TEXT_MUTED, 1)

    big = "--" if avg is None else f"{avg:.2f}"
    _center_text(frame, big, cx, y + 168, 2.6, color, 3)

    if avg is not None:
        bar_w = cw - 120
        theme.progress_bar(frame, cx - bar_w // 2, y + 190, bar_w, 10, avg, color)
        _center_text(frame, _balance_rating(avg), cx, y + 224, 0.7, color, 1)

    # Divider + secondary stats: time and holds.
    theme.panel(frame, x + 40, y + 244, cw - 80, 1, radius=0,
                color=theme.BORDER, alpha=0.6, border=None)
    _draw_summary_stat(frame, "TIME", hud.timer or "--", x + cw // 4, y + 288)
    _draw_summary_stat(frame, "HOLDS", f"{hud.used}/{hud.total}",
                       x + 3 * cw // 4, y + 288)


def _draw_summary_stat(frame, label, value, cx, y):
    _center_text(frame, value, cx, y, 0.9, theme.TEXT, 2)
    _center_text(frame, label, cx, y - 30, 0.42, theme.TEXT_MUTED, 1)


def _draw_metric_card(frame, card: MetricCard, x, y, w, h) -> None:
    frac = None
    if card.current is not None:
        lo, hi = card.value_range
        frac = (card.current - lo) / ((hi - lo) or 1.0)

    # Header row: label (muted) + value (colored by level).
    theme.text(frame, card.label.upper(), (x, y + 16), 0.5, theme.TEXT_MUTED, 1)
    value_color = charts.gauge_color(frac) if frac is not None else theme.TEXT_DIM
    value_str = "--" if card.current is None else f"{card.current:.2f}"
    vw, _ = theme.text_size(value_str, 0.95, 2)
    theme.text(frame, value_str, (x + w - vw, y + 22), 0.95, value_color, 2)

    # Sparkline in the middle (the tall area is the point of the wide card).
    spark_y = y + 30
    spark_h = h - 30 - 10
    charts.sparkline(frame, card.history, x, spark_y, w, spark_h,
                     card.value_range, color=value_color)

    # Gauge at the bottom.
    if frac is not None:
        theme.progress_bar(frame, x, y + h - 6, w, 6, frac, value_color)


_CONTROLS_MAX_ROWS = 3
_CONTROLS_COL_GAP = 30
_CONTROLS_SCALE = 0.58


def _controls_columns(controls: list[str]):
    ncols = (len(controls) + _CONTROLS_MAX_ROWS - 1) // _CONTROLS_MAX_ROWS
    cols = [controls[i * _CONTROLS_MAX_ROWS:(i + 1) * _CONTROLS_MAX_ROWS]
            for i in range(ncols)]
    widths = [max(theme.text_size(s, _CONTROLS_SCALE, 1)[0] for s in col)
              for col in cols]
    return cols, widths


def _controls_width(controls: list[str]) -> int:
    if not controls:
        return 0
    _, widths = _controls_columns(controls)
    return sum(widths) + _CONTROLS_COL_GAP * (len(widths) - 1)


def _draw_controls(frame, controls: list[str], right_x, y, h) -> None:
    """Right-aligned control hints, wrapped into columns of at most 3 rows."""
    if not controls:
        return
    line_h, col_gap, scale = 27, _CONTROLS_COL_GAP, _CONTROLS_SCALE
    cols, widths = _controls_columns(controls)
    total_w = sum(widths) + col_gap * (len(widths) - 1)

    cx = right_x - total_w
    for col, cw in zip(cols, widths):
        col_h = line_h * len(col)
        start_y = y + max(0, (h - col_h) // 2) + 15
        for r, line in enumerate(col):
            tw, _ = theme.text_size(line, scale, 1)
            theme.text(frame, line, (cx + cw - tw, start_y + r * line_h),
                       scale, theme.TEXT_MUTED, 1)
        cx += cw + col_gap
