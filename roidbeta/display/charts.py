"""Chart primitives drawn with cv2 (no matplotlib, so the capture loop is safe).

These are low-level building blocks (sparkline, gauge) styled with the theme.
The dashboard composes them into metric cards.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import config
from . import theme


def _points(series, x, y, w, h, value_range):
    lo, hi = value_range
    span = (hi - lo) or 1.0
    recent = series[-config.CHART_HISTORY_POINTS:]
    n = len(recent)
    pts = []
    for i, v in enumerate(recent):
        if v is None:
            pts.append(None)
            continue
        px = x + int(i * w / max(1, n - 1))
        norm = (max(lo, min(hi, v)) - lo) / span
        py = y + h - int(norm * h)
        pts.append((px, py))
    return pts


def sparkline(
    frame: np.ndarray, series, x: int, y: int, w: int, h: int,
    value_range=(0.0, 1.0), color=theme.PRIMARY, fill: bool = True,
) -> None:
    """Line chart of series inside (x, y, w, h), with a soft area fill.

    None samples break the line (and its fill) into separate segments.
    """
    pts = _points(series, x, y, w, h, value_range)
    baseline = y + h

    # Contiguous runs of non-None points.
    segments: list[list[tuple[int, int]]] = []
    run: list[tuple[int, int]] = []
    for p in pts:
        if p is None:
            if len(run) > 1:
                segments.append(run)
            run = []
        else:
            run.append(p)
    if len(run) > 1:
        segments.append(run)

    if fill and segments:
        overlay = frame.copy()
        for seg in segments:
            poly = np.array(seg + [(seg[-1][0], baseline), (seg[0][0], baseline)],
                            dtype=np.int32)
            cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    for seg in segments:
        cv2.polylines(frame, [np.array(seg, np.int32)], False, color, 2, cv2.LINE_AA)
    # Highlight the latest sample.
    for p in reversed(pts):
        if p is not None:
            cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)
            cv2.circle(frame, p, 3, theme.TEXT, 1, cv2.LINE_AA)
            break


def gauge_color(frac: float) -> tuple[int, int, int]:
    """Green when high, amber mid, red low (frac in [0, 1])."""
    if frac >= 0.66:
        return theme.SUCCESS
    if frac >= 0.33:
        return theme.WARNING
    return theme.DANGER
