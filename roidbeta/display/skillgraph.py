"""Skill-graph radar: compare climbers as video-game-style "class" shapes.

Draws a radar/spider chart with one axis per skill (Balance, Smoothness, Speed,
Reach) and one filled polygon per climber, in that climber's label color. Several
climbers overlaid on the same radar make their strengths and trade-offs obvious
at a glance.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .. import config
from . import theme

# Distinct label colors for climbers (BGR), reused in order.
CLIMBER_COLORS = [
    (90, 130, 245),   # red
    (230, 170, 70),   # blue
    (110, 200, 110),  # green
    (60, 200, 250),   # amber
    (210, 140, 230),  # violet
    (210, 200, 90),   # teal
]


def _axis_points(center, radius, n, values):
    """Pixel points for a polygon whose vertices are values (0..1) along n axes."""
    cx, cy = center
    pts = []
    for i in range(n):
        ang = -math.pi / 2 + 2 * math.pi * i / n   # start at top, clockwise
        r = radius * max(0.0, min(1.0, values[i]))
        pts.append((int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))))
    return pts


def draw_skill_graph(frame, center, radius, entries, axes=config.SKILL_AXES) -> None:
    """Draw the radar grid, axis labels, and one polygon per entry.

    entries: list of (label, color, axis_values) where axis_values maps each axis
    name to a 0..1 score.
    """
    cx, cy = center
    n = len(axes)

    # Concentric grid rings.
    for ring in (0.25, 0.5, 0.75, 1.0):
        pts = _axis_points(center, radius, n, [ring] * n)
        cv2.polylines(frame, [np.array(pts, np.int32)], True, theme.BORDER, 1, cv2.LINE_AA)

    # Spokes + axis labels.
    for i, name in enumerate(axes):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        ex, ey = int(cx + radius * math.cos(ang)), int(cy + radius * math.sin(ang))
        cv2.line(frame, (cx, cy), (ex, ey), theme.BORDER, 1, cv2.LINE_AA)
        lx, ly = int(cx + (radius + 22) * math.cos(ang)), int(cy + (radius + 22) * math.sin(ang))
        tw, _ = theme.text_size(name, 0.45, 1)
        theme.text(frame, name, (lx - tw // 2, ly + 4), 0.45, theme.TEXT_MUTED, 1)

    # One translucent polygon per climber, drawn on top of the grid.
    for label, color, values in entries:
        vals = [values.get(a, 0.0) for a in axes]
        pts = np.array(_axis_points(center, radius, n, vals), np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, color, -1, cv2.LINE_AA)
