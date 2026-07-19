"""Session comparison screen: skill-graph radar plus per-metric leaderboards.

Takes already-computed data (so it stays free of session/scoring imports): a list
of climber entries and a list of per-metric leaders. main assembles these from
the session's SkillProfiles.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import skillgraph, theme


def draw_comparison(frame: np.ndarray, entries, leaders) -> None:
    """Render the results screen.

    entries: list of (label, color, axis_values) for the radar and legend.
    leaders: list of (metric_label, winner_label, winner_text) rows.
    """
    fh, fw = frame.shape[:2]
    theme.text(frame, "SESSION RESULTS", (40, 54), 0.9, theme.TEXT, 2)
    theme.text(frame, "who climbs like which class?", (40, 82), 0.55,
               theme.TEXT_MUTED, 1)

    # Radar on the left half.
    radar_cx = fw // 3
    radar_cy = fh // 2 + 20
    radius = min(fw // 4, fh // 3) - 30
    skillgraph.draw_skill_graph(frame, (radar_cx, radar_cy), radius, entries)

    # Right panel: legend + per-metric leaders.
    px = fw * 2 // 3 - 10
    pw = fw - px - 40
    py = 120
    ph = fh - py - 90
    theme.panel(frame, px, py, pw, ph, radius=14, color=theme.BG, alpha=0.85)

    ix = px + 24
    iy = py + 40
    theme.text(frame, "CLIMBERS", (ix, iy), 0.5, theme.TEXT_MUTED, 1)
    iy += 30
    for label, color, _ in entries:
        cv2.circle(frame, (ix + 6, iy - 5), 7, color, -1, cv2.LINE_AA)
        theme.text(frame, label, (ix + 22, iy), 0.6, theme.TEXT, 1)
        iy += 30

    iy += 14
    theme.text(frame, "CATEGORY LEADERS", (ix, iy), 0.5, theme.TEXT_MUTED, 1)
    iy += 32
    for metric, winner_label, winner_text in leaders:
        theme.text(frame, metric.upper(), (ix, iy), 0.5, theme.TEXT_MUTED, 1)
        theme.text(frame, f"{winner_label}  {winner_text}", (ix + 130, iy),
                   0.55, theme.TEXT, 1)
        iy += 30

    theme.text(frame, "R  new session      Q  quit", (40, fh - 44), 0.6,
               theme.TEXT_MUTED, 1)
