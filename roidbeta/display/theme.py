"""Visual design system for the overlay.

A small set of colors and drawing primitives so every panel, label, and gauge
shares one look: dark translucent surfaces, rounded corners, a restrained accent
palette, and clean anti-aliased type. Higher-level overlay code composes these
instead of calling cv2 drawing directly, which keeps the UI consistent.

All colors are BGR (OpenCV order).
"""

from __future__ import annotations

import cv2
import numpy as np

# --- palette (BGR) ---
BG = (26, 24, 22)          # near-black slate for panels
SURFACE = (44, 40, 36)     # raised surface
SURFACE_HI = (60, 55, 50)  # gauge/track fill
BORDER = (86, 80, 74)      # subtle hairline border
TEXT = (240, 240, 244)     # primary text
TEXT_MUTED = (168, 162, 158)  # secondary / labels / hints
TEXT_DIM = (120, 116, 112)

PRIMARY = (240, 190, 90)   # sky-blue accent
SUCCESS = (120, 210, 100)  # green
WARNING = (70, 180, 250)   # amber
DANGER = (96, 96, 240)     # red
VIOLET = (220, 150, 190)   # review / reference

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_PLAIN = cv2.FONT_HERSHEY_SIMPLEX


def text_size(s: str, scale: float, thickness: int = 1, font=FONT) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(s, font, scale, thickness)
    return w, h


def text(
    frame: np.ndarray,
    s: str,
    org: tuple[int, int],
    scale: float = 0.6,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
    font=FONT,
    shadow: bool = True,
) -> None:
    """Draw anti-aliased text with a subtle 1px shadow for legibility."""
    x, y = org
    if shadow:
        cv2.putText(frame, s, (x + 1, y + 1), font, scale, BG, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _round_rect(img, x, y, w, h, r, color, filled=True):
    r = int(max(0, min(r, w // 2, h // 2)))
    if filled:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r),
                       (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x + r, y), (x + w - r, y), color, 1, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, 1, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, 1, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, 1, cv2.LINE_AA)
        for cx, cy, a0 in ((x + r, y + r, 180), (x + w - r, y + r, 270),
                           (x + w - r, y + h - r, 0), (x + r, y + h - r, 90)):
            cv2.ellipse(img, (cx, cy), (r, r), a0, 0, 90, color, 1, cv2.LINE_AA)


def panel(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    radius: int = 12,
    color: tuple[int, int, int] = SURFACE,
    alpha: float = 0.80,
    border: tuple[int, int, int] | None = BORDER,
) -> None:
    """Draw a rounded translucent panel, clipped to the frame."""
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    overlay = frame.copy()
    _round_rect(overlay, x, y, w, h, radius, color, filled=True)
    roi = (slice(y0, y1), slice(x0, x1))
    frame[roi] = cv2.addWeighted(overlay[roi], alpha, frame[roi], 1.0 - alpha, 0)
    if border is not None:
        _round_rect(frame, x, y, w, h, radius, border, filled=False)


def pill(
    frame: np.ndarray, x: int, y: int, label: str,
    accent: tuple[int, int, int], scale: float = 0.5,
) -> int:
    """Draw a solid rounded status badge. Returns its width for layout."""
    tw, th = text_size(label, scale, 1)
    pad_x, pad_y = 10, 6
    w, h = tw + pad_x * 2, th + pad_y * 2
    panel(frame, x, y, w, h, radius=h // 2, color=accent, alpha=0.95, border=None)
    text(frame, label, (x + pad_x, y + h - pad_y - 1), scale, BG, 1, shadow=False)
    return w


def progress_bar(
    frame: np.ndarray, x: int, y: int, w: int, h: int, frac: float,
    color: tuple[int, int, int] = PRIMARY,
    track: tuple[int, int, int] = SURFACE_HI,
) -> None:
    """Draw a rounded progress track with a filled portion."""
    frac = max(0.0, min(1.0, frac))
    r = h // 2
    _round_rect(frame, x, y, w, h, r, track, filled=True)
    fill_w = int(w * frac)
    if fill_w >= 2 * r:
        _round_rect(frame, x, y, fill_w, h, r, color, filled=True)
    elif fill_w > 0:
        cv2.circle(frame, (x + r, y + r), r, color, -1)
