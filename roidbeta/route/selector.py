"""Human-in-the-loop route selection by color.

A route is one color. The user clicks a representative hold on a frozen frame;
we sample the HSV neighborhood of that click from the actual pixels (never a
hardcoded color name) and build a mask from the sampled range. Multiple clicks
union their ranges, so a route whose holds vary slightly in shade can be
captured by clicking a few of them.

This is the load-bearing simplification of the whole system, so it is
deliberately interactive and tunable rather than automatic.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .. import config
from .holds import Hold, RouteMask, extract_holds

# Bounds of the OpenCV HSV space: hue is 0..179, sat and val are 0..255.
_HUE_MAX = 179
_CHANNEL_MAX = 255

HsvRange = tuple[np.ndarray, np.ndarray]  # (lower, upper) inclusive inRange bounds


@dataclass(frozen=True)
class RouteSelection:
    """The frozen result of a selection: the route mask and the frame it used."""

    route: RouteMask
    frame: np.ndarray  # the snapshot the user selected on, for reference/overlay


def sample_hsv_ranges(
    frame_bgr: np.ndarray,
    x: int,
    y: int,
    patch_radius: int = config.HSV_SAMPLE_PATCH_RADIUS_PX,
    hue_tol: int = config.HSV_HUE_TOLERANCE,
    sat_tol: int = config.HSV_SAT_TOLERANCE,
    val_tol: int = config.HSV_VAL_TOLERANCE,
) -> list[HsvRange]:
    """Sample HSV around (x, y) and return inRange bounds for that color.

    Uses the median HSV of a patch so one stray pixel does not skew the range.
    Hue wraps at 180, so a range straddling the wrap (common for red) is split
    into two inRange bounds; sat and val are simply clamped. Returns one or two
    (lower, upper) pairs.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    x0, x1 = max(0, x - patch_radius), min(w, x + patch_radius + 1)
    y0, y1 = max(0, y - patch_radius), min(h, y + patch_radius + 1)
    patch = hsv[y0:y1, x0:x1].reshape(-1, 3)

    hue_c, sat_c, val_c = (int(v) for v in np.median(patch, axis=0))

    sat_lo = max(0, sat_c - sat_tol)
    sat_hi = min(_CHANNEL_MAX, sat_c + sat_tol)
    val_lo = max(0, val_c - val_tol)
    val_hi = min(_CHANNEL_MAX, val_c + val_tol)

    hue_lo = hue_c - hue_tol
    hue_hi = hue_c + hue_tol

    ranges: list[HsvRange] = []
    if hue_lo < 0:
        # Wraps below 0: [0, hue_hi] and [180 + hue_lo, 179].
        ranges.append(_range(0, sat_lo, val_lo, hue_hi, sat_hi, val_hi))
        ranges.append(_range(_HUE_MAX + 1 + hue_lo, sat_lo, val_lo, _HUE_MAX, sat_hi, val_hi))
    elif hue_hi > _HUE_MAX:
        # Wraps above 179: [hue_lo, 179] and [0, hue_hi - 180].
        ranges.append(_range(hue_lo, sat_lo, val_lo, _HUE_MAX, sat_hi, val_hi))
        ranges.append(_range(0, sat_lo, val_lo, hue_hi - _HUE_MAX - 1, sat_hi, val_hi))
    else:
        ranges.append(_range(hue_lo, sat_lo, val_lo, hue_hi, sat_hi, val_hi))
    return ranges


def _range(hl: int, sl: int, vl: int, hh: int, sh: int, vh: int) -> HsvRange:
    return (
        np.array([hl, sl, vl], dtype=np.uint8),
        np.array([hh, sh, vh], dtype=np.uint8),
    )


def build_mask(
    frame_bgr: np.ndarray,
    ranges: list[HsvRange],
    morph_kernel_px: int = config.MASK_MORPH_KERNEL_PX,
) -> np.ndarray:
    """Build a cleaned binary mask from the union of HSV ranges.

    Each range contributes an inRange mask; they are OR'd together, then opened
    (remove speckle) and closed (fill holds' interior gaps from chalk/shadow).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    if morph_kernel_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_px, morph_kernel_px)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _overlay_selection(frame_bgr: np.ndarray, mask: np.ndarray, holds) -> np.ndarray:
    """Preview: tint masked pixels and draw each extracted hold target."""
    preview = frame_bgr.copy()
    tint = np.zeros_like(preview)
    tint[mask > 0] = (0, 255, 0)
    preview = cv2.addWeighted(preview, 1.0, tint, 0.4, 0)
    for hold in holds:
        cv2.circle(preview, (hold.x, hold.y), hold.radius, (0, 165, 255), 2)
        cv2.circle(preview, (hold.x, hold.y), 3, (0, 0, 255), -1)
    return preview


class RouteSelector:
    """Interactive, one-time route selection on a frozen frame.

    Click holds to accumulate their color; the mask and detected holds update
    live. Keys: r reset, Enter/c confirm, q/Esc cancel. Returns a frozen
    RouteSelection, or None if cancelled.
    """

    def __init__(self, window_name: str = config.OVERLAY_WINDOW_NAME) -> None:
        self._window_name = window_name

    def select(self, frame_bgr: np.ndarray) -> RouteSelection | None:
        frame = frame_bgr.copy()  # freeze the snapshot; the wall does not move
        ranges: list[HsvRange] = []

        def on_mouse(event, mx, my, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                ranges.extend(sample_hsv_ranges(frame, mx, my))

        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, on_mouse)
        try:
            while True:
                if ranges:
                    mask = build_mask(frame, ranges)
                    holds = extract_holds(mask)
                else:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    holds = ()

                cv2.imshow(self._window_name, _overlay_selection(frame, mask, holds))
                key = cv2.waitKey(20) & 0xFF
                if key == ord("r"):
                    ranges.clear()
                elif key in (ord("q"), 27):  # cancel
                    return None
                elif key in (ord("c"), 13) and holds:  # confirm (needs at least one hold)
                    route = RouteMask(
                        mask=mask, holds=holds, frame_shape=frame.shape[:2]
                    )
                    return RouteSelection(route=route, frame=frame)
        finally:
            cv2.setMouseCallback(self._window_name, lambda *a: None)
