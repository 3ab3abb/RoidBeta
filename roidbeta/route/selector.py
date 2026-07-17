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

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .. import config
from .holds import Hold, HoldRole, RouteMask, extract_holds

# Bounds of the OpenCV HSV space: hue is 0..179, sat and val are 0..255.
_HUE_MAX = 179
_CHANNEL_MAX = 255

HsvRange = tuple[np.ndarray, np.ndarray]  # (lower, upper) inclusive inRange bounds


def normalize_lighting(frame_bgr: np.ndarray) -> np.ndarray:
    """Even out uneven gym lighting before color work.

    Applies CLAHE (contrast-limited adaptive histogram equalization) to the L
    channel in LAB space and returns the result as BGR. This lifts shadowed and
    chalk-dulled holds toward their true color and tames warm glare, so the same
    sampled range catches a hold across the whole wall instead of only where the
    light is even. Sampling and masking must both run on the normalized frame so
    the sampled range and the mask agree.
    """
    if not config.COLOR_NORMALIZE_ENABLED:
        return frame_bgr
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=(config.CLAHE_TILE_GRID, config.CLAHE_TILE_GRID),
    )
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


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


def _nearest_index(holds, x: int, y: int, tol: int) -> int | None:
    """Return the index of the hold within tol pixels of (x, y), closest first."""
    best = None
    best_d = float(tol)
    for i, hold in enumerate(holds):
        d = float(np.hypot(hold.x - x, hold.y - y))
        if d <= best_d:
            best_d = d
            best = i
    return best


def _nearest_hold(holds, x: int, y: int, tol: int):
    """Return the hold within tol pixels of (x, y) that is closest, or None."""
    idx = _nearest_index(holds, x, y, tol)
    return None if idx is None else holds[idx]


def _apply_exclusions(holds, excluded, tol: int):
    """Drop auto holds the user removed, matched by proximity to an exclusion."""
    if not excluded:
        return holds
    kept = []
    for hold in holds:
        if any(np.hypot(hold.x - ex, hold.y - ey) <= tol for ex, ey in excluded):
            continue
        kept.append(hold)
    return kept


def _assign_roles(holds, top_point, start_points, tol: int):
    """Stamp START/TOP roles onto the holds nearest the designated positions.

    Roles are stored as positions (not hold identities) so they survive the auto
    holds being re-extracted every frame: whichever hold is nearest a designated
    point gets the role. The TOP takes precedence over START on the same hold.
    """
    holds = list(holds)
    if top_point is not None:
        idx = _nearest_index(holds, top_point[0], top_point[1], tol)
        if idx is not None:
            holds[idx] = replace(holds[idx], role=HoldRole.TOP)
    for sx, sy in start_points:
        idx = _nearest_index(holds, sx, sy, tol)
        if idx is not None and holds[idx].role is HoldRole.NORMAL:
            holds[idx] = replace(holds[idx], role=HoldRole.START)
    return tuple(holds)


def _combine_holds(auto, manual, top_point, start_points, tol: int):
    """Merge color-extracted and hand-placed holds, assign roles, order top-down."""
    combined = list(auto) + list(manual)
    combined.sort(key=lambda h: h.y)
    return _assign_roles(combined, top_point, start_points, tol)


_SAMPLE_TINT = (0, 255, 0)          # green tint over the color mask
_AUTO_COLOR = (0, 165, 255)         # orange for color-extracted holds
_MANUAL_COLOR = (255, 0, 255)       # magenta for hand-placed debug holds
_START_COLOR = (0, 255, 0)          # green ring for the start hold(s)
_TOP_COLOR = (0, 255, 255)          # yellow ring for the top hold
_CENTER_COLOR = (0, 0, 255)


def _overlay_selection(frame_bgr, mask, holds, mode: str, manual_radius: int):
    """Preview: color tint, hold targets with roles, mode, and instructions."""
    preview = frame_bgr.copy()
    tint = np.zeros_like(preview)
    tint[mask > 0] = _SAMPLE_TINT
    preview = cv2.addWeighted(preview, 1.0, tint, 0.4, 0)

    for hold in holds:
        color = _MANUAL_COLOR if hold.manual else _AUTO_COLOR
        cv2.circle(preview, (hold.x, hold.y), hold.radius, color, 2)
        cv2.circle(preview, (hold.x, hold.y), 3, _CENTER_COLOR, -1)
        if hold.role is HoldRole.TOP:
            cv2.circle(preview, (hold.x, hold.y), hold.radius + 6, _TOP_COLOR, 2)
            _label(preview, "TOP", hold)
        elif hold.role is HoldRole.START:
            cv2.circle(preview, (hold.x, hold.y), hold.radius + 6, _START_COLOR, 2)
            _label(preview, "START", hold)

    n_manual = sum(1 for h in holds if h.manual)
    has_top = any(h.role is HoldRole.TOP for h in holds)
    n_start = sum(1 for h in holds if h.role is HoldRole.START)
    lclick = "place manual hold" if mode == "manual" else "sample color"
    lines = [
        f"holds: {len(holds)} ({n_manual} manual)   start:{n_start}   "
        f"top:{'yes' if has_top else 'NO'}",
        f"L-click: {lclick}   R-click: remove   M: manual place {'ON' if mode == 'manual' else 'off'}",
        "hover a hold + T: set TOP   S: toggle START   [ ]: manual radius",
        "R: reset   Enter/C: confirm (needs TOP)   Q/Esc: cancel",
    ]
    for i, text in enumerate(lines):
        y = 24 * (i + 1)
        cv2.putText(preview, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def _label(frame, text: str, hold) -> None:
    org = (hold.x + hold.radius + 6, hold.y)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


class RouteSelector:
    """Interactive, one-time route selection on a frozen frame.

    Left-click always samples a hold's color into the mask (the primary action);
    press M to toggle a manual-placement mode where left-click instead drops a
    hold by hand for holds the color mask cannot see. Role designation does not
    hijack the click: hover the cursor over a hold and press T to set it as the
    TOP (finish) hold, or S to toggle it as a START hold. Right-click removes the
    hold under the cursor. [ / ] size the manual hold, R resets, Enter/C confirms
    (a TOP hold is required), Q/Esc cancels. Returns a RouteSelection or None.
    """

    def __init__(self, window_name: str = config.OVERLAY_WINDOW_NAME) -> None:
        self._window_name = window_name

    def select(self, frame_bgr: np.ndarray) -> RouteSelection | None:
        frame = frame_bgr.copy()  # freeze the snapshot; the wall does not move
        # Sample and mask on the lighting-normalized frame; show the original so
        # the user still sees the wall as it looks.
        work = normalize_lighting(frame)
        ranges: list[HsvRange] = []
        manual_holds: list[Hold] = []
        excluded: list[tuple[int, int]] = []  # centroids of removed auto holds
        start_points: list[tuple[int, int]] = []
        tol = config.HOLD_UNSELECT_TOLERANCE_PX
        # Mutable containers so the mouse callback can share state with the loop.
        manual_mode = {"on": False}
        manual_radius = {"value": config.MANUAL_HOLD_RADIUS_PX}
        top_point: dict = {"value": None}
        cursor = {"pos": (0, 0)}
        holds_now: dict = {"holds": ()}      # latest combined holds (for role keys)
        auto_now: dict = {"holds": ()}       # latest auto holds (for right-click)

        def on_mouse(event, mx, my, _flags, _param):
            if event == cv2.EVENT_MOUSEMOVE:
                cursor["pos"] = (mx, my)
            elif event == cv2.EVENT_LBUTTONDOWN:
                if manual_mode["on"]:
                    manual_holds.append(
                        Hold(x=mx, y=my, radius=manual_radius["value"], manual=True)
                    )
                else:
                    ranges.extend(sample_hsv_ranges(work, mx, my))
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Prefer removing a hand-placed hold; else exclude an auto hold.
                mh = _nearest_hold(manual_holds, mx, my, tol)
                if mh is not None:
                    manual_holds.remove(mh)
                else:
                    ah = _nearest_hold(auto_now["holds"], mx, my, tol)
                    if ah is not None:
                        excluded.append((ah.x, ah.y))
                # Also clear any role designations on the removed hold.
                _drop_near(start_points, mx, my, tol)
                if top_point["value"] is not None and (
                    np.hypot(top_point["value"][0] - mx, top_point["value"][1] - my) <= tol
                ):
                    top_point["value"] = None

        def mark_top() -> None:
            cx, cy = cursor["pos"]
            hold = _nearest_hold(holds_now["holds"], cx, cy, tol)
            if hold is not None:
                top_point["value"] = (hold.x, hold.y)

        def toggle_start() -> None:
            cx, cy = cursor["pos"]
            hold = _nearest_hold(holds_now["holds"], cx, cy, tol)
            if hold is None:
                return
            existing = _nearest_index(
                [Hold(sx, sy, 0) for sx, sy in start_points], cx, cy, tol
            )
            if existing is not None:
                start_points.pop(existing)   # toggle off
            else:
                start_points.append((hold.x, hold.y))

        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, on_mouse)
        try:
            while True:
                if ranges:
                    mask = build_mask(work, ranges)
                    auto = _apply_exclusions(extract_holds(mask), excluded, tol)
                else:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    auto = []
                auto_now["holds"] = tuple(auto)
                holds = _combine_holds(
                    auto, manual_holds, top_point["value"], start_points, tol
                )
                holds_now["holds"] = holds

                mode = "manual" if manual_mode["on"] else "sample"
                preview = _overlay_selection(
                    frame, mask, holds, mode, manual_radius["value"]
                )
                cv2.imshow(self._window_name, preview)
                key = cv2.waitKey(20) & 0xFF
                has_top = any(h.role is HoldRole.TOP for h in holds)
                if key == ord("m"):
                    manual_mode["on"] = not manual_mode["on"]
                elif key == ord("t"):
                    mark_top()
                elif key == ord("s"):
                    toggle_start()
                elif key == ord("["):
                    manual_radius["value"] = max(4, manual_radius["value"] - 4)
                elif key == ord("]"):
                    manual_radius["value"] += 4
                elif key == ord("r"):
                    ranges.clear()
                    manual_holds.clear()
                    excluded.clear()
                    start_points.clear()
                    top_point["value"] = None
                elif key in (ord("q"), 27):  # cancel
                    return None
                elif key in (ord("c"), 13) and holds and has_top:  # confirm
                    route = RouteMask(
                        mask=mask, holds=holds, frame_shape=frame.shape[:2]
                    )
                    return RouteSelection(route=route, frame=frame)
        finally:
            cv2.setMouseCallback(self._window_name, lambda *a: None)


def _drop_near(points: list[tuple[int, int]], x: int, y: int, tol: int) -> None:
    """Remove any point within tol of (x, y) from the list, in place."""
    points[:] = [p for p in points if np.hypot(p[0] - x, p[1] - y) > tol]
