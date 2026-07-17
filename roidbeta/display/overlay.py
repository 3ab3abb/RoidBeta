"""Live overlay drawing.

Thin, swappable display layer. Draws pose keypoints now; holds, score, and
metrics are added as later build steps land. Every function takes a BGR frame
and draws in place, returning the same frame for convenience.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import config
from . import theme
from ..pose.keypoints import (
    FACE_LANDMARKS,
    FOOT_LANDMARKS,
    HAND_LANDMARKS,
    LEFT_SIDE,
    RIGHT_SIDE,
    Landmark,
    PoseFrame,
    center_of_mass,
)
from ..route.holds import Hold, HoldRole

# MediaPipe Pose skeleton as landmark index pairs. Kept local to the display
# layer so scoring never depends on how the body is drawn.
_POSE_CONNECTIONS: tuple[tuple[Landmark, Landmark], ...] = (
    (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER),
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
    (Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
    (Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP),
    (Landmark.LEFT_HIP, Landmark.RIGHT_HIP),
    (Landmark.LEFT_HIP, Landmark.LEFT_KNEE),
    (Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
    (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
    (Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
    # Feet themselves are drawn as filled, enlarged triangles in _draw_feet, not
    # as raw skeleton lines (which read too small and tilt with keypoint noise).
)

# Skeleton is color-coded by body side so left and right limbs are easy to tell
# apart. Torso cross-links (shoulder-shoulder, hip-hip, shoulder-hip) are center.
_LEFT_COLOR = (80, 255, 60)         # green for the left side
_RIGHT_COLOR = (255, 140, 0)        # blue for the right side
_CENTER_COLOR = (0, 220, 255)       # amber for the torso
_CONTACT_COLOR = (0, 165, 255)      # orange for hand/foot contact joints
_COM_FILL = (0, 0, 255)             # red center-of-mass marker
_COM_RING = (255, 255, 255)         # white ring around the CoM marker
_ENVELOPE_COLOR = (255, 255, 120)   # pale cyan convex-hull envelope
_MIN_VISIBILITY = config.KEYPOINT_MIN_VISIBILITY  # skip low-confidence points


def _connection_color(a: Landmark, b: Landmark) -> tuple[int, int, int]:
    if a in LEFT_SIDE and b in LEFT_SIDE:
        return _LEFT_COLOR
    if a in RIGHT_SIDE and b in RIGHT_SIDE:
        return _RIGHT_COLOR
    return _CENTER_COLOR


_HOLD_COLOR = (255, 255, 0)         # cyan for unused route holds
_USED_HOLD_COLOR = (0, 255, 0)      # green for holds already used
_TOUCHED_HOLD_COLOR = (0, 165, 255) # orange while a keypoint is inside a hold
_TOP_HOLD_COLOR = (0, 255, 255)     # yellow ring on the top hold (completion target)
_START_HOLD_COLOR = (0, 255, 0)     # green ring on the start hold(s)
_MANUAL_HOLD_COLOR = (255, 0, 255)  # magenta accent for hand-placed debug holds


def _to_pixel(kp, width: int, height: int) -> tuple[int, int]:
    return int(kp.x * width), int(kp.y * height)


_HUD_FONT = theme.FONT


def draw_hud(frame: np.ndarray, lines: list[str], title_accent=theme.PRIMARY) -> np.ndarray:
    """Draw a stack of lines in a themed rounded panel at top-left.

    The first line is treated as a title (accented), the rest as body text. Used
    for simple screens like clip review.
    """
    if not lines:
        return frame

    pad = 14
    line_h = 26
    width = max(theme.text_size(t, 0.55, 1)[0] for t in lines) + 2 * pad
    height = line_h * len(lines) + pad + 4

    theme.panel(frame, 16, 16, width, height, radius=12, color=theme.BG, alpha=0.82)
    for i, line in enumerate(lines):
        y = 16 + pad + line_h * (i + 1) - 8
        color = title_accent if i == 0 else theme.TEXT_MUTED
        scale = 0.6 if i == 0 else 0.5
        theme.text(frame, line, (16 + pad, y), scale, color, 1)
    return frame


def draw_holds(
    frame: np.ndarray,
    holds: tuple[Hold, ...],
    used_indices: frozenset[int] = frozenset(),
    touched_indices: frozenset[int] = frozenset(),
) -> np.ndarray:
    """Draw route holds, colored by scoring state.

    Unused holds are cyan, used holds green, and a hold with a keypoint inside it
    this frame is drawn thicker in orange. The designated START hold gets a green
    ring and label, the TOP hold a yellow ring and label, and hand-placed (manual)
    holds a magenta accent so debug targets are obvious.
    """
    for i, hold in enumerate(holds):
        if i in used_indices:
            color = _USED_HOLD_COLOR
        else:
            color = _HOLD_COLOR
        thickness = 3 if i in touched_indices else 2
        if i in touched_indices:
            color = _TOUCHED_HOLD_COLOR

        cv2.circle(frame, (hold.x, hold.y), hold.radius, color, thickness)
        cv2.circle(frame, (hold.x, hold.y), 2, color, -1)

        if hold.manual:
            cv2.circle(frame, (hold.x, hold.y), hold.radius + 3, _MANUAL_HOLD_COLOR, 1)
        if hold.role is HoldRole.START:
            cv2.circle(frame, (hold.x, hold.y), hold.radius + 6, _START_HOLD_COLOR, 2)
            _hold_label(frame, "START", hold)
        elif hold.role is HoldRole.TOP:
            cv2.circle(frame, (hold.x, hold.y), hold.radius + 6, _TOP_HOLD_COLOR, 2)
            _hold_label(frame, "TOP", hold)
    return frame


_SUPPORT_COLOR = (255, 200, 0)      # base-of-support polygon
_BALANCE_GOOD = (0, 255, 0)         # CoM over the base
_BALANCE_BAD = (0, 0, 255)          # CoM outside the base


def draw_balance(frame: np.ndarray, balance) -> np.ndarray:
    """Draw the base-of-support polygon and a CoM-to-base balance indicator."""
    if balance.com is None or len(balance.support_points) < 2:
        return frame

    pts = np.array([(int(x), int(y)) for x, y in balance.support_points], np.int32)
    hull = cv2.convexHull(pts)
    cv2.polylines(frame, [hull], True, _SUPPORT_COLOR, 2, cv2.LINE_AA)

    xs = [p[0] for p in balance.support_points]
    ys = [p[1] for p in balance.support_points]
    base_center = (int((min(xs) + max(xs)) / 2), int(sum(ys) / len(ys)))
    com = (int(balance.com[0]), int(balance.com[1]))
    color = _BALANCE_GOOD if balance.over_base else _BALANCE_BAD
    cv2.line(frame, base_center, com, color, 2, cv2.LINE_AA)
    return frame


_TRAIL_COLOR = (0, 255, 255)        # live CoM trail (yellow)
_REFERENCE_COLOR = (200, 200, 200)  # reference (ghost) CoM trail (grey)


def draw_trajectory(
    frame: np.ndarray,
    points,
    color: tuple[int, int, int] = _TRAIL_COLOR,
    fade: bool = True,
) -> np.ndarray:
    """Draw a CoM path as a connected trail.

    points is a per-frame list of (x, y) pixel positions, with None for frames
    where the CoM could not be estimated (those break the line into segments).
    When fade is True the trail dims toward its older end, so the recent path
    reads brighter; a reference trail is drawn flat (fade=False).
    """
    recent = points[-config.TRAJECTORY_MAX_POINTS:]
    n = len(recent)
    thickness = config.TRAJECTORY_THICKNESS
    prev = None
    for i, p in enumerate(recent):
        if p is None:
            prev = None
            continue
        cur = (int(p[0]), int(p[1]))
        if prev is not None:
            if fade:
                t = 0.3 + 0.7 * (i / max(1, n - 1))
                seg = tuple(int(c * t) for c in color)
            else:
                seg = color
            cv2.line(frame, prev, cur, seg, thickness, cv2.LINE_AA)
        prev = cur
    return frame


def _hold_label(frame: np.ndarray, text: str, hold: Hold) -> None:
    org = (hold.x + hold.radius + 6, hold.y + 4)
    theme.text(frame, text, org, 0.44, theme.TEXT, 1)


def draw_pose(frame: np.ndarray, pose: PoseFrame | None) -> np.ndarray:
    """Draw the envelope, color-coded skeleton, joints, and the CoM marker.

    Face landmarks are excluded: only the body from the shoulders down is drawn.
    """
    if pose is None:
        return frame

    height, width = frame.shape[:2]
    thickness = config.POSE_SKELETON_THICKNESS

    # Envelope first, so the skeleton and joints sit on top of it.
    body_points = [
        _to_pixel(pose.get(lm), width, height)
        for lm in Landmark
        if lm not in FACE_LANDMARKS and pose.get(lm).visibility >= _MIN_VISIBILITY
    ]
    _draw_envelope(frame, body_points)
    _draw_feet(frame, pose, width, height)

    for a, b in _POSE_CONNECTIONS:
        ka, kb = pose.get(a), pose.get(b)
        if ka.visibility < _MIN_VISIBILITY or kb.visibility < _MIN_VISIBILITY:
            continue
        cv2.line(
            frame,
            _to_pixel(ka, width, height),
            _to_pixel(kb, width, height),
            _connection_color(a, b),
            thickness,
            cv2.LINE_AA,
        )

    contact = set(HAND_LANDMARKS) | set(FOOT_LANDMARKS)
    for landmark in Landmark:
        if landmark in FACE_LANDMARKS:
            continue  # no anchor points on the face
        kp = pose.get(landmark)
        if kp.visibility < _MIN_VISIBILITY:
            continue
        if landmark in contact:
            color, radius = _CONTACT_COLOR, config.POSE_CONTACT_JOINT_RADIUS
        else:
            side = _LEFT_COLOR if landmark in LEFT_SIDE else (
                _RIGHT_COLOR if landmark in RIGHT_SIDE else _CENTER_COLOR
            )
            color, radius = side, config.POSE_JOINT_RADIUS
        cv2.circle(frame, _to_pixel(kp, width, height), radius, color, -1)

    _draw_com(frame, pose, width, height)
    return frame


_FEET = (
    (_LEFT_COLOR, (Landmark.LEFT_ANKLE, Landmark.LEFT_HEEL, Landmark.LEFT_FOOT_INDEX)),
    (_RIGHT_COLOR, (Landmark.RIGHT_ANKLE, Landmark.RIGHT_HEEL, Landmark.RIGHT_FOOT_INDEX)),
)


def _draw_feet(frame: np.ndarray, pose: PoseFrame, width: int, height: int) -> None:
    """Draw each foot as a filled, enlarged ankle-heel-toe triangle.

    The heel and toe are pushed out from the ankle by POSE_FOOT_DRAW_SCALE so the
    foot reads at a sensible size instead of the tiny raw keypoint spread.
    """
    scale = config.POSE_FOOT_DRAW_SCALE
    for color, landmarks in _FEET:
        ankle, heel, toe = (pose.get(lm) for lm in landmarks)
        if any(k.visibility < _MIN_VISIBILITY for k in (ankle, heel, toe)):
            continue
        ax, ay = _to_pixel(ankle, width, height)

        def scaled(kp):
            px, py = _to_pixel(kp, width, height)
            return int(ax + (px - ax) * scale), int(ay + (py - ay) * scale)

        pts = np.array([(ax, ay), scaled(heel), scaled(toe)], np.int32)
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)
        cv2.polylines(frame, [pts], True, color,
                      config.POSE_SKELETON_THICKNESS, cv2.LINE_AA)


def _draw_envelope(frame: np.ndarray, points: list[tuple[int, int]]) -> None:
    """Wrap the skeleton in a colored convex-hull envelope (fill + outline)."""
    if len(points) < 3:
        return
    hull = cv2.convexHull(np.array(points, dtype=np.int32))

    alpha = config.POSE_ENVELOPE_ALPHA
    if alpha > 0:
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, hull, _ENVELOPE_COLOR)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.polylines(
        frame, [hull], True, _ENVELOPE_COLOR,
        config.POSE_ENVELOPE_THICKNESS, cv2.LINE_AA,
    )


def _draw_com(frame: np.ndarray, pose: PoseFrame, width: int, height: int) -> None:
    """Draw the approximate center-of-mass as a labeled dot, if estimable."""
    com = center_of_mass(pose)
    if com is None:
        return
    cx, cy = int(com[0] * width), int(com[1] * height)
    radius = config.COM_DOT_RADIUS
    cv2.circle(frame, (cx, cy), radius, _COM_FILL, -1)
    cv2.circle(frame, (cx, cy), radius, _COM_RING, 2, cv2.LINE_AA)
    cv2.putText(frame, "CoM", (cx + radius + 4, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COM_RING, 1, cv2.LINE_AA)
