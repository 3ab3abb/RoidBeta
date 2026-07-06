"""Live overlay drawing.

Thin, swappable display layer. Draws pose keypoints now; holds, score, and
metrics are added as later build steps land. Every function takes a BGR frame
and draws in place, returning the same frame for convenience.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..pose.keypoints import FOOT_LANDMARKS, HAND_LANDMARKS, Landmark, PoseFrame
from ..route.holds import Hold

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
    (Landmark.LEFT_ANKLE, Landmark.LEFT_FOOT_INDEX),
    (Landmark.LEFT_ANKLE, Landmark.LEFT_HEEL),
    (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
    (Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
    (Landmark.RIGHT_ANKLE, Landmark.RIGHT_FOOT_INDEX),
    (Landmark.RIGHT_ANKLE, Landmark.RIGHT_HEEL),
)

_SKELETON_COLOR = (200, 200, 200)   # BGR, light grey bones
_JOINT_COLOR = (0, 200, 0)          # green joints
_CONTACT_COLOR = (0, 165, 255)      # orange for hand/foot contact points
_MIN_VISIBILITY = 0.5               # do not draw low-confidence (occluded) points


_HOLD_COLOR = (255, 255, 0)         # cyan for frozen route holds
_TOP_HOLD_COLOR = (0, 255, 255)     # yellow for the top hold (completion target)


def _to_pixel(kp, width: int, height: int) -> tuple[int, int]:
    return int(kp.x * width), int(kp.y * height)


_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_TEXT_COLOR = (255, 255, 255)
_HUD_BG_COLOR = (0, 0, 0)


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a stack of status/instruction lines in a panel at top-left."""
    if not lines:
        return frame

    pad = 8
    line_h = 24
    width = max(cv2.getTextSize(t, _HUD_FONT, 0.6, 1)[0][0] for t in lines) + 2 * pad
    height = line_h * len(lines) + pad

    panel = frame[0:height, 0:width]
    tint = np.full_like(panel, _HUD_BG_COLOR)
    frame[0:height, 0:width] = cv2.addWeighted(panel, 0.4, tint, 0.6, 0)

    for i, text in enumerate(lines):
        y = pad + line_h * (i + 1) - 6
        cv2.putText(frame, text, (pad, y), _HUD_FONT, 0.6, _HUD_TEXT_COLOR, 1, cv2.LINE_AA)
    return frame


def draw_holds(
    frame: np.ndarray,
    holds: tuple[Hold, ...],
    top_hold: Hold | None = None,
) -> np.ndarray:
    """Draw frozen route holds as contact circles; highlight the top hold."""
    for hold in holds:
        is_top = top_hold is not None and hold is top_hold
        color = _TOP_HOLD_COLOR if is_top else _HOLD_COLOR
        cv2.circle(frame, (hold.x, hold.y), hold.radius, color, 2)
        cv2.circle(frame, (hold.x, hold.y), 2, color, -1)
    return frame


def draw_pose(frame: np.ndarray, pose: PoseFrame | None) -> np.ndarray:
    """Draw the skeleton, joints, and hand/foot contact points on the frame."""
    if pose is None:
        return frame

    height, width = frame.shape[:2]

    for a, b in _POSE_CONNECTIONS:
        ka, kb = pose.get(a), pose.get(b)
        if ka.visibility < _MIN_VISIBILITY or kb.visibility < _MIN_VISIBILITY:
            continue
        cv2.line(
            frame,
            _to_pixel(ka, width, height),
            _to_pixel(kb, width, height),
            _SKELETON_COLOR,
            2,
        )

    contact = set(HAND_LANDMARKS) | set(FOOT_LANDMARKS)
    for landmark in Landmark:
        kp = pose.get(landmark)
        if kp.visibility < _MIN_VISIBILITY:
            continue
        color = _CONTACT_COLOR if landmark in contact else _JOINT_COLOR
        radius = 6 if landmark in contact else 3
        cv2.circle(frame, _to_pixel(kp, width, height), radius, color, -1)

    return frame
