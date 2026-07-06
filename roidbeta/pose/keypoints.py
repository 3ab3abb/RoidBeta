"""Stable keypoint representation passed between modules.

Raw MediaPipe result objects never cross a module boundary. The pose layer
converts once, here, into a plain, documented structure. Downstream code
(scoring, display) depends only on this.

Coordinates are normalized [0, 1] relative to frame width and height, matching
MediaPipe. Conversion to pixel space happens at a single boundary in
scoring/contact.py, per the convention in config.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Landmark(IntEnum):
    """MediaPipe Pose landmark indices, named.

    Full 33-point model. Only the subset used by scoring is documented in
    detail, but the whole mapping is kept so nothing downstream re-derives it.
    """

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Keypoints that count as a "hand" or "foot" for hold-contact scoring (Layer A).
# Wrists stand in for hands; foot-index (toe) and heel stand in for feet.
HAND_LANDMARKS = (Landmark.LEFT_WRIST, Landmark.RIGHT_WRIST)
FOOT_LANDMARKS = (
    Landmark.LEFT_FOOT_INDEX,
    Landmark.RIGHT_FOOT_INDEX,
    Landmark.LEFT_HEEL,
    Landmark.RIGHT_HEEL,
)


@dataclass(frozen=True)
class Keypoint:
    """One landmark in normalized frame coordinates.

    x, y are in [0, 1]. visibility is MediaPipe's confidence in [0, 1]; low
    visibility means the point is likely occluded or off-frame and should be
    treated cautiously by scoring.
    """

    x: float
    y: float
    visibility: float


@dataclass(frozen=True)
class PoseFrame:
    """A full 33-landmark pose for a single frame.

    keypoints is indexed by Landmark. When MediaPipe finds no person, the pose
    layer emits None rather than a PoseFrame, so an empty PoseFrame is never a
    valid state.
    """

    keypoints: tuple[Keypoint, ...]

    def get(self, landmark: Landmark) -> Keypoint:
        return self.keypoints[int(landmark)]
