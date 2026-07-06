"""Hold centroid extraction and the frozen route target set.

A hold is reduced to (x, y, radius) in pixel coordinates. After route selection
these targets are cached and never re-segmented (the wall does not move), so the
scorer works against this frozen structure, not against a live mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .. import config


@dataclass(frozen=True)
class Hold:
    """A single hold target in pixel coordinates of the captured frame.

    radius is the contact radius: a hand/foot keypoint within this distance of
    (x, y) counts as touching the hold. It is derived from the hold's masked
    blob, so larger holds get a larger radius.
    """

    x: int
    y: int
    radius: int


@dataclass(frozen=True)
class RouteMask:
    """A frozen route: the binary mask it was built from plus its hold targets.

    holds is ordered top-to-bottom on the wall (smallest y first), so holds[0]
    is the top hold used for completion detection.
    """

    mask: np.ndarray
    holds: tuple[Hold, ...]
    frame_shape: tuple[int, int]  # (height, width) of the frame the mask came from

    @property
    def top_hold(self) -> Hold | None:
        return self.holds[0] if self.holds else None


def extract_holds(
    mask: np.ndarray,
    min_area_px: int = config.HOLD_MIN_AREA_PX,
) -> tuple[Hold, ...]:
    """Find hold blobs in a binary mask and reduce each to (x, y, radius).

    Blobs smaller than min_area_px are discarded as noise. The centroid is the
    blob's moment center; the radius is the max distance from that center to the
    blob outline, so a keypoint "within radius" covers the whole blob. Returned
    ordered top-to-bottom (ascending y).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holds: list[Hold] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        # Radius that covers the whole blob: farthest outline point from center.
        pts = contour.reshape(-1, 2).astype(np.float64)
        radius = float(np.max(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))

        holds.append(Hold(x=int(round(cx)), y=int(round(cy)), radius=int(round(radius))))

    holds.sort(key=lambda h: h.y)
    return tuple(holds)
