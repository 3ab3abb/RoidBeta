"""Hold centroid extraction and the frozen route target set.

A hold is reduced to (x, y, radius) in pixel coordinates. After route selection
these targets are cached and never re-segmented (the wall does not move), so the
scorer works against this frozen structure, not against a live mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .. import config


class HoldRole(Enum):
    """A hold's role on the route, designated by the user during selection.

    The setter decides these, not geometry: a route's start is not always the
    lowest hold and its top is not always the highest, so they are marked
    explicitly rather than inferred.
    """

    NORMAL = "normal"
    START = "start"
    TOP = "top"


@dataclass(frozen=True)
class Hold:
    """A single hold target in pixel coordinates of the captured frame.

    radius is the contact radius: a hand/foot keypoint within this distance of
    (x, y) counts as touching the hold. For color-extracted holds it is derived
    from the masked blob, so larger holds get a larger radius.

    manual is True for holds a user placed by hand during selection (for holds
    the color mask cannot see). They score identically; the flag only lets the
    overlay mark them so it is clear which targets are hand-placed.

    role marks the designated START or TOP hold. Completion is judged against
    the TOP hold only.
    """

    x: int
    y: int
    radius: int
    manual: bool = False
    role: HoldRole = HoldRole.NORMAL


@dataclass(frozen=True)
class RouteMask:
    """A frozen route: the binary mask it was built from plus its hold targets.

    holds is ordered top-to-bottom on the wall (smallest y first). The start and
    top holds are whichever ones the user designated, not the geometric extremes.
    """

    mask: np.ndarray
    holds: tuple[Hold, ...]
    frame_shape: tuple[int, int]  # (height, width) of the frame the mask came from

    @property
    def top_hold(self) -> Hold | None:
        """The designated finish hold, or None if none was designated."""
        for hold in self.holds:
            if hold.role is HoldRole.TOP:
                return hold
        return None

    @property
    def top_index(self) -> int | None:
        for i, hold in enumerate(self.holds):
            if hold.role is HoldRole.TOP:
                return i
        return None

    @property
    def start_holds(self) -> tuple[Hold, ...]:
        """The designated start hold(s); a boulder start is often two holds."""
        return tuple(h for h in self.holds if h.role is HoldRole.START)


def _hold_from_region(region: np.ndarray, min_area_px: int) -> Hold | None:
    """Reduce a single binary blob to a Hold (centroid + covering radius)."""
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area_px:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    pts = contour.reshape(-1, 2).astype(np.float64)
    radius = float(np.max(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
    return Hold(x=int(round(cx)), y=int(round(cy)), radius=int(round(radius)))


def _split_markers(mask: np.ndarray) -> tuple[int, np.ndarray] | None:
    """Seed watershed from distance-transform peaks, one per hold center.

    Returns (num_seeds, markers) or None when there are not at least two seeds
    (nothing to split, so the caller falls back to plain contours).
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.HOLD_SPLIT_PEAK_WINDOW_PX, config.HOLD_SPLIT_PEAK_WINDOW_PX),
    )
    dilated = cv2.dilate(dist, kernel)
    # A pixel is a center seed if it is a local max of the distance transform and
    # far enough from an edge to be a real hold center (not a thin bridge).
    peaks = ((dist >= dilated - 1e-5) & (dist >= config.HOLD_SPLIT_MIN_CENTER_DIST_PX))
    num_peaks, seeds = cv2.connectedComponents(peaks.astype(np.uint8))
    if num_peaks <= 2:  # 0/1 real seeds (label 0 is background): nothing to split
        return None
    return num_peaks, seeds


def extract_holds(
    mask: np.ndarray,
    min_area_px: int = config.HOLD_MIN_AREA_PX,
) -> tuple[Hold, ...]:
    """Find hold blobs in a binary mask and reduce each to (x, y, radius).

    Touching holds of the same color are split via a distance-transform watershed
    so they become separate centroids instead of one merged blob. Isolated blobs
    pass through unchanged. Blobs smaller than min_area_px are discarded as noise.
    Returned ordered top-to-bottom (ascending y).
    """
    if mask.max() == 0:
        return ()

    holds: list[Hold] = []
    split = _split_markers(mask)
    if split is not None:
        num_peaks, seeds = split
        # Watershed marker convention: label 1 = sure background (outside the
        # mask), labels 2..num_peaks = the hold-center seeds, 0 = unknown (the
        # mask interior to be flooded and divided between the seeds).
        markers = np.where(seeds > 0, seeds + 1, 0).astype(np.int32)
        markers[mask == 0] = 1
        cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
        for label in range(2, num_peaks + 1):
            region = np.where(markers == label, 255, 0).astype(np.uint8)
            hold = _hold_from_region(region, min_area_px)
            if hold is not None:
                holds.append(hold)
    else:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            region = np.zeros_like(mask)
            cv2.drawContours(region, [contour], -1, 255, -1)
            hold = _hold_from_region(region, min_area_px)
            if hold is not None:
                holds.append(hold)

    holds.sort(key=lambda h: h.y)
    return tuple(holds)
