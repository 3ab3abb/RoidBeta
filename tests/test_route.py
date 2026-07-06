"""Tests for route selection: HSV sampling, mask building, hold extraction.

These synthesize frames with known colored blobs so the color -> mask -> centroid
pipeline is verified objectively without a camera. Run:
    python -m pytest tests/test_route.py
"""

import cv2
import numpy as np

from roidbeta.route.holds import extract_holds
from roidbeta.route.selector import build_mask, sample_hsv_ranges


def _frame_with_circle(color_bgr, center, radius=20, size=200):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(frame, center, radius, color_bgr, -1)
    return frame


def test_sample_and_mask_recovers_blue_hold():
    blue = (255, 0, 0)  # BGR
    frame = _frame_with_circle(blue, center=(50, 50))

    ranges = sample_hsv_ranges(frame, 50, 50)
    mask = build_mask(frame, ranges)

    # The mask should light up roughly where the blue circle is and nowhere near
    # the black background corner.
    assert mask[50, 50] > 0
    assert mask[5, 5] == 0


def test_extract_holds_finds_centroids_ordered_top_to_bottom():
    blue = (255, 0, 0)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(frame, (150, 120), 20, blue, -1)  # lower on the wall
    cv2.circle(frame, (50, 40), 20, blue, -1)    # higher on the wall

    ranges = sample_hsv_ranges(frame, 50, 40)
    mask = build_mask(frame, ranges)
    holds = extract_holds(mask, min_area_px=100)

    assert len(holds) == 2
    # Ordered ascending y: the higher hold first.
    assert holds[0].y < holds[1].y
    # Centroids land near the drawn centers (allow a few px from morphology).
    assert abs(holds[0].x - 50) <= 4 and abs(holds[0].y - 40) <= 4
    assert abs(holds[1].x - 150) <= 4 and abs(holds[1].y - 120) <= 4
    # Radius is in the ballpark of the drawn radius (20).
    assert 14 <= holds[0].radius <= 26


def test_extract_holds_discards_small_blobs():
    blue = (255, 0, 0)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 20, blue, -1)  # real hold
    cv2.circle(frame, (10, 10), 2, blue, -1)     # speckle, below min area

    ranges = sample_hsv_ranges(frame, 100, 100)
    mask = build_mask(frame, ranges)
    holds = extract_holds(mask, min_area_px=150)

    assert len(holds) == 1
    assert abs(holds[0].x - 100) <= 4 and abs(holds[0].y - 100) <= 4


def test_red_hue_wraparound_is_handled():
    red = (0, 0, 255)  # BGR -> HSV hue near 0, so the tolerance range wraps
    frame = _frame_with_circle(red, center=(100, 100))

    ranges = sample_hsv_ranges(frame, 100, 100)
    # Red near hue 0 with tolerance should split into two inRange bounds.
    assert len(ranges) == 2

    mask = build_mask(frame, ranges)
    holds = extract_holds(mask, min_area_px=100)
    assert len(holds) == 1
    assert abs(holds[0].x - 100) <= 4 and abs(holds[0].y - 100) <= 4
