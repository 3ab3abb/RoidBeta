"""Tests for pose-derived quantities (the center-of-mass approximation).

Pure math over synthetic poses; no MediaPipe. Run:
    python -m pytest tests/test_pose.py
"""

import pytest

from roidbeta.pose.keypoints import Keypoint, Landmark, PoseFrame, center_of_mass


def _pose(points):
    kps = [Keypoint(0.0, 0.0, 0.0) for _ in range(33)]
    for landmark, (x, y) in points.items():
        kps[int(landmark)] = Keypoint(x, y, 1.0)
    return PoseFrame(keypoints=tuple(kps))


def test_com_weights_hips_over_shoulders():
    # Shoulders midpoint at y=0.2, hips midpoint at y=0.6. Weighted 0.4/0.6.
    pose = _pose({
        Landmark.LEFT_SHOULDER: (0.4, 0.2),
        Landmark.RIGHT_SHOULDER: (0.6, 0.2),
        Landmark.LEFT_HIP: (0.4, 0.6),
        Landmark.RIGHT_HIP: (0.6, 0.6),
    })
    com = center_of_mass(pose)
    assert com is not None
    assert com[0] == pytest.approx(0.5)
    assert com[1] == pytest.approx(0.6 * 0.6 + 0.4 * 0.2)  # pulled toward hips


def test_com_falls_back_to_shoulders_when_hips_hidden():
    pose = _pose({
        Landmark.LEFT_SHOULDER: (0.4, 0.2),
        Landmark.RIGHT_SHOULDER: (0.6, 0.2),
    })
    com = center_of_mass(pose)
    assert com == pytest.approx((0.5, 0.2))


def test_com_none_when_torso_not_visible():
    pose = _pose({Landmark.NOSE: (0.5, 0.1)})  # no hips or shoulders
    assert center_of_mass(pose) is None
