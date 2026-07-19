"""Tests for the smoothness metric and the per-attempt skill profile."""

import numpy as np

from roidbeta.scoring.metrics import compute_smoothness
from roidbeta.scoring.profile import build_profile
from roidbeta.scoring.scorer import AttemptHistory
from roidbeta.route.holds import Hold, HoldRole, RouteMask
from roidbeta.scoring.contact import ContactTracker
from roidbeta.pose.keypoints import Keypoint, Landmark, PoseFrame


def test_smoothness_smooth_path_scores_higher_than_jerky():
    smooth = [(float(i), 100.0) for i in range(60)]                 # straight line
    jerky = [(float(i), 100.0 + (30 if i % 2 else -30)) for i in range(60)]  # zig-zag
    s_smooth = compute_smoothness(smooth)
    s_jerky = compute_smoothness(jerky)
    assert s_smooth is not None and s_jerky is not None
    assert s_smooth > s_jerky
    assert 0.0 <= s_jerky <= 1.0 and 0.0 <= s_smooth <= 1.0


def test_smoothness_none_when_too_few_points():
    assert compute_smoothness([(0.0, 0.0), (1.0, 1.0)]) is None
    assert compute_smoothness([None, None]) is None


def test_smoothness_ignores_gaps_between_runs():
    # A big jump across a None gap must not be counted as one huge acceleration.
    traj = [(float(i), 0.0) for i in range(20)] + [None] + \
           [(500.0 + i, 0.0) for i in range(20)]
    s = compute_smoothness(traj)
    assert s is not None and s > 0.9  # both runs are straight -> very smooth


def test_build_profile_reduces_attempt_to_axes():
    W, H = 100, 200
    route = RouteMask(mask=np.zeros((H, W), np.uint8),
                      holds=(Hold(50, 20, 12, role=HoldRole.TOP), Hold(50, 180, 12)),
                      frame_shape=(H, W))
    hist = AttemptHistory(
        com_trajectory=[(float(i), 100.0) for i in range(40)],
        balance_history=[0.8] * 30 + [None] * 5 + [0.6] * 5,
    )
    # Fabricate a contact state: reached the top of a 2-hold route.
    tracker = ContactTracker(route, contact_frames=1, completion_seconds=0.0)
    both = [Keypoint(0, 0, 0.0) for _ in range(33)]
    both[int(Landmark.LEFT_WRIST)] = Keypoint(50 / W, 20 / H, 1.0)
    both[int(Landmark.RIGHT_WRIST)] = Keypoint(50 / W, 20 / H, 1.0)
    contact = tracker.update(PoseFrame(tuple(both)), W, H, now=0.0)

    p = build_profile("P1", hist, contact, time_s=20.0)
    assert p.label == "P1"
    assert 0.0 <= p.balance <= 1.0
    assert p.smoothness is not None and p.smoothness > 0.9  # straight CoG path
    assert p.speed >= 0.0
    axes = p.axis_values()
    assert set(axes) == {"Balance", "Smoothness", "Speed", "Reach"}
    assert all(0.0 <= v <= 1.0 for v in axes.values())
