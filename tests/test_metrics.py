"""Tests for Layer B movement-quality metrics (CoM over base of support).

Synthetic poses over a known set of holds; no camera or MediaPipe. Run:
    python -m pytest tests/test_metrics.py
"""

import numpy as np

from roidbeta.pose.keypoints import Keypoint, Landmark, PoseFrame
from roidbeta.route.holds import Hold
from roidbeta.scoring.metrics import compute_balance

WIDTH, HEIGHT = 200, 400


def _pose(points):
    kps = [Keypoint(0.0, 0.0, 0.0) for _ in range(33)]
    for landmark, (px, py) in points.items():
        kps[int(landmark)] = Keypoint(px / WIDTH, py / HEIGHT, 1.0)
    return PoseFrame(keypoints=tuple(kps))


# Two foot holds spread apart near the bottom, forming a wide support base.
FOOT_HOLDS = (Hold(x=60, y=300, radius=15), Hold(x=140, y=300, radius=15))


def _torso(cx):
    """Shoulders and hips centered at column cx (so the CoM lands near cx)."""
    return {
        Landmark.LEFT_SHOULDER: (cx - 15, 150), Landmark.RIGHT_SHOULDER: (cx + 15, 150),
        Landmark.LEFT_HIP: (cx - 12, 240), Landmark.RIGHT_HIP: (cx + 12, 240),
    }


def test_balanced_when_com_over_base_center():
    pose = _pose({
        **_torso(cx=100),  # CoM near x=100, the base center
        Landmark.LEFT_FOOT_INDEX: (60, 300),
        Landmark.RIGHT_FOOT_INDEX: (140, 300),
    })
    b = compute_balance(pose, FOOT_HOLDS, WIDTH, HEIGHT)
    assert b.over_base
    assert len(b.support_points) == 2
    assert b.balance_score is not None and b.balance_score > 0.9


def test_off_balance_when_com_outside_base():
    pose = _pose({
        **_torso(cx=190),  # CoM far to the right of the base (60..140)
        Landmark.LEFT_FOOT_INDEX: (60, 300),
        Landmark.RIGHT_FOOT_INDEX: (140, 300),
    })
    b = compute_balance(pose, FOOT_HOLDS, WIDTH, HEIGHT)
    assert not b.over_base
    assert b.balance_score is not None and b.balance_score < 0.5


def test_none_when_too_few_support_points():
    pose = _pose({
        **_torso(cx=100),
        Landmark.LEFT_FOOT_INDEX: (60, 300),  # only one foot on a hold
    })
    b = compute_balance(pose, FOOT_HOLDS, WIDTH, HEIGHT)
    assert b.balance_score is None
    assert not b.over_base


def test_none_without_pose():
    b = compute_balance(None, FOOT_HOLDS, WIDTH, HEIGHT)
    assert b.com is None and b.balance_score is None


from roidbeta.scoring.metrics import BalanceTracker


def _feet_on_holds():
    return _pose({
        **_torso(cx=100),
        Landmark.LEFT_FOOT_INDEX: (60, 300),
        Landmark.RIGHT_FOOT_INDEX: (140, 300),
    })


def test_tracker_smooths_score_with_ema():
    t = BalanceTracker(alpha=0.5, hysteresis=0)
    # First balanced frame sets the EMA; a second, off-balance frame moves it only
    # halfway (alpha=0.5), so the smoothed value lies between the two raw scores.
    s1 = t.update(_feet_on_holds(), FOOT_HOLDS, WIDTH, HEIGHT)
    off = _pose({**_torso(cx=190),
                 Landmark.LEFT_FOOT_INDEX: (60, 300),
                 Landmark.RIGHT_FOOT_INDEX: (140, 300)})
    s2 = t.update(off, FOOT_HOLDS, WIDTH, HEIGHT)
    assert s2.balance_score is not None
    assert s2.balance_score > 0.0            # not the raw ~0 off-balance value
    assert s2.balance_score < s1.balance_score


def test_tracker_hysteresis_keeps_base_through_brief_flicker():
    t = BalanceTracker(alpha=1.0, hysteresis=5, gap_max=0)
    t.update(_feet_on_holds(), FOOT_HOLDS, WIDTH, HEIGHT)   # both feet on holds
    # One foot briefly leaves the hold; hysteresis should keep the base alive.
    one_foot = _pose({**_torso(cx=100),
                      Landmark.LEFT_FOOT_INDEX: (60, 300),
                      Landmark.RIGHT_FOOT_INDEX: (10, 10)})  # off any hold
    s = t.update(one_foot, FOOT_HOLDS, WIDTH, HEIGHT)
    assert s.balance_score is not None       # base retained, not a gap


def test_tracker_holds_score_through_gap_when_com_stable():
    t = BalanceTracker(alpha=1.0, hysteresis=0, gap_max=5, gap_stable_px=60)
    t.update(_feet_on_holds(), FOOT_HOLDS, WIDTH, HEIGHT)
    # Both feet off holds (no base), but the CoM barely moves: carry the score.
    airborne = _pose({**_torso(cx=102)})  # no feet on holds
    s = t.update(airborne, FOOT_HOLDS, WIDTH, HEIGHT)
    assert s.balance_score is not None       # held via CoM-stability gate


def test_tracker_gaps_when_com_jumps():
    t = BalanceTracker(alpha=1.0, hysteresis=0, gap_max=5, gap_stable_px=30)
    t.update(_feet_on_holds(), FOOT_HOLDS, WIDTH, HEIGHT)
    # No base and the CoM leaps far: an honest gap, not a fabricated value.
    dynamic = _pose({
        Landmark.LEFT_SHOULDER: (185, 150), Landmark.RIGHT_SHOULDER: (195, 150),
        Landmark.LEFT_HIP: (185, 240), Landmark.RIGHT_HIP: (195, 240),
    })
    s = t.update(dynamic, FOOT_HOLDS, WIDTH, HEIGHT)
    assert s.balance_score is None
