"""Tests for Layer A hold-contact scoring.

Synthetic poses over a known route verify the contact/used/completion logic and
the normalized->pixel conversion boundary, with no camera or MediaPipe. Run:
    python -m pytest tests/test_contact.py
"""

import numpy as np

from roidbeta.pose.keypoints import Keypoint, Landmark, PoseFrame
from roidbeta.route.holds import Hold, HoldRole, RouteMask
from roidbeta.scoring.contact import ContactTracker

WIDTH, HEIGHT = 100, 200

# Route: three holds stacked on the wall, sorted top-to-bottom (index 0 = top).
# The top hold is explicitly designated (not inferred from geometry).
TOP = Hold(x=50, y=20, radius=12, role=HoldRole.TOP)     # index 0 -> rank 3
MID = Hold(x=50, y=100, radius=12)                       # index 1 -> rank 2
BOTTOM = Hold(x=50, y=180, radius=12, role=HoldRole.START)  # index 2 -> rank 1
ROUTE = RouteMask(mask=np.zeros((HEIGHT, WIDTH), np.uint8),
                  holds=(TOP, MID, BOTTOM), frame_shape=(HEIGHT, WIDTH))


def _pose(points):
    """Build a 33-landmark pose; only the given landmarks are visible."""
    kps = [Keypoint(0.0, 0.0, 0.0) for _ in range(33)]
    for landmark, (px, py) in points.items():
        kps[int(landmark)] = Keypoint(px / WIDTH, py / HEIGHT, 1.0)
    return PoseFrame(keypoints=tuple(kps))


def _tracker():
    return ContactTracker(ROUTE, contact_frames=3, completion_frames=3)


def test_hold_used_after_n_consecutive_frames():
    t = _tracker()
    hand_on_mid = _pose({Landmark.RIGHT_WRIST: (MID.x, MID.y)})

    s = t.update(hand_on_mid, WIDTH, HEIGHT)
    assert 1 in s.touched_indices and s.used_count == 0  # streak building
    t.update(hand_on_mid, WIDTH, HEIGHT)
    s = t.update(hand_on_mid, WIDTH, HEIGHT)  # third frame reaches contact_frames

    assert 1 in s.used_indices
    assert s.used_count == 1
    assert s.highest_rank == 2  # MID is rank 2 from the ground


def test_streak_resets_on_a_miss():
    t = _tracker()
    on = _pose({Landmark.RIGHT_WRIST: (MID.x, MID.y)})
    off = _pose({Landmark.RIGHT_WRIST: (5, 5)})  # far from any hold

    t.update(on, WIDTH, HEIGHT)
    t.update(on, WIDTH, HEIGHT)
    t.update(off, WIDTH, HEIGHT)  # breaks the streak before it reaches 3
    s = t.update(on, WIDTH, HEIGHT)

    assert s.used_count == 0  # has to start the count over


def test_low_visibility_keypoints_are_ignored():
    t = _tracker()
    kps = [Keypoint(0.0, 0.0, 0.0) for _ in range(33)]
    # Hand exactly on MID but with low visibility (occluded): should not count.
    kps[int(Landmark.RIGHT_WRIST)] = Keypoint(MID.x / WIDTH, MID.y / HEIGHT, 0.1)
    pose = PoseFrame(keypoints=tuple(kps))

    for _ in range(5):
        s = t.update(pose, WIDTH, HEIGHT)
    assert s.used_count == 0
    assert not s.touched_indices


def test_two_hands_on_top_hold_completes():
    t = _tracker()
    both_hands_on_top = _pose({
        Landmark.LEFT_WRIST: (TOP.x - 3, TOP.y),
        Landmark.RIGHT_WRIST: (TOP.x + 3, TOP.y),
    })

    for _ in range(3):
        s = t.update(both_hands_on_top, WIDTH, HEIGHT)
    assert s.completed
    assert t.completed
    assert s.highest_rank == 3  # topped out


def test_one_hand_on_top_hold_does_not_complete():
    t = _tracker()
    one_hand_on_top = _pose({Landmark.LEFT_WRIST: (TOP.x, TOP.y)})

    for _ in range(6):
        s = t.update(one_hand_on_top, WIDTH, HEIGHT)

    assert not s.completed        # a send needs two hands controlling the top
    assert 0 in s.used_indices    # but one hand still marks the hold used


def test_feet_on_top_hold_do_not_complete():
    t = _tracker()
    foot_on_top = _pose({
        Landmark.LEFT_FOOT_INDEX: (TOP.x - 3, TOP.y),
        Landmark.RIGHT_FOOT_INDEX: (TOP.x + 3, TOP.y),
    })

    for _ in range(5):
        s = t.update(foot_on_top, WIDTH, HEIGHT)

    assert not s.completed          # feet never count as a send
    assert 0 in s.used_indices      # but the feet still mark the hold used


def test_no_pose_breaks_streak_but_keeps_used():
    t = _tracker()
    on = _pose({Landmark.RIGHT_WRIST: (MID.x, MID.y)})
    for _ in range(3):
        t.update(on, WIDTH, HEIGHT)  # MID becomes used

    s = t.update(None, WIDTH, HEIGHT)  # dropped pose frame
    assert s.used_count == 1           # used holds are retained
    assert not s.touched_indices       # nothing touched this frame


def test_sequential_hands_complete_the_top():
    # A realistic top: one hand matches, then the other arrives later. It should
    # still complete even though both hands are never on the hold at once.
    t = _tracker()  # completion_frames=3
    left = _pose({Landmark.LEFT_WRIST: (TOP.x, TOP.y)})
    right = _pose({Landmark.RIGHT_WRIST: (TOP.x, TOP.y)})

    for _ in range(4):
        s = t.update(left, WIDTH, HEIGHT)   # left hand matches first
    assert not s.completed                  # one hand only, not a send yet
    for _ in range(4):
        s = t.update(right, WIDTH, HEIGHT)  # right hand matches later
    assert s.completed                      # both have now matched -> send
