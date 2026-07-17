"""Layer A: hold-contact progress.

The objective substrate of the score. Given the frozen route holds and a live
pose, it tracks which holds a hand or foot has used, the highest hold reached,
and whether the climber has topped out.

This module owns the single coordinate-conversion boundary named in config.py:
pose keypoints arrive normalized [0, 1]; hold centroids are pixel coordinates.
Keypoints are converted to pixels here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from ..pose.keypoints import FOOT_LANDMARKS, HAND_LANDMARKS, PoseFrame
from ..route.holds import RouteMask


@dataclass(frozen=True)
class ContactState:
    """Snapshot of Layer A progress after one update.

    highest_rank is the highest hold reached counted from the ground: the bottom
    hold is rank 1 and the top hold is rank total_holds, so it reads naturally as
    "reached hold 7/10". None before any hold is used.
    """

    total_holds: int
    used_count: int
    highest_rank: int | None
    touched_indices: frozenset[int]  # holds a keypoint is inside this frame
    used_indices: frozenset[int]     # holds confirmed used (held for N frames)
    completed: bool


def _pixel_points(pose: PoseFrame, landmarks, width: int, height: int, min_vis: float):
    """Convert the given landmarks to pixel (x, y), dropping low-visibility ones."""
    points = []
    for landmark in landmarks:
        kp = pose.get(landmark)
        if kp.visibility < min_vis:
            continue
        points.append((kp.x * width, kp.y * height))
    return points


class ContactTracker:
    """Stateful Layer A tracker, updated once per processed pose frame.

    A hold becomes "used" once a hand or foot keypoint stays within its radius
    for contact_frames consecutive frames. Completion is stricter: a hand
    keypoint must stay within the top hold for completion_frames consecutive
    frames (feet on the top hold do not count as a send).
    """

    def __init__(
        self,
        route: RouteMask,
        contact_frames: int = config.HOLD_CONTACT_FRAMES_N,
        completion_frames: int = config.TOP_HOLD_COMPLETION_FRAMES_N,
        top_hands_required: int = config.TOP_HOLD_HANDS_REQUIRED,
        completion_radius_scale: float = config.TOP_HOLD_COMPLETION_RADIUS_SCALE,
        min_visibility: float = config.KEYPOINT_MIN_VISIBILITY,
    ) -> None:
        self._holds = route.holds
        self._top_index = route.top_index  # the user-designated finish hold
        self._contact_frames = contact_frames
        self._completion_frames = completion_frames
        self._top_hands_required = top_hands_required
        self._completion_radius_scale = completion_radius_scale
        self._min_vis = min_visibility

        self._streak = [0] * len(self._holds)   # consecutive touched frames per hold
        self._used: set[int] = set()
        # Completion is per-hand: each wrist independently "matches" the top hold
        # once it stays on it long enough, and matches persist. A send is when
        # enough hands have matched, so a one-hand-then-the-other top counts.
        self._hand_top_streak = {lm: 0 for lm in HAND_LANDMARKS}
        self._matched_hands: set = set()
        self._completed = False

    def _touching(self, hold, points) -> bool:
        return self._count_inside(hold, points) > 0

    def _count_inside(self, hold, points, radius: float | None = None) -> int:
        r = hold.radius if radius is None else radius
        r2 = r * r
        return sum(
            1 for px, py in points
            if (px - hold.x) ** 2 + (py - hold.y) ** 2 <= r2
        )

    def update(self, pose: PoseFrame | None, width: int, height: int) -> ContactState:
        touched: set[int] = set()

        if pose is not None and self._holds:
            hands = _pixel_points(pose, HAND_LANDMARKS, width, height, self._min_vis)
            feet = _pixel_points(pose, FOOT_LANDMARKS, width, height, self._min_vis)
            hands_and_feet = hands + feet

            for i, hold in enumerate(self._holds):
                if self._touching(hold, hands_and_feet):
                    touched.add(i)
                    self._streak[i] += 1
                    if self._streak[i] >= self._contact_frames:
                        self._used.add(i)
                else:
                    self._streak[i] = 0

            # Completion: each hand matches the top hold independently (with an
            # enlarged radius so a realistic top-out registers). A hand that has
            # matched stays matched, so matching one hand then the other counts.
            if self._top_index is not None:
                top = self._holds[self._top_index]
                radius = top.radius * self._completion_radius_scale
                for hand_lm in HAND_LANDMARKS:
                    kp = pose.get(hand_lm)
                    if kp.visibility < self._min_vis:
                        self._hand_top_streak[hand_lm] = 0
                        continue
                    point = (kp.x * width, kp.y * height)
                    if self._count_inside(top, [point], radius):
                        self._hand_top_streak[hand_lm] += 1
                        if self._hand_top_streak[hand_lm] >= self._completion_frames:
                            self._matched_hands.add(hand_lm)
                    else:
                        self._hand_top_streak[hand_lm] = 0
                if len(self._matched_hands) >= self._top_hands_required:
                    self._completed = True
        else:
            # No usable pose this frame: break contact streaks, keep results.
            self._streak = [0] * len(self._holds)
            for lm in self._hand_top_streak:
                self._hand_top_streak[lm] = 0

        return self._state(touched)

    def _state(self, touched: set[int]) -> ContactState:
        highest_rank = None
        if self._used:
            highest_index = min(self._used)  # smallest index = highest on wall
            highest_rank = len(self._holds) - highest_index
        return ContactState(
            total_holds=len(self._holds),
            used_count=len(self._used),
            highest_rank=highest_rank,
            touched_indices=frozenset(touched),
            used_indices=frozenset(self._used),
            completed=self._completed,
        )

    @property
    def completed(self) -> bool:
        return self._completed
