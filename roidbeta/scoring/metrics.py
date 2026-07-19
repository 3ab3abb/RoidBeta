"""Layer B: movement-quality metrics.

Each metric here is a heuristic proxy, not ground truth, and is computed only
during an attempt and anchored to Layer A (which holds are in contact). They are
kept as small, independently testable functions so they can be tuned or toggled
one at a time.

First metric: center-of-mass over the base of support. In climbing, keeping your
weight (center of mass) over the polygon of your supporting contact points is the
essence of balance; letting the CoM drift outside that base means you are hanging
off your arms or barn-dooring. This works from any camera angle because it is
computed in the image plane from points we already track.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from ..pose.keypoints import (
    FOOT_LANDMARKS,
    HAND_LANDMARKS,
    Landmark,
    PoseFrame,
    center_of_mass,
)
from ..route.holds import Hold


@dataclass(frozen=True)
class BalanceState:
    """CoM-over-base-of-support balance for one frame.

    com and support_points are in pixel coordinates. balance_score is 0..1, with
    1 meaning the CoM sits right over the center of the support base and 0 meaning
    it is a full half-base-width (or more) off to one side. All fields are None
    when there is not enough visible support to judge balance.
    """

    com: tuple[float, float] | None
    support_points: tuple[tuple[float, float], ...]
    over_base: bool
    balance_offset_px: float | None   # signed: CoM x minus support-base center x
    balance_score: float | None


def _pixel(kp, width: int, height: int) -> tuple[float, float]:
    return kp.x * width, kp.y * height


def _shoulder_width_px(pose: PoseFrame, width: int, min_vis: float) -> float | None:
    ls, rs = pose.get(Landmark.LEFT_SHOULDER), pose.get(Landmark.RIGHT_SHOULDER)
    if ls.visibility < min_vis or rs.visibility < min_vis:
        return None
    return abs(ls.x - rs.x) * width


def _com_px(pose: PoseFrame, width, height, min_vis) -> tuple[float, float] | None:
    com_norm = center_of_mass(pose, min_vis)
    if com_norm is None:
        return None
    return com_norm[0] * width, com_norm[1] * height


def _support_inside(pose, holds, width, height, min_vis):
    """Return (landmark, (px, py)) for each hand/foot keypoint inside a hold."""
    found = []
    for landmark in (*HAND_LANDMARKS, *FOOT_LANDMARKS):
        kp = pose.get(landmark)
        if kp.visibility < min_vis:
            continue
        px, py = _pixel(kp, width, height)
        if any((px - h.x) ** 2 + (py - h.y) ** 2 <= h.radius * h.radius for h in holds):
            found.append((landmark, (px, py)))
    return found


def _score_from(com, support_points, shoulder_px):
    """Balance from a CoM and >=2 support positions. Returns (over, offset, score)."""
    xs = [p[0] for p in support_points]
    base_min, base_max = min(xs), max(xs)
    base_center = (base_min + base_max) / 2.0
    offset = com[0] - base_center
    over_base = base_min <= com[0] <= base_max
    # Normalize by the base half-width, falling back to shoulder width when the
    # support points are nearly vertical (a degenerate, tiny base).
    half_base = (base_max - base_min) / 2.0
    norm = max(half_base, (shoulder_px or 0.0) * 0.5, 1.0)
    return over_base, offset, max(0.0, 1.0 - abs(offset) / norm)


def compute_balance(
    pose: PoseFrame | None,
    holds: tuple[Hold, ...],
    width: int,
    height: int,
    min_visibility: float = config.KEYPOINT_MIN_VISIBILITY,
) -> BalanceState:
    """Per-frame, stateless balance (no smoothing). Used for tests and by the tracker."""
    if pose is None:
        return BalanceState(None, (), False, None, None)
    com = _com_px(pose, width, height, min_visibility)
    if com is None:
        return BalanceState(None, (), False, None, None)

    support = [pos for _, pos in _support_inside(pose, holds, width, height, min_visibility)]
    if len(support) < config.BALANCE_MIN_SUPPORT_POINTS:
        return BalanceState(com, tuple(support), False, None, None)

    shoulder = _shoulder_width_px(pose, width, min_visibility)
    over_base, offset, score = _score_from(com, support, shoulder)
    return BalanceState(com, tuple(support), over_base, offset, score)


class BalanceTracker:
    """Stateful balance with three stabilizers over the raw per-frame value.

    1. Sticky support: a hand/foot stays in the base for a few frames after it
       leaves a hold radius, so brief contact flicker does not collapse the base.
    2. EMA smoothing: the score is exponentially smoothed to remove jitter.
    3. Gap hold: during a frame with too few contacts, the last smoothed score is
       carried forward, but only while the CoM stays near where it last had a
       valid base (the trail's stability is the confidence signal) and only for a
       bounded number of frames. A dynamic move (CoM jumps) leaves an honest gap.
    """

    def __init__(
        self,
        alpha: float = config.BALANCE_EMA_ALPHA,
        hysteresis: int = config.BALANCE_SUPPORT_HYSTERESIS_FRAMES,
        gap_max: int = config.BALANCE_GAP_HOLD_MAX_FRAMES,
        gap_stable_px: float = config.BALANCE_GAP_COM_STABLE_PX,
        min_visibility: float = config.KEYPOINT_MIN_VISIBILITY,
    ) -> None:
        self._alpha = alpha
        self._hysteresis = hysteresis
        self._gap_max = gap_max
        self._gap_stable_px = gap_stable_px
        self._min_vis = min_visibility

        self._age: dict = {}   # landmark -> frames since last seen inside a hold
        self._pos: dict = {}   # landmark -> last in-hold pixel position
        self._ema: float | None = None
        self._last_com: tuple[float, float] | None = None
        self._last_over = False
        self._last_offset: float | None = None
        self._gap = 0

    def _effective_support(self, pose, holds, width, height):
        inside = dict(_support_inside(pose, holds, width, height, self._min_vis))
        for landmark in (*HAND_LANDMARKS, *FOOT_LANDMARKS):
            if landmark in inside:
                self._age[landmark] = 0
                self._pos[landmark] = inside[landmark]
            else:
                self._age[landmark] = self._age.get(landmark, 999) + 1
        return [self._pos[lm] for lm in self._pos
                if self._age.get(lm, 999) <= self._hysteresis]

    def update(self, pose, holds, width, height) -> BalanceState:
        com = _com_px(pose, width, height, self._min_vis) if pose is not None else None
        support = self._effective_support(pose, holds, width, height) if pose else []

        if com is not None and len(support) >= config.BALANCE_MIN_SUPPORT_POINTS:
            shoulder = _shoulder_width_px(pose, width, self._min_vis)
            over, offset, raw = _score_from(com, support, shoulder)
            self._ema = raw if self._ema is None else (
                self._alpha * raw + (1 - self._alpha) * self._ema)
            self._last_com, self._last_over, self._last_offset = com, over, offset
            self._gap = 0
            return BalanceState(com, tuple(support), over, offset, self._ema)

        # No valid base this frame: carry the last score while the CoM is stable.
        if (self._ema is not None and self._gap < self._gap_max
                and com is not None and self._last_com is not None
                and _dist(com, self._last_com) <= self._gap_stable_px):
            self._gap += 1
            return BalanceState(com, (), self._last_over, self._last_offset, self._ema)

        return BalanceState(com, tuple(support), False, None, None)


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def compute_smoothness(
    trajectory,
    scale: float = config.SMOOTHNESS_SCALE,
) -> float | None:
    """Score how smooth a CoG path is, in [0, 1] (1 = glassy, 0 = jerky).

    Heuristic proxy: the mean acceleration magnitude along the path relative to
    the mean speed (roughness). Jerky, stop-start movement has high acceleration
    per unit motion; a flowing climb has low. None gaps split the path so a jump
    across a gap is not counted as acceleration. Returns None if there is too
    little movement to judge.
    """
    # Contiguous runs of tracked points.
    runs: list[list] = []
    run: list = []
    for p in trajectory:
        if p is None:
            if len(run) >= 3:
                runs.append(run)
            run = []
        else:
            run.append(p)
    if len(run) >= 3:
        runs.append(run)
    if not runs:
        return None

    speeds: list[float] = []
    accels: list[float] = []
    for r in runs:
        for i in range(len(r) - 2):
            speeds.append(_dist(r[i + 1], r[i]))
            # Acceleration = second difference; magnitude captures both speed and
            # direction changes, so a zig-zag path scores as jerky as a stutter.
            ax = r[i + 2][0] - 2 * r[i + 1][0] + r[i][0]
            ay = r[i + 2][1] - 2 * r[i + 1][1] + r[i][1]
            accels.append((ax * ax + ay * ay) ** 0.5)
    if not speeds or sum(speeds) == 0:
        return None

    mean_speed = sum(speeds) / len(speeds)
    mean_accel = sum(accels) / len(accels)
    if mean_speed < 1e-6:
        return None
    roughness = mean_accel / mean_speed
    return float(1.0 / (1.0 + roughness / scale))
