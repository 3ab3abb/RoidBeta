"""Per-attempt skill profile: the axes a climber is compared on.

A SkillProfile reduces one attempt to a few interpretable numbers (balance,
smoothness, speed, reach) so multiple climbers on the same route can be compared
on a skill-graph radar and a per-metric leaderboard. Raw values are kept; the
0..1 axis values used for the radar are derived from them so scales stay tunable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from .contact import ContactState
from .metrics import compute_smoothness
from .scorer import AttemptHistory


@dataclass(frozen=True)
class SkillProfile:
    """One climber's attempt, reduced to comparable metrics.

    balance and smoothness are 0..1; speed is holds-reached-per-second (raw);
    reach is the fraction of the route reached (0..1); time_s is wall time.
    balance/smoothness are None when there was too little signal to compute them.
    """

    label: str
    balance: float | None
    smoothness: float | None
    speed: float
    reach: float
    completed: bool
    time_s: float

    def axis_values(self) -> dict[str, float]:
        """Normalized 0..1 values per skill-graph axis (absolute, tunable scales)."""
        speed_axis = min(1.0, self.speed / config.SPEED_FULL_SCALE_HPS) if \
            config.SPEED_FULL_SCALE_HPS else 0.0
        return {
            "Balance": self.balance or 0.0,
            "Smoothness": self.smoothness or 0.0,
            "Speed": speed_axis,
            "Reach": self.reach,
        }


def build_profile(
    label: str,
    history: AttemptHistory,
    contact: ContactState,
    time_s: float,
) -> SkillProfile:
    """Reduce an attempt (its history + final contact state + time) to a profile."""
    balances = [v for v in history.balance_history if v is not None]
    balance = sum(balances) / len(balances) if balances else None
    smoothness = compute_smoothness(history.com_trajectory)

    total = contact.total_holds or 1
    reach = (contact.highest_rank or 0) / total
    # Speed as progress rate: holds reached per second (higher = faster).
    speed = (contact.highest_rank or 0) / time_s if time_s > 1e-3 else 0.0

    return SkillProfile(
        label=label,
        balance=balance,
        smoothness=smoothness,
        speed=speed,
        reach=reach,
        completed=contact.completed,
        time_s=time_s,
    )
