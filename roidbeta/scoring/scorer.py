"""Attempt scorer: combines the scoring layers and holds attempt state.

Wraps Layer A (hold-contact progress) and Layer B (movement-quality metrics). It
is created when an attempt starts and updated once per processed pose frame while
ATTEMPT_ACTIVE, never in any other state.

Besides the per-frame ScoreState, the scorer keeps time series (the CoM path and
the balance-score history) so the live charts can plot them and so a later
reference comparison (Layer C, DTW) has a trajectory to align.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..pose.keypoints import PoseFrame
from ..route.holds import RouteMask
from .contact import ContactState, ContactTracker
from .metrics import BalanceState, BalanceTracker


@dataclass(frozen=True)
class ScoreState:
    """Everything the overlay needs to render the live score for one frame."""

    contact: ContactState
    balance: BalanceState

    @property
    def completed(self) -> bool:
        return self.contact.completed


@dataclass
class AttemptHistory:
    """Per-frame time series accumulated across an attempt.

    com_trajectory holds the pixel CoM per processed frame (None when it could
    not be estimated), which is the raw material for the Layer C reference
    comparison. balance_history holds the balance score per frame for the live
    chart. Both are appended in lockstep, one entry per processed frame.
    """

    com_trajectory: list[tuple[float, float] | None] = field(default_factory=list)
    balance_history: list[float | None] = field(default_factory=list)


class Scorer:
    """Drives Layer A and Layer B for a single attempt and records its history."""

    def __init__(self, route: RouteMask) -> None:
        self._route = route
        self._contact = ContactTracker(route)
        self._balance = BalanceTracker()
        self.history = AttemptHistory()

    def update(
        self, pose: PoseFrame | None, width: int, height: int, now: float = 0.0
    ) -> ScoreState:
        contact_state = self._contact.update(pose, width, height, now)
        balance_state = self._balance.update(pose, self._route.holds, width, height)

        self.history.com_trajectory.append(balance_state.com)
        self.history.balance_history.append(balance_state.balance_score)

        return ScoreState(contact=contact_state, balance=balance_state)

    @property
    def completed(self) -> bool:
        return self._contact.completed
