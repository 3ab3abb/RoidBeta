"""Explicit state machine for an attempt.

    IDLE -> ROUTE_SELECT -> READY -> ATTEMPT_ACTIVE -> COMPLETED
                              ^                             |
                              +-------------- RESET --------+

Modeling the modes explicitly prevents mode-confusion bugs: the scorer only runs
in ATTEMPT_ACTIVE, the route mask is frozen on leaving ROUTE_SELECT, and illegal
transitions raise instead of silently corrupting state.

This module is deliberately free of OpenCV and MediaPipe so the control flow can
be unit tested on its own. It stores the frozen route and attempt timing; the
caller drives transitions in response to keyboard input and scorer events.
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Callable

from .route.holds import RouteMask


class State(Enum):
    SETUP = auto()          # choose solo vs session and the climber count
    ROUTE_SELECT = auto()
    READY = auto()
    ATTEMPT_ACTIVE = auto()
    COMPLETED = auto()
    REVIEW = auto()         # replaying the recorded attempt to keep or discard it
    COMPARISON = auto()     # session results: skill graph + leaderboards


# Allowed transitions. Anything not listed here is rejected.
_ALLOWED: dict[State, frozenset[State]] = {
    State.SETUP: frozenset({State.ROUTE_SELECT}),
    State.ROUTE_SELECT: frozenset({State.READY, State.SETUP}),  # SETUP = cancel
    State.READY: frozenset({State.ATTEMPT_ACTIVE, State.ROUTE_SELECT, State.SETUP}),
    State.ATTEMPT_ACTIVE: frozenset({State.COMPLETED, State.READY}),  # READY = abort
    State.COMPLETED: frozenset(
        {State.READY, State.ROUTE_SELECT, State.REVIEW, State.COMPARISON}
    ),
    State.REVIEW: frozenset({State.COMPLETED}),
    State.COMPARISON: frozenset({State.SETUP}),  # SETUP = new session
}


class InvalidTransition(RuntimeError):
    """Raised when a transition is not allowed from the current state."""


class StateMachine:
    """Tracks the attempt mode, the frozen route, and attempt timing.

    clock is injectable so timing is testable without wall-clock sleeps.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._state = State.SETUP
        self._route: RouteMask | None = None
        self._attempt_start: float | None = None
        self._attempt_end: float | None = None

    @property
    def state(self) -> State:
        return self._state

    @property
    def route(self) -> RouteMask | None:
        return self._route

    def can(self, target: State) -> bool:
        return target in _ALLOWED[self._state]

    def _transition(self, target: State) -> None:
        if not self.can(target):
            raise InvalidTransition(f"{self._state.name} -> {target.name}")
        self._state = target

    # --- transitions ---

    def begin_route_select(self) -> None:
        """Enter ROUTE_SELECT from SETUP, READY, or COMPLETED to pick a route."""
        self._transition(State.ROUTE_SELECT)

    def cancel_route_select(self) -> None:
        """Back out of ROUTE_SELECT to SETUP without choosing a route."""
        self._transition(State.SETUP)

    def set_route(self, route: RouteMask) -> None:
        """Freeze the selected route and advance ROUTE_SELECT -> READY."""
        self._transition(State.READY)
        self._route = route  # frozen on exit from ROUTE_SELECT; never re-segmented

    def start_attempt(self) -> None:
        """Begin scoring: READY -> ATTEMPT_ACTIVE. Records the start time."""
        self._transition(State.ATTEMPT_ACTIVE)
        self._attempt_start = self._clock()
        self._attempt_end = None

    def complete(self) -> None:
        """Top reached: ATTEMPT_ACTIVE -> COMPLETED. Freezes the elapsed time."""
        self._transition(State.COMPLETED)
        self._attempt_end = self._clock()

    def begin_review(self) -> None:
        """Enter REVIEW from COMPLETED to replay the recorded attempt."""
        self._transition(State.REVIEW)

    def end_review(self) -> None:
        """Return to COMPLETED after the clip has been kept or discarded."""
        self._transition(State.COMPLETED)

    def reset(self) -> None:
        """Return to READY keeping the same route (abort, retry, or next climber)."""
        self._transition(State.READY)
        self._attempt_start = None
        self._attempt_end = None

    def begin_comparison(self) -> None:
        """Session finished: COMPLETED -> COMPARISON to show the results."""
        self._transition(State.COMPARISON)

    def new_session(self) -> None:
        """Leave COMPARISON back to SETUP to start a fresh session."""
        self._transition(State.SETUP)
        self._route = None
        self._attempt_start = None
        self._attempt_end = None

    @property
    def attempt_elapsed_s(self) -> float | None:
        """Seconds since the attempt started; frozen once COMPLETED.

        None outside of an attempt (before start / after reset).
        """
        if self._attempt_start is None:
            return None
        end = self._attempt_end if self._attempt_end is not None else self._clock()
        return end - self._attempt_start
