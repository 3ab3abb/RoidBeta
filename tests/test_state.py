"""Tests for the attempt state machine.

Pure control-flow tests: legal transitions advance, illegal ones raise, the
route freezes on selection, and attempt timing behaves. No camera or pose. Run:
    python -m pytest tests/test_state.py
"""

import numpy as np
import pytest

from roidbeta.route.holds import Hold, RouteMask
from roidbeta.state import InvalidTransition, State, StateMachine


def _fake_route(n=3):
    holds = tuple(Hold(x=10, y=10 * (i + 1), radius=8) for i in range(n))
    mask = np.zeros((100, 100), dtype=np.uint8)
    return RouteMask(mask=mask, holds=holds, frame_shape=(100, 100))


def test_happy_path_reaches_completed():
    m = StateMachine()
    assert m.state is State.SETUP

    m.begin_route_select()
    assert m.state is State.ROUTE_SELECT

    route = _fake_route()
    m.set_route(route)
    assert m.state is State.READY
    assert m.route is route  # frozen route stored

    m.start_attempt()
    assert m.state is State.ATTEMPT_ACTIVE

    m.complete()
    assert m.state is State.COMPLETED


def test_illegal_transition_raises():
    m = StateMachine()
    # Cannot start an attempt straight from SETUP.
    with pytest.raises(InvalidTransition):
        m.start_attempt()
    # Cannot complete before an attempt is active.
    m.begin_route_select()
    m.set_route(_fake_route())
    with pytest.raises(InvalidTransition):
        m.complete()


def test_reset_returns_to_ready_and_keeps_route():
    m = StateMachine()
    m.begin_route_select()
    route = _fake_route()
    m.set_route(route)
    m.start_attempt()
    m.reset()
    assert m.state is State.READY
    assert m.route is route  # same route retained
    assert m.attempt_elapsed_s is None  # timing cleared


def test_new_route_from_completed_reselects():
    m = StateMachine()
    m.begin_route_select()
    m.set_route(_fake_route())
    m.start_attempt()
    m.complete()
    m.begin_route_select()
    assert m.state is State.ROUTE_SELECT


def test_cancel_route_select_returns_to_setup():
    m = StateMachine()
    m.begin_route_select()
    m.cancel_route_select()
    assert m.state is State.SETUP
    assert m.route is None


def test_session_flow_completed_to_comparison_to_setup():
    m = StateMachine()
    m.begin_route_select()
    m.set_route(_fake_route())
    m.start_attempt()
    m.complete()
    m.begin_comparison()
    assert m.state is State.COMPARISON
    m.new_session()
    assert m.state is State.SETUP
    assert m.route is None  # a new session drops the old route


def test_next_climber_resets_to_ready_from_completed():
    m = StateMachine()
    m.begin_route_select()
    route = _fake_route()
    m.set_route(route)
    m.start_attempt()
    m.complete()
    m.reset()  # advance to the next climber, same route
    assert m.state is State.READY
    assert m.route is route


def test_attempt_elapsed_uses_clock_and_freezes_on_complete():
    now = {"t": 100.0}
    m = StateMachine(clock=lambda: now["t"])

    m.begin_route_select()
    m.set_route(_fake_route())
    assert m.attempt_elapsed_s is None  # no attempt yet

    m.start_attempt()          # start at t=100
    now["t"] = 102.5
    assert m.attempt_elapsed_s == pytest.approx(2.5)

    m.complete()               # freeze at t=102.5
    now["t"] = 999.0
    assert m.attempt_elapsed_s == pytest.approx(2.5)  # stays frozen
