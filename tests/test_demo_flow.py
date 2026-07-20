"""End-to-end flow test: the exact sequence main() drives, without a GUI.

Guards the demo path: setup -> route -> climber 1 attempt -> complete -> next
climber -> complete -> comparison. Catches runtime bugs in the session wiring
that unit tests of individual pieces would miss.
"""

import numpy as np

from roidbeta.main import _comparison_data, _hud_state
from roidbeta.pose.keypoints import Keypoint, Landmark, PoseFrame
from roidbeta.route.holds import Hold, HoldRole, RouteMask
from roidbeta.scoring import Scorer, build_profile
from roidbeta.session import Session
from roidbeta.state import State, StateMachine

W, H = 200, 400
TOP = Hold(100, 40, 20, role=HoldRole.TOP)
MID = Hold(100, 200, 20)
START = Hold(100, 360, 20, role=HoldRole.START)
ROUTE = RouteMask(mask=np.zeros((H, W), np.uint8),
                  holds=(TOP, MID, START), frame_shape=(H, W))


def _pose_at(points):
    kps = [Keypoint(0.0, 0.0, 0.0) for _ in range(33)]
    for lm, (px, py) in points.items():
        kps[int(lm)] = Keypoint(px / W, py / H, 1.0)
    return PoseFrame(keypoints=tuple(kps))


def _both_hands_on(hold):
    return _pose_at({
        Landmark.LEFT_WRIST: (hold.x - 5, hold.y),
        Landmark.RIGHT_WRIST: (hold.x + 5, hold.y),
        Landmark.LEFT_SHOULDER: (hold.x - 20, hold.y + 60),
        Landmark.RIGHT_SHOULDER: (hold.x + 20, hold.y + 60),
        Landmark.LEFT_HIP: (hold.x - 15, hold.y + 140),
        Landmark.RIGHT_HIP: (hold.x + 15, hold.y + 140),
        Landmark.LEFT_FOOT_INDEX: (hold.x - 15, hold.y + 240),
        Landmark.RIGHT_FOOT_INDEX: (hold.x + 15, hold.y + 240),
    })


def _climb_one_attempt(machine, clock, seconds_per_frame=0.1):
    """Drive one climber from READY through a real top-out, as main does."""
    machine.start_attempt()
    scorer = Scorer(ROUTE)
    score = None
    # Move up the route, then control the top until the countdown completes.
    sequence = [START] * 10 + [MID] * 10 + [TOP] * 60
    for hold in sequence:
        clock["t"] += seconds_per_frame
        now = machine.attempt_elapsed_s or 0.0
        score = scorer.update(_both_hands_on(hold), W, H, now)
        if scorer.completed:
            machine.complete()
            break
    return scorer, score


def test_full_two_climber_session_reaches_comparison():
    clock = {"t": 0.0}
    machine = StateMachine(clock=lambda: clock["t"])
    session = Session.create(2)

    # SETUP -> ROUTE_SELECT -> READY
    assert machine.state is State.SETUP
    machine.begin_route_select()
    machine.set_route(ROUTE)
    assert machine.state is State.READY

    for expected_index in (0, 1):
        assert session.index == expected_index
        scorer, score = _climb_one_attempt(machine, clock)
        assert machine.state is State.COMPLETED, "controlled top should complete"
        assert score.contact.completed

        # main records the profile on entering COMPLETED
        profile = build_profile(session.current.label, scorer.history,
                                score.contact, machine.attempt_elapsed_s or 0.0)
        session.record(profile)

        # HUD must render for this state without blowing up
        hud = _hud_state(machine, score, scorer, False, session)
        assert hud.title == "COMPLETED"

        if session.is_last:
            machine.begin_comparison()
        else:
            session.advance()
            machine.reset()

    assert machine.state is State.COMPARISON
    assert len(session.results) == 2

    entries, leaders = _comparison_data(session)
    assert len(entries) == 2
    assert all(set(v) == {"Balance", "Smoothness", "Speed", "Reach"}
               for _, _, v in entries)
    assert leaders, "there should be category leaders"
    # Every leader names a real climber.
    labels = {c.label for c in session.climbers}
    assert all(w in labels for _, w, _ in leaders)

    machine.new_session()
    assert machine.state is State.SETUP


def test_solo_flow_never_needs_comparison():
    clock = {"t": 0.0}
    machine = StateMachine(clock=lambda: clock["t"])
    session = Session.create(1)
    assert session.is_solo

    machine.begin_route_select()
    machine.set_route(ROUTE)
    scorer, score = _climb_one_attempt(machine, clock)
    assert machine.state is State.COMPLETED
    # Solo goes back out to a new route rather than a comparison.
    machine.begin_route_select()
    assert machine.state is State.ROUTE_SELECT


def test_retry_replaces_the_profile_not_appends():
    clock = {"t": 0.0}
    machine = StateMachine(clock=lambda: clock["t"])
    session = Session.create(2)
    machine.begin_route_select()
    machine.set_route(ROUTE)

    scorer, score = _climb_one_attempt(machine, clock)
    session.record(build_profile("Climber 1", scorer.history, score.contact, 10.0))
    machine.reset()  # retry
    scorer, score = _climb_one_attempt(machine, clock)
    session.record(build_profile("Climber 1", scorer.history, score.contact, 5.0))
    assert len(session.results) == 1 and session.results[0].time_s == 5.0
