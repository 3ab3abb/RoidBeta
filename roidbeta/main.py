"""Roidbeta entry point.

Build order progress:
  step 1 (done): live frames on screen via the Continuity Camera source.
  step 2 (done): MediaPipe pose keypoints drawn live on the climber.
  step 3 (done): human-in-the-loop route selection + frozen holds drawn live.
  step 4 (this): explicit state machine driving the flow with keyboard controls.

    IDLE -> ROUTE_SELECT -> READY -> ATTEMPT_ACTIVE -> COMPLETED
                              ^                             |
                              +-------------- RESET --------+

Pose runs only in ATTEMPT_ACTIVE (the scorer, step 5, will live there too).
Completion is triggered manually with C for now; step 5 replaces that with real
top-hold detection.

Run:
    python -m roidbeta.main
Keys are shown live in the on-screen HUD per state. Q or Esc quits.
"""

from __future__ import annotations

import time

import cv2

from . import config
from .capture import ContinuityCameraSource
from .display import draw_holds, draw_hud, draw_pose
from .pose import PoseEstimator
from .route import RouteSelector
from .state import State, StateMachine

_QUIT_KEYS = (ord("q"), 27)


def _wait_for_first_frame(source: ContinuityCameraSource, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        seq, frame = source.latest()
        if seq != 0 and frame is not None:
            return frame
        time.sleep(0.02)
    return None


def _hud_lines(machine: StateMachine) -> list[str]:
    state = machine.state
    if state is State.IDLE:
        return ["IDLE", "S: select route   Q: quit"]
    if state is State.READY:
        n = len(machine.route.holds) if machine.route else 0
        return [f"READY   route: {n} holds", "SPACE: start   N: new route   Q: quit"]
    if state is State.ATTEMPT_ACTIVE:
        elapsed = machine.attempt_elapsed_s or 0.0
        return [f"ATTEMPT   {elapsed:4.1f}s", "C: complete   R: reset   Q: quit"]
    if state is State.COMPLETED:
        elapsed = machine.attempt_elapsed_s or 0.0
        return [f"COMPLETED in {elapsed:.1f}s", "R: retry   N: new route   Q: quit"]
    return []


def run() -> None:
    with ContinuityCameraSource() as source, PoseEstimator() as pose_estimator:
        if _wait_for_first_frame(source) is None:
            print("No frames from the camera. Is the iPhone connected?")
            return

        machine = StateMachine()
        selector = RouteSelector()
        pose = None
        last_seq = 0

        while True:
            # ROUTE_SELECT owns its own blocking interactive window.
            if machine.state is State.ROUTE_SELECT:
                _, snapshot = source.latest()
                selection = selector.select(snapshot.copy())
                if selection is None:
                    machine.cancel_route_select()
                else:
                    machine.set_route(selection.route)
                continue

            seq, frame = source.latest()
            if seq == 0 or frame is None:
                if cv2.waitKey(10) & 0xFF in _QUIT_KEYS:
                    break
                continue

            if machine.state is State.ATTEMPT_ACTIVE and seq != last_seq:
                last_seq = seq
                pose = pose_estimator.estimate(frame)

            if machine.route is not None:
                draw_holds(frame, machine.route.holds, machine.route.top_hold)
            if machine.state is State.ATTEMPT_ACTIVE:
                draw_pose(frame, pose)
            draw_hud(frame, _hud_lines(machine))

            cv2.imshow(config.OVERLAY_WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in _QUIT_KEYS:
                break
            if _handle_key(key, machine):
                pose = None  # drop stale skeleton across state changes

    cv2.destroyAllWindows()


def _handle_key(key: int, machine: StateMachine) -> bool:
    """Apply a keypress to the machine. Returns True if the state changed."""
    state = machine.state
    if state is State.IDLE and key == ord("s"):
        machine.begin_route_select()
        return True
    if state is State.READY:
        if key == ord(" "):
            machine.start_attempt()
            return True
        if key == ord("n"):
            machine.begin_route_select()
            return True
    if state is State.ATTEMPT_ACTIVE:
        if key == ord("c"):  # temporary manual completion; step 5 automates this
            machine.complete()
            return True
        if key == ord("r"):
            machine.reset()
            return True
    if state is State.COMPLETED:
        if key == ord("r"):
            machine.reset()
            return True
        if key == ord("n"):
            machine.begin_route_select()
            return True
    return False


if __name__ == "__main__":
    run()
