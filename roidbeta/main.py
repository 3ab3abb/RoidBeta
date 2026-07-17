"""Roidbeta entry point.

Build order progress:
  step 1 (done): live frames on screen via the Continuity Camera source.
  step 2 (done): MediaPipe pose keypoints drawn live on the climber.
  step 3 (done): human-in-the-loop route selection + frozen holds drawn live.
  step 4 (done): explicit state machine driving the flow with keyboard controls.
  step 5 (done): Layer A hold-contact scoring with automatic top-out completion.
  plus: the attempt is recorded and can be replayed and kept or discarded.

    IDLE -> ROUTE_SELECT -> READY -> ATTEMPT_ACTIVE -> COMPLETED -> REVIEW
                              ^                             |          |
                              +-------- RESET --------------+----------+

The annotated attempt is recorded to a temp file while ATTEMPT_ACTIVE. From
COMPLETED, P replays it in this same window; in REVIEW, S saves the clip and D
discards it. Pose and the scorer run only in ATTEMPT_ACTIVE.

Run live (Continuity Camera):
    python -m roidbeta.main
Run on a recorded video (for testing the pipeline):
    python -m roidbeta.main --video path/to/clip.mp4 [--loop] [--fps 30]
A video source starts paused on the first frame so you can select the route,
then plays through once the attempt starts. Keys are shown live in the HUD per
state. Q or Esc quits.
"""

from __future__ import annotations

import argparse
import time

import cv2

from . import config
from .capture import ContinuityCameraSource, FrameSource, VideoFileSource
from .display import (
    HudState,
    MetricCard,
    draw_balance,
    draw_bottom_bar,
    draw_hint,
    draw_holds,
    draw_pose,
    draw_results_card,
    draw_status_card,
    draw_top_banner,
    draw_trajectory,
)
from .display import theme
from .pose import PoseEstimator
from .recording import AttemptRecorder
from .reference import clear_trajectory, load_trajectory, save_trajectory
from .route import RouteSelector
from .scoring import Scorer, ScoreState
from .state import State, StateMachine

_QUIT_KEYS = (ord("q"), 27)
_REFERENCE_TRAIL_COLOR = (200, 200, 200)  # grey ghost trail for the reference path


def _wait_for_first_frame(source: FrameSource, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        seq, frame = source.latest()
        if seq != 0 and frame is not None:
            return frame
        time.sleep(0.02)
    return None


def _metric_cards(scorer: Scorer | None, score: ScoreState | None) -> list[MetricCard]:
    """One MetricCard per Layer B metric for the bottom bar."""
    if scorer is None:
        return []
    current = score.balance.balance_score if score else None
    return [MetricCard("Balance", scorer.history.balance_history, current, (0.0, 1.0))]


def _average_balance(scorer: Scorer | None) -> float | None:
    """Mean balance over the attempt, ignoring frames where it was unscorable."""
    if scorer is None:
        return None
    vals = [v for v in scorer.history.balance_history if v is not None]
    return sum(vals) / len(vals) if vals else None


def _hud_state(
    machine: StateMachine,
    score: ScoreState | None,
    scorer: Scorer | None,
    has_reference: bool,
) -> HudState:
    """Build the dashboard model for the current state."""
    state = machine.state
    contact = score.contact if score else None
    reached = (contact.highest_rank or 0) if contact else 0
    used = contact.used_count if contact else 0
    total = contact.total_holds if contact else 0

    if state is State.IDLE:
        return HudState("IDLE", theme.TEXT_MUTED, has_reference=has_reference,
                        controls=["S  select route", "Q  quit"])
    if state is State.READY:
        n = len(machine.route.holds) if machine.route else 0
        return HudState("READY", theme.PRIMARY, total=n, has_reference=has_reference,
                        controls=["SPACE  start", "N  new route", "Q  quit"])
    if state is State.ATTEMPT_ACTIVE:
        return HudState(
            "ATTEMPT", theme.WARNING,
            timer=f"{machine.attempt_elapsed_s or 0.0:.1f}s",
            reached=reached, used=used, total=total, show_progress=True,
            has_reference=has_reference, metrics=_metric_cards(scorer, score),
            controls=["C  complete", "R  reset", "Q  quit"],
        )
    if state is State.COMPLETED:
        return HudState(
            "COMPLETED", theme.SUCCESS,
            timer=f"{machine.attempt_elapsed_s or 0.0:.1f}s",
            reached=reached, used=used, total=total, show_progress=True,
            has_reference=has_reference, avg_balance=_average_balance(scorer),
            metrics=_metric_cards(scorer, score),
            controls=["P  replay", "F  set reference", "X  clear reference",
                      "R  retry", "N  new route", "Q  quit"],
        )
    return HudState(state.name, theme.TEXT_MUTED)


def _review_clip(recorder: AttemptRecorder | None) -> str:
    """Replay the recorded attempt on a loop until the user decides.

    Returns "save", "discard", or "quit". "discard" immediately if there is no
    clip to show.
    """
    if recorder is None or not recorder.has_frames:
        return "discard"
    recorder.finalize()  # ensure the temp file is closed before reading it

    cap = cv2.VideoCapture(str(recorder.temp_path))
    if not cap.isOpened():
        return "discard"
    delay = max(1, int(1000 / recorder.fps))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:  # reached the end: loop back to the start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            draw_top_banner(frame, "REVIEW",
                            "S save    D discard    R restart    Q quit")
            cv2.imshow(config.OVERLAY_WINDOW_NAME, frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("s"):
                return "save"
            if key == ord("d"):
                return "discard"
            if key == ord("r"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif key in _QUIT_KEYS:
                return "quit"
    finally:
        cap.release()


def run(source: FrameSource) -> None:
    with source, PoseEstimator() as pose_estimator:
        if _wait_for_first_frame(source) is None:
            print("No frames from the source. Is the camera connected / video valid?")
            return

        machine = StateMachine()
        selector = RouteSelector()
        pose = None
        scorer: Scorer | None = None
        score: ScoreState | None = None
        recorder: AttemptRecorder | None = None
        summary_written = False  # have the end-of-clip results frames been appended
        reference = load_trajectory(config.REFERENCE_PATH)  # ghost trail, if any
        last_seq = 0

        while True:
            # A finite (video) source plays only during an active attempt, and
            # holds its frame the rest of the time so the route can be selected.
            if machine.state is State.ATTEMPT_ACTIVE:
                source.resume()
            else:
                source.pause()

            # ROUTE_SELECT and REVIEW each own their own blocking sub-loop.
            if machine.state is State.ROUTE_SELECT:
                _, snapshot = source.latest()
                selection = selector.select(snapshot.copy())
                if selection is None:
                    machine.cancel_route_select()
                else:
                    machine.set_route(selection.route)
                continue

            if machine.state is State.REVIEW:
                decision = _review_clip(recorder)
                if decision == "quit":
                    break
                if decision == "save" and recorder is not None:
                    print(f"Saved clip: {recorder.keep()}")
                    recorder = None  # moved to its permanent path
                machine.end_review()
                continue

            seq, frame = source.latest()
            if seq == 0 or frame is None:
                if cv2.waitKey(10) & 0xFF in _QUIT_KEYS:
                    break
                continue

            was_active = machine.state is State.ATTEMPT_ACTIVE
            new_frame = False
            if was_active:
                if recorder is None:  # first frame of the attempt
                    recorder = AttemptRecorder()
                    summary_written = False
                if scorer is None:
                    scorer = Scorer(machine.route)
                if seq != last_seq:
                    last_seq = seq
                    new_frame = True
                    pose = pose_estimator.estimate(frame)
                    height, width = frame.shape[:2]
                    score = scorer.update(pose, width, height)
                    if scorer.completed:
                        machine.complete()

            # Draw the frozen route, colored by scoring state where available.
            if machine.route is not None:
                used = score.contact.used_indices if score else frozenset()
                touched = score.contact.touched_indices if score else frozenset()
                draw_holds(frame, machine.route.holds, used, touched)
            # CoM trails: the reference (ghost) under the current attempt's path.
            # Shown during the attempt and while COMPLETED (the scorer persists).
            if scorer is not None:
                if reference:
                    draw_trajectory(frame, reference, fade=False,
                                    color=_REFERENCE_TRAIL_COLOR)
                draw_trajectory(frame, scorer.history.com_trajectory)
            if was_active:
                draw_pose(frame, pose)
                if score is not None:
                    draw_balance(frame, score.balance)
            hud = _hud_state(machine, score, scorer, reference is not None)
            draw_status_card(frame, hud)
            if hud.metrics:  # bottom metrics bar appears once the attempt starts
                draw_bottom_bar(frame, hud)
            else:
                draw_hint(frame, hud.controls)  # slim key hint before that
            if machine.state is State.COMPLETED:
                draw_results_card(frame, hud)

            # Record the displayed attempt frame (what the user saw).
            if was_active and new_frame and recorder is not None:
                recorder.add(frame)
            # On completion, append a few seconds of the results card so the saved
            # clip (and the replay) ends on the score window, then close the file.
            if (machine.state is State.COMPLETED and recorder is not None
                    and not summary_written):
                # Hold the results frame for a few seconds at the measured fps, so
                # the summary lasts CLIP_SUMMARY_SECONDS regardless of frame rate.
                n_summary = int(config.CLIP_SUMMARY_SECONDS * recorder.fps)
                for _ in range(n_summary):
                    recorder.add(frame, realtime=False)
                recorder.finalize()
                summary_written = True

            cv2.imshow(config.OVERLAY_WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in _QUIT_KEYS:
                break
            if machine.state is State.COMPLETED and key == ord("f") and scorer is not None:
                # Remember this attempt's CoM path as the reference for later ones.
                reference = list(scorer.history.com_trajectory)
                path = save_trajectory(config.REFERENCE_PATH, reference)
                print(f"Saved reference trajectory ({len(reference)} frames): {path}")
            elif machine.state is State.COMPLETED and key == ord("x"):
                if clear_trajectory(config.REFERENCE_PATH):
                    print("Cleared reference trajectory.")
                reference = None
            elif _handle_key(key, machine):
                pose = None
                # Leaving the attempt/completed flow entirely: drop scoring and
                # discard the unsaved recording.
                if machine.state in (State.READY, State.IDLE, State.ROUTE_SELECT):
                    scorer = None
                    score = None
                    if recorder is not None:
                        recorder.discard()
                        recorder = None

        if recorder is not None:
            recorder.discard()  # quit without saving: clean up the temp file

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
        if key == ord("c"):  # debug force-complete; normally the scorer completes
            machine.complete()
            return True
        if key == ord("r"):
            machine.reset()
            return True
    if state is State.COMPLETED:
        if key == ord("p"):
            machine.begin_review()
            return True
        if key == ord("r"):
            machine.reset()
            return True
        if key == ord("n"):
            machine.begin_route_select()
            return True
    return False


def _build_source(args: argparse.Namespace) -> FrameSource:
    """Pick the frame source from CLI args: a video file, or the live camera."""
    if args.video:
        return VideoFileSource(args.video, loop=args.loop, fps=args.fps or None)
    return ContinuityCameraSource()


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="roidbeta", description=__doc__)
    parser.add_argument(
        "--video", metavar="PATH",
        help="run the pipeline on a recorded video file instead of the live camera",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="loop the video file (repeats from the start on reaching the end)",
    )
    parser.add_argument(
        "--fps", type=float, default=0.0,
        help="override video playback fps (0 = use the file's own fps)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run(_build_source(_parse_args(argv)))


if __name__ == "__main__":
    main()
