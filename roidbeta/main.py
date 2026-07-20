"""Roidbeta entry point.

Flow:
    SETUP -> ROUTE_SELECT -> READY -> ATTEMPT_ACTIVE -> COMPLETED -> REVIEW
                               ^                            |          |
                               +----- RESET (retry /--------+----------+
                                       next climber)        |
                                                            v (session, last done)
                                                        COMPARISON -> SETUP

SETUP chooses solo (classic single climber) or a session of several climbers who
take turns on one shared route and are compared on a skill-graph radar. A send is
a controlled top: both hands on the top hold for a 3s countdown (shown big).

Live camera uses wall-clock time and drops stale frames. A --video source is
frame-accurate: every frame is processed in order, and timing plus the exported
clip come from the video's own fps, so nothing runs fast or loses frames.

Run live:
    python -m roidbeta.main
Run on a recording:
    python -m roidbeta.main --video path/to/clip.mov [--loop] [--fps 30]
"""

from __future__ import annotations

import argparse

import cv2

from . import config
from .capture import ContinuityCameraSource, FrameSource, VideoFileSource
from .display import (
    HudState,
    MetricCard,
    draw_balance,
    draw_bottom_bar,
    draw_climber_tag,
    draw_comparison,
    draw_countdown,
    draw_hint,
    draw_holds,
    draw_pose,
    draw_results_card,
    draw_setup_screen,
    draw_status_card,
    draw_top_banner,
    draw_trajectory,
    theme,
)
from .pose import PoseEstimator
from .recording import AttemptRecorder
from .reference import clear_trajectory, load_trajectory, save_trajectory
from .route import RouteSelector
from .scoring import Scorer, ScoreState, build_profile
from .session import Session
from .state import State, StateMachine

_QUIT_KEYS = (ord("q"), 27)
_REFERENCE_TRAIL_COLOR = (200, 200, 200)  # grey ghost trail for the reference path
_UP_KEYS = (ord("+"), ord("="), 82)       # 82 = arrow-up in many highgui builds
_DOWN_KEYS = (ord("-"), ord("_"), 84)     # 84 = arrow-down


def _wait_for_first_frame(source: FrameSource, timeout_s: float = 10.0):
    import time
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        seq, frame = source.latest()
        if seq != 0 and frame is not None:
            return frame
        time.sleep(0.02)
    return None


def _metric_cards(scorer: Scorer | None, score: ScoreState | None) -> list[MetricCard]:
    if scorer is None:
        return []
    current = score.balance.balance_score if score else None
    return [MetricCard("Balance", scorer.history.balance_history, current, (0.0, 1.0))]


def _average_balance(scorer: Scorer | None) -> float | None:
    if scorer is None:
        return None
    vals = [v for v in scorer.history.balance_history if v is not None]
    return sum(vals) / len(vals) if vals else None


def _hud_state(machine, score, scorer, has_reference, session) -> HudState:
    state = machine.state
    contact = score.contact if score else None
    reached = (contact.highest_rank or 0) if contact else 0
    used = contact.used_count if contact else 0
    total = contact.total_holds if contact else 0
    is_session = session is not None and not session.is_solo

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
        next_ctrl = "N  next climber" if is_session else "N  new route"
        return HudState(
            "COMPLETED", theme.SUCCESS,
            timer=f"{machine.attempt_elapsed_s or 0.0:.1f}s",
            reached=reached, used=used, total=total, show_progress=True,
            has_reference=has_reference, avg_balance=_average_balance(scorer),
            metrics=_metric_cards(scorer, score),
            controls=["P  replay", "F  set ref", "X  clear ref",
                      "R  retry", next_ctrl, "Q  quit"],
        )
    return HudState(state.name, theme.TEXT_MUTED)


def _comparison_data(session: Session):
    """Build (entries, leaders) for the comparison screen from a session."""
    entries = [
        (c.label, c.color, p.axis_values())
        for c, p in zip(session.climbers, session.results)
    ]
    pairs = list(zip(session.climbers, session.results))
    leaders = []
    if pairs:
        def leader(name, value_fn, fmt):
            c, p = max(pairs, key=lambda cp: value_fn(cp[1]))
            return (name, c.label, fmt(value_fn(p)))

        leaders.append(leader("Balance", lambda p: p.balance or 0.0, lambda v: f"{v:.2f}"))
        leaders.append(leader("Smoothness", lambda p: p.smoothness or 0.0, lambda v: f"{v:.2f}"))
        leaders.append(leader("Speed", lambda p: p.speed, lambda v: f"{v:.2f}/s"))
        leaders.append(leader("Reach", lambda p: p.reach, lambda v: f"{int(v * 100)}%"))
        topped = [(c, p) for c, p in pairs if p.completed]
        if topped:
            c, p = min(topped, key=lambda cp: cp[1].time_s)
            leaders.append(("Time", c.label, f"{p.time_s:.1f}s"))
    return entries, leaders


def _review_clip(recorder: AttemptRecorder | None) -> str:
    if recorder is None or not recorder.has_frames:
        return "discard"
    recorder.finalize()
    cap = cv2.VideoCapture(str(recorder.temp_path))
    if not cap.isOpened():
        return "discard"
    delay = max(1, int(1000 / recorder.fps))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
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


def run(source: FrameSource, pose_variant: str = config.POSE_MODEL_VARIANT) -> None:
    with source, PoseEstimator(variant=pose_variant) as pose_estimator:
        if _wait_for_first_frame(source) is None:
            print("No frames from the source. Is the camera connected / video valid?")
            return

        machine = StateMachine(clock=source.position_seconds)
        selector = RouteSelector()
        session: Session | None = None
        setup = {"mode": "solo", "count": 2}
        pose = None
        scorer: Scorer | None = None
        score: ScoreState | None = None
        recorder: AttemptRecorder | None = None
        summary_written = False
        profile_saved = False
        reference = load_trajectory(config.REFERENCE_PATH)
        last_seq = 0

        def reset_attempt():
            nonlocal scorer, score, recorder, summary_written, profile_saved, pose
            if recorder is not None:
                recorder.discard()
            scorer = None
            score = None
            pose = None
            recorder = None
            summary_written = False
            profile_saved = False

        while True:
            # --- SETUP: mode select + climber count ---
            if machine.state is State.SETUP:
                _, base = source.latest()
                frame = base.copy()
                draw_setup_screen(frame, setup["mode"], setup["count"])
                cv2.imshow(config.OVERLAY_WINDOW_NAME, frame)
                key = cv2.waitKey(20) & 0xFF
                if key in _QUIT_KEYS:
                    break
                if key == ord("s"):
                    setup["mode"] = "solo"
                elif key == ord("m"):
                    setup["mode"] = "session"
                    setup["count"] = max(2, setup["count"])
                elif key in _UP_KEYS and setup["mode"] == "session":
                    setup["count"] = min(6, setup["count"] + 1)
                elif key in _DOWN_KEYS and setup["mode"] == "session":
                    setup["count"] = max(2, setup["count"] - 1)
                elif key in (13, ord(" ")):  # confirm
                    count = 1 if setup["mode"] == "solo" else setup["count"]
                    session = Session.create(count)
                    machine.begin_route_select()
                continue

            # --- ROUTE_SELECT: blocking sub-loop ---
            if machine.state is State.ROUTE_SELECT:
                _, snapshot = source.latest()
                selection = selector.select(snapshot.copy())
                if selection is None:
                    machine.cancel_route_select()
                else:
                    machine.set_route(selection.route)
                continue

            # --- REVIEW: blocking replay ---
            if machine.state is State.REVIEW:
                decision = _review_clip(recorder)
                if decision == "quit":
                    break
                if decision == "save" and recorder is not None:
                    print(f"Saved clip: {recorder.keep()}")
                    recorder = None
                machine.end_review()
                continue

            # --- COMPARISON: session results ---
            if machine.state is State.COMPARISON:
                _, base = source.latest()
                frame = base.copy()
                entries, leaders = _comparison_data(session)
                draw_comparison(frame, entries, leaders)
                cv2.imshow(config.OVERLAY_WINDOW_NAME, frame)
                key = cv2.waitKey(20) & 0xFF
                if key in _QUIT_KEYS:
                    break
                if key == ord("r"):
                    machine.new_session()
                    session = None
                continue

            # --- live states: step the source, then read the current frame ---
            if machine.state is State.ATTEMPT_ACTIVE:
                if source.is_realtime:
                    source.resume()
                else:
                    source.advance()
            elif source.is_realtime:
                source.pause()

            seq, base = source.latest()
            if seq == 0 or base is None:
                if cv2.waitKey(10) & 0xFF in _QUIT_KEYS:
                    break
                continue
            frame = base.copy()

            was_active = machine.state is State.ATTEMPT_ACTIVE
            new_frame = False
            if was_active:
                if recorder is None:
                    recorder = AttemptRecorder(
                        fps=source.nominal_fps or config.REPLAY_FPS,
                        measured=source.is_realtime,
                    )
                    summary_written = False
                if scorer is None:
                    scorer = Scorer(machine.route)
                now = machine.attempt_elapsed_s or 0.0
                if seq != last_seq:
                    last_seq = seq
                    new_frame = True
                    pose = pose_estimator.estimate(frame)
                    h, w = frame.shape[:2]
                    score = scorer.update(pose, w, h, now)
                    if scorer.completed:
                        machine.complete()
                # A finished video ends the attempt (may be a non-send).
                if (machine.state is State.ATTEMPT_ACTIVE
                        and not source.is_realtime and source.is_finished):
                    machine.complete()

            # --- draw ---
            if machine.route is not None:
                used = score.contact.used_indices if score else frozenset()
                touched = score.contact.touched_indices if score else frozenset()
                draw_holds(frame, machine.route.holds, used, touched)
            if scorer is not None:
                if reference:
                    draw_trajectory(frame, reference, fade=False,
                                    color=_REFERENCE_TRAIL_COLOR)
                draw_trajectory(frame, scorer.history.com_trajectory)
            if was_active:
                draw_pose(frame, pose)
                if score is not None:
                    draw_balance(frame, score.balance)

            hud = _hud_state(machine, score, scorer, reference is not None, session)
            draw_status_card(frame, hud)
            if session is not None and not session.is_solo and machine.state in (
                State.READY, State.ATTEMPT_ACTIVE, State.COMPLETED
            ):
                c = session.current
                draw_climber_tag(frame, f"{c.label}  ({session.position_label})", c.color)
            if hud.metrics:
                draw_bottom_bar(frame, hud)
            else:
                draw_hint(frame, hud.controls)
            if was_active and score is not None and score.contact.top_countdown is not None:
                draw_countdown(frame, score.contact.top_countdown,
                               config.TOP_HOLD_COMPLETION_SECONDS)
            if machine.state is State.COMPLETED:
                draw_results_card(frame, hud)

            # --- record ---
            if was_active and new_frame and recorder is not None:
                recorder.add(frame)
            if (machine.state is State.COMPLETED and recorder is not None
                    and not summary_written):
                for _ in range(int(config.CLIP_SUMMARY_SECONDS * recorder.fps)):
                    recorder.add(frame, realtime=False)
                recorder.finalize()
                summary_written = True
            # Store this climber's profile once, on entering COMPLETED.
            if (machine.state is State.COMPLETED and session is not None
                    and scorer is not None and score is not None and not profile_saved):
                session.record(build_profile(
                    session.current.label, scorer.history, score.contact,
                    machine.attempt_elapsed_s or 0.0,
                ))
                profile_saved = True

            # --- keys ---
            cv2.imshow(config.OVERLAY_WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in _QUIT_KEYS:
                break
            state = machine.state
            if state is State.READY:
                if key == ord(" "):
                    reset_attempt()
                    machine.start_attempt()
                elif key == ord("n"):
                    reset_attempt()
                    machine.begin_route_select()
            elif state is State.ATTEMPT_ACTIVE:
                if key == ord("c"):        # manual finish (debug / mantle tops)
                    machine.complete()
                elif key == ord("r"):
                    machine.reset()
                    reset_attempt()
            elif state is State.COMPLETED:
                if key == ord("p"):
                    machine.begin_review()
                elif key == ord("f") and scorer is not None:
                    reference = list(scorer.history.com_trajectory)
                    print(f"Saved reference: "
                          f"{save_trajectory(config.REFERENCE_PATH, reference)}")
                elif key == ord("x"):
                    if clear_trajectory(config.REFERENCE_PATH):
                        print("Cleared reference trajectory.")
                    reference = None
                elif key == ord("r"):      # retry the same climber
                    machine.reset()
                    reset_attempt()
                elif key == ord("n"):
                    if session is not None and not session.is_solo:
                        if session.is_last:
                            machine.begin_comparison()
                        else:
                            session.advance()
                            machine.reset()
                        reset_attempt()
                    else:
                        machine.begin_route_select()
                        reset_attempt()

        if recorder is not None:
            recorder.discard()

    cv2.destroyAllWindows()


def _build_source(args: argparse.Namespace) -> FrameSource:
    if args.video:
        return VideoFileSource(args.video, loop=args.loop, fps=args.fps or None)
    return ContinuityCameraSource(camera_index=args.camera)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="roidbeta", description=__doc__)
    parser.add_argument("--video", metavar="PATH",
                        help="run on a recorded video file instead of the live camera")
    parser.add_argument("--loop", action="store_true", help="loop the video file")
    parser.add_argument("--fps", type=float, default=0.0,
                        help="override video fps (0 = the file's own fps)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        metavar="N",
                        help="camera index to open (try 1 if it grabs the wrong one)")
    parser.add_argument("--pose", default=config.POSE_MODEL_VARIANT,
                        choices=("lite", "full", "heavy"),
                        help="pose model: lite is fastest, heavy most accurate")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run(_build_source(args), pose_variant=args.pose)


if __name__ == "__main__":
    main()
