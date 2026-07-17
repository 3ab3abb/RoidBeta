# Roidβ

Real-time computer-vision bouldering scorer. Point a camera at a climber, select
a route by color, and Roidβ tracks the climber's pose against the route and scores
the attempt live: hold-contact progress first, then movement quality on top.

Part of the **Roid** family (RoidOME, RoidOTA, Roidβ). "β" is the climbing term for
the *beta* of a route (its movement/sequence information).

See [Roidβ.md](Roid%CE%B2.md) for the full design rationale and constraints.

---

## What it does

1. Reads a live feed (iPhone via Continuity Camera) or a recorded video file.
2. You select a route: each route is one color, so you click a representative
   hold, its HSV neighborhood is sampled from the actual frame, and a hold mask
   is built and frozen. You also designate the START and TOP holds.
3. You start an attempt. The system tracks pose (MediaPipe, 33 keypoints) and
   scores movement in real time.
4. A live overlay shows the skeleton, the route holds (colored by contact state),
   a center-of-gravity trail, a balance chart, and a running score/time.
5. On top-out it auto-completes, freezes the clock, and shows a session summary.
   The annotated attempt can be replayed and saved as a clip.

---

## Requirements

- macOS on Apple Silicon (for Continuity Camera; video-file mode works anywhere).
- Python 3.10–3.12 (mediapipe supports this range). Development used pyenv 3.10.13.
- Dependencies in [requirements.txt](requirements.txt): opencv-python, mediapipe,
  numpy.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The MediaPipe pose model (`pose_landmarker_*.task`) is downloaded automatically on
first run into `models/` (gitignored).

## Running

Live camera:

```bash
.venv/bin/python -m roidbeta.main
```

On a recorded video (for testing the pipeline without the wall):

```bash
.venv/bin/python -m roidbeta.main --video path/to/clip.mov [--loop] [--fps 30]
```

A video source starts paused on the first frame so you can select the route,
then plays through once the attempt starts. `.mov` and `.mp4` both work; if an
iPhone HEVC `.mov` fails to open, transcode once with
`ffmpeg -i in.mov -c:v libx264 -pix_fmt yuv420p -an out.mp4`.

### Optional: a `BETA` shell function (fish)

```fish
function BETA
    clear
    cd /path/to/Roidβ
    if test (count $argv) -gt 0; and not test -f $argv[1]
        echo "BETA: file not found: $argv[1]"
    else if test (count $argv) -gt 0
        .venv/bin/python -m roidbeta.main --video (path resolve $argv[1]) $argv[2..-1]
    else
        .venv/bin/python -m roidbeta.main
    end
    cd -
end
```

`BETA` launches live; `BETA clip.mov` runs on a file.

---

## Controls

**Route selection** (on a frozen frame):

| Input | Action |
| --- | --- |
| Left-click | Sample a hold's color into the route mask |
| `M` | Toggle manual placement (left-click drops a hold the color mask misses) |
| Hover a hold + `T` | Set it as the TOP (finish) hold |
| Hover a hold + `S` | Toggle it as a START hold |
| Right-click | Remove the hold under the cursor |
| `[` / `]` | Shrink / grow the manual hold radius |
| `R` | Reset | `Enter`/`C` confirm (a TOP is required) | `Q`/`Esc` cancel |

**Session flow:**

| State | Keys |
| --- | --- |
| IDLE | `S` select route |
| READY | `SPACE` start · `N` new route |
| ATTEMPT | `C` complete (manual fallback) · `R` reset |
| COMPLETED | `P` replay · `F` set reference · `X` clear reference · `R` retry · `N` new route |
| REVIEW | `S` save · `D` discard · `R` restart |

`Q`/`Esc` quits from the live states.

---

## Scoring

Scoring is the actual project. The score is a defined construct, not a measured
truth, built in two layers.

**Layer A — hold-contact progress (objective).** A hold is *used* once a hand or
foot stays within its radius for N frames. Tracks the highest hold reached, the
count of holds used, and time. A send (completion) is two hands matching the
designated TOP hold (each hand counts independently, so a one-hand-then-the-other
top registers); completion freezes the clock and shows the summary.

**Layer B — movement quality (heuristic proxies).** Currently one metric:
**center-of-mass over the base of support** — how well the CoM sits over the
polygon of current contact points (balance). It is stabilized with EMA smoothing,
sticky support (a contact stays in the base briefly after leaving a hold), and a
CoM-stability–gated gap hold, so it reads steadily instead of flickering. This is
a horizontal-plane proxy from a single camera, not ground truth.

A CoM **trajectory trail** is drawn live; save an attempt's trail as a **reference**
(`F`) and it is overlaid as a ghost on later attempts for visual comparison
(same-camera, same-route). A DTW similarity score against the reference is future
work.

---

## Architecture

```
FrameSource (Continuity Camera | video file)
    -> single-slot latest-frame buffer (capture thread; drops stale frames)
        -> RouteSelector (one-time, human-in-the-loop HSV sampling + frozen holds)
        -> PoseEstimator (MediaPipe Tasks PoseLandmarker, per frame)
            -> Scorer (Layer A contact + Layer B balance, holds attempt state)
                -> Display overlay + metrics dashboard (live)
```

An explicit state machine (`state.py`) drives the flow:
`IDLE → ROUTE_SELECT → READY → ATTEMPT_ACTIVE → COMPLETED → REVIEW`, with RESET
back to READY. Pose and the scorer run only in `ATTEMPT_ACTIVE`.

All tunables (HSV tolerances, contact frame counts, metric weights, display
sizes) live in [config.py](roidbeta/config.py).

## Layout

```
roidbeta/
  main.py            entry point, state-machine wiring, CLI (--video)
  state.py           FSM
  config.py          all tunable parameters
  capture/           FrameSource interface, Continuity Camera, video file, buffer
  route/             HSV selection, hold extraction (watershed split), roles
  pose/              MediaPipe wrapper, keypoint structure, center-of-mass
  scoring/           Layer A contact, Layer B metrics, attempt scorer
  display/           theme + overlay + charts + dashboard (skeleton, holds, HUD)
  recording.py       attempt clip recording (measured-fps) and storage
  reference.py       save/load/clear the reference CoM trajectory
tests/               pytest suite (no camera needed)
```

## Tests

```bash
.venv/bin/python -m pytest tests/
```

The suite runs without a camera (synthetic frames and poses).
