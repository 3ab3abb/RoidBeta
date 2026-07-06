# Roidβ

Real-time computer vision bouldering scorer. Captures a live climbing attempt, tracks the
climber against a selected route, and scores movement in real time.

Part of the **Roid** family of projects (RoidOME, RoidOTA, Roidβ). "β" is the climbing term
for the beta of a route (the sequence and movement information for sending it).

---

## What the system does

1. Live feed comes from an iPhone into a Mac (M1) via Continuity Camera.
2. User selects a route. Each route is a single color, so selection is color-based:
   the user clicks one representative hold, we sample its HSV neighborhood from the actual
   frame, and build a hold mask from that sampled range.
3. User starts an attempt.
4. The system tracks the climber's pose and scores movement in real time: hold-contact
   progress first, then movement-quality metrics layered on top.
5. A live overlay shows current hold reached, score, and metrics.

---

## Hard constraints (do not violate)

- **Single machine.** Everything runs on the M1. No Windows box, no cross-machine
  streaming, no remote training service. The v1 is inference-only. There is nothing to train.
- **No em dashes** anywhere in code comments, docstrings, commit messages, or generated docs.
- **Python only** for v1. Apple Vision (Swift) is a documented future optimization, not a v1 dependency.
- **Off-the-shelf pose.** Do not train or fine-tune a pose model. Consume MediaPipe Pose.
- **Frozen route mask.** After route selection, hold centroids are cached as
  `(x, y, radius)` targets and are NOT re-segmented every frame. The wall does not move.

---

## Stack

- **Language:** Python 3.11+
- **Capture:** Continuity Camera, read as a native `AVCaptureDevice` via OpenCV
  (`cv2.VideoCapture`). Wrap it behind a `FrameSource` interface so an RTSP source can be
  swapped in later without touching downstream code.
- **Route segmentation:** OpenCV, HSV color space, human-in-the-loop color sampling.
- **Pose estimation:** MediaPipe Pose (33 keypoints, includes hands and feet, needed for
  grip and foot-placement detection).
- **Scoring:** pure Python heuristics over keypoint time-series.
- **Display:** OpenCV window with drawn overlay for v1. Keep the display layer thin and
  swappable.

---

## Architecture

```
FrameSource (Continuity Camera)
        |  frames
        v
   Ingest / single-slot buffer  ---------------+
        |                                       |
        v                                       | shared latest frame
  RouteSelector (one-time, human-in-loop)       |
   -> hold centroids + binary mask              |
        |                                       v
        |                                  PoseEstimator (per frame)
        |                                   -> 33 keypoints
        |                                       |
        v                                       |
        +----------------> Scorer <-------------+
                              |  state: holds used, current hold,
                              |  metrics, score, completion
                              v
                          Display / overlay (live)
```

### Threading model (state this up front, do not build a synchronous loop)

Capture, pose, and display must not run in lockstep. Pose is the bottleneck
(~20-30 fps on M1 with MediaPipe).

- **Capture thread:** pushes frames to a single-slot buffer. Always overwrite. Drop stale
  frames, never queue-lag.
- **Pose/scoring thread:** consumes the latest available frame, never a backlog.
- **Display thread (or main loop):** renders the latest scored state.

The overlay must feel live even when pose stutters. A synchronous
capture->pose->score->draw loop will feel laggy and is the wrong design.

### State machine (model explicitly)

```
IDLE -> ROUTE_SELECT -> READY -> ATTEMPT_ACTIVE -> COMPLETED
                          ^                             |
                          +-------------- RESET --------+
```

- Route mask is built during `ROUTE_SELECT` and frozen on exit.
- The scorer only runs in `ATTEMPT_ACTIVE`.
- `COMPLETED` fires when a hand keypoint reaches the top hold and holds for N frames.
- `RESET` returns to `READY` keeping the same route, or to `ROUTE_SELECT` to pick a new one.

An explicit FSM prevents scoring before the attempt starts and similar mode-confusion bugs.

---

## Scoring design

Scoring is the actual project. There is no objective ground truth for "good climbing
movement," so the score is a defined construct, not a measured truth. Build it in two layers.

### Layer A: hold-contact progress (the substrate, build first)

Well-defined and objective. Uses route centroids from selection and hand/foot keypoints from pose.

- A hold is **used** when a hand or foot keypoint stays within its radius for N frames.
- Track: highest hold reached, count of holds matched, time-to-top, completion flag.
- This is the foundation. Every Layer B metric is computed relative to which hold the
  climber is on and what phase of the move they are in, so A must exist before B is meaningful.

### Layer B: movement quality (layer on top, deliver in same v1)

Heuristic proxies over keypoint time-series. Each is a proxy, not truth. Implement them one
at a time on top of a working Layer A, and keep each behind a clearly named function so they
can be toggled and tuned independently.

Candidate metrics:
- **Center-of-mass sway:** penalize excessive lateral COM movement between holds.
- **Hip-to-wall distance:** reward keeping hips close to the wall (good technique). Estimated
  from pose depth cues and hip-keypoint position; document that this is approximate from a
  single camera.
- **Regrip count:** count readjustments on a hold. Repeated hand-on/hand-off of the same hold
  is wasted energy.
- **Static vs dynamic ratio:** measure smoothness of trajectories vs jerky/dynamic movement.
- **Trajectory smoothness:** derivative of keypoint paths.

Each Layer B metric must be:
1. Computed only in `ATTEMPT_ACTIVE`.
2. Anchored to the current hold / movement phase from Layer A.
3. Documented in code as a heuristic proxy with its tuning parameters exposed.

### Explicitly out of scope for v1

- Comparative scoring against a reference climber (DTW alignment against a recorded send).
  This is the eventual Layer C and the only part that might justify a trained model. Not now.

---

## Known risks and how the design handles them

- **Color segmentation is the load-bearing simplification.** Gym lighting is warm and uneven,
  chalk desaturates holds, and similar hues (red/pink, green/yellow-green) bleed. Mitigation:
  human-in-the-loop HSV sampling from the live frame, not hardcoded color names. Never promise
  automatic route detection.
- **Occlusion is the biggest accuracy risk.** A wall-facing camera hides half the climber's
  keypoints. This is fixed by camera placement (off-axis: side and slightly elevated), not by
  code. Do not try to solve occlusion in software for v1.
- **Movement scoring is heuristic.** Layer A is objective; Layer B is proxies. Keep them
  clearly separated so the demo can lead with the trustworthy number and present the
  interesting metrics as experimental.

---

## Project layout (proposed)

```
roidbeta/
  __init__.py
  main.py                 # entry point, wires up the state machine and threads
  state.py                # FSM: IDLE / ROUTE_SELECT / READY / ATTEMPT_ACTIVE / COMPLETED
  capture/
    frame_source.py       # FrameSource interface
    continuity.py         # Continuity Camera implementation
    buffer.py             # single-slot latest-frame buffer
  route/
    selector.py           # human-in-the-loop HSV sampling + mask build
    holds.py              # centroid extraction, (x, y, radius) targets
  pose/
    estimator.py          # MediaPipe wrapper, returns normalized keypoints
  scoring/
    contact.py            # Layer A: hold-contact progress
    metrics.py            # Layer B: movement-quality metrics (one function each)
    scorer.py             # combines A + B, holds attempt state
  display/
    overlay.py            # draws holds, keypoints, score, current metrics
  config.py               # tunable params: HSV tolerances, N-frame thresholds, metric weights
```

---

## Conventions

- Keypoints passed between modules as a stable, documented structure (index -> name mapping
  from MediaPipe). Do not pass raw MediaPipe objects across module boundaries; convert once at
  the pose layer.
- All tunables (HSV tolerance, contact frame count N, metric weights) live in `config.py`,
  never hardcoded inline.
- Coordinates: decide once (normalized 0-1 vs pixel) and document it in `config.py`. Pose is
  normalized; route mask is pixel. Convert at a single boundary, not ad hoc.
- No em dashes.

---

## Build order

1. `FrameSource` + Continuity Camera + single-slot buffer. Prove live frames on screen.
2. `PoseEstimator` overlay. Prove keypoints draw live on the climber.
3. `RouteSelector` HSV sampling + centroids. Prove the selected route's holds are masked.
4. FSM wiring. Prove mode transitions with keyboard controls.
5. **Layer A** hold-contact scoring. Prove "reached hold 7/10 in 45s" live. This is the
   minimum demoable product.
6. **Layer B** metrics, one at a time on top of A.
7. Overlay polish for the team demo.

Do not proceed to Layer B until Layer A works end to end live.
