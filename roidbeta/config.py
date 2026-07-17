"""Tunable parameters for RoidBeta. Nothing tunable should be hardcoded outside this file."""

# --- Capture ---
CAMERA_INDEX = 0  # index of the Continuity Camera device as seen by AVFoundation
CAPTURE_TARGET_FPS = 30

# --- Coordinate convention ---
# Pose keypoints (MediaPipe) are normalized [0, 1] relative to frame width/height.
# Route hold centroids and radii are pixel coordinates in the captured frame.
# Any code comparing the two must convert pose keypoints to pixel space first,
# at the boundary in scoring/contact.py. Do not convert ad hoc elsewhere.

# --- Pose estimation (MediaPipe Tasks PoseLandmarker) ---
# The legacy mp.solutions.pose API was removed in recent mediapipe; v1 uses the
# MediaPipe Tasks PoseLandmarker, which needs an off-the-shelf .task model asset.
# Variant trades accuracy for speed: "lite" | "full" | "heavy". "full" is a good
# default on M1. The asset is downloaded once and cached under models/.
POSE_MODEL_VARIANT = "full"
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
POSE_MIN_PRESENCE_CONFIDENCE = 0.5

# --- Route selection (HSV sampling) ---
HSV_SAMPLE_PATCH_RADIUS_PX = 8      # neighborhood around a clicked point to sample HSV from
HSV_HUE_TOLERANCE = 12
HSV_SAT_TOLERANCE = 60
HSV_VAL_TOLERANCE = 60
MASK_MORPH_KERNEL_PX = 5             # kernel size for open/close cleanup of the raw color mask
HOLD_MIN_AREA_PX = 150               # discard mask blobs smaller than this when extracting holds

# Lighting normalization (CLAHE on the L channel in LAB) evens out gym lighting
# so chalked/shadowed holds are not missed and warm glare does not bleed.
COLOR_NORMALIZE_ENABLED = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = 8

# Watershed splitting of touching same-color holds into separate centroids.
HOLD_SPLIT_MIN_CENTER_DIST_PX = 4   # min distance-transform value for a blob center seed
HOLD_SPLIT_PEAK_WINDOW_PX = 13      # min separation between two hold centers

# Debug / manual selection aids:
MANUAL_HOLD_RADIUS_PX = 28           # radius of a hand-placed hold (for holds the color mask misses)
HOLD_UNSELECT_TOLERANCE_PX = 30      # click-to-hold distance that counts as "remove this hold"

# --- Hold contact (Layer A) ---
KEYPOINT_MIN_VISIBILITY = 0.5        # ignore keypoints below this confidence (likely occluded)
HOLD_CONTACT_FRAMES_N = 8            # frames a hand/foot keypoint must stay in a hold radius
TOP_HOLD_COMPLETION_FRAMES_N = 12    # frames the top hold must be matched (controlled) to complete
TOP_HOLD_HANDS_REQUIRED = 2          # hands that must be on the top hold to count a send (match)
# Completion is judged with a slightly enlarged top-hold radius, since hands
# rarely land dead-center and keypoints jitter; this makes topping out reliable.
TOP_HOLD_COMPLETION_RADIUS_SCALE = 1.6

# --- Movement quality (Layer B) ---
BALANCE_MIN_SUPPORT_POINTS = 2       # contact points needed before balance is scored
# Stabilizers for the balance metric (reduce flicker and fill short gaps):
BALANCE_EMA_ALPHA = 0.3              # smoothing weight for new samples (0..1, lower = smoother)
BALANCE_SUPPORT_HYSTERESIS_FRAMES = 6  # keep a contact in the base this many frames after it leaves
BALANCE_GAP_HOLD_MAX_FRAMES = 12     # max frames to carry the last score through a gap
BALANCE_GAP_COM_STABLE_PX = 45       # only carry it while the CoM stays within this of its last base
COM_SWAY_WINDOW_FRAMES = 30
REGRIP_COOLDOWN_FRAMES = 5           # min frames between counting a new regrip on the same hold
TRAJECTORY_SMOOTHNESS_WINDOW_FRAMES = 10

# --- Display ---
OVERLAY_WINDOW_NAME = "RoidBeta"
POSE_SKELETON_THICKNESS = 5         # bone line thickness for the drawn skeleton
POSE_JOINT_RADIUS = 4               # drawn radius of a regular joint
POSE_CONTACT_JOINT_RADIUS = 7       # drawn radius of hand/foot (contact) joints
POSE_FOOT_DRAW_SCALE = 1.9          # enlarge the drawn foot triangle (raw feet read too small)
COM_DOT_RADIUS = 9                  # drawn radius of the center-of-mass marker
POSE_ENVELOPE_THICKNESS = 2         # outline thickness of the convex-hull envelope
POSE_ENVELOPE_ALPHA = 0.15          # fill opacity of the envelope (0 = outline only)

# Live metrics chart panel:
CHART_PANEL_HEIGHT = 150            # height in px of the bottom metrics panel
CHART_HISTORY_POINTS = 150         # most recent samples shown in a sparkline
CHART_PANEL_ALPHA = 0.55           # panel background opacity

# Center-of-gravity trajectory trail:
TRAJECTORY_THICKNESS = 2
TRAJECTORY_MAX_POINTS = 400        # cap on how many recent CoM points are drawn

# A saved reference attempt's CoM path, drawn as a ghost trail for comparison.
# Same-camera, same-route only (raw pixel coordinates are not normalized).
REFERENCE_PATH = "data/reference_trajectory.json"

# --- Clip recording / review ---
# The annotated attempt is recorded to a temp file while ATTEMPT_ACTIVE, replayed
# from disk in REVIEW, then kept (moved into CLIP_DIR) or discarded.
CLIP_DIR = "clips"                  # where saved attempt clips land (relative to cwd)
CLIP_FOURCC = "mp4v"                # OpenCV VideoWriter fourcc; mp4v pairs with .mp4
CLIP_FILE_EXT = ".mp4"
REPLAY_FPS = CAPTURE_TARGET_FPS     # playback rate for the review loop
CLIP_SUMMARY_SECONDS = 3            # seconds of the results card held at the end of a saved clip
