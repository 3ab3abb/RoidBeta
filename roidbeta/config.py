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

# --- Hold contact (Layer A) ---
HOLD_CONTACT_FRAMES_N = 8            # frames a hand/foot keypoint must stay in a hold radius
TOP_HOLD_COMPLETION_FRAMES_N = 15    # frames a hand keypoint must hold at the top hold to complete

# --- Movement quality (Layer B) ---
COM_SWAY_WINDOW_FRAMES = 30
REGRIP_COOLDOWN_FRAMES = 5           # min frames between counting a new regrip on the same hold
TRAJECTORY_SMOOTHNESS_WINDOW_FRAMES = 10

# --- Display ---
OVERLAY_WINDOW_NAME = "RoidBeta"
