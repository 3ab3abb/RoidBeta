"""MediaPipe Tasks PoseLandmarker wrapper.

Consumes an off-the-shelf MediaPipe pose model (no training, no fine-tuning per
the project constraints) and returns our stable PoseFrame structure. This is the
only place that touches raw MediaPipe objects.

Uses the Tasks PoseLandmarker in VIDEO running mode, which expects frames in
capture order with monotonically increasing timestamps. That matches our
single-slot buffer: we only ever process the latest frame, forward in time.
"""

from __future__ import annotations

import time

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .. import config
from .keypoints import Keypoint, PoseFrame
from .model import ensure_model


class PoseEstimator:
    """Runs MediaPipe PoseLandmarker on BGR frames, returns normalized keypoints.

    Construct once and reuse on the pose/scoring thread. Not thread-safe: a
    single landmarker in VIDEO mode must be driven from one thread with
    non-decreasing timestamps.
    """

    def __init__(
        self,
        variant: str = config.POSE_MODEL_VARIANT,
        min_detection_confidence: float = config.POSE_MIN_DETECTION_CONFIDENCE,
        min_presence_confidence: float = config.POSE_MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = config.POSE_MIN_TRACKING_CONFIDENCE,
    ) -> None:
        model_path = ensure_model(variant)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._start_ns: int | None = None

    def _timestamp_ms(self) -> int:
        # VIDEO mode requires monotonically increasing timestamps in ms. Derive
        # them from a monotonic clock anchored to the first estimate() call.
        now = time.monotonic_ns()
        if self._start_ns is None:
            self._start_ns = now
        return (now - self._start_ns) // 1_000_000

    def estimate(self, frame_bgr: np.ndarray) -> PoseFrame | None:
        """Return the pose for a BGR frame, or None if no person is found."""
        # MediaPipe expects RGB. Convert at this boundary only.
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms())

        if not result.pose_landmarks:
            return None

        # num_poses=1, so take the first (and only) detected pose.
        landmarks = result.pose_landmarks[0]
        keypoints = tuple(
            Keypoint(x=lm.x, y=lm.y, visibility=lm.visibility)
            for lm in landmarks
        )
        return PoseFrame(keypoints=keypoints)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
