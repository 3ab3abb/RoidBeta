"""Continuity Camera FrameSource.

Reads the iPhone Continuity Camera on the Mac as a native AVCaptureDevice
through OpenCV's AVFoundation backend. Runs a dedicated capture thread that
pushes frames into a single-slot buffer, always overwriting, so consumers never
see a backlog.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from .. import config
from .buffer import LatestFrameBuffer
from .frame_source import FrameSource


class ContinuityCameraSource(FrameSource):
    """Live BGR frames from the Continuity Camera via cv2.VideoCapture.

    The device is opened with the AVFoundation backend so Continuity Camera is
    picked up as a native capture device. If the iPhone is not the device at
    CAMERA_INDEX, adjust config.CAMERA_INDEX.
    """

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        target_fps: int = config.CAPTURE_TARGET_FPS,
    ) -> None:
        self._camera_index = camera_index
        self._target_fps = target_fps
        self._buffer = LatestFrameBuffer()
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("ContinuityCameraSource already started")

        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self._camera_index}. "
                "Is the iPhone connected and Continuity Camera enabled?"
            )
        cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        self._cap = cap

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="capture", daemon=True
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        assert self._cap is not None
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                # A transient read failure should not kill the thread; keep the
                # last good frame in the buffer and try again.
                continue
            self._buffer.put(frame)

    def latest(self) -> tuple[int, np.ndarray | None]:
        return self._buffer.get()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
