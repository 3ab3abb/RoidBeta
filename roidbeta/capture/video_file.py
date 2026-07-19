"""Frame-accurate video-file FrameSource for testing on recorded footage.

Unlike the live camera (which runs a thread and drops stale frames to stay
real-time), a recording is for analysis, so this source is pulled one frame at a
time via advance() and never drops a frame. Its time comes from the media (frame
index / fps), so the attempt timer and the exported clip are exact regardless of
how fast the machine processes.

It drops in wherever the camera source goes; the pipeline (route selection, pose,
scoring, overlay, recording) runs unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

from .buffer import LatestFrameBuffer
from .frame_source import FrameSource


class VideoFileSource(FrameSource):
    """Plays a video file frame by frame through the FrameSource interface.

    loop restarts from the beginning on reaching the end; otherwise the last
    frame is held and is_finished becomes True. fps overrides the media fps for
    timing and the exported clip.
    """

    def __init__(self, path: str, loop: bool = False, fps: float | None = None) -> None:
        self._path = path
        self._loop = loop
        self._fps_override = fps
        self._buffer = LatestFrameBuffer()
        self._cap: cv2.VideoCapture | None = None
        self._fps = 30.0
        self._frame_index = 0
        self._finished = False

    def start(self) -> None:
        if self._cap is not None:
            raise RuntimeError("VideoFileSource already started")
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self._path}")
        self._cap = cap
        self._fps = self._fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Video file has no frames: {self._path}")
        self._buffer.put(frame)
        self._frame_index = 1

    def advance(self) -> None:
        """Read the next frame in order. Sets is_finished at the end (no loop)."""
        if self._cap is None or self._finished:
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_index = 0
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self._finished = True
                    return
            else:
                self._finished = True
                return
        self._buffer.put(frame)
        self._frame_index += 1

    def latest(self) -> tuple[int, np.ndarray | None]:
        return self._buffer.get()

    @property
    def is_realtime(self) -> bool:
        return False

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def nominal_fps(self) -> float | None:
        return self._fps

    def position_seconds(self) -> float:
        return self._frame_index / self._fps if self._fps else 0.0

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
