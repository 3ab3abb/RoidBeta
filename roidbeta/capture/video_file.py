"""Video-file FrameSource for testing the pipeline on pre-recorded footage.

Drops in wherever the Continuity Camera source goes, so the whole pipeline (route
selection, pose, scoring, overlay, recording) runs unchanged on a recorded clip.

It mimics a live camera: a thread paces frames at the file's fps into the
single-slot buffer, dropping stale frames just like the live source. It starts
paused holding the first frame, so you can select the route before the clip
plays, then resume() when the attempt starts.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .buffer import LatestFrameBuffer
from .frame_source import FrameSource


class VideoFileSource(FrameSource):
    """Plays a video file through the FrameSource interface.

    loop replays from the start on reaching the end (handy for repeated testing);
    when not looping, the last frame is held and is_finished becomes True. fps
    overrides the playback pace; 0 means "as fast as the consumer reads".
    """

    def __init__(self, path: str, loop: bool = False, fps: float | None = None) -> None:
        self._path = path
        self._loop = loop
        self._fps_override = fps
        self._buffer = LatestFrameBuffer()
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # start paused on the first frame
        self._finished = False
        self._frame_interval = 0.0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("VideoFileSource already started")

        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self._path}")
        self._cap = cap

        fps = self._fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_interval = 1.0 / fps if fps > 0 else 0.0

        # Preload the first frame so route selection has something to show while
        # the source is still paused.
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Video file has no frames: {self._path}")
        self._buffer.put(frame)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._play_loop, name="video-file", daemon=True
        )
        self._thread.start()

    def _play_loop(self) -> None:
        assert self._cap is not None
        while not self._stop_event.is_set():
            if self._paused.is_set():
                time.sleep(0.01)
                continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self._loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._finished = True
                self._paused.set()  # hold the last frame at the end
                continue

            self._buffer.put(frame)
            if self._frame_interval:
                time.sleep(self._frame_interval)

    def latest(self) -> tuple[int, np.ndarray | None]:
        return self._buffer.get()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        # A finished, non-looping video stays put; there is nothing left to play.
        if not self._finished:
            self._paused.clear()

    @property
    def is_finished(self) -> bool:
        return self._finished

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
