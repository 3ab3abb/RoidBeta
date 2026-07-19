"""FrameSource interface.

Everything downstream reads frames through this interface, never directly from
OpenCV. That keeps the Continuity Camera swappable for an RTSP source later
without touching pose, scoring, or display code.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np


class FrameSource(ABC):
    """A source of BGR video frames.

    Implementations are expected to run their own capture thread and expose the
    latest frame through read(). Callers must not assume every produced frame is
    delivered: a live source drops stale frames rather than building a backlog.
    """

    @abstractmethod
    def start(self) -> None:
        """Open the underlying device and begin producing frames."""

    @abstractmethod
    def latest(self) -> tuple[int, np.ndarray | None]:
        """Return (seq, frame) for the latest available frame.

        seq is a monotonic counter that increments each time a new frame is
        produced; it is 0 before the first frame. A consumer compares seq
        against the last value it saw to know whether to reprocess. frame is a
        BGR uint8 ndarray of shape (height, width, 3), or None before the first
        frame.
        """

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Convenience wrapper: (ok, frame) for the latest frame.

        ok is False when no frame is available yet. Prefer latest() when you
        need to avoid reprocessing an unchanged frame.
        """
        seq, frame = self.latest()
        return (seq != 0 and frame is not None), frame

    @abstractmethod
    def stop(self) -> None:
        """Stop producing frames and release the underlying device."""

    # Playback control. Meaningful only for finite sources (a video file); live
    # sources cannot be paused and treat these as no-ops.

    def pause(self) -> None:
        """Hold the current frame instead of advancing. No-op for live sources."""

    def resume(self) -> None:
        """Resume advancing frames. No-op for live sources."""

    def advance(self) -> None:
        """Pull the next frame (frame-accurate sources). No-op for threaded ones."""

    @property
    def is_finished(self) -> bool:
        """True when a finite source has reached its end. Always False if live."""
        return False

    @property
    def is_realtime(self) -> bool:
        """True for a live source; False for a frame-accurate (video) source.

        A real-time source is driven by its own thread and is paced by the wall
        clock; a frame-accurate source is pulled one frame at a time via advance()
        and its time comes from the media, so no frames are dropped.
        """
        return True

    @property
    def nominal_fps(self) -> float | None:
        """The source's known frame rate, or None if it must be measured (live)."""
        return None

    def position_seconds(self) -> float:
        """Current time in seconds: wall clock for live, media time for video.

        Used as the state machine's clock so attempt timing is exact on recorded
        video (independent of processing speed) and real time on the live camera.
        """
        return time.monotonic()

    def __enter__(self) -> "FrameSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
