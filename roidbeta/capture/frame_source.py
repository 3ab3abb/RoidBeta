"""FrameSource interface.

Everything downstream reads frames through this interface, never directly from
OpenCV. That keeps the Continuity Camera swappable for an RTSP source later
without touching pose, scoring, or display code.
"""

from __future__ import annotations

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

    @property
    def is_finished(self) -> bool:
        """True when a finite source has reached its end. Always False if live."""
        return False

    def __enter__(self) -> "FrameSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
