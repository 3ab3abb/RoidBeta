"""Single-slot latest-frame buffer.

The capture thread writes into this buffer and always overwrites the previous
frame. The consumer (pose/scoring) reads the most recent frame and never a
backlog. This is deliberately not a queue: we want to drop stale frames so the
overlay stays live even when pose stutters.
"""

from __future__ import annotations

import threading

import numpy as np


class LatestFrameBuffer:
    """Holds exactly one frame: the most recent one written.

    Thread-safe for a single-writer / multi-reader pattern. Each write bumps a
    monotonic sequence number so a reader can tell whether the frame changed
    since it last looked, without comparing pixels.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq: int = 0

    def put(self, frame: np.ndarray) -> None:
        """Overwrite the stored frame with the newest one."""
        with self._lock:
            self._frame = frame
            self._seq += 1

    def get(self) -> tuple[int, np.ndarray | None]:
        """Return (seq, frame) for the current frame.

        seq is 0 until the first frame arrives. Callers that only want new
        frames should compare seq against the last value they saw.
        """
        with self._lock:
            return self._seq, self._frame
