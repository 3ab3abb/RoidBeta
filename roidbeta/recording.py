"""Attempt clip recording and storage.

An attempt is recorded to a temporary video file as it happens (annotated frames,
so the clip shows the skeleton, holds, and score, not raw video). After the
attempt the user reviews the clip and either keeps it (re-encoded to the clips
directory at the true frame rate) or discards it (the temp file is deleted).

The real capture/processing rate is lower and more variable than the target fps
(pose is the bottleneck), so writing at a fixed fps makes playback run fast. We
therefore measure the actual rate at which real-time frames arrive and use that
for both the review playback and the saved file. Appended end-of-clip summary
frames are excluded from that measurement so they do not skew it.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import config

_MIN_FPS, _MAX_FPS = 5.0, 60.0


class AttemptRecorder:
    """Writes annotated attempt frames to a temp file, then keeps or discards it.

    Lifecycle: add() frames while the attempt runs, finalize() to close the file,
    then keep() to store it or discard() to delete it. keep()/discard() finalize
    automatically. Pass realtime=False for frames that are not part of the live
    attempt (the held summary at the end) so they don't affect the measured fps.
    """

    def __init__(
        self,
        fps: float = config.REPLAY_FPS,
        clip_dir: str = config.CLIP_DIR,
        fourcc: str = config.CLIP_FOURCC,
        ext: str = config.CLIP_FILE_EXT,
        measured: bool = True,
    ) -> None:
        # measured=True times real-time frames from the wall clock (live camera);
        # measured=False uses the given fps as-is (a video's known frame rate).
        self._measured = measured
        self._fallback_fps = float(fps)
        self._clip_dir = Path(clip_dir)
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._ext = ext
        self._stamp = time.strftime("%Y%m%d_%H%M%S")
        self._writer: cv2.VideoWriter | None = None
        self._temp_path = self._clip_dir / f".recording_{self._stamp}{ext}"
        self._size: tuple[int, int] | None = None  # (width, height)
        self._frame_count = 0
        # Wall-clock timing of real-time frames only, for the measured fps.
        self._realtime_count = 0
        self._first_t: float | None = None
        self._last_t: float | None = None

    def add(self, frame: np.ndarray, realtime: bool = True) -> None:
        """Append one frame, opening the writer on the first call.

        realtime frames update the fps measurement; pass False for filler frames
        (the end-of-clip summary) that are written faster than real time.
        """
        if self._writer is None:
            self._clip_dir.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self._size = (width, height)
            self._writer = cv2.VideoWriter(
                str(self._temp_path), self._fourcc, self._fallback_fps, (width, height)
            )
            if not self._writer.isOpened():
                raise RuntimeError(
                    f"Could not open a video writer for {self._temp_path}. "
                    f"Is the {config.CLIP_FOURCC!r} codec available?"
                )
        self._writer.write(frame)
        self._frame_count += 1

        if realtime:
            now = time.monotonic()
            if self._first_t is None:
                self._first_t = now
            self._last_t = now
            self._realtime_count += 1

    def finalize(self) -> None:
        """Close the temp file so it can be replayed, kept, or discarded."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    @property
    def has_frames(self) -> bool:
        return self._frame_count > 0

    @property
    def fps(self) -> float:
        """Frame rate to write at: the video's known fps, or the measured rate."""
        if not self._measured:
            return self._fallback_fps
        if (self._realtime_count >= 2 and self._first_t is not None
                and self._last_t is not None and self._last_t > self._first_t):
            measured = (self._realtime_count - 1) / (self._last_t - self._first_t)
            return min(_MAX_FPS, max(_MIN_FPS, measured))
        return self._fallback_fps

    @property
    def temp_path(self) -> Path:
        return self._temp_path

    def keep(self) -> Path:
        """Finalize and store the clip at its true fps. Returns the saved path.

        The temp file was written with a placeholder fps header, so it is
        re-encoded to the measured fps so external players run it at real speed.
        """
        self.finalize()
        dest = self._clip_dir / f"attempt_{self._stamp}{self._ext}"
        fps = self.fps

        cap = cv2.VideoCapture(str(self._temp_path))
        writer = None
        if cap.isOpened() and self._size is not None:
            writer = cv2.VideoWriter(str(dest), self._fourcc, fps, self._size)
        if writer is None or not writer.isOpened():
            # Re-encode unavailable: fall back to moving the file as-is.
            cap.release()
            self._temp_path.replace(dest)
            return dest

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        self._temp_path.unlink(missing_ok=True)
        return dest

    def discard(self) -> None:
        """Finalize and delete the temp file."""
        self.finalize()
        self._temp_path.unlink(missing_ok=True)
