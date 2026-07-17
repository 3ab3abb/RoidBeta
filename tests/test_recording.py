"""Tests for attempt clip recording lifecycle.

Exercises the temp-file/keep/discard flow with synthetic frames and no camera.
Writes into a temporary clip directory so it never touches the real clips/. Run:
    python -m pytest tests/test_recording.py
"""

import numpy as np
import pytest

from roidbeta.recording import AttemptRecorder


def _frames(n=6, size=(64, 48)):
    h, w = size
    return [np.full((h, w, 3), i % 255, dtype=np.uint8) for i in range(n)]


def test_records_to_temp_then_keeps(tmp_path):
    rec = AttemptRecorder(fps=10, clip_dir=str(tmp_path))
    assert not rec.has_frames

    for f in _frames():
        rec.add(f)
    assert rec.has_frames
    if not rec.temp_path.exists():
        pytest.skip("VideoWriter codec unavailable in this environment")

    dest = rec.keep()
    assert dest.exists()
    assert not rec.temp_path.exists()   # temp moved, not left behind
    assert dest.parent == tmp_path
    assert dest.stat().st_size > 0


def test_discard_removes_temp(tmp_path):
    rec = AttemptRecorder(fps=10, clip_dir=str(tmp_path))
    for f in _frames():
        rec.add(f)
    if not rec.temp_path.exists():
        pytest.skip("VideoWriter codec unavailable in this environment")

    rec.discard()
    assert not rec.temp_path.exists()
    # No permanent clips were created.
    assert not any(p.name.startswith("attempt_") for p in tmp_path.iterdir())


def test_no_frames_has_frames_false(tmp_path):
    rec = AttemptRecorder(fps=10, clip_dir=str(tmp_path))
    assert not rec.has_frames
    rec.discard()  # safe to discard an empty recorder
