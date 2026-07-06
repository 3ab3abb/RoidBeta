"""Tests for the single-slot latest-frame buffer.

These cover the overwrite / sequence semantics without needing a camera. Run:
    python -m pytest tests/test_buffer.py
"""

import threading

import numpy as np

from roidbeta.capture.buffer import LatestFrameBuffer


def test_empty_buffer_returns_seq_zero_and_none():
    buf = LatestFrameBuffer()
    seq, frame = buf.get()
    assert seq == 0
    assert frame is None


def test_put_bumps_sequence_and_returns_latest():
    buf = LatestFrameBuffer()
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.ones((2, 2, 3), dtype=np.uint8)

    buf.put(first)
    seq1, got1 = buf.get()
    assert seq1 == 1
    assert got1 is first

    buf.put(second)
    seq2, got2 = buf.get()
    assert seq2 == 2
    assert got2 is second  # always the newest, never a backlog


def test_concurrent_writes_leave_a_consistent_latest():
    buf = LatestFrameBuffer()
    frames = [np.full((1, 1, 3), i % 256, dtype=np.uint8) for i in range(500)]

    def writer(chunk):
        for f in chunk:
            buf.put(f)

    mid = len(frames) // 2
    threads = [
        threading.Thread(target=writer, args=(frames[:mid],)),
        threading.Thread(target=writer, args=(frames[mid:],)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    seq, frame = buf.get()
    assert seq == len(frames)  # every put counted exactly once
    assert frame is not None
