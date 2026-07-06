"""Capture layer: frame sources and the single-slot latest-frame buffer."""

from .buffer import LatestFrameBuffer
from .continuity import ContinuityCameraSource
from .frame_source import FrameSource

__all__ = ["FrameSource", "LatestFrameBuffer", "ContinuityCameraSource"]
