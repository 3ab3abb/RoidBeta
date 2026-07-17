"""Save and load a reference CoM trajectory.

A reference is one attempt's center-of-mass path, kept so later attempts can draw
it as a ghost trail for visual comparison. It is stored as JSON: a list of
[x, y] pixel positions, with null for frames where the CoM was not estimated.

The coordinates are raw pixels, so a reference only lines up on the same camera
framing and route it was recorded on. Cross-setup comparison would need
normalization (a later step, alongside the DTW similarity score).
"""

from __future__ import annotations

import json
from pathlib import Path

Trajectory = list[tuple[float, float] | None]


def save_trajectory(path: str, points: Trajectory) -> Path:
    """Write a CoM trajectory to path as JSON. Returns the written path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = [None if p is None else [float(p[0]), float(p[1])] for p in points]
    dest.write_text(json.dumps(data))
    return dest


def load_trajectory(path: str) -> Trajectory | None:
    """Read a CoM trajectory from path, or None if the file does not exist."""
    src = Path(path)
    if not src.exists():
        return None
    data = json.loads(src.read_text())
    return [None if d is None else (float(d[0]), float(d[1])) for d in data]


def clear_trajectory(path: str) -> bool:
    """Delete the saved reference. Returns True if a file was removed."""
    src = Path(path)
    if src.exists():
        src.unlink()
        return True
    return False
