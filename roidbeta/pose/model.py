"""Off-the-shelf PoseLandmarker model asset management.

MediaPipe Tasks needs a .task model file. These are pretrained, off-the-shelf
assets published by Google; downloading one is not training or fine-tuning and
respects the project constraint. The asset is cached under models/ so it is
fetched at most once.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# Repo-local cache directory for model assets.
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

_MODEL_URLS = {
    "lite": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "full": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "heavy": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
}


def ensure_model(variant: str) -> Path:
    """Return a local path to the PoseLandmarker asset, downloading if absent.

    Raises ValueError for an unknown variant and lets network errors propagate:
    a missing model is a hard startup failure, not something to silently swallow.
    """
    if variant not in _MODEL_URLS:
        raise ValueError(
            f"Unknown pose model variant {variant!r}. "
            f"Expected one of {sorted(_MODEL_URLS)}."
        )

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _MODELS_DIR / f"pose_landmarker_{variant}.task"
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    url = _MODEL_URLS[variant]
    tmp = dest.with_suffix(".task.partial")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (trusted Google-hosted asset)
    tmp.replace(dest)
    return dest
