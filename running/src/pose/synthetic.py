"""Ground-truth poses for a synthetic clip, for testing the pipeline end to end.

scripts/make_synthetic_clip.py writes the clip and, beside it, a sidecar
``<clip>.poses.json`` holding the exact keypoints it drew. This backend reads
that sidecar, so the analysis can be checked against a known cadence.
"""

from __future__ import annotations

import json
from pathlib import Path


def sidecar_for(cfg) -> Path:
    return cfg.INPUT_VIDEO.with_suffix(cfg.INPUT_VIDEO.suffix + ".poses.json")


def run(video: Path, info, *, cfg, console) -> tuple[list[dict], dict]:
    path = sidecar_for(cfg)
    if not path.is_file():
        raise RuntimeError(f"No ground-truth sidecar at {path}. "
                           "Run scripts/make_synthetic_clip.py first.")
    blob = json.loads(path.read_text())
    frames = [f for f in blob["frames"] if f["index"] < info.n_frames]
    console.print(f"  ground truth: {len(frames)} frames from [dim]{path.name}[/]")
    return frames, {"backend": "synthetic", "truth": blob.get("truth")}


def cache_key(cfg) -> dict:
    return {"backend": "synthetic"}
