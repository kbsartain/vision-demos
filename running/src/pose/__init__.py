"""Pose backends and the one frame format they all produce.

Every backend returns ``frames``: a list, one entry per source frame that was
posed, of

    {"index": int, "persons": [{"bbox_xywh": [x, y, w, h], "kpts_xy": [[x, y] * 17],
                                "track_id": int | None}]}

with coordinates normalized to the frame (x by width, y by height) and ``(0, 0)``
meaning the keypoint was not seen. That is the VLM Run gateway format for
ViTPose, so a cached gateway response and a local YOLO run are interchangeable
downstream.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

BACKENDS = {"yolo": "src.pose.yolo", "vitpose": "src.pose.vitpose", "vlmrun": "src.pose.vlmrun",
            "synthetic": "src.pose.synthetic"}


def get_backend(name: str):
    try:
        return importlib.import_module(BACKENDS[name])
    except KeyError:
        raise ValueError(f"Unknown POSE_BACKEND {name!r}; pick one of {sorted(BACKENDS)}") from None


def cache_path(video: Path, cache_dir: Path, *, key: dict) -> Path:
    stat = video.stat()
    blob = json.dumps({"video": video.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, **key},
                      sort_keys=True, default=str)
    return cache_dir / f"poses.{hashlib.sha256(blob.encode()).hexdigest()[:12]}.json"


def save_cache(path: Path, frames: list[dict], meta: dict, *, stamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"created": stamp, "frames": frames, "meta": meta}))


def load_cache(path: Path):
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
        return blob["frames"], blob.get("meta") or {}, blob.get("created", "unknown")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _area(b):
    return float(b[2] * b[3])


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def dedupe(frames: list[dict], *, max_persons, iou: float) -> int:
    """One body should be one subject.

    Keeps the box that best continues the previous frame (then the largest),
    drops overlaps above *iou*, and caps the count at *max_persons*.
    """
    previous, dropped = None, 0
    for frame in frames:
        persons = frame.get("persons") or []
        if not persons:
            continue
        order = sorted(persons, key=lambda p: (
            -(_iou(p["bbox_xywh"], previous) if previous is not None else 0.0),
            -_area(p["bbox_xywh"])))
        kept: list[dict] = []
        for person in order:
            if all(_iou(person["bbox_xywh"], k["bbox_xywh"]) < iou for k in kept):
                kept.append(person)
        if max_persons:
            kept = kept[:max_persons]
        dropped += len(persons) - len(kept)
        frame["persons"] = kept
        if kept:
            previous = kept[0]["bbox_xywh"]
    return dropped


def n_posed(frames) -> int:
    return sum(1 for f in frames if f.get("persons"))
