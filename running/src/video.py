"""Probing, cached conversion and final encoding of the clip, through ffmpeg."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def __str__(self) -> str:
        return (f"{self.width}x{self.height} @ {self.fps:.2f} fps, "
                f"{self.n_frames} frames, {self.duration:.1f}s")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{proc.stderr[-3000:]}")
    return proc


def probe_source(path: Path) -> dict:
    """ffprobe the source, including the rotation flag OpenCV would ignore."""
    proc = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_frames,duration,pix_fmt",
        "-show_entries", "stream_side_data=rotation",
        "-of", "json", str(path),
    ])
    stream = json.loads(proc.stdout)["streams"][0]
    rotation = 0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "pix_fmt": stream.get("pix_fmt"),
        "n_frames": int(stream.get("nb_frames") or 0),
        "duration": float(stream.get("duration") or 0.0),
        "rotation": rotation,
    }


def cache_path(src: Path, cache_dir: Path, *, target_height, trim_seconds, crf) -> Path:
    """A name that changes whenever the source or any conversion setting does."""
    stat = src.stat()
    key = json.dumps({"name": src.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "target_height": target_height, "trim_seconds": trim_seconds, "crf": crf},
                     sort_keys=True)
    return cache_dir / f"{src.stem}.{hashlib.sha256(key.encode()).hexdigest()[:12]}.mp4"


def convert(src: Path, dst: Path, *, target_height, trim_seconds, crf: int) -> None:
    """Transcode to H.264 MP4, applying rotation and only ever downscaling."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if trim_seconds:
        cmd += ["-t", str(trim_seconds)]
    if target_height:
        cmd += ["-vf", f"scale=-2:'min({target_height},ih)'"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-an", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg conversion failed:\n{proc.stderr[-3000:]}")
    tmp.replace(dst)


def inspect(path: Path) -> VideoInfo:
    """Geometry as OpenCV will decode it, which is what the renderer uses."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    info = VideoInfo(path=path, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
                     n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def encode_h264(src: Path, dst: Path, *, crf: int) -> None:
    """OpenCV writes MPEG-4 Part 2; this makes the result playable everywhere."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
          "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p", str(dst)])
