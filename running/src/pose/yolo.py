"""Local pose estimation with ultralytics YOLO-pose. No API key, runs offline.

YOLO11-pose predicts the same 17 COCO keypoints ViTPose does, in the same order,
so its output drops straight into the gait analysis.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn


def _device(spec: str) -> str:
    if str(spec).lower() != "auto":
        return spec
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run(video: Path, info, *, cfg, console) -> tuple[list[dict], dict]:
    from ultralytics import YOLO

    device = _device(cfg.YOLO_DEVICE)
    model = YOLO(cfg.YOLO_MODEL)
    console.print(f"  model [bold]{cfg.YOLO_MODEL}[/] on [bold]{device}[/], imgsz {cfg.YOLO_IMGSZ}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")

    p = int(cfg.PRECISION)
    frames: list[dict] = []
    batch: list = []
    indices: list[int] = []

    def flush():
        results = model.predict(batch, imgsz=cfg.YOLO_IMGSZ, conf=cfg.YOLO_CONF, device=device,
                                verbose=False)
        for idx, res in zip(indices, results):
            persons = []
            if res.keypoints is not None and res.boxes is not None and len(res.boxes):
                xyn = res.keypoints.xyn.cpu().numpy()
                kconf = res.keypoints.conf
                kconf = kconf.cpu().numpy() if kconf is not None else None
                boxes = res.boxes.xyxyn.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                for j in range(len(boxes)):
                    kpts = []
                    for k in range(xyn.shape[1]):
                        x, y = float(xyn[j, k, 0]), float(xyn[j, k, 1])
                        seen = ((kconf is None or float(kconf[j, k]) >= cfg.YOLO_KPT_CONF)
                                and (x > 0 or y > 0))
                        kpts.append([round(x, p), round(y, p)] if seen else [0.0, 0.0])
                    x0, y0, x1, y1 = (float(v) for v in boxes[j])
                    persons.append({
                        "bbox_xywh": [round(x0, p), round(y0, p), round(x1 - x0, p), round(y1 - y0, p)],
                        "kpts_xy": kpts,
                        "score": round(float(scores[j]), 4),
                        "track_id": None,
                    })
            frames.append({"index": idx, "persons": persons})
        batch.clear()
        indices.clear()

    columns = (TextColumn("[cyan]pose[/]"), BarColumn(), TaskProgressColumn(),
               TextColumn("{task.completed}/{task.total} frames"), TimeElapsedColumn())
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task("pose", total=info.n_frames)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            batch.append(frame)
            indices.append(i)
            i += 1
            if len(batch) >= max(1, int(cfg.YOLO_BATCH)):
                flush()
                progress.update(task, completed=i)
        if batch:
            flush()
            progress.update(task, completed=i)
    cap.release()
    return frames, {"backend": "yolo", "model": cfg.YOLO_MODEL, "imgsz": cfg.YOLO_IMGSZ,
                    "device": device, "conf": cfg.YOLO_CONF, "kpt_conf": cfg.YOLO_KPT_CONF}


def cache_key(cfg) -> dict:
    return {"backend": "yolo", "model": cfg.YOLO_MODEL, "imgsz": cfg.YOLO_IMGSZ,
            "conf": cfg.YOLO_CONF, "kpt_conf": cfg.YOLO_KPT_CONF, "precision": cfg.PRECISION}
