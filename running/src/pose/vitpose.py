"""ViTPose+ locally, through Hugging Face transformers. The model the original
demo ran on the VLM Run gateway, with no API key and no upload.

Top-down: a fast YOLO detector finds the person box on every frame, then
ViTPose estimates the 17 COCO keypoints inside each box. The weights download
from the Hugging Face hub on first use (about 1.2 GB for plus-large).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
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
    import torch
    from PIL import Image
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    from ultralytics import YOLO

    device = _device(cfg.YOLO_DEVICE)
    detector = YOLO(cfg.VITPOSE_DETECTOR)
    processor = AutoProcessor.from_pretrained(cfg.VITPOSE_MODEL)
    model = VitPoseForPoseEstimation.from_pretrained(cfg.VITPOSE_MODEL).to(device).eval()
    is_plus = "plus" in str(cfg.VITPOSE_MODEL).lower()
    console.print(f"  model [bold]{cfg.VITPOSE_MODEL}[/] on [bold]{device}[/], boxes from {cfg.VITPOSE_DETECTOR}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    p = int(cfg.PRECISION)
    frames: list[dict] = []
    columns = (TextColumn("[cyan]pose[/]"), BarColumn(), TaskProgressColumn(),
               TextColumn("{task.completed}/{task.total} frames"), TimeElapsedColumn())
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task("pose", total=info.n_frames)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            det = detector.predict(frame, imgsz=cfg.YOLO_IMGSZ, conf=cfg.YOLO_CONF, device=device,
                                   classes=[0], verbose=False)[0]
            persons = []
            if det.boxes is not None and len(det.boxes):
                xyxy = det.boxes.xyxy.cpu().numpy()
                order = np.argsort(-(xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]))
                xyxy = xyxy[order][: max(1, int(cfg.MAX_PERSONS or 1))]
                boxes = np.stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = processor(image, boxes=[boxes], return_tensors="pt").to(device)
                with torch.no_grad():
                    if is_plus:
                        inputs["dataset_index"] = torch.zeros(len(boxes), dtype=torch.int64, device=device)
                    outputs = model(**inputs)
                results = processor.post_process_pose_estimation(outputs, boxes=[boxes])[0]
                for box, res in zip(boxes, results):
                    kp = res["keypoints"].cpu().numpy()
                    sc = res["scores"].cpu().numpy()
                    kpts = []
                    for k in range(kp.shape[0]):
                        if float(sc[k]) >= cfg.VITPOSE_KPT_CONF:
                            kpts.append([round(float(kp[k, 0]) / w, p), round(float(kp[k, 1]) / h, p)])
                        else:
                            kpts.append([0.0, 0.0])
                    persons.append({
                        "bbox_xywh": [round(float(box[0]) / w, p), round(float(box[1]) / h, p),
                                      round(float(box[2]) / w, p), round(float(box[3]) / h, p)],
                        "kpts_xy": kpts, "track_id": None,
                    })
            frames.append({"index": i, "persons": persons})
            i += 1
            progress.update(task, completed=i)
    cap.release()
    return frames, {"backend": "vitpose", "model": cfg.VITPOSE_MODEL, "detector": cfg.VITPOSE_DETECTOR,
                    "device": device, "kpt_conf": cfg.VITPOSE_KPT_CONF}


def cache_key(cfg) -> dict:
    return {"backend": "vitpose", "model": cfg.VITPOSE_MODEL, "detector": cfg.VITPOSE_DETECTOR,
            "imgsz": cfg.YOLO_IMGSZ, "conf": cfg.YOLO_CONF, "kpt_conf": cfg.VITPOSE_KPT_CONF,
            "precision": cfg.PRECISION}
