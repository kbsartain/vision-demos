"""Make a synthetic side-on treadmill runner with a known cadence, plus a
ground-truth keypoint sidecar, so the whole pipeline can be tested without a
camera.

    python scripts/make_synthetic_clip.py [--seconds 15] [--fps 30] [--cadence 178]
                                          [--drift 3] [--noise 0.0025] [--facing left]

Writes data/input/synthetic.mp4 and data/input/synthetic.mp4.poses.json. Point
config.INPUT_VIDEO at the clip and set POSE_BACKEND = "synthetic" to read the
exact keypoints back, or leave POSE_BACKEND = "yolo" to see what a detector
makes of a stick figure (usually nothing: this is a pipeline test, not a
detector test).

The runner is a two-link leg driven by an ankle path and a hip path in leg
lengths, with the knee solved by inverse kinematics. Contact flexion comes out
near 20 degrees and the swing peak near 100, roughly what a real runner shows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.skeleton import KPT_NAMES  # noqa: E402
from src.video import encode_h264  # noqa: E402

W, H = 608, 1080            # portrait, like a phone held sideways-on to a treadmill
LEG_PX = 0.36 * H           # hip-to-ankle, straight
GROUND_Y = 0.84 * H
THIGH = SHANK = 0.5         # leg lengths

# The ankle path over one stride, in leg lengths: x forward of the hip, y above
# the ground. Stance is the flat run (phase 0 to ~0.4); the rest is the swing.
ANKLE_PATH = [
    (0.00, 0.24, 0.00), (0.10, 0.105, 0.00), (0.20, -0.03, 0.00), (0.30, -0.165, 0.01),
    (0.40, -0.30, 0.06), (0.52, -0.38, 0.22), (0.64, -0.25, 0.36), (0.76, 0.00, 0.32),
    (0.88, 0.20, 0.16), (1.00, 0.24, 0.00),
]


def periodic_table(points, n=720, sigma=0.018):
    """Dense periodic (x, y) over phase in [0, 1), lightly smoothed so corners round off."""
    phis = np.array([p[0] for p in points])
    xs, ys = np.array([p[1] for p in points]), np.array([p[2] for p in points])
    grid = np.linspace(0.0, 1.0, n, endpoint=False)
    x, y = np.interp(grid, phis, xs), np.interp(grid, phis, ys)
    k = int(round(sigma * n)) * 3
    t = np.arange(-k, k + 1)
    kernel = np.exp(-0.5 * (t / (sigma * n)) ** 2)
    kernel /= kernel.sum()
    x = np.array([np.sum(np.take(x, (i + t) % n) * kernel) for i in range(n)])
    y = np.array([np.sum(np.take(y, (i + t) % n) * kernel) for i in range(n)])
    return grid, x, y


GRID, AX, AY = periodic_table(ANKLE_PATH)


def ankle_at(phase: float) -> tuple[float, float]:
    p = phase % 1.0
    i = p * len(GRID)
    i0, i1 = int(math.floor(i)) % len(GRID), (int(math.floor(i)) + 1) % len(GRID)
    f = i - math.floor(i)
    return float(AX[i0] * (1 - f) + AX[i1] * f), float(AY[i0] * (1 - f) + AY[i1] * f)


def hip_at(phase: float) -> tuple[float, float]:
    """Hip: a small forward-back sway and a bounce, lowest at mid-stance of either leg."""
    return 0.015 * math.sin(2 * math.pi * phase), 0.94 - 0.02 * math.cos(4 * math.pi * (phase - 0.2))


def knee_ik(hip, ankle):
    """Knee for a two-link leg, bending forward (the natural way)."""
    dx, dy = ankle[0] - hip[0], ankle[1] - hip[1]
    d = math.hypot(dx, dy)
    d_eff = min(d, THIGH + SHANK - 1e-4)
    ux, uy = dx / d, dy / d
    a = (THIGH ** 2 - SHANK ** 2 + d_eff ** 2) / (2 * d_eff)
    h = math.sqrt(max(THIGH ** 2 - a ** 2, 0.0))
    px, py = hip[0] + a * ux, hip[1] + a * uy
    nx, ny = -uy, ux                     # perpendicular; forward for a downward-pointing leg
    if nx < 0:
        nx, ny = -nx, -ny
    return px + h * nx, py + h * ny


def flexion_deg(hip, knee, ankle) -> float:
    v1 = (hip[0] - knee[0], hip[1] - knee[1])
    v2 = (ankle[0] - knee[0], ankle[1] - knee[1])
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    return 180.0 - math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def body_at(phase_right: float) -> dict:
    """All 17 COCO keypoints in world units (x forward, y up, leg lengths), keyed by name."""
    phase_left = phase_right + 0.5
    hx, hy = hip_at(phase_right)
    hip = (hx, hy)
    out = {}
    for foot, phase in (("right", phase_right), ("left", phase_left)):
        ankle = ankle_at(phase)
        ankle = (hx + ankle[0], ankle[1])
        knee = knee_ik(hip, ankle)
        out[f"{foot}_hip"] = (hx, hy + (0.012 if foot == "left" else 0.0))
        out[f"{foot}_knee"] = knee
        out[f"{foot}_ankle"] = ankle
    sx, sy = hx + 0.06, hy + 0.60
    out["left_shoulder"], out["right_shoulder"] = (sx, sy + 0.012), (sx, sy)
    nose = (sx + 0.11, sy + 0.23)
    out["nose"] = nose
    out["left_eye"], out["right_eye"] = (nose[0] - 0.02, nose[1] + 0.035), (nose[0] - 0.005, nose[1] + 0.04)
    out["left_ear"], out["right_ear"] = (sx - 0.005, sy + 0.245), (sx + 0.01, sy + 0.25)
    for side, phase in (("right", phase_left), ("left", phase_right)):
        theta = math.radians(35.0 * math.cos(2 * math.pi * phase))
        ex, ey = sx + 0.30 * math.sin(theta), sy - 0.30 * math.cos(theta)
        tf = theta + math.radians(85.0)
        out[f"{side}_elbow"] = (ex, ey)
        out[f"{side}_wrist"] = (ex + 0.28 * math.sin(tf), ey - 0.28 * math.cos(tf))
    return out


def to_image(pt, facing: float) -> tuple[float, float]:
    """World (forward, up) -> image pixels. facing is -1 (screen left) or +1."""
    return 0.5 * W + facing * pt[0] * LEG_PX, GROUND_Y - pt[1] * LEG_PX


def draw_scene(img, t: float, belt_speed_px: float) -> None:
    img[:] = (172, 214, 222)                                   # pale gym wall, BGR
    cv2.rectangle(img, (0, int(0.60 * H)), (W, int(0.72 * H)), (150, 190, 200), -1)
    cv2.rectangle(img, (0, int(GROUND_Y + 0.05 * H)), (W, H), (70, 70, 74), -1)   # floor
    deck_y0, deck_y1 = int(GROUND_Y + 0.006 * H), int(GROUND_Y + 0.05 * H)
    cv2.rectangle(img, (int(0.05 * W), deck_y0), (int(0.95 * W), deck_y1), (60, 60, 62), -1)
    cv2.rectangle(img, (int(0.05 * W), deck_y0), (int(0.95 * W), deck_y0 + 8), (30, 30, 32), -1)
    # belt stripes moving backwards
    period = 60
    shift = int((t * belt_speed_px) % period)
    for x in range(int(0.05 * W) - period + shift, int(0.95 * W), period):
        cv2.line(img, (max(x, int(0.05 * W)), deck_y0 + 8), (min(x + 24, int(0.95 * W)), deck_y0 + 8),
                 (110, 110, 112), 3)
    # console post at the front (screen left, since the runner faces left)
    cv2.rectangle(img, (int(0.06 * W), int(0.42 * H)), (int(0.10 * W), deck_y0), (90, 90, 92), -1)
    cv2.rectangle(img, (int(0.02 * W), int(0.38 * H)), (int(0.22 * W), int(0.43 * H)), (40, 40, 42), -1)


def draw_runner(img, kp: dict) -> None:
    def p(name):
        return tuple(int(round(v)) for v in kp[name])

    skin, shirt, shorts, shoe = (150, 180, 220), (190, 120, 60), (40, 40, 45), (245, 245, 245)
    # far leg first, then torso, then near leg (right = near)
    for foot, col in (("left", (35, 35, 40)), ("right", shorts)):
        cv2.line(img, p(f"{foot}_hip"), p(f"{foot}_knee"), col, 22, cv2.LINE_AA)
        cv2.line(img, p(f"{foot}_knee"), p(f"{foot}_ankle"), skin, 16, cv2.LINE_AA)
        cv2.circle(img, p(f"{foot}_ankle"), 14, shoe, -1, cv2.LINE_AA)
    torso = np.array([p("left_shoulder"), p("right_shoulder"), p("right_hip"), p("left_hip")], np.int32)
    cv2.polylines(img, [torso], True, shirt, 44, cv2.LINE_AA)
    cv2.fillPoly(img, [torso], shirt, cv2.LINE_AA)
    for side in ("left", "right"):
        cv2.line(img, p(f"{side}_shoulder"), p(f"{side}_elbow"), skin, 14, cv2.LINE_AA)
        cv2.line(img, p(f"{side}_elbow"), p(f"{side}_wrist"), skin, 12, cv2.LINE_AA)
    head = ((np.array(p("left_ear")) + np.array(p("right_ear"))) / 2).astype(int)
    cv2.circle(img, tuple(head), 34, skin, -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(head[0]), int(head[1] - 14)), (38, 20), 0, 180, 360, (40, 40, 45), -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--cadence", type=float, default=178.0, help="mean steps per minute")
    ap.add_argument("--drift", type=float, default=3.0, help="slow sinusoidal cadence swing, spm")
    ap.add_argument("--noise", type=float, default=0.0025, help="keypoint jitter, normalized units")
    ap.add_argument("--facing", choices=("left", "right"), default="left")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "input" / "synthetic.mp4")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    facing = -1.0 if args.facing == "left" else 1.0
    n = int(round(args.seconds * args.fps))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = args.out.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    img = np.zeros((H, W, 3), np.uint8)

    frames, cadence_series, strikes = [], [], []
    phase = 0.15                       # start mid-stance so the first strike is not at t=0
    prev_phase = phase
    contact_flex = []
    belt_speed = 0.54 * LEG_PX / (0.4 * 120.0 / args.cadence)   # px/s: stance span over stance time
    for i in range(n):
        t = i / args.fps
        c = args.cadence + args.drift * math.sin(2 * math.pi * t / 9.0)
        cadence_series.append(round(c, 3))
        kp = body_at(phase)
        # exact strikes: the right foot lands when the phase crosses an integer, the left at .5
        for foot, offset in (("right", 0.0), ("left", 0.5)):
            if math.floor(prev_phase - offset) != math.floor(phase - offset) and i > 0:
                strikes.append({"foot": foot, "time": round(t, 4)})
                h, k, a = kp[f"{foot}_hip"], kp[f"{foot}_knee"], kp[f"{foot}_ankle"]
                contact_flex.append(flexion_deg(h, k, a))
        px = {name: to_image(pt, facing) for name, pt in kp.items()}
        draw_scene(img, t, belt_speed)
        draw_runner(img, px)
        writer.write(img)

        kpts = []
        for name in KPT_NAMES:
            x, y = px[name]
            x = x / W + rng.normal(0.0, args.noise)
            y = y / H + rng.normal(0.0, args.noise)
            kpts.append([round(float(x), 4), round(float(y), 4)])
        xs, ys = [k[0] for k in kpts], [k[1] for k in kpts]
        bx, by = min(xs) - 0.02, min(ys) - 0.03
        frames.append({"index": i, "persons": [{
            "bbox_xywh": [round(bx, 4), round(by, 4), round(max(xs) - bx + 0.02, 4), round(max(ys) - by + 0.03, 4)],
            "kpts_xy": kpts, "track_id": 0}]})
        prev_phase = phase
        phase += c / 120.0 / args.fps

    writer.release()
    encode_h264(raw, args.out, crf=20)
    raw.unlink(missing_ok=True)

    truth = {
        "fps": args.fps, "frames": n, "facing_x": facing,
        "cadence_spm_mean": round(float(np.mean(cadence_series)), 3),
        "cadence_series_spm": cadence_series,
        "strikes": strikes,
        "contact_flexion_deg": {"mean": round(float(np.mean(contact_flex)), 2),
                                "sd": round(float(np.std(contact_flex, ddof=1)), 2)} if len(contact_flex) > 1 else None,
        "noise": args.noise,
    }
    sidecar = args.out.with_suffix(args.out.suffix + ".poses.json")
    sidecar.write_text(json.dumps({"frames": frames, "truth": truth}))
    print(f"wrote {args.out} ({n} frames @ {args.fps} fps) and {sidecar.name}")
    print(f"truth: cadence {truth['cadence_spm_mean']} spm, {len(strikes)} strikes, "
          f"contact flexion {truth['contact_flexion_deg']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
