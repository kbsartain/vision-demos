"""COCO-17 skeleton: names, edges, colours, and the overlay drawn on the video."""

from __future__ import annotations

from collections import deque

import cv2

from .comet import Comet

KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KPT_INDEX = {name: i for i, name in enumerate(KPT_NAMES)}
FACE_KPTS = frozenset({0, 1, 2, 3, 4})

# (a, b, side): the side picks the colour.
SKELETON = [
    (15, 13, "L"), (13, 11, "L"), (16, 14, "R"), (14, 12, "R"),
    (11, 12, "T"), (5, 11, "L"), (6, 12, "R"), (5, 6, "T"),
    (5, 7, "L"), (7, 9, "L"), (6, 8, "R"), (8, 10, "R"),
    (0, 1, "H"), (0, 2, "H"), (1, 3, "H"), (2, 4, "H"), (3, 5, "H"), (4, 6, "H"),
]

# BGR. Left limbs cyan, right limbs orange: the panel keeps the same two colours.
SIDE_COLORS = {
    "L": (255, 200, 0),
    "R": (0, 140, 255),
    "T": (80, 220, 80),
    "H": (200, 80, 220),
}
FOOT_COLORS = {"left": (255, 200, 0), "right": (0, 140, 255)}
ANKLES = {"left": KPT_INDEX["left_ankle"], "right": KPT_INDEX["right_ankle"]}


def visible_parts(draw_face: bool):
    if draw_face:
        return SKELETON, frozenset(range(len(KPT_NAMES)))
    edges = [(a, b, s) for a, b, s in SKELETON if a not in FACE_KPTS and b not in FACE_KPTS]
    return edges, frozenset(i for i in range(len(KPT_NAMES)) if i not in FACE_KPTS)


def _pixels(person: dict, width: int, height: int):
    return [(int(round(x * width)), int(round(y * height))) for x, y in person.get("kpts_xy", [])]


def draw_person(img, person: dict, width: int, height: int, *, edges, points,
                thickness: int = 3, radius: int = 4, draw_bbox: bool = False,
                bbox_color=(255, 160, 40)):
    """Draw one person from normalized coordinates. (0, 0) means the joint was not seen."""
    pts = _pixels(person, width, height)

    def ok(i):
        return 0 <= i < len(pts) and i in points and pts[i] != (0, 0)

    if draw_bbox and "bbox_xywh" in person:
        bx, by, bw, bh = person["bbox_xywh"]
        cv2.rectangle(img, (int(bx * width), int(by * height)),
                      (int((bx + bw) * width), int((by + bh) * height)),
                      bbox_color, max(1, thickness - 1), cv2.LINE_AA)
    for a, b, side in edges:
        if ok(a) and ok(b):
            cv2.line(img, pts[a], pts[b], SIDE_COLORS[side], thickness, cv2.LINE_AA)
    for i in range(len(pts)):
        if ok(i):
            cv2.circle(img, pts[i], radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pts[i], radius, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def ankle_pixel(person: dict, foot: str, width: int, height: int):
    kpts = person.get("kpts_xy") or []
    i = ANKLES[foot]
    if i >= len(kpts) or tuple(kpts[i]) == (0, 0):
        return None
    x, y = kpts[i]
    return int(round(x * width)), int(round(y * height))


def draw_foot_markers(img, person, width, height, *, radius: int):
    """Enlarge exactly the two joints the whole analysis is measured from."""
    for foot, color in FOOT_COLORS.items():
        p = ankle_pixel(person, foot, width, height)
        if p is None:
            continue
        cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, p, radius, (255, 255, 255), max(1, radius // 4), cv2.LINE_AA)
    return img


def draw_contact_flash(img, person, width, height, *, foot: str, progress: float,
                       radius: int, thickness: int):
    """A ring that expands and fades over a foot strike; progress runs 0 to 1."""
    p = ankle_pixel(person, foot, width, height)
    if p is None:
        return img
    t = float(min(max(progress, 0.0), 1.0))
    faded = tuple(int(c * (1.0 - t)) for c in FOOT_COLORS[foot])
    cv2.circle(img, p, radius + int(radius * 1.8 * t), faded, max(1, thickness), cv2.LINE_AA)
    return img


class AnkleTrails:
    """The recent path of each ankle, drawn as a comet behind the marker on the video."""

    def __init__(self, feet, *, length: int, width: int, taper=0.18, opacity=0.85,
                 fade=1.35, glow=2.6, glow_opacity=0.28, softness=0.6, wind=(0.0, 0.0)):
        self._pts = {foot: deque(maxlen=max(2, length)) for foot in feet}
        self._comet = Comet(width=width, taper=taper, opacity=opacity, fade=fade, glow=glow,
                            glow_opacity=glow_opacity, softness=softness, wind=wind)

    def update(self, person, width: int, height: int):
        for foot, history in self._pts.items():
            history.append(None if person is None else ankle_pixel(person, foot, width, height))
        return self

    def draw(self, img):
        return self._comet.draw(img, [(list(h), FOOT_COLORS[f]) for f, h in self._pts.items()])
