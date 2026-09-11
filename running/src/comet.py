"""Comet trails: a path that tapers, fades and streams downwind from its head.

Used twice, over different coordinate systems: behind each ankle on the video,
and in the panel's ankle view. The compositing therefore lives here once.

The fade is a real alpha blend through a blurred mask, not additive light, so
it reads the same over a bright gym wall and over the near-black panel.

The wind is a drawing device: each point of the tail is pushed downwind in
proportion to its age, so the head always sits exactly on the joint and only
the spent part of the trail drifts. A planted foot then streams a clean streak
instead of collapsing into a blob.
"""

from __future__ import annotations

import cv2
import numpy as np


def catmull_rom(points, per_span: int):
    """Dense points on a Catmull-Rom spline through *points* (passes through all)."""
    n = len(points)
    if n < 3 or per_span <= 1:
        return list(points)
    out = []
    for i in range(n - 1):
        p0 = points[max(i - 1, 0)]
        p1, p2 = points[i], points[i + 1]
        p3 = points[min(i + 2, n - 1)]
        for k in range(per_span):
            t = k / per_span
            t2, t3 = t * t, t * t * t
            b0 = -0.5 * t3 + t2 - 0.5 * t
            b1 = 1.5 * t3 - 2.5 * t2 + 1.0
            b2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
            b3 = 0.5 * t3 - 0.5 * t2
            out.append((p0[0] * b0 + p1[0] * b1 + p2[0] * b2 + p3[0] * b3,
                        p0[1] * b0 + p1[1] * b1 + p2[1] * b2 + p3[1] * b3))
    out.append(points[-1])
    return out


def wind_vector(setting, heading: float, *, speed: float, fps: float) -> tuple[float, float]:
    """Downwind drift per frame, in px. Downwind is the direction of travel reversed."""
    setting = str(setting).lower()
    sign = {"left": -1.0, "right": 1.0, "none": 0.0}.get(setting, -float(np.sign(heading)))
    if not sign or speed <= 0 or fps <= 0:
        return (0.0, 0.0)
    return (sign * float(speed) / float(fps), 0.0)


class Comet:
    """One trail compositor per drawing surface. Scratch buffers are reused."""

    def __init__(self, *, width: int, taper: float = 0.18, opacity: float = 0.85,
                 fade: float = 1.35, glow: float = 2.6, glow_opacity: float = 0.28,
                 softness: float = 0.6, wind=(0.0, 0.0), smooth: int = 1):
        self.width = max(1, int(width))
        self.wind = (float(wind[0]), float(wind[1]))
        self.taper = float(min(max(taper, 0.0), 1.0))
        self.opacity = float(min(max(opacity, 0.0), 1.0))
        self.fade = max(float(fade), 0.1)
        self.glow = max(float(glow), 1.0)
        self.glow_opacity = float(min(max(glow_opacity, 0.0), 1.0))
        self.spread = max(0, round(self.width * float(softness)))
        self.smooth = max(1, int(smooth))
        self._layer = self._mask = self._box = None

    def _segments(self, points):
        """Yield (a, b, head_fraction) per drawable segment, oldest first, raked by age.

        A None in the path is a gap: nothing is drawn across it. Age is counted
        in real frames whatever the smoothing, so wind and fade are unchanged by it.
        """
        span = max(len(points) - 1, 1)
        last = len(points) - 1
        wx, wy = self.wind

        def blown(p, age):
            if not (wx or wy):
                return (int(round(p[0])), int(round(p[1])))
            return (int(round(p[0] + wx * age)), int(round(p[1] + wy * age)))

        i, n = 0, len(points)
        while i < n:
            if points[i] is None:
                i += 1
                continue
            j = i
            while j + 1 < n and points[j + 1] is not None:
                j += 1
            run = points[i:j + 1]
            if self.smooth <= 1 or len(run) < 3:
                dense, step = run, 1.0
            else:
                dense, step = catmull_rom(run, self.smooth), 1.0 / self.smooth
            age = last - i
            for k in range(len(dense) - 1):
                a_age, b_age = age, age - step
                f = 1.0 - b_age / span
                yield blown(dense[k], a_age), blown(dense[k + 1], b_age), f
                age = b_age
            i = j + 1

    def draw(self, img, paths):
        """Composite every (points, color) trail in *paths* in one blend."""
        h, w = img.shape[:2]
        if self._layer is None or self._layer.shape[:2] != (h, w):
            self._layer = np.zeros_like(img)
            self._mask = np.zeros((h, w), np.uint8)
        elif self._box is not None:
            x0, y0, x1, y1 = self._box
            self._layer[y0:y1, x0:x1] = 0
            self._mask[y0:y1, x0:x1] = 0
        self._box = None

        paths = [(list(self._segments(pts)), color) for pts, color in paths if color is not None]
        xs: list[int] = []
        ys: list[int] = []
        for scale, opacity in ((self.glow, self.glow_opacity), (1.0, self.opacity)):
            for segments, color in paths:
                for a, b, f in segments:
                    taper = self.taper + (1.0 - self.taper) * f
                    weight = max(1, round(self.width * taper * scale))
                    alpha = round(255 * opacity * f ** self.fade)
                    if alpha <= 0:
                        continue
                    cv2.line(self._layer, a, b, color, weight + 2 * self.spread)
                    cv2.line(self._mask, a, b, alpha, weight, cv2.LINE_AA)
                    xs += [a[0], b[0]]
                    ys += [a[1], b[1]]
        if not xs:
            return img

        pad = max(2, round(self.width * self.glow) + 2 * self.spread)
        x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
        x1, y1 = min(w, max(xs) + pad), min(h, max(ys) + pad)
        if x1 <= x0 or y1 <= y0:
            return img
        self._box = (x0, y0, x1, y1)

        mask = self._mask[y0:y1, x0:x1]
        if self.spread:
            k = 2 * self.spread + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        roi = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = (roi * (1.0 - alpha)
                             + self._layer[y0:y1, x0:x1].astype(np.float32) * alpha).astype(np.uint8)
        return img
