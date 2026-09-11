"""The live side panel: cadence headline, cadence-over-time graph, the average
knee at foot strike, and the ankle path view.

Drawn with OpenCV and Pillow rather than matplotlib, because it changes every
frame. Everything static (titles, axes, ticks, legend, credit) is drawn once
into a background image; each frame copies it and draws only what moves.

Nothing is drawn before it happened: every trace, number and shape reads only
frames at or before the one being written.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import text as textmod
from .comet import Comet, catmull_rom, wind_vector
from .gait import FEET
from .skeleton import FOOT_COLORS

BG = (18, 18, 20)
GRID = (40, 40, 46)
AXIS = (90, 90, 98)
DIM = (150, 150, 158)
INK = (245, 245, 245)
MUTED = (110, 110, 118)
MEAN_COLOR = (235, 111, 31)      # BGR: a saturated blue that reads on near-black


class Box:
    """A plot rectangle plus the data range it maps."""

    def __init__(self, px0: int, px1: int, py0: int, py1: int):
        self.px0, self.px1, self.py0, self.py1 = px0, px1, py0, py1
        self.x0 = self.x1 = self.y0 = self.y1 = 0.0

    @property
    def width(self) -> int:
        return self.px1 - self.px0

    @property
    def height(self) -> int:
        return self.py1 - self.py0

    def sx(self, x: float) -> int:
        t = (float(x) - self.x0) / max(self.x1 - self.x0, 1e-9)
        return int(round(self.px0 + self.width * float(np.clip(t, 0.0, 1.0))))

    def sy(self, y: float) -> int:
        t = (float(y) - self.y0) / max(self.y1 - self.y0, 1e-9)
        return int(round(self.py1 - self.height * float(np.clip(t, 0.0, 1.0))))


def nice_step(span: float, target: int, candidates) -> float:
    for value in candidates:
        if span / value <= target:
            return float(value)
    return float(candidates[-1])


class GaitPanel:
    """Draws the panel for a given frame. One instance per render."""

    def __init__(self, analysis, width: int, height: int, *, cfg, font=None):
        self.a = analysis
        self.w, self.h = width, height
        self.cfg = cfg
        self.font = font
        self.u = width / 720.0
        self.fps = analysis.fps or 30.0

        s = self._s
        self.size_title = s(cfg.PANEL_TITLE_SIZE)
        self.size_number = s(cfg.PANEL_NUMBER_SIZE)
        self.size_unit = s(cfg.PANEL_UNIT_SIZE)
        self.size_sub = s(cfg.PANEL_SUB_SIZE)
        self.size_graph_title = s(cfg.PANEL_GRAPH_TITLE_SIZE)
        self.size_axis_label = s(cfg.PANEL_AXIS_LABEL_SIZE)
        self.size_tick = s(cfg.PANEL_TICK_SIZE)
        self.size_knee_key = s(cfg.PANEL_KNEE_KEY_SIZE)
        self.size_knee_value = s(cfg.PANEL_KNEE_VALUE_SIZE)

        # ── horizontal extent ────────────────────────────────────────────────
        side_margin = s(cfg.PANEL_SIDE_MARGIN)
        self.y_label_dx = s(52)
        gutter = side_margin + self.y_label_dx + self.size_axis_label // 2
        avail_w = (width - side_margin) - gutter
        draw_x0, draw_x1 = side_margin, width - side_margin

        self.ankle_series, ankle_shape = self._ankle_series()

        # ── vertical layout: fixed elements first, the two drawings share the rest ──
        gap_number, gap_sub, gap_plot = s(14), s(10), s(14)
        gap_section = s(cfg.PANEL_SECTION_GAP)
        below_axis = self.size_tick + s(10) + self.size_axis_label
        knee_label_h = self.size_knee_value + s(9)
        credit = (s(cfg.ATTRIBUTION_SIZE) + 2 * s(cfg.ATTRIBUTION_MARGIN)) if cfg.ATTRIBUTION else s(14)
        margin_top, margin_bottom = s(cfg.PANEL_TOP_MARGIN), s(cfg.PANEL_BOTTOM_MARGIN)

        fixed = (self.size_title + gap_number + self.size_number + gap_sub + self.size_sub
                 + gap_section + self.size_graph_title + gap_plot + below_axis
                 + gap_section + self.size_graph_title + gap_plot
                 + gap_section + self.size_graph_title + gap_plot)
        room = height - credit - fixed - margin_top - margin_bottom

        graph_h = max(s(50), min(int(round(avail_w / max(float(cfg.PANEL_GRAPH_ASPECT), 0.2))),
                                 int(room * 0.5)))
        rest = room - graph_h
        span_x, span_y = ankle_shape
        ideal = int(round((draw_x1 - draw_x0) * span_y / max(span_x, 1e-9)))
        ankle_h = max(s(60), min(ideal, int(rest * cfg.PANEL_ANKLE_MAX_SHARE)))
        knee_h = max(s(60), rest - ankle_h)

        px0, px1 = gutter, gutter + avail_w
        y = margin_top
        self.y_title = y
        y += self.size_title + gap_number
        self.y_number = y + self.size_number // 2
        y += self.size_number + gap_sub
        self.y_sub = y
        y += self.size_sub + gap_section

        self.y_graph_title = y
        y += self.size_graph_title + gap_plot
        self.plot = Box(px0, px1, y, y + graph_h)
        y += graph_h + below_axis + gap_section

        self.y_knee_title = y
        y += self.size_graph_title + gap_plot
        self.knee = Box(draw_x0, draw_x1, y, y + knee_h)
        self.knee_label_h = knee_label_h
        y += knee_h + gap_section

        self.y_ankle_title = y
        y += self.size_graph_title + gap_plot
        self.ankle = Box(draw_x0, draw_x1, y, y + ankle_h)

        # ── data ranges ──────────────────────────────────────────────────────
        self.plot.x0, self.plot.x1 = 0.0, analysis.n_frames / self.fps
        self.mean_samples = analysis.mean_samples()
        finite = np.array([v for _, v in self.mean_samples if np.isfinite(v)])
        if finite.size:
            span = max(float(np.ptp(finite)) * 1.25, float(cfg.PANEL_Y_MIN_SPAN))
            mid = 0.5 * (finite.min() + finite.max())
            lo, hi = mid - span / 2, mid + span / 2
            step = nice_step(span, 4, (5, 10, 20, 25, 50))
        else:
            step, lo, hi = 20.0, 140.0, 200.0
        self.plot.y0, self.plot.y1 = float(lo), float(hi)
        self.y_step = step

        self.knee_track = self._knee_track()
        self.ankle_track = self._ankle_window()
        self.background = self._background()

    def _s(self, px: float) -> int:
        return max(1, int(round(px * self.u)))

    # ── precomputed per-frame state ──────────────────────────────────────────
    def _knee_track(self) -> dict:
        """Per-frame knee geometry, eased so it drifts toward each new average."""
        feet = [f for f in FEET if f in self.cfg.PANEL_KNEE_FEET] or list(FEET)
        alpha = float(self.cfg.PANEL_KNEE_EASING)
        keys = ("thigh", "shank", "thigh_len", "shank_len", "flexion", "flexion_sd", "thigh_sd")
        track = {f: {k: np.full(self.a.n_frames, np.nan) for k in keys} for f in feet}
        for foot in feet:
            state = None
            for i in range(self.a.n_frames):
                shape = self.a.knee_shape(foot, until=i / self.fps, window=self.cfg.PANEL_KNEE_WINDOW)
                if shape is None or not np.isfinite(shape.flexion):
                    continue
                target = {"thigh": shape.thigh_angle, "shank": shape.shank_angle,
                          "thigh_len": shape.thigh_length, "shank_len": shape.shank_length,
                          "flexion": shape.flexion, "flexion_sd": shape.flexion_sd,
                          "thigh_sd": shape.thigh_sd}
                if state is None:
                    state = dict(target)
                else:
                    nxt = {}
                    for k in keys:
                        goal, prev = target[k], state[k]
                        nxt[k] = goal if not (np.isfinite(goal) and np.isfinite(prev)) \
                            else prev + alpha * (goal - prev)
                    state = nxt
                for k in keys:
                    track[foot][k][i] = state[k]
            shape = self.a.knee_shape(foot)
            track[foot]["bend"] = shape.bend_sign if shape is not None else 1.0
        return track

    def _ankle_series(self):
        """Both ankles per frame in leg lengths (y up), and the padded span of the space they use."""
        cfg = self.cfg
        feet = [f for f in FEET if f in cfg.PANEL_ANKLE_FEET] or list(FEET)
        mode = str(cfg.PANEL_ANKLE_FRAME).lower()
        if mode == "auto":
            mode = "hip" if cfg.FOOT_REFERENCE == "hip" else "image"
        self.ankle_mode = mode
        leg = self.a.leg_length or 1e-6
        hip_x = hip_y = 0.0
        if mode == "hip":
            hip_x = np.nanmean(np.vstack([self.a.limb[f]["hip"][0] for f in FEET]), axis=0)
            hip_y = np.nanmean(np.vstack([self.a.limb[f]["hip"][1] for f in FEET]), axis=0)
        track = {}
        for foot in feet:
            ax, ay = self.a.limb[foot]["ankle"]
            track[foot] = ((np.asarray(ax, float) - hip_x) / leg, -(np.asarray(ay, float) - hip_y) / leg)

        self.ankle_ground = (-float(np.mean(list(self.a.ground.values()))) / leg
                             if mode == "image" and cfg.PANEL_ANKLE_GROUND else None)
        xs = np.concatenate([t[0] for t in track.values()]) if track else np.zeros(0)
        ys = np.concatenate([t[1] for t in track.values()]) if track else np.zeros(0)
        xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
        if not xs.size or not ys.size:
            xs = ys = np.array([0.0, 1.0])
        lo_y, hi_y = float(ys.min()), float(ys.max())
        if self.ankle_ground is not None:
            lo_y, hi_y = min(lo_y, self.ankle_ground), max(hi_y, self.ankle_ground)
        pad = 1.0 + 2 * float(cfg.PANEL_ANKLE_PAD)
        self._ankle_bounds = (float(xs.min()), float(xs.max()), lo_y, hi_y)

        # A lightly eased copy of the path for the trail to run through; the dot
        # itself is always the exact reading. Causal.
        alpha = float(cfg.PANEL_ANKLE_PATH_EASE)
        self.ankle_track_eased = {}
        for foot, (fx, fy) in track.items():
            ex, ey = np.array(fx, dtype=float), np.array(fy, dtype=float)
            for i in range(1, len(ex)):
                if not (np.isfinite(ex[i]) and np.isfinite(ey[i])):
                    continue
                if np.isfinite(ex[i - 1]) and np.isfinite(ey[i - 1]):
                    ex[i] = alpha * ex[i] + (1 - alpha) * ex[i - 1]
                    ey[i] = alpha * ey[i] + (1 - alpha) * ey[i - 1]
            self.ankle_track_eased[foot] = (ex, ey)
        return track, (max(float(xs.max() - xs.min()), 1e-6) * pad, max(hi_y - lo_y, 1e-6) * pad)

    def _ankle_window(self) -> dict:
        """Fit the ankle box to the path at equal scale, and build the trail machinery."""
        cfg, box = self.cfg, self.ankle
        x_lo, x_hi, y_lo, y_hi = self._ankle_bounds
        pad = 1.0 + 2 * float(cfg.PANEL_ANKLE_PAD)
        x_span, y_span = max(x_hi - x_lo, 1e-6) * pad, max(y_hi - y_lo, 1e-6) * pad
        px_per_unit = min(box.width / x_span, box.height / y_span)
        half_x, half_y = box.width / (2 * px_per_unit), box.height / (2 * px_per_unit)
        x_mid, y_mid = 0.5 * (x_hi + x_lo), 0.5 * (y_hi + y_lo)
        box.x0, box.x1 = x_mid - half_x, x_mid + half_x
        box.y0, box.y1 = y_mid - half_y, y_mid + half_y

        self.ankle_dense_step = K = max(1, int(cfg.PANEL_ANKLE_TRAIL_SMOOTH))
        self.ankle_dense = {}
        for foot, (ex, ey) in self.ankle_track_eased.items():
            pts = list(zip(ex.tolist(), ey.tolist()))
            finite = all(np.isfinite(x) and np.isfinite(y) for x, y in pts)
            self.ankle_dense[foot] = catmull_rom(pts, K) if finite and K > 1 else pts

        self.ankle_comet = Comet(
            width=max(2, self._s(cfg.PANEL_ANKLE_TRAIL_WIDTH)), taper=cfg.PANEL_ANKLE_TRAIL_TAPER,
            opacity=cfg.PANEL_ANKLE_TRAIL_OPACITY, fade=cfg.PANEL_ANKLE_TRAIL_FADE,
            glow=cfg.PANEL_ANKLE_GLOW, glow_opacity=cfg.PANEL_ANKLE_GLOW_OPACITY,
            softness=cfg.PANEL_ANKLE_SOFTNESS,
            wind=wind_vector(cfg.TRAIL_WIND, self.a.heading, speed=cfg.PANEL_ANKLE_WIND_SPEED * self.u,
                             fps=self.fps),
            smooth=K,
        )
        self.ankle_trail = max(2, round(cfg.PANEL_ANKLE_TRAIL_SECONDS * self.fps))
        self.ankle_settle_frames = max(1, round(cfg.PANEL_ANKLE_PATH_SETTLE_SECONDS * self.fps))

        halflife = max(float(cfg.PANEL_ANKLE_MEMORY_HALFLIFE_SECONDS), 1e-3)
        self._memory_decay = 0.5 ** (1.0 / max(halflife * self.fps, 1e-6))
        self._visited_upto = -1
        self._visited_pen = np.zeros((box.height, box.width), np.uint8)
        self._heat = None
        return self.ankle_series

    # ── the static background ────────────────────────────────────────────────
    def _background(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), BG, np.uint8)
        f, cfg, box, s = self.font, self.cfg, self.plot, self._s

        textmod.draw(img, cfg.PANEL_GRAPH_TITLE, f, size=self.size_graph_title,
                     xy=(self.w // 2, self.y_graph_title), color=DIM, anchor="ct")
        first = np.ceil(box.y0 / self.y_step) * self.y_step
        for value in np.arange(first, box.y1 + 1e-9, self.y_step):
            textmod.draw(img, f"{int(round(value))}", f, size=self.size_tick,
                         xy=(box.px0 - s(9), box.sy(value)), color=MUTED, anchor="rm")
        cv2.line(img, (box.px0, box.py0), (box.px0, box.py1), AXIS, 1, cv2.LINE_AA)
        cv2.line(img, (box.px0, box.py1), (box.px1, box.py1), AXIS, 1, cv2.LINE_AA)
        x_step = nice_step(box.x1 - box.x0, 7, (1, 2, 5, 10, 15, 30, 60))
        for sec in np.arange(0, box.x1 + 1e-9, x_step):
            textmod.draw(img, f"{int(sec)}", f, size=self.size_tick,
                         xy=(box.sx(sec), box.py1 + s(8)), color=MUTED, anchor="ct")
        textmod.draw(img, cfg.PANEL_X_LABEL, f, size=self.size_axis_label,
                     xy=(self.w // 2, box.py1 + s(10) + self.size_tick), color=DIM, anchor="ct")
        textmod.draw(img, cfg.PANEL_Y_LABEL, f, size=self.size_axis_label,
                     xy=(box.px0 - self.y_label_dx, (box.py0 + box.py1) // 2), color=DIM,
                     anchor="cm", rotate=90)

        textmod.draw(img, cfg.PANEL_KNEE_TITLE, f, size=self.size_graph_title,
                     xy=(self.w // 2, self.y_knee_title), color=DIM, anchor="ct")
        feet = [x for x in FEET if x in cfg.PANEL_KNEE_FEET] or list(FEET)
        if len(feet) > 1:
            mid_x = (self.knee.px0 + self.knee.px1) // 2
            for y in range(self.knee.py0, self.knee.py1 - self.knee_label_h, s(10)):
                cv2.line(img, (mid_x, y), (mid_x, min(y + s(5), self.knee.py1)), GRID, 1, cv2.LINE_AA)

        textmod.draw(img, cfg.PANEL_ANKLE_TITLE, f, size=self.size_graph_title,
                     xy=(self.w // 2, self.y_ankle_title), color=DIM, anchor="ct")
        box = self.ankle
        legend_row = self.size_tick + s(7)
        ly = box.py0 + s(8) + self.size_tick // 2
        for i, foot in enumerate(("left", "right")):
            name = cfg.PANEL_KNEE_LEFT if foot == "left" else cfg.PANEL_KNEE_RIGHT
            ty, tx = ly + i * legend_row, box.px1 - s(6)
            tw = textmod.measure(name, f, self.size_tick)[0]
            textmod.draw(img, name, f, size=self.size_tick, xy=(tx, ty), color=MUTED, anchor="rm")
            cv2.circle(img, (tx - tw - s(6) - s(3), ty), s(3), FOOT_COLORS[foot], -1, cv2.LINE_AA)

        if self.ankle_ground is not None:
            gy = box.sy(self.ankle_ground)
            for x in range(box.px0, box.px1, s(12)):
                cv2.line(img, (x, gy), (min(x + s(5), box.px1), gy), AXIS, 1, cv2.LINE_AA)
        elif self.ankle_mode == "hip" and box.x0 < 0.0 < box.x1:
            hx = box.sx(0.0)
            for y in range(box.py0, box.py1, s(12)):
                cv2.line(img, (hx, y), (hx, min(y + s(5), box.py1)), AXIS, 1, cv2.LINE_AA)
            textmod.draw(img, cfg.PANEL_ANKLE_HIP_LABEL, f, size=self.size_tick,
                         xy=(hx + s(6), box.py0), color=MUTED, anchor="lt")

        if cfg.ATTRIBUTION:
            textmod.draw(img, cfg.ATTRIBUTION, f, size=s(cfg.ATTRIBUTION_SIZE),
                         xy=(self.w - s(cfg.ATTRIBUTION_MARGIN), self.h - s(cfg.ATTRIBUTION_MARGIN)),
                         color=INK, anchor="rb", opacity=cfg.ATTRIBUTION_OPACITY)
        return img

    # ── per-frame content ────────────────────────────────────────────────────
    def _headline(self, img, frame: int) -> None:
        f, cfg = self.font, self.cfg
        textmod.draw(img, cfg.PANEL_TITLE, f, size=self.size_title, xy=(self.w // 2, self.y_title),
                     color=INK, anchor="ct")
        live = self.a.cadence_at(frame)
        defined = bool(np.isfinite(live))
        number = f"{live:.0f}" if defined else "—"
        unit = cfg.PANEL_UNIT if defined else ""
        wide = textmod.measure(number, f, self.size_number)[0]
        unit_w = textmod.measure(unit, f, self.size_unit)[0] if unit else 0
        pad = self._s(9) if unit else 0
        left = self.w // 2 - (wide + pad + unit_w) // 2
        textmod.draw(img, number, f, size=self.size_number, xy=(left, self.y_number), color=INK, anchor="lm")
        if unit:
            textmod.draw(img, unit, f, size=self.size_unit,
                         xy=(left + wide + pad, self.y_number + self.size_number // 2), color=MUTED, anchor="lb")
        landed = self.a.steps_by(frame)
        template = cfg.PANEL_STEPS_LABEL_ONE if landed == 1 else cfg.PANEL_STEPS_LABEL
        textmod.draw(img, template.format(steps=landed), f, size=self.size_sub,
                     xy=(self.w // 2, self.y_sub), color=MUTED, anchor="ct")

    def _trace(self, img, samples, frame: int, color, *, thickness: int) -> None:
        """One series as a smooth curve up to the current frame, held flat to the playhead."""
        box = self.plot
        now = frame / self.fps
        visible = [(t, v) for t, v in samples if t <= now and np.isfinite(v)]
        if not visible:
            return
        pixels = [(box.sx(t), box.sy(v)) for t, v in
                  catmull_rom(visible, max(2, int(self.cfg.PANEL_TRACE_SEGMENTS)))]
        for a, b in zip(pixels[:-1], pixels[1:]):
            cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
        end = pixels[-1]
        live = (box.sx(now), end[1])
        if live[0] > end[0]:
            cv2.line(img, end, live, color, thickness, cv2.LINE_AA)
        cv2.circle(img, live, self._s(5), color, -1, cv2.LINE_AA)
        cv2.circle(img, live, self._s(5), BG, self._s(1), cv2.LINE_AA)

    def _graph(self, img, frame: int) -> None:
        box = self.plot
        x = box.sx(frame / self.fps)
        cv2.line(img, (x, box.py0), (x, box.py1), (86, 86, 94), 1, cv2.LINE_AA)
        self._trace(img, self.mean_samples, frame, MEAN_COLOR,
                    thickness=self._s(self.cfg.PANEL_TRACE_THICKNESS))

    def _knee(self, img, frame: int) -> None:
        """The mean limb at foot strike, its spread shaded as a fan around it."""
        cfg, box = self.cfg, self.knee
        feet = [f for f in FEET if f in cfg.PANEL_KNEE_FEET] or list(FEET)
        share = box.width // len(feet)
        draw_h = box.height - self.knee_label_h

        for i, foot in enumerate(feet):
            cx = box.px0 + share // 2 + i * share
            track = self.knee_track.get(foot)
            if track is None or not np.isfinite(track["flexion"][frame]):
                continue
            g = {k: float(track[k][frame]) for k in
                 ("thigh", "shank", "thigh_len", "shank_len", "flexion", "flexion_sd", "thigh_sd")}
            bend = float(track["bend"])
            color = FOOT_COLORS[foot]
            total = g["thigh_len"] + g["shank_len"]
            scale = (draw_h * cfg.PANEL_KNEE_FILL) / max(total, 1e-6)

            def limb(thigh, shank):
                knee_off = np.array([np.cos(thigh), np.sin(thigh)]) * g["thigh_len"] * scale
                ankle_off = knee_off + np.array([np.cos(shank), np.sin(shank)]) * g["shank_len"] * scale
                return knee_off, ankle_off

            knee_off, ankle_off = limb(g["thigh"], g["shank"])
            pts = np.vstack([[0.0, 0.0], knee_off, ankle_off])
            centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
            origin = np.array([cx, box.py0 + draw_h / 2]) - centre

            sd_flex = g["flexion_sd"] if np.isfinite(g["flexion_sd"]) else 0.0
            spread = max(1, int(cfg.PANEL_KNEE_ENVELOPE_STEPS))
            band = self._s(cfg.PANEL_KNEE_THICKNESS)
            y0, y1 = box.py0, box.py0 + draw_h
            x0, x1 = box.px0 + i * share, box.px0 + (i + 1) * share
            coverage = np.zeros((y1 - y0, x1 - x0), np.float32)
            layer = np.zeros_like(coverage, np.uint8)
            shifted = origin - np.array([x0, y0])
            for k in np.linspace(-cfg.PANEL_KNEE_FAN_SD, cfg.PANEL_KNEE_FAN_SD, spread):
                thigh_k = g["thigh"] + k * g["thigh_sd"]
                flex_k = g["flexion"] + k * sd_flex
                k_off, a_off = limb(thigh_k, thigh_k + bend * np.radians(flex_k))
                layer[:] = 0
                pt = [tuple(np.round(shifted + o).astype(int)) for o in ([0.0, 0.0], k_off, a_off)]
                cv2.line(layer, pt[0], pt[1], 1, band, cv2.LINE_AA)
                cv2.line(layer, pt[1], pt[2], 1, band, cv2.LINE_AA)
                coverage += layer
            alpha = np.clip(coverage * cfg.PANEL_KNEE_FAN_STEP_OPACITY, 0.0, cfg.PANEL_KNEE_FAN_OPACITY)[..., None]
            region = img[y0:y1, x0:x1].astype(np.float32)
            img[y0:y1, x0:x1] = (region * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)

            hip = tuple(np.round(origin).astype(int))
            kn = tuple(np.round(origin + knee_off).astype(int))
            an = tuple(np.round(origin + ankle_off).astype(int))
            cv2.line(img, hip, kn, color, band, cv2.LINE_AA)
            cv2.line(img, kn, an, color, band, cv2.LINE_AA)
            for point, r in ((hip, 5), (an, 5), (kn, 7)):
                cv2.circle(img, point, self._s(r), color, -1, cv2.LINE_AA)
                cv2.circle(img, point, self._s(r), BG, self._s(1), cv2.LINE_AA)

            # The digits hold on a slow grid; the shape keeps easing every frame.
            hold = max(1, int(round(cfg.PANEL_KNEE_LABEL_HOLD_SECONDS * self.fps)))
            at = (frame // hold) * hold
            if not np.isfinite(track["flexion"][at]):
                at = frame
            name = cfg.PANEL_KNEE_LEFT if foot == "left" else cfg.PANEL_KNEE_RIGHT
            places = int(cfg.PANEL_KNEE_DECIMALS)
            sd_at = float(track["flexion_sd"][at])
            value = (cfg.PANEL_KNEE_FORMAT if np.isfinite(sd_at) else cfg.PANEL_KNEE_FORMAT_NO_SD).format(
                deg=f"{float(track['flexion'][at]):.{places}f}", sd=f"{sd_at:.{places}f}")
            label_y = box.py1 - self.knee_label_h // 2
            name_w = textmod.measure(name, self.font, self.size_knee_key)[0]
            value_w = textmod.measure(value, self.font, self.size_knee_value)[0]
            gap = self._s(9)
            x = cx - (name_w + gap + value_w) // 2
            textmod.draw(img, name, self.font, size=self.size_knee_key, xy=(x, label_y), color=color, anchor="lm")
            textmod.draw(img, value, self.font, size=self.size_knee_value, xy=(x + name_w + gap, label_y),
                         color=INK, anchor="lm")

    # ── the ankles ───────────────────────────────────────────────────────────
    def _ankle_point(self, foot: str, i: int, eased: bool = False):
        xs, ys = (self.ankle_track_eased if eased else self.ankle_track)[foot]
        if i < 0 or i >= len(xs) or not (np.isfinite(xs[i]) and np.isfinite(ys[i])):
            return None
        return self.ankle.sx(xs[i]), self.ankle.sy(ys[i])

    def _dense_span(self, foot: str, i: int):
        """Box-relative points covering real frame (i-1, i] off the one whole-clip curve."""
        box, dense, step = self.ankle, self.ankle_dense.get(foot), self.ankle_dense_step
        if not dense:
            return None
        lo, hi = max(0, (i - 1) * step), min(len(dense) - 1, i * step)
        if hi <= lo:
            return None
        return [(int(round(box.sx(x) - box.px0)), int(round(box.sy(y) - box.py0))) for x, y in dense[lo:hi + 1]]

    def _visited_cloud(self, img, frame: int) -> None:
        """Where each ankle has recently been, as a coverage buffer that fades with a half-life."""
        opacity = float(self.cfg.PANEL_ANKLE_MEMORY_OPACITY)
        if opacity <= 0:
            return
        box = self.ankle
        if self._heat is None or frame < self._visited_upto:
            self._heat = {foot: np.zeros((box.height, box.width), np.float32) for foot in self.ankle_track}
            self._visited_upto = -1
        pen = self._visited_pen
        width = max(1, self._s(self.cfg.PANEL_ANKLE_MEMORY_WIDTH))
        for i in range(self._visited_upto + 1, frame + 1):
            for foot, heat in self._heat.items():
                heat *= self._memory_decay
                poly = self._dense_span(foot, i)
                if poly is None:
                    continue
                pen[:] = 0
                for p0, p1 in zip(poly[:-1], poly[1:]):
                    cv2.line(pen, p0, p1, 255, width, cv2.LINE_AA)
                np.maximum(heat, pen.astype(np.float32) * (1.0 / 255.0), out=heat)
        self._visited_upto = frame
        region = self.background[box.py0:box.py1, box.px0:box.px1].astype(np.float32)
        for foot, heat in self._heat.items():
            alpha = (heat * opacity)[..., None]
            region = region * (1.0 - alpha) + np.array(FOOT_COLORS[foot], np.float32) * alpha
        img[box.py0:box.py1, box.px0:box.px1] = region.astype(np.uint8)

    def _ankles(self, img, frame: int) -> None:
        box = self.ankle
        self._visited_cloud(img, frame)
        window = img[box.py0:box.py1, box.px0:box.px1]
        settle = max(1, self.ankle_settle_frames)
        paths = []
        for foot in self.ankle_track:
            points = []
            for i in range(max(0, frame - self.ankle_trail + 1), frame + 1):
                age = frame - i
                exact, eased = self._ankle_point(foot, i), self._ankle_point(foot, i, eased=True)
                if exact is None or eased is None:
                    point = None
                elif age >= settle:
                    point = eased
                else:
                    w = 0.5 * (1.0 + np.cos(np.pi * age / settle))
                    point = (exact[0] * w + eased[0] * (1.0 - w), exact[1] * w + eased[1] * (1.0 - w))
                points.append(None if point is None else (point[0] - box.px0, point[1] - box.py0))
            paths.append((points, FOOT_COLORS[foot]))
        self.ankle_comet.draw(window, paths)
        for foot in self.ankle_track:
            point = self._ankle_point(foot, frame)
            if point is None:
                continue
            r = self._s(self.cfg.PANEL_ANKLE_DOT)
            cv2.circle(img, point, r, FOOT_COLORS[foot], -1, cv2.LINE_AA)
            cv2.circle(img, point, r, BG, self._s(1), cv2.LINE_AA)

    def draw(self, frame_index: int) -> np.ndarray:
        img = self.background.copy()
        frame = int(np.clip(frame_index, 0, max(self.a.n_frames - 1, 0)))
        self._headline(img, frame)
        self._graph(img, frame)
        self._knee(img, frame)
        self._ankles(img, frame)
        return img
