"""Ankle clearance, foot strikes, cadence, and the knee at contact.

One signal per leg: how far that ankle sits above its own stance level, in units
of the runner's leg length. It is near zero while the foot is planted and peaks
at mid-swing, so a run reads as two interleaved waves.

The ankle, because COCO-17 has no foot: the ankles are the lowest two of the 17
keypoints. So the zero is a *calibrated* stance level per leg, not the floor,
and a "foot strike" is the ankle crossing down through a threshold a little
above that level.

    1. ankle height above its own stance level, in leg lengths
    2. a strike is the downward crossing of CONTACT_FRACTION of a typical swing
    3. intervals are timed on mid-stance (robust to the threshold), and
    4. cadence is 120 / mean(stride), stride = same foot to the same foot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .skeleton import KPT_INDEX

FEET = ("left", "right")


# ── signal helpers ───────────────────────────────────────────────────────────
def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k < 3:
        return x
    k += (k + 1) % 2
    pad = np.pad(x, k // 2, mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(x))])


def box_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average; edge padded so it does not shift peak times."""
    if k < 2:
        return x
    k += (k + 1) % 2
    pad = np.pad(x, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def interp_nan(x: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x)
    if valid.all() or not valid.any():
        return x
    idx = np.arange(len(x))
    out = x.copy()
    out[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return out


def joint_series(frames: list[dict], name: str) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of one joint on the first person, per returned frame. (0, 0) -> NaN."""
    j = KPT_INDEX[name]
    xs = np.full(len(frames), np.nan)
    ys = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        persons = frame.get("persons") or []
        if not persons:
            continue
        kpts = persons[0].get("kpts_xy") or []
        if j >= len(kpts) or tuple(kpts[j]) == (0, 0):
            continue
        xs[i], ys[i] = float(kpts[j][0]), float(kpts[j][1])
    return xs, ys


def densify(values: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    """Put a per-returned-frame series onto a per-source-frame grid."""
    out = np.full(n, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    out[:] = np.interp(np.arange(n), indices[finite], values[finite], left=np.nan, right=np.nan)
    return out


def robust_peak(signal: np.ndarray) -> float:
    """Typical swing height: the median of the local peaks, not the maximum."""
    finite = signal[np.isfinite(signal)]
    if not finite.size:
        return 0.0
    top = float(np.max(finite))
    if top <= 0:
        return 0.0
    peaks = [signal[i] for i in range(1, len(signal) - 1)
             if np.isfinite(signal[i - 1:i + 2]).all()
             and signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]
             and signal[i] > 0.4 * top]
    if len(peaks) >= 3:
        return float(np.median(peaks))
    return float(np.percentile(finite, 95))


def runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs as (start, end_exclusive, label)."""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i, int(labels[start])))
            start = i
    return out


def cross_time(signal, before: int, after: int, level: float, fps: float) -> float:
    """Sub-frame time at which the signal crosses *level* between two frames."""
    v0, v1 = signal[before], signal[after]
    if not (np.isfinite(v0) and np.isfinite(v1)) or v1 == v0:
        return after / fps
    frac = float(np.clip((level - v0) / (v1 - v0), 0.0, 1.0))
    return (before + frac) / fps


def joint_xy(frames, name, indices, n):
    """One joint on the per-source-frame grid, gaps interpolated, plus an observed mask."""
    xs, ys = joint_series(frames, name)
    dx, dy = densify(xs, indices, n), densify(ys, indices, n)
    observed = np.isfinite(dx) & np.isfinite(dy)
    return interp_nan(dx), interp_nan(dy), observed


def leg_length(limb: dict) -> float:
    """Median thigh + median shank, averaged over both legs, in frame units."""
    per_leg = []
    for joints in limb.values():
        total = 0.0
        for a, b in (("hip", "knee"), ("knee", "ankle")):
            d = np.hypot(joints[b][0] - joints[a][0], joints[b][1] - joints[a][1])
            d = d[np.isfinite(d)]
            if not d.size:
                total = np.nan
                break
            total += float(np.median(d))
        if np.isfinite(total):
            per_leg.append(total)
    return float(np.mean(per_leg)) if per_leg else 0.0


def knee_flexion(hip, knee, ankle) -> np.ndarray:
    """Knee flexion in degrees per frame; 0 is a straight leg. Projected, not anatomical."""
    v1 = np.vstack([hip[0] - knee[0], hip[1] - knee[1]])
    v2 = np.vstack([ankle[0] - knee[0], ankle[1] - knee[1]])
    n1, n2 = np.hypot(*v1), np.hypot(*v2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (v1 * v2).sum(axis=0) / (n1 * n2)
    return 180.0 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def heading(limb: dict, planted: dict, nose_x: np.ndarray) -> tuple[float, float]:
    """Direction of travel in image x (+1 towards +x), and the runner's facing.

    Measured off the planted foot: relative to the hips it travels backwards
    through every stance, on a belt or on a road. Facing (nose vs mid-hip) is
    the cross-check and the fallback.
    """
    drift = []
    for foot, joints in limb.items():
        rel = np.asarray(joints["ankle"][0]) - np.asarray(joints["hip"][0])
        stance = np.asarray(planted[foot])
        keep = stance[1:] & stance[:-1]
        drift += [v for v in np.diff(rel)[keep] if np.isfinite(v)]
    head = -float(np.sign(np.mean(drift))) if drift else 0.0
    hips = [np.asarray(j["hip"][0]) for j in limb.values()]
    mid_hip_x = np.nanmean(np.vstack(hips), axis=0) if hips else np.zeros(0)
    lead = np.asarray(nose_x) - mid_hip_x
    lead = lead[np.isfinite(lead)]
    facing = float(np.sign(np.mean(lead))) if lead.size else 0.0
    return head or facing, facing


def vertex_min(series: np.ndarray, centre: int, reach: int) -> tuple[float, float]:
    """Sub-frame minimum near *centre*, as (frame, value).

    The lowest frame in the window is found first, then a least-squares parabola
    is fitted through it and its two neighbours either side and the vertex taken.
    A parabola over the whole window would read high whenever the curve is a
    sharp V (the knee extends fast into contact and flexes fast out of it);
    fitting only the bottom keeps the noise averaging without that bias.
    """
    lo, hi = max(0, centre - reach), min(len(series), centre + reach + 1)
    finite = [i for i in range(lo, hi) if np.isfinite(series[i])]
    if not finite:
        return float(centre), float("nan")
    best = min(finite, key=lambda i: series[i])
    f_lo, f_hi = max(0, best - 2), min(len(series), best + 3)
    window = series[f_lo:f_hi]
    if len(window) < 3 or not np.isfinite(window).all():
        return float(best), float(series[best])
    t = np.arange(f_lo, f_hi, dtype=float) - best
    a2, a1, a0 = np.polyfit(t, window, 2)
    if a2 <= 0:
        return float(best), float(series[best])
    peak = float(np.clip(-a1 / (2.0 * a2), t[0], t[-1]))
    return best + peak, float(a2 * peak ** 2 + a1 * peak + a0)


def segment_angle(a, b) -> np.ndarray:
    """Direction from a to b in radians, image coordinates (y down)."""
    return np.arctan2(b[1] - a[1], b[0] - a[0])


def circ_mean(angles: np.ndarray) -> float:
    finite = angles[np.isfinite(angles)]
    if not finite.size:
        return float("nan")
    return float(np.arctan2(np.mean(np.sin(finite)), np.mean(np.cos(finite))))


def circ_sd(angles: np.ndarray) -> float:
    finite = angles[np.isfinite(angles)]
    if finite.size < 2:
        return 0.0
    r = np.hypot(np.mean(np.cos(finite)), np.mean(np.sin(finite)))
    return float(np.sqrt(-2.0 * np.log(max(r, 1e-12))))


def _round(value, places: int):
    """JSON has no NaN; None is what a missing measurement is."""
    return None if value is None or not np.isfinite(value) else round(float(value), places)


# ── data ─────────────────────────────────────────────────────────────────────
@dataclass
class KneeShape:
    """Mean hip-knee-ankle geometry at foot strike for one leg.

    Held as two segment directions plus two lengths, so the drawn limb keeps
    its size and all the variation lands in the angles.
    """

    foot: str
    n: int
    thigh_angle: float
    shank_angle: float
    thigh_length: float
    shank_length: float
    flexion: float
    flexion_sd: float
    thigh_sd: float = 0.0
    bend_sign: float = 1.0

    def as_dict(self) -> dict:
        return {
            "foot": self.foot, "strikes_averaged": self.n,
            "flexion_deg": _round(self.flexion, 2), "flexion_sd_deg": _round(self.flexion_sd, 2),
            "thigh_deg": _round(np.degrees(self.thigh_angle), 2),
            "shank_deg": _round(np.degrees(self.shank_angle), 2),
            "thigh_deg_sd": _round(np.degrees(self.thigh_sd), 2),
            "thigh_length_legs": _round(self.thigh_length, 4),
            "shank_length_legs": _round(self.shank_length, 4),
        }


@dataclass
class Step:
    """One foot strike and the swing before it. Intervals are timed on mid_stance."""

    number: int
    foot: str
    time: float
    mid_stance: float
    frame: int
    peak_time: float
    peak_clearance: float
    contact_seconds: float
    knee_frame: int = 0
    knee_time: float = 0.0
    knee_offset: float = 0.0
    knee_flexion: float = float("nan")
    interval: float = float("nan")
    stride: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "number": self.number, "foot": self.foot,
            "time": round(self.time, 4), "mid_stance": _round(self.mid_stance, 4),
            "peak_time": round(self.peak_time, 4), "peak_clearance": _round(self.peak_clearance, 4),
            "contact_seconds": _round(self.contact_seconds, 4),
            "knee_flexion_deg": _round(self.knee_flexion, 2),
            "knee_sample_offset_ms": round(self.knee_offset * 1000.0, 1),
            "step_interval": _round(self.interval, 4), "stride_seconds": _round(self.stride, 4),
        }


@dataclass
class GaitAnalysis:
    """Everything the panel, the plots and the report read from."""

    fps: float
    n_frames: int
    clearance: dict
    raw: dict
    amplitude: dict
    contact_level: float
    leg_length: float
    ground: dict
    hip_height: np.ndarray
    knee_flexion: dict = field(default_factory=dict)
    limb: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    cadence_live: np.ndarray = field(default_factory=lambda: np.zeros(0))
    foot_cadence_live: dict = field(default_factory=dict)
    warnings: dict = field(default_factory=dict)
    airborne: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    heading: float = 0.0
    facing: float = 0.0
    frame_aspect: float = 1.0

    @property
    def count(self) -> int:
        return len(self.steps)

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) / (self.fps or 30.0)

    @property
    def intervals(self) -> list[float]:
        return [s.interval for s in self.steps if np.isfinite(s.interval)]

    def steps_for(self, foot: str) -> list:
        return [s for s in self.steps if s.foot == foot]

    def mean_cadence_by(self, until=None) -> float:
        """Steps per minute over the strides completed by *until* (120 / mean stride)."""
        strides = [s.stride for s in self.steps if np.isfinite(s.stride)
                   and (until is None or s.mid_stance <= until)]
        if strides:
            return 120.0 / float(np.mean(strides))
        marks = [s.mid_stance for s in self.steps if np.isfinite(s.mid_stance)
                 and (until is None or s.mid_stance <= until)]
        if len(marks) < 2:
            return float("nan")
        return 60.0 * (len(marks) - 1) / (marks[-1] - marks[0])

    @property
    def mean_cadence(self) -> float:
        return self.mean_cadence_by(None)

    @property
    def mean_stride(self) -> float:
        strides = [s.stride for s in self.steps if np.isfinite(s.stride)]
        return float(np.mean(strides)) if strides else float("nan")

    @property
    def flight_ratio(self) -> float:
        return float(np.mean(self.airborne)) if len(self.airborne) else float("nan")

    def stride_spread(self) -> float:
        strides = [s.stride for s in self.steps if np.isfinite(s.stride)]
        return float(np.std(strides, ddof=1)) if len(strides) > 1 else float("nan")

    def step_spread(self) -> float:
        iv = self.intervals
        return float(np.std(iv, ddof=1)) if len(iv) > 1 else float("nan")

    def step_interval(self, foot: str) -> float:
        """Mean interval of the steps that land on this foot."""
        v = [s.interval for s in self.steps if s.foot == foot and np.isfinite(s.interval)]
        return float(np.mean(v)) if v else float("nan")

    def foot_stride_rate(self, foot: str) -> float:
        strides = [s.stride for s in self.steps_for(foot) if np.isfinite(s.stride)]
        return 120.0 / float(np.mean(strides)) if strides else float("nan")

    def phase_split(self) -> tuple[float, float]:
        left, right = self.step_interval("left"), self.step_interval("right")
        total = left + right
        if not np.isfinite(total) or total <= 0:
            return float("nan"), float("nan")
        return 100.0 * left / total, 100.0 * right / total

    def contact_mean(self, foot: str) -> float:
        v = [s.contact_seconds for s in self.steps_for(foot) if np.isfinite(s.contact_seconds)]
        return float(np.mean(v)) if v else float("nan")

    def balance(self) -> tuple[float, float]:
        left, right = self.contact_mean("left"), self.contact_mean("right")
        if not (np.isfinite(left) and np.isfinite(right)) or left + right <= 0:
            return float("nan"), float("nan")
        return 100.0 * left / (left + right), 100.0 * right / (left + right)

    def vertical_oscillation(self) -> float:
        """Median hip rise and fall within a stride, in leg lengths."""
        fps = self.fps or 30.0
        spans = []
        for foot in FEET:
            steps = self.steps_for(foot)
            for a, b in zip(steps, steps[1:]):
                if not (np.isfinite(a.mid_stance) and np.isfinite(b.mid_stance)):
                    continue
                i0, i1 = int(round(a.mid_stance * fps)), int(round(b.mid_stance * fps))
                w = self.hip_height[max(0, i0):min(self.n_frames, i1 + 1)]
                w = w[np.isfinite(w)]
                if w.size > 2:
                    spans.append(float(w.max() - w.min()))
        return float(np.median(spans)) if spans else float("nan")

    def peak_flexion(self, foot: str) -> float:
        series = self.knee_flexion.get(foot)
        if series is None or not len(series):
            return float("nan")
        finite = series[np.isfinite(series)]
        return float(np.percentile(finite, 98)) if finite.size else float("nan")

    def segment_ratio(self, foot: str) -> float:
        """Shank / thigh length at the strikes: a keypoint-quality check, ~1.0 in a real leg."""
        if foot not in self.limb:
            return float("nan")
        L = self.limb[foot]
        thigh = np.hypot(L["knee"][0] - L["hip"][0], L["knee"][1] - L["hip"][1])
        shank = np.hypot(L["ankle"][0] - L["knee"][0], L["ankle"][1] - L["knee"][1])
        at = np.array([s.knee_time * (self.fps or 30.0) for s in self.steps_for(foot)])
        if not at.size:
            return float("nan")
        ratio = np.interp(at, np.arange(self.n_frames), shank / thigh)
        ratio = ratio[np.isfinite(ratio)]
        return float(np.median(ratio)) if ratio.size else float("nan")

    def knee_shape(self, foot: str, until=None, window=None):
        """Mean limb geometry at this foot's strikes landed by *until* (last *window* strikes)."""
        if foot not in self.limb:
            return None
        cut = np.inf if until is None else float(until)
        landed = [s for s in self.steps_for(foot) if s.knee_time <= cut and np.isfinite(s.knee_flexion)]
        if window:
            landed = landed[-int(window):]
        if not landed:
            return None
        hip, knee, ankle = (self.limb[foot][j] for j in ("hip", "knee", "ankle"))
        grid = np.arange(self.n_frames)
        at = np.array([s.knee_time * (self.fps or 30.0) for s in landed])

        def pick(j):
            return np.interp(at, grid, j[0]), np.interp(at, grid, j[1])

        h, k, a = pick(hip), pick(knee), pick(ankle)
        thigh, shank = segment_angle(h, k), segment_angle(k, a)
        good = np.isfinite(thigh) & np.isfinite(shank)
        if not good.any():
            return None
        flex = np.array([s.knee_flexion for s in landed])
        flex = flex[np.isfinite(flex)]
        thigh_mean, shank_mean = circ_mean(thigh[good]), circ_mean(shank[good])
        flexion = float(np.mean(flex)) if flex.size else float("nan")
        bend = 1.0
        # The shank is reconstructed from the reported flexion, so the drawing and
        # the number beside it cannot disagree.
        if np.isfinite(flexion) and np.isfinite(thigh_mean):
            turn = np.radians(flexion)
            options = ((thigh_mean + turn, 1.0), (thigh_mean - turn, -1.0))
            shank_mean, bend = min(options, key=lambda c: abs(np.arctan2(
                np.sin(c[0] - shank_mean), np.cos(c[0] - shank_mean))))
        return KneeShape(
            foot=foot, n=int(good.sum()), thigh_angle=thigh_mean, shank_angle=shank_mean,
            thigh_length=float(np.median(np.hypot(k[0] - h[0], k[1] - h[1])[good])) / self.leg_length,
            shank_length=float(np.median(np.hypot(a[0] - k[0], a[1] - k[1])[good])) / self.leg_length,
            flexion=flexion,
            flexion_sd=float(np.std(flex, ddof=1)) if flex.size > 1 else float("nan"),
            thigh_sd=circ_sd(thigh[good]), bend_sign=bend,
        )

    def cadence_at(self, frame: int) -> float:
        if not len(self.cadence_live) or frame >= len(self.cadence_live):
            return float("nan")
        return float(self.cadence_live[frame])

    def mean_samples(self) -> list[tuple[float, float]]:
        return running_mean_samples(self.steps)

    def running_cadence_series(self, foot=None) -> np.ndarray:
        samples = (running_mean_samples(self.steps) if foot is None
                   else foot_window_samples(self.steps, 2, foot))
        return densify_samples(samples, self.fps, self.n_frames)

    def steps_by(self, frame: int) -> int:
        t = frame / (self.fps or 30.0)
        return sum(1 for s in self.steps if s.time <= t)

    def summary(self) -> dict:
        left_pct, right_pct = self.balance()
        iv = self.intervals
        shapes = {f: self.knee_shape(f) for f in FEET}
        return {
            "steps": self.count,
            "cadence_spm": _round(self.mean_cadence, 1),
            "mean_step_seconds": _round(float(np.mean(iv)) if iv else float("nan"), 4),
            "step_seconds_sd": _round(self.step_spread(), 4),
            "stride_seconds_sd": _round(self.stride_spread(), 4),
            "mean_stride_seconds": _round(self.mean_stride, 4),
            "step_seconds_per_foot": {f: _round(self.step_interval(f), 4) for f in FEET},
            "stride_rate_spm_per_foot": {f: _round(self.foot_stride_rate(f), 2) for f in FEET},
            "phase_split_percent": dict(zip(FEET, [_round(v, 1) for v in self.phase_split()])),
            "contact_seconds": {f: _round(self.contact_mean(f), 4) for f in FEET},
            "contact_balance_percent": {"left": _round(left_pct, 1), "right": _round(right_pct, 1)},
            "flight_ratio": _round(self.flight_ratio, 4),
            "heading_x": self.heading, "facing_x": self.facing,
            "peak_clearance_legs": {
                f: _round(float(np.mean([s.peak_clearance for s in self.steps_for(f)]))
                          if self.steps_for(f) else float("nan"), 4) for f in FEET},
            "vertical_oscillation_legs": _round(self.vertical_oscillation(), 4),
            "knee_flexion_at_strike_deg": {
                f: {"mean": _round(s.flexion, 2), "sd": _round(s.flexion_sd, 2), "strikes": s.n}
                for f, s in shapes.items() if s is not None},
            "knee_peak_flexion_deg": {f: _round(self.peak_flexion(f), 2) for f in FEET},
            "shank_thigh_ratio_at_contact": {f: _round(self.segment_ratio(f), 3) for f in FEET},
            "knee_shape_at_strike": {f: s.as_dict() for f, s in shapes.items() if s is not None},
            "steps_per_foot": {f: len(self.steps_for(f)) for f in FEET},
            "quality": self.warnings,
            "steps_detail": [s.as_dict() for s in self.steps],
        }

    def series(self) -> dict:
        return {
            "time": [round(float(v), 4) for v in self.times],
            "clearance": {f: [_round(float(v), 5) for v in self.clearance[f]] for f in FEET},
            "hip_height": [_round(float(v), 5) for v in self.hip_height],
            "knee_flexion_deg": {f: [_round(float(v), 3) for v in self.knee_flexion[f]]
                                 for f in FEET if f in self.knee_flexion},
            "cadence_spm": [_round(float(v), 2) for v in self.cadence_live],
            "airborne": [bool(v) for v in self.airborne],
        }


def signal_config(cfg) -> dict:
    return {
        "foot_reference": cfg.FOOT_REFERENCE, "per_foot_ground": cfg.PER_FOOT_GROUND,
        "ground_percentile": cfg.GROUND_PERCENTILE,
        "smooth_median_frames": cfg.SMOOTH_MEDIAN_FRAMES, "smooth_mean_frames": cfg.SMOOTH_MEAN_FRAMES,
        "swing_peak_fraction": cfg.SWING_PEAK_FRACTION, "contact_fraction": cfg.CONTACT_FRACTION,
        "min_contact_seconds": cfg.MIN_CONTACT_SECONDS, "cadence_smoothing_alpha": cfg.CADENCE_SMOOTHING,
    }


# ── the analysis ─────────────────────────────────────────────────────────────
def analyze(frames: list[dict], fps: float, n_source_frames: int, *, cfg, aspect: float) -> GaitAnalysis:
    """Build both clearance signals and find every foot strike.

    *aspect* is the frame width / height: the model normalizes x by the width
    and y by the height, so x is rescaled here to put both axes in one unit
    before any angle or length is measured.
    """
    indices = np.array([f["index"] for f in frames], dtype=float)
    n = max(n_source_frames, 1)

    raw_limb = {foot: {joint: joint_xy(frames, f"{foot}_{joint}", indices, n)
                       for joint in ("hip", "knee", "ankle")} for foot in FEET}
    scale_x = float(aspect)
    limb = {foot: {j: (v[0] * scale_x, v[1]) for j, v in joints.items()}
            for foot, joints in raw_limb.items()}
    observed = {foot: np.logical_and.reduce([raw_limb[foot][j][2] for j in ("hip", "knee", "ankle")])
                for foot in FEET}

    ankle_y = {foot: densify(joint_series(frames, f"{foot}_ankle")[1], indices, n) for foot in FEET}
    hip_y = {foot: densify(joint_series(frames, f"{foot}_hip")[1], indices, n) for foot in FEET}

    leg = leg_length(limb)
    if leg <= 1e-6:
        raise RuntimeError("Could not measure a leg length: hips or ankles were never detected.")

    finite_ankles = {foot: ankle_y[foot][np.isfinite(ankle_y[foot])] for foot in FEET}
    if any(not v.size for v in finite_ankles.values()):
        raise RuntimeError("One ankle was never detected; both are needed.")
    if cfg.PER_FOOT_GROUND:
        ground = {foot: float(np.percentile(finite_ankles[foot], cfg.GROUND_PERCENTILE)) for foot in FEET}
    else:
        shared = float(np.percentile(np.concatenate(list(finite_ankles.values())), cfg.GROUND_PERCENTILE))
        ground = {foot: shared for foot in FEET}

    mid_hip = np.nanmean(np.vstack([hip_y[foot] for foot in FEET]), axis=0)

    raw, clearance, amplitude = {}, {}, {}
    for foot in FEET:
        series = (ground[foot] - ankle_y[foot]) / leg
        if cfg.FOOT_REFERENCE == "hip":
            series = (mid_hip - ankle_y[foot]) / leg
            series = series - np.nanpercentile(series, 100 - cfg.GROUND_PERCENTILE)
        raw[foot] = series
        filled = interp_nan(series)
        smooth = box_filter(median_filter(filled, cfg.SMOOTH_MEDIAN_FRAMES), cfg.SMOOTH_MEAN_FRAMES)
        smooth[~np.isfinite(series) & ~np.isfinite(filled)] = np.nan
        clearance[foot] = smooth
        amplitude[foot] = robust_peak(smooth)

    positive = [v for v in amplitude.values() if v > 0]
    swing = min(positive) if positive else 0.0
    contact_level = cfg.CONTACT_FRACTION * swing
    high = cfg.SWING_PEAK_FRACTION * swing

    strikes: list[Step] = []
    for foot in FEET:
        strikes += find_strikes(clearance[foot], foot, fps, high=high, low=contact_level,
                                min_frames=max(1, int(round(cfg.MIN_CONTACT_SECONDS * fps))))
    strikes.sort(key=lambda s: s.time)

    last_by_foot: dict[str, float] = {}
    for i, step in enumerate(strikes):
        step.number = i + 1
        step.frame = int(min(n - 1, round(step.time * fps)))
        if not np.isfinite(step.mid_stance):
            continue
        if i and np.isfinite(strikes[i - 1].mid_stance):
            step.interval = step.mid_stance - strikes[i - 1].mid_stance
        if step.foot in last_by_foot:
            step.stride = step.mid_stance - last_by_foot[step.foot]
        last_by_foot[step.foot] = step.mid_stance

    planted = {foot: np.isfinite(clearance[foot]) & (clearance[foot] < contact_level) for foot in FEET}
    airborne = np.ones(n, dtype=bool)
    for foot in FEET:
        airborne &= ~planted[foot]

    # The nose is scaled like the limb, so the facing check compares like with like.
    head, facing = heading(limb, planted,
                           densify(joint_series(frames, "nose")[0], indices, n) * scale_x)

    flexion = {foot: knee_flexion(limb[foot]["hip"], limb[foot]["knee"], limb[foot]["ankle"])
               for foot in FEET}

    # The knee is read at its own most-extended point near the strike, where the
    # rate is zero, rather than on the steep loading slope just after contact.
    reach = max(1, int(round(cfg.KNEE_SEARCH_SECONDS * fps)))
    edge_hits = 0
    for step in strikes:
        series, seen = flexion[step.foot], observed[step.foot]
        step.knee_frame, step.knee_time, step.knee_offset = step.frame, step.frame / fps, 0.0
        lo, hi = max(0, step.frame - reach), min(len(series), step.frame + reach + 1)
        if not seen[lo:hi].all():
            continue
        at, value = vertex_min(series, step.frame, reach)
        if abs(at - step.frame) >= reach - 1e-6:
            edge_hits += 1
        step.knee_time = at / fps
        step.knee_frame = int(np.clip(round(at), 0, n - 1))
        step.knee_offset = (at - step.frame) / fps
        step.knee_flexion = value

    warnings = {
        "untimed_strikes": sum(1 for s in strikes if not np.isfinite(s.mid_stance)),
        "consecutive_same_foot_strikes": sum(1 for a, b in zip(strikes, strikes[1:]) if a.foot == b.foot),
        "knee_landmark_on_window_edge": edge_hits,
        "implausible_stance_flexion_frames": implausible_stance(flexion, strikes, fps, cfg.KNEE_STANCE_LIMIT_DEG),
        "frames_interpolated": {foot: int(n - int(observed[foot].sum())) for foot in FEET},
        "heading_disagrees_with_facing": int(bool(head and facing and head != facing)),
    }

    return GaitAnalysis(
        fps=fps, n_frames=n, clearance=clearance, raw=raw, amplitude=amplitude,
        contact_level=contact_level, leg_length=leg, ground=ground,
        hip_height=(float(np.mean(list(ground.values()))) - mid_hip) / leg,
        knee_flexion=flexion, limb=limb, steps=strikes,
        cadence_live=densify_samples(stride_ema_samples(strikes, cfg.CADENCE_SMOOTHING), fps, n),
        foot_cadence_live={foot: densify_samples(
            foot_window_samples(strikes, cfg.CADENCE_FOOT_WINDOW_STEPS, foot), fps, n) for foot in FEET},
        warnings=warnings, airborne=airborne, heading=head, facing=facing, frame_aspect=scale_x,
    )


def find_strikes(signal: np.ndarray, foot: str, fps: float, *, high: float, low: float,
                 min_frames: int) -> list[Step]:
    """Every foot strike on one clearance signal.

    Stance and swing are runs either side of the *low* threshold; a "swing" that
    never reaches *high* is stance wobble and is absorbed, and a "stance" shorter
    than *min_frames* is the signal clipping the line mid-swing.
    """
    if high <= 0 or len(signal) < 3:
        return []
    finite = np.isfinite(signal)
    labels = (finite & (signal < low)).astype(int)
    for _ in range(2):
        changed = False
        for start, end, label in runs(labels):
            window = signal[start:end]
            window = window[np.isfinite(window)]
            if label == 0 and (not window.size or float(window.max()) < high):
                labels[start:end] = 1
                changed = True
            elif label == 1 and end - start < min_frames:
                labels[start:end] = 0
                changed = True
        if not changed:
            break

    steps: list[Step] = []
    stances = [(a, b) for a, b, label in runs(labels) if label == 1]
    for start, end in stances:
        if start == 0:
            continue
        strike = cross_time(signal, start - 1, start, low, fps)
        toe_off = cross_time(signal, end - 1, end, low, fps) if end < len(signal) else float("nan")
        prev_end = max((b for a, b in stances if b <= start), default=0)
        window = signal[prev_end:start]
        finite_w = window[np.isfinite(window)]
        peak = float(finite_w.max()) if finite_w.size else float("nan")
        peak_i = prev_end + int(np.nanargmax(window)) if finite_w.size else start
        mid = (strike + toe_off) / 2 if np.isfinite(toe_off) else float("nan")
        steps.append(Step(number=0, foot=foot, time=strike, mid_stance=mid, frame=0,
                          peak_time=peak_i / fps, peak_clearance=peak,
                          contact_seconds=toe_off - strike if np.isfinite(toe_off) else float("nan")))
    return steps


def implausible_stance(flexion: dict, steps: list, fps: float, limit: float) -> int:
    """Frames inside a ground contact with the knee bent past *limit*: the pose failing."""
    bad = 0
    for step in steps:
        if not np.isfinite(step.contact_seconds):
            continue
        series = flexion.get(step.foot)
        if series is None:
            continue
        lo = int(round(step.time * fps))
        hi = int(round((step.time + step.contact_seconds) * fps))
        window = series[max(0, lo):min(len(series), hi + 1)]
        bad += int(np.sum(np.isfinite(window) & (window > limit)))
    return bad


def stride_ema_samples(steps: list, alpha: float) -> list[tuple[float, float]]:
    """The live reading, once per step: an EMA over overlapping strides."""
    marks = [s.mid_stance for s in steps]
    if len(marks) < 3:
        return []
    out, average = [], None
    for i in range(2, len(marks)):
        if not np.isfinite(marks[i - 2:i + 1]).all():
            continue
        stride = marks[i] - marks[i - 2]
        average = stride if average is None else alpha * stride + (1.0 - alpha) * average
        out.append((marks[i], 120.0 / average))
    return out


def running_mean_samples(steps: list) -> list[tuple[float, float]]:
    """The mean cadence so far, sampled at whole strides (even step counts)."""
    marks = [s.mid_stance for s in steps]
    out = []
    for i in range(2, len(marks), 2):
        if not (np.isfinite(marks[i]) and np.isfinite(marks[0])):
            continue
        out.append((marks[i], 60.0 * i / (marks[i] - marks[0])))
    return out


def foot_window_samples(steps: list, window: int, foot: str) -> list[tuple[float, float]]:
    marks = [s.mid_stance for s in steps]
    out, intervals = [], []
    for i in range(1, len(marks)):
        if steps[i].foot != foot or not (np.isfinite(marks[i]) and np.isfinite(marks[i - 1])):
            continue
        intervals.append(marks[i] - marks[i - 1])
        k = min(int(window), len(intervals))
        out.append((marks[i], 60.0 / float(np.mean(intervals[-k:]))))
    return out


def densify_samples(samples, fps: float, n: int) -> np.ndarray:
    """Hold each reading from its own moment until the next one."""
    out = np.full(n, np.nan)
    for t, value in samples:
        start = int(min(n, np.ceil(t * (fps or 30.0))))
        out[start:] = value
    return out


def summary_text(analysis: GaitAnalysis, *, video_seconds: float, source: str = "") -> str:
    """The human-readable gait report saved beside gait.json."""
    a = analysis
    lines = ["Running gait summary", "=" * 58]
    if source:
        lines.append(f"{'source':<24}{source}")
    side = {1.0: "towards +x (screen right)", -1.0: "towards -x (screen left)"}
    head = side.get(a.heading, "undecided")
    if a.facing and a.heading != a.facing:
        head += "   <- disagrees with the facing; see quality"
    lines += [
        f"{'clip length':<24}{video_seconds:.2f}s",
        f"{'foot strikes':<24}{a.count}   ({len(a.steps_for('left'))} left, {len(a.steps_for('right'))} right)",
        f"{'direction of travel':<24}{head}",
    ]
    if a.count < 2:
        lines.append("\nToo few foot strikes to measure a cadence. Check clearance.png.")
        return "\n".join(lines) + "\n"

    iv = a.intervals
    left_pct, right_pct = a.balance()
    vo = a.vertical_oscillation()
    spread = a.stride_spread()
    lines += [
        f"{'cadence':<24}{a.mean_cadence:.1f} steps/min",
        f"{'step time':<24}{float(np.mean(iv)):.3f}s" if iv else f"{'step time':<24}-",
        f"{'stride time':<24}{a.mean_stride:.3f}s   (one full cycle, per foot)",
        f"{'stride variability':<24}"
        + (f"{spread * 1000:.1f} ms   ({spread / a.mean_stride:.1%} of the stride)" if np.isfinite(spread) else "-"),
        "",
        "Left / right",
        "-" * 58,
        f"{'stride rate':<24}{a.foot_stride_rate('left'):.2f} spm left, {a.foot_stride_rate('right'):.2f} spm right   <- equal by construction",
        f"{'phase split':<24}{a.phase_split()[0]:.1f}% / {a.phase_split()[1]:.1f}%",
        f"{'contact time':<24}{a.contact_mean('left'):.3f}s left, {a.contact_mean('right'):.3f}s right",
        f"{'contact balance':<24}{left_pct:.1f}% / {right_pct:.1f}%   <- proxy; near/far geometry leaks in",
        "",
        "Knee",
        "-" * 58,
        f"{'flexion at contact':<24}"
        + ", ".join(f"{a.knee_shape(f).flexion:.1f} +/- {a.knee_shape(f).flexion_sd:.1f} deg {f}"
                    for f in FEET if a.knee_shape(f) is not None),
        f"{'peak flexion in cycle':<24}" + ", ".join(f"{a.peak_flexion(f):.1f} deg {f}" for f in FEET),
        f"{'shank/thigh at contact':<24}" + ", ".join(f"{a.segment_ratio(f):.2f} {f}" for f in FEET)
        + "   <- should be ~1.0",
        f"{'':<24}(projected angle in the image plane; 0 = straight leg)",
        "",
        "Whole body",
        "-" * 58,
        f"{'airborne':<24}{a.flight_ratio * 100:.0f}% of the clip   (neither foot planted)",
        f"{'vertical oscillation':<24}" + (f"{vo:.3f} leg lengths per stride" if np.isfinite(vo) else "-"),
        f"{'leg length':<24}{a.leg_length:.3f} of the frame height",
        "",
        "Per step",
        "-" * 58,
        f"{'#':>3} {'foot':<6} {'strike':>8} {'step':>7} {'stride':>7} {'contact':>8} {'clearance':>10} {'knee':>7}",
    ]

    def cell(value, width, places, unit=""):
        if value is None or not np.isfinite(value):
            return f"{'-':>{width}}"
        return f"{value:>{width - len(unit)}.{places}f}{unit}"

    for s in a.steps:
        lines.append(f"{s.number:>3} {s.foot:<6} {cell(s.time, 8, 3, 's')} {cell(s.interval, 7, 3, 's')} "
                     f"{cell(s.stride, 7, 3, 's')} {cell(s.contact_seconds, 8, 3, 's')} "
                     f"{cell(s.peak_clearance, 10, 3)} {cell(s.knee_flexion, 7, 1)}")

    lines += [
        "",
        "Notes. Cadence is timed stride by stride on one foot at a time, so any",
        "per-foot landmark offset cancels. Left/right figures do not have that",
        "property: from one side-on camera the near and far ankle cross the strike",
        "threshold at different rates, so read each leg against itself over time",
        "rather than against the other. Knee flexion is a projected angle, and",
        "the swing peak can only be an under-read as the shank turns out of the",
        "image plane. COCO-17 has no foot, so strike pattern is not available.",
    ]
    return "\n".join(lines) + "\n"
