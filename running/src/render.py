"""Drawing the poses back onto the frames beside the live panel, plus the still plots."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .comet import wind_vector
from .gait import FEET
from .panel import GaitPanel
from .skeleton import (AnkleTrails, draw_contact_flash, draw_foot_markers, draw_person,
                       visible_parts)
from .text import resolve_font
from .video import VideoInfo

PLOT_COLORS = {"left": "#12b0d9", "right": "#f2820c"}


def render(info: VideoInfo, by_index: dict, n_returned: int, dst: Path, *, cfg, console,
           analysis=None) -> dict:
    """Write the overlay video (clip on the left, panel on the right) and return stats."""
    edges, points = visible_parts(cfg.DRAW_FACE)
    stroke = info.width / 720
    thickness = max(1, round(cfg.LINE_THICKNESS * stroke))
    radius = max(1, round(cfg.POINT_RADIUS * stroke))
    foot_radius = max(2, round(cfg.FOOT_MARKER_RADIUS * stroke))
    flash_thickness = max(1, round(cfg.FLASH_THICKNESS * stroke))

    trails = None
    wind = (0.0, 0.0)
    if cfg.TRAIL_ON_ANKLES:
        wind = wind_vector(cfg.TRAIL_WIND, analysis.heading if analysis else 0.0,
                           speed=cfg.TRAIL_WIND_SPEED * stroke, fps=info.fps)
        trails = AnkleTrails(cfg.TRAIL_FEET, length=max(2, round(cfg.TRAIL_SECONDS * info.fps)),
                             width=max(2, round(cfg.TRAIL_WIDTH * stroke)), taper=cfg.TRAIL_TAPER,
                             opacity=cfg.TRAIL_OPACITY, fade=cfg.TRAIL_FADE, glow=cfg.TRAIL_GLOW,
                             glow_opacity=cfg.TRAIL_GLOW_OPACITY, softness=cfg.TRAIL_SOFTNESS, wind=wind)

    stride = max(1, round(info.n_frames / max(n_returned, 1)))
    hold_limit = stride - 1

    panel, out_w, font_name = None, info.width, None
    if cfg.SIDE_PANEL and analysis is not None:
        font = resolve_font(cfg.PANEL_FONT, cfg.PANEL_FONT_INDEX)
        font_name = font[2] if font else "OpenCV Hershey"
        panel = GaitPanel(analysis, info.width, info.height, cfg=cfg, font=font)
        out_w = info.width * 2
        console.print(f"  side panel: live, {out_w}x{info.height}, {font_name}")

    flashes: dict[int, list[str]] = {}
    flash_frames = max(1, round(cfg.FLASH_SECONDS * info.fps))
    if analysis is not None and cfg.FLASH_ON_CONTACT:
        for step in analysis.steps:
            flashes.setdefault(step.frame, []).append(step.foot)

    cap = cv2.VideoCapture(str(info.path))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (out_w, info.height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open a writer for {dst}")

    last_persons, held_for = None, 0
    n_drawn = n_held = 0
    columns = (TextColumn("[cyan]rendering[/]"), BarColumn(), TaskProgressColumn(),
               TextColumn("{task.completed}/{task.total} frames"), TimeElapsedColumn())
    try:
        with Progress(*columns, console=console, transient=True) as progress:
            task = progress.add_task("render", total=info.n_frames)
            for i in range(info.n_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                if i in by_index:
                    persons, held = by_index[i], False
                    last_persons, held_for = persons, 0
                elif last_persons is not None and held_for < hold_limit:
                    persons, held = last_persons, True
                    held_for += 1
                else:
                    persons, held = [], False
                h, w = frame.shape[:2]

                if trails is not None:
                    trails.update(persons[0] if persons else None, w, h).draw(frame)
                for person in persons:
                    draw_person(frame, person, w, h, edges=edges, points=points, thickness=thickness,
                                radius=radius, draw_bbox=cfg.DRAW_BBOX, bbox_color=cfg.BBOX_COLOR)
                if persons and cfg.HIGHLIGHT_FEET:
                    draw_foot_markers(frame, persons[0], w, h, radius=foot_radius)
                if persons and flashes:
                    for age in range(flash_frames):
                        for foot in flashes.get(i - age, ()):
                            draw_contact_flash(frame, persons[0], w, h, foot=foot, progress=age / flash_frames,
                                               radius=foot_radius, thickness=flash_thickness)
                if persons:
                    n_drawn += 1
                    n_held += bool(held)
                if panel is not None:
                    frame = np.hstack([frame, panel.draw(i)])
                writer.write(frame)
                progress.update(task, advance=1)
    finally:
        cap.release()
        writer.release()

    return {"frames_written": info.n_frames, "frames_with_pose": n_drawn, "frames_held": n_held,
            "hold_limit": hold_limit, "output_width": out_w, "side_panel": panel is not None,
            "ankle_trails": trails is not None, "trail_wind_px_per_frame": round(wind[0], 3),
            "panel_font": font_name}


def plot_clearance(analysis, dst: Path, *, cfg, title: str = "") -> None:
    """Both ankles, every strike, both thresholds, and the cadence under it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = analysis.times
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for foot in FEET:
        ax.plot(t, analysis.raw[foot], color=PLOT_COLORS[foot], lw=0.8, alpha=0.35, label=f"{foot} raw")
        ax.plot(t, analysis.clearance[foot], color=PLOT_COLORS[foot], lw=2.0, label=f"{foot} smoothed")
    ax.axhline(analysis.contact_level, color="#8a8f98", ls="--", lw=1.2,
               label=f"strike threshold ({analysis.contact_level:.3f})")
    for step in analysis.steps:
        ax.plot(step.time, analysis.contact_level, "o", color=PLOT_COLORS[step.foot], ms=6, mec="white", mew=0.8)
        ax.axvline(step.time, color=PLOT_COLORS[step.foot], lw=0.6, alpha=0.25)
    ax.fill_between(t, *ax.get_ylim(), where=analysis.airborne, color="#adb5bd", alpha=0.10, label="airborne")
    ax.set_ylabel("ankle height\n(leg lengths above its own stance level)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=6, fontsize=8.5, frameon=False)
    ax.set_title(title or f"{analysis.count} foot strikes", fontsize=13, fontweight="bold", pad=26)

    for foot in FEET:
        ax2.plot(t, analysis.running_cadence_series(foot), color=PLOT_COLORS[foot], lw=2.0,
                 drawstyle="steps-post", label=f"{foot}, mean so far")
    ax2.plot(t, analysis.running_cadence_series(), color="#212529", lw=2.6, drawstyle="steps-post",
             label="both, mean so far")
    ax2.plot(t, analysis.cadence_live, color="#868e96", lw=1.0, alpha=0.8, drawstyle="steps-post",
             label=f"live (EMA alpha={cfg.CADENCE_SMOOTHING})")
    if np.isfinite(analysis.mean_cadence):
        ax2.axhline(analysis.mean_cadence, color="#1864ab", ls="--", lw=1.2, alpha=0.7,
                    label=f"clip mean {analysis.mean_cadence:.1f} spm")
    ax2.set_ylabel("cadence\n(steps / min)")
    ax2.set_xlabel("time (s)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=3)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.08, hspace=0.12)
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def plot_steps(analysis, dst: Path, *, cfg, title: str = "") -> None:
    """Step time and contact time per step, the two feet separated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not analysis.steps:
        return
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    for foot in FEET:
        steps = analysis.steps_for(foot)
        xs = [s.number for s in steps]
        ax.plot(xs, [s.interval for s in steps], "o-", color=PLOT_COLORS[foot], ms=7, lw=1.4, label=f"{foot} foot")
        ax2.plot(xs, [s.contact_seconds for s in steps], "o-", color=PLOT_COLORS[foot], ms=7, lw=1.4,
                 label=f"{foot} foot")
    iv = analysis.intervals
    if iv:
        mean = float(np.mean(iv))
        ax.axhline(mean, color="#495057", ls="--", lw=1.1, label=f"mean {mean:.3f}s ({analysis.mean_cadence:.0f} spm)")
    ax.set_ylabel("step time (s)\nsince the other foot landed")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper right")
    ax.set_title(title or "Per step", fontsize=13, fontweight="bold")
    left, right = analysis.balance()
    ax2.set_ylabel("contact time (s)\n(proxy)")
    ax2.set_xlabel("step number")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, frameon=False, ncol=3, loc="upper right",
               title=f"balance {left:.1f}% / {right:.1f}%", title_fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10, hspace=0.10)
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def plot_knee(analysis, dst: Path, *, cfg, title: str = "") -> bool:
    """Every detected knee at contact, the mean limb through stance, and flexion vs stance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not analysis.steps or not analysis.limb:
        return False
    offset = {"left": 0.0, "right": 0.85}
    fps = analysis.fps or 30.0
    grid = np.arange(analysis.n_frames)
    fig, (ax_hit, ax_stance, ax_flex) = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1, 1.35]})

    def limb_at(foot, t):
        L = analysis.limb[foot]
        i = t * fps
        hx, hy = np.interp(i, grid, L["hip"][0]), np.interp(i, grid, L["hip"][1])
        return [((np.interp(i, grid, L[j][0]) - hx) / analysis.leg_length,
                 (np.interp(i, grid, L[j][1]) - hy) / analysis.leg_length) for j in ("knee", "ankle")]

    for foot in FEET:
        steps = analysis.steps_for(foot)
        if not steps:
            continue
        dx = offset[foot]
        for s in steps:
            (kx, ky), (ax_, ay) = limb_at(foot, s.knee_time)
            ax_hit.plot([dx, dx + kx, dx + ax_], [0, ky, ay], color=PLOT_COLORS[foot], lw=1.0, alpha=0.35,
                        solid_capstyle="round")
        shape = analysis.knee_shape(foot)
        mx = np.mean([limb_at(foot, s.knee_time) for s in steps], axis=0)
        label = (f"{foot}: {shape.flexion:.1f} +/- {shape.flexion_sd:.1f} deg (n={len(steps)})"
                 if shape is not None else foot)
        ax_hit.plot([dx, dx + mx[0][0], dx + mx[1][0]], [0, mx[0][1], mx[1][1]], color=PLOT_COLORS[foot], lw=4.0,
                    solid_capstyle="round", label=label)
        ax_hit.plot([dx], [0], "o", color=PLOT_COLORS[foot], ms=9, mec="white", mew=1.2)
    ax_hit.set_title("At contact: every strike overlaid\n(thin = one strike, bold = the mean)", fontsize=11,
                     fontweight="bold")

    fracs = np.linspace(0.0, 1.0, 5)
    for foot in FEET:
        steps = [s for s in analysis.steps_for(foot) if np.isfinite(s.contact_seconds)]
        if not steps:
            continue
        dx = offset[foot]
        for k, frac in enumerate(fracs):
            pts = np.mean([limb_at(foot, s.time + frac * s.contact_seconds) for s in steps], axis=0)
            ax_stance.plot([dx, dx + pts[0][0], dx + pts[1][0]], [0, pts[0][1], pts[1][1]], color=PLOT_COLORS[foot],
                           lw=2.6, alpha=0.30 + 0.70 * frac, solid_capstyle="round",
                           label=f"{foot} {frac:.0%} of stance" if k in (0, len(fracs) - 1) else None)
        ax_stance.plot([dx], [0], "o", color=PLOT_COLORS[foot], ms=9, mec="white", mew=1.2)
    ax_stance.set_title("Through ground contact: mean limb\n(faint = touchdown, solid = toe-off)", fontsize=11,
                        fontweight="bold")
    for ax in (ax_hit, ax_stance):
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("leg lengths")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8.5, frameon=False, loc="lower center")
    ax_hit.set_ylabel("leg lengths below the hip")

    pct = np.linspace(0, 100, 51)
    for foot in FEET:
        kf = analysis.knee_flexion[foot]
        stack = []
        for s in analysis.steps_for(foot):
            if not np.isfinite(s.contact_seconds):
                continue
            ts = s.time + (pct / 100.0) * s.contact_seconds
            stack.append(np.interp(ts * fps, np.arange(len(kf)), kf))
        if not stack:
            continue
        stack = np.array(stack)
        for row in stack:
            ax_flex.plot(pct, row, color=PLOT_COLORS[foot], lw=0.8, alpha=0.30)
        m, sd = stack.mean(axis=0), stack.std(axis=0)
        ax_flex.plot(pct, m, color=PLOT_COLORS[foot], lw=3.0, label=f"{foot} mean")
        ax_flex.fill_between(pct, m - sd, m + sd, color=PLOT_COLORS[foot], alpha=0.15)
    ax_flex.set_xlabel("% of ground contact (0 = strike, 100 = toe-off)")
    ax_flex.set_ylabel("knee flexion (deg), 0 = straight leg")
    ax_flex.grid(alpha=0.25)
    ax_flex.legend(fontsize=9, frameon=False, loc="upper left")
    ax_flex.set_title("Flexion through contact: one line per strike\n(band = +/- 1 sd)", fontsize=11,
                      fontweight="bold")
    fig.suptitle(title or "Knee at foot strike", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.11, wspace=0.24)
    fig.savefig(dst, dpi=150)
    plt.close(fig)
    return True
