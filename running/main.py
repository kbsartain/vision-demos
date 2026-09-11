"""Pose-estimate a running clip, measure the gait, and render it beside a live panel.

    python main.py

Everything is configured in config.py. One timestamped directory per run lands
under data/output/ with the overlay video, the plots, gait.json and summary.txt.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from src import gait, pose, render, video

console = Console()


def rule(step: int, title: str) -> None:
    console.rule(f"[bold cyan]{step}[/] {title}", align="left")


def kv(pairs: dict, title: str | None = None) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    for key, value in pairs.items():
        table.add_row(key, str(value))
    console.print(Panel(table, title=title, title_align="left", expand=False) if title else table)


def rel(path: Path) -> Path:
    try:
        return path.relative_to(cfg.PROJECT_DIR)
    except ValueError:
        return path


def main() -> int:
    t_start = time.perf_counter()
    console.print()
    console.rule("[bold]Running analysis[/]", align="center")

    # ── 1. Source ────────────────────────────────────────────────────────────
    rule(1, "Source video")
    if not cfg.INPUT_VIDEO.is_file():
        console.print(f"[red]Not found:[/] {cfg.INPUT_VIDEO}\n"
                      "  Drop a side-on running clip under data/input/ and point INPUT_VIDEO at it.")
        return 1
    src_info = video.probe_source(cfg.INPUT_VIDEO)
    kv({
        "path": rel(cfg.INPUT_VIDEO),
        "size": f"{cfg.INPUT_VIDEO.stat().st_size / 1e6:.1f} MB",
        "codec": f"{src_info['codec']} ({src_info['pix_fmt']})",
        "stored": f"{src_info['width']}x{src_info['height']}",
        "rotation": f"{src_info['rotation']} deg",
        "frames": f"{src_info['n_frames']} ({src_info['duration']:.1f}s)",
    })

    # ── 2. Convert ───────────────────────────────────────────────────────────
    rule(2, "Convert to MP4")
    timings: dict[str, float] = {}

    def prepare(height, label):
        path = video.cache_path(cfg.INPUT_VIDEO, cfg.CACHE_DIR, target_height=height,
                                trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF)
        if path.is_file() and not cfg.FORCE_RECONVERT:
            console.print(f"  [green]cache hit[/] ({label}): [dim]{rel(path)}[/]")
        else:
            console.print(f"  [yellow]converting[/] ({label})")
            t0 = time.perf_counter()
            video.convert(cfg.INPUT_VIDEO, path, target_height=height, trim_seconds=cfg.TRIM_SECONDS,
                          crf=cfg.CONVERT_CRF)
            timings["convert"] = timings.get("convert", 0.0) + time.perf_counter() - t0
            console.print(f"  done -> [dim]{rel(path)}[/]")
        return path, video.inspect(path)

    mp4, info = prepare(cfg.INFERENCE_HEIGHT, "inference")
    if cfg.EXPORT_HEIGHT == cfg.INFERENCE_HEIGHT:
        export_mp4, export_info = mp4, info
    else:
        export_mp4, export_info = prepare(cfg.EXPORT_HEIGHT, "export")
        if export_info.n_frames != info.n_frames:
            console.print("  [yellow]warning[/] frame counts differ between renditions; the overlay may drift")
    console.print(f"  inference [bold]{info}[/]")
    if export_mp4 is not mp4:
        console.print(f"  export    [bold]{export_info}[/]")

    # ── 3. Poses ─────────────────────────────────────────────────────────────
    rule(3, f"Pose estimation ({cfg.POSE_BACKEND})")
    backend = pose.get_backend(cfg.POSE_BACKEND)
    poses_cache = pose.cache_path(mp4, cfg.CACHE_DIR, key=backend.cache_key(cfg))
    cached = pose.load_cache(poses_cache) if cfg.REUSE_POSES else None
    if cached is not None:
        frames, meta, created = cached
        console.print(f"  [green]cache hit[/]: poses from [dim]{created}[/] ([dim]{poses_cache.name}[/])")
        console.print("  [dim]set REUSE_POSES = False in config.py to re-run detection[/]")
        pose_seconds = float(meta.get("seconds") or 0.0)
    else:
        t0 = time.perf_counter()
        frames, meta = backend.run(mp4, info, cfg=cfg, console=console)
        pose_seconds = time.perf_counter() - t0
        meta["seconds"] = round(pose_seconds, 3)
        pose.save_cache(poses_cache, frames, meta, stamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        console.print(f"  [green]done in {pose_seconds:.1f}s[/] ({len(frames) / max(pose_seconds, 1e-9):.1f} fps)")
    timings["pose"] = pose_seconds

    dropped = pose.dedupe(frames, max_persons=cfg.MAX_PERSONS, iou=cfg.PERSON_IOU_THRESHOLD)
    if dropped:
        console.print(f"  [dim]{dropped} duplicate detections suppressed[/]")
    by_index = {f["index"]: f["persons"] for f in frames}
    kv({
        "frames returned": f"{len(frames)} of {info.n_frames}",
        "frames with a person": pose.n_posed(frames),
        "backend": json.dumps({k: v for k, v in meta.items() if k not in ("truth",)}, default=str),
    }, title="response")

    # ── 4. Gait ──────────────────────────────────────────────────────────────
    rule(4, "Gait")
    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    run_dir = cfg.OUTPUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "poses.json").write_text(json.dumps({"frames": frames, "meta": meta}, default=str))

    analysis = None
    if cfg.ANALYZE_GAIT:
        analysis = gait.analyze(frames, info.fps, export_info.n_frames, cfg=cfg, aspect=info.width / info.height)
        if analysis.count >= 2:
            left, right = analysis.balance()
            kv({
                "cadence": f"[bold green]{analysis.mean_cadence:.1f} steps/min[/]",
                "steps": f"{analysis.count}  ({len(analysis.steps_for('left'))} left, "
                         f"{len(analysis.steps_for('right'))} right)",
                "step time": f"{np.mean(analysis.intervals):.3f}s" if analysis.intervals else "-",
                "stride time": f"{analysis.mean_stride:.3f}s",
                "stride var": f"{analysis.stride_spread() * 1000:.1f} ms",
                "contact": f"{analysis.contact_mean('left'):.3f}s left / {analysis.contact_mean('right'):.3f}s right"
                           f"  [dim]({left:.1f}% / {right:.1f}%)[/]",
                "airborne": f"{analysis.flight_ratio * 100:.0f}% of the clip",
                "knee at strike": ", ".join(
                    f"{analysis.knee_shape(f).flexion:.1f} +/- {analysis.knee_shape(f).flexion_sd:.1f} deg {f}"
                    for f in gait.FEET if analysis.knee_shape(f) is not None) or "-",
                "vertical osc.": f"{analysis.vertical_oscillation():.3f} leg lengths",
            }, title="gait")
        else:
            console.print("  [yellow]too few foot strikes detected[/]. Check clearance.png")
        w = analysis.warnings
        if w.get("consecutive_same_foot_strikes"):
            console.print(f"  [yellow]warning[/] {w['consecutive_same_foot_strikes']} consecutive same-foot strikes: "
                          "a strike on the other foot was missed")
        if w.get("implausible_stance_flexion_frames"):
            console.print(f"  [yellow]warning[/] {w['implausible_stance_flexion_frames']} frame(s) with a knee bent "
                          f"past {cfg.KNEE_STANCE_LIMIT_DEG:.0f} deg while planted; usually a left/right swap")
        if w.get("heading_disagrees_with_facing"):
            console.print("  [yellow]warning[/] measured direction of travel disagrees with the facing")

    # ── 5. Render ────────────────────────────────────────────────────────────
    rule(5, "Render")
    raw = run_dir / "_raw.mp4"
    t0 = time.perf_counter()
    stats = render.render(export_info, by_index, len(frames), raw, cfg=cfg, console=console, analysis=analysis)
    timings["render"] = time.perf_counter() - t0
    console.print(f"  drew poses on [bold]{stats['frames_with_pose']}/{stats['frames_written']}[/] frames"
                  + (f" ({stats['frames_held']} held)" if stats["frames_held"] else ""))
    out_video = run_dir / f"{cfg.INPUT_VIDEO.stem}_pose.mp4"
    t0 = time.perf_counter()
    video.encode_h264(raw, out_video, crf=cfg.OUTPUT_CRF)
    timings["encode"] = time.perf_counter() - t0
    raw.unlink(missing_ok=True)
    console.print(f"  video  -> [dim]{rel(out_video)}[/] ({out_video.stat().st_size / 1e6:.1f} MB)")

    outputs = {"video": out_video.name, "poses": "poses.json"}
    if analysis is not None:
        if cfg.SAVE_CLEARANCE_PLOT:
            render.plot_clearance(analysis, run_dir / "clearance.png", cfg=cfg,
                                  title=f"Ankle height: {analysis.count} strikes, {analysis.mean_cadence:.1f} steps/min")
            outputs["clearance"] = "clearance.png"
            console.print(f"  graph  -> [dim]{rel(run_dir / 'clearance.png')}[/]")
        if cfg.SAVE_KNEE_PLOT and analysis.steps:
            if render.plot_knee(analysis, run_dir / "knee.png", cfg=cfg,
                                title=f"Knee at foot strike: {analysis.count} strikes"):
                outputs["knee"] = "knee.png"
                console.print(f"  graph  -> [dim]{rel(run_dir / 'knee.png')}[/]")
        if cfg.SAVE_STEPS_PLOT and analysis.steps:
            render.plot_steps(analysis, run_dir / "steps.png", cfg=cfg, title="Per step: timing and contact, by foot")
            outputs["steps"] = "steps.png"
            console.print(f"  graph  -> [dim]{rel(run_dir / 'steps.png')}[/]")
        (run_dir / "gait.json").write_text(json.dumps({
            "fps": info.fps, "signal": gait.signal_config(cfg),
            "leg_length": round(analysis.leg_length, 5), "frame_aspect": round(analysis.frame_aspect, 5),
            "ground": {f: round(v, 5) for f, v in analysis.ground.items()},
            "contact_level": round(analysis.contact_level, 5),
            **analysis.summary(), "series": analysis.series(),
        }, indent=2))
        (run_dir / "summary.txt").write_text(gait.summary_text(
            analysis, video_seconds=info.duration, source=cfg.INPUT_VIDEO.name), encoding="utf-8")
        outputs.update({"gait": "gait.json", "summary": "summary.txt"})
        console.print(f"  gait   -> [dim]{rel(run_dir / 'gait.json')}[/] + summary.txt")

    # ── 6. Run record ────────────────────────────────────────────────────────
    total = time.perf_counter() - t_start
    snapshot = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items()
                if k.isupper() and not k.startswith("_")}
    (run_dir / "run.json").write_text(json.dumps({
        "stamp": stamp, "total_seconds": round(total, 2),
        "input": {"path": str(cfg.INPUT_VIDEO), **src_info},
        "converted": {"path": str(mp4), "width": info.width, "height": info.height, "fps": info.fps,
                      "n_frames": info.n_frames},
        "poses": {"backend": cfg.POSE_BACKEND, "meta": meta, "frames_returned": len(frames),
                  "frames_with_person": pose.n_posed(frames), "cached": cached is not None},
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "render": stats,
        "gait": analysis.summary() if analysis else None,
        "outputs": outputs,
        "config": snapshot,
    }, indent=2, default=str))
    console.print()
    console.rule(f"[bold green]done[/] in {total:.1f}s", align="left")
    console.print(f"  [bold]{rel(run_dir)}/[/]\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        sys.exit(130)
