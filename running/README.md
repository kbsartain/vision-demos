# Running analysis from a single side-on video

A computer-vision running analyser: pose-estimate a treadmill clip, measure
cadence and every foot strike, average the knee shape at contact, and render it
all back beside the clip as a live panel.

```
+------------------------+-------------------------------+
|                        |            CADENCE            |
|   the clip, with the   |           178 spm             |
|   skeleton drawn on,   |           12 steps            |
|   both ankles marked,  |     AVG CADENCE OVER TIME     |
|   comet trails, and a  |     [graph with playhead]     |
|   flash at each strike |  AVG KNEE SHAPE AT FOOT STRIKE|
|                        |   [mean limb, +/-1 sd fan]    |
|                        |   ANKLE PATH VISUALIZATION    |
|                        |  [both ankles, cloud + comet] |
+------------------------+-------------------------------+
```

This is a from-scratch replica of Jeremy Park's `running` demo in
[jeremyipark/vision-demos](https://github.com/jeremyipark/vision-demos)
(Apache-2.0), built from the LinkedIn video and the method described in that
repo. The main difference: it runs **locally with no API key**. The default
backend is the same ViTPose+ Large model the original used, loaded through
Hugging Face transformers; a faster YOLO11-pose backend and the VLM Run
gateway are also available.

Tested on the clip from the post itself (the left half of the video, 404x720,
15 s): 45 strikes, 180.5 spm, right knee 20.8 ± 2.8° at contact, against the
44 strikes, 177 to 182 spm and 22.7 ± 1.4° shown on the original panel.
YOLO11-pose on the same clip lost most strikes: its ankle keypoints barely
lift during swing, so keep `POSE_BACKEND = "vitpose"` for real footage.

## Run it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

(For CPU-only torch, install `torch torchvision` first from
`https://download.pytorch.org/whl/cpu`; for a CUDA GPU use the matching index.)
You also need `ffmpeg` and `ffprobe` on the PATH.

Drop a clip under `data/input/` and point `INPUT_VIDEO` in `config.py` at it.
Film from the side with both ankles in frame; a treadmill is ideal because the
camera can stay still. For a handheld camera set `FOOT_REFERENCE = "hip"`.
`PANEL_KNEE_FEET` picks which leg the knee block draws (the near one by default).

```bash
.venv/Scripts/python.exe main.py
```

### No clip handy? Test with a synthetic runner

```bash
.venv/Scripts/python.exe scripts/make_synthetic_clip.py --cadence 178
```

then set `INPUT_VIDEO = DATA_DIR / "input" / "synthetic.mp4"` and
`POSE_BACKEND = "synthetic"` in `config.py`. The generator writes the exact
keypoints it drew, so the measured cadence and knee angle can be checked
against known values.

## Output

One timestamped directory per run under `data/output/`:

```
20260911-140501/
├── <clip>_pose.mp4     # H.264 overlay video + live panel
├── clearance.png       # both ankles' height, every strike, the cadence
├── knee.png            # every detected knee, for debugging
├── steps.png           # step and contact time per step, by foot
├── gait.json           # per-step timings + the full per-frame signals
├── summary.txt         # the gait report, human-readable
├── poses.json          # the raw keypoints
└── run.json            # config snapshot + provenance
```

Converted MP4s and keypoints are cached in `data/cache/`, keyed on the source
and the settings, so a rendering tweak never re-runs the model.

## How the cadence is measured

1. Pose on every frame, 17 COCO keypoints.
2. Each ankle's height above **that ankle's own stance level** (the 90th
   percentile of its image height), in units of the runner's leg length
   (median thigh + median shank).
3. A foot strike is the downward crossing of 18% of a typical swing peak. A
   swing that never clears 55% of the peak is stance wobble; a stance shorter
   than 100 ms is the signal clipping the line.
4. Intervals are timed on interpolated mid-stance, which does not move when
   the threshold is slightly wrong. Cadence is `120 / mean(stride)`, same foot
   to same foot, so any per-leg landmark offset cancels.

The knee at foot strike is sampled at its own most-extended point within
100 ms of the strike (a parabola vertex), not on the steep loading slope
after it. Knee flexion is a projected angle, and only comparable for one leg
against itself over time from a single camera.

## Layout

```
config.py                 every knob, snapshotted into run.json
main.py                   the pipeline
src/gait.py               signal, strikes, cadence, knee shape, report
src/panel.py              the live side panel
src/render.py             overlay + panel per frame, and the still plots
src/comet.py              comet trails (taper, fade, wind)
src/skeleton.py           COCO-17 drawing
src/text.py               TrueType text on frames via Pillow
src/video.py              ffprobe / ffmpeg conversion and encoding
src/pose/vitpose.py       local ViTPose+ Large via transformers (default)
src/pose/yolo.py          local ultralytics YOLO11-pose backend
src/pose/vlmrun.py        VLM Run gateway (ViTPose+ Large) backend
src/pose/synthetic.py     ground-truth sidecar backend for tests
scripts/make_synthetic_clip.py
```
