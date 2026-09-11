"""Every knob for the running analysis. Edit values here, then `python main.py`.

There are no command-line flags on purpose: each run snapshots these values into
run.json, so any output can always be traced back to the settings that made it.
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
# RUN_INPUT_VIDEO and RUN_POSE_BACKEND in the environment override the two
# settings below, so a test run never needs this file edited.
INPUT_VIDEO = Path(os.environ.get("RUN_INPUT_VIDEO") or DATA_DIR / "input" / "running.mp4")

# ── Conversion (any container -> H.264 MP4) ──────────────────────────────────
# Always converted: ffmpeg applies the phone's rotation flag, OpenCV ignores it.
INFERENCE_HEIGHT = 1080   # what the pose model sees
EXPORT_HEIGHT = 1080      # what gets rendered; the panel is this tall too
TRIM_SECONDS = None       # e.g. 3.0 for a quick test
CONVERT_CRF = 23
FORCE_RECONVERT = False
REUSE_POSES = True        # reuse cached keypoints instead of re-running the model

# ── Pose backend ─────────────────────────────────────────────────────────────
# "vitpose"   ViTPose+ Large locally via Hugging Face transformers: the model
#             the original demo used, no key. Best keypoints; ~2 s/frame on CPU.
# "yolo"      local ultralytics YOLO11-pose. Fast, but on the demo clip its
#             ankle keypoints barely lifted in swing and most strikes were lost.
# "vlmrun"    ViTPose+ Large through the VLM Run gateway. Needs VLMRUN_API_KEY.
# "synthetic" reads the ground-truth sidecar written by
#             scripts/make_synthetic_clip.py, for testing the pipeline.
POSE_BACKEND = os.environ.get("RUN_POSE_BACKEND") or "vitpose"

YOLO_MODEL = os.environ.get("RUN_YOLO_MODEL") or "yolo11m-pose.pt"   # n / s / m / l / x; weights download on first use
YOLO_IMGSZ = int(os.environ.get("RUN_YOLO_IMGSZ") or 640)   # raise for a small subject; 1280 helps a phone clip
YOLO_CONF = 0.25                 # detection confidence
YOLO_KPT_CONF = 0.30             # keypoints below this are treated as not seen
YOLO_DEVICE = "auto"             # "auto", "cpu", "cuda:0"
YOLO_BATCH = 8

VITPOSE_MODEL = os.environ.get("RUN_VITPOSE_MODEL") or "usyd-community/vitpose-plus-large"
VITPOSE_DETECTOR = "yolo11n-pose.pt"   # any ultralytics model with a person class
VITPOSE_KPT_CONF = 0.30

VLMRUN_MODEL = "usyd-community/vitpose-plus-large"
VLMRUN_BASE_URL = "https://gateway.vlm.run/v1/openai"
VLMRUN_TIMEOUT = 1800.0

EVERY_FRAME = True        # pose on every frame; a foot strike is a 2-3 frame event
PRECISION = 4             # decimals kept on normalized coordinates

# ── Render ───────────────────────────────────────────────────────────────────
DRAW_FACE = False
MAX_PERSONS = 1
PERSON_IOU_THRESHOLD = 0.6
DRAW_BBOX = False
BBOX_COLOR = (255, 160, 40)    # BGR
LINE_THICKNESS = 3             # px at a 720px-wide frame; scales with the export
POINT_RADIUS = 4
OUTPUT_CRF = 20

HIGHLIGHT_FEET = True
FOOT_MARKER_RADIUS = 9
FLASH_ON_CONTACT = True        # an expanding ring on the ankle at each strike
FLASH_SECONDS = 0.25
FLASH_THICKNESS = 3

# The comet trail behind each ankle on the video.
TRAIL_ON_ANKLES = True
TRAIL_SECONDS = 0.32
TRAIL_FEET = ("left", "right")
TRAIL_WIDTH = 9
TRAIL_TAPER = 0.18
TRAIL_OPACITY = 0.85
TRAIL_FADE = 1.35
TRAIL_GLOW = 2.6
TRAIL_GLOW_OPACITY = 0.28
TRAIL_SOFTNESS = 0.6
# The tail streams downwind (direction of travel, reversed). "auto" measures the
# heading from the planted foot; "left"/"right" force it; "none" switches it off.
TRAIL_WIND = "auto"
TRAIL_WIND_SPEED = 220         # px/s at a 720px-wide frame

# ── The gait signal ──────────────────────────────────────────────────────────
# One signal per leg: ankle height above that ankle's own stance level, in leg
# lengths. COCO-17 has no foot, so the ankle is the lowest joint there is.
ANALYZE_GAIT = True
FOOT_REFERENCE = "ground"      # "ground" for a static camera, "hip" for a moving one
GROUND_PERCENTILE = 90         # the stance level, as a percentile of ankle y
PER_FOOT_GROUND = True         # the near and far ankle do not project to one line
SMOOTH_MEDIAN_FRAMES = 3
SMOOTH_MEAN_FRAMES = 3

# ── Foot strikes ─────────────────────────────────────────────────────────────
SWING_PEAK_FRACTION = 0.55     # a swing must clear this share of a typical peak
CONTACT_FRACTION = 0.18        # planted below this share of a typical peak
MIN_CONTACT_SECONDS = 0.10     # shorter "stances" are the signal clipping the line
CADENCE_SMOOTHING = 0.20       # EMA weight on the newest stride (~9 steps)
CADENCE_FOOT_WINDOW_STEPS = 2

# ── Side panel ───────────────────────────────────────────────────────────────
SIDE_PANEL = True
PANEL_Y_MIN_SPAN = 20.0        # spm; keeps a steady runner from reading as a seismograph
PANEL_TRACE_SEGMENTS = 6
PANEL_TRACE_THICKNESS = 3.0
PANEL_SIDE_MARGIN = 60
PANEL_GRAPH_ASPECT = 3.2

# Type sizes are px at a 720px-wide panel and scale with the export.
PANEL_TITLE_SIZE = 26
PANEL_NUMBER_SIZE = 78
PANEL_UNIT_SIZE = 28
PANEL_SUB_SIZE = 22
PANEL_GRAPH_TITLE_SIZE = 26
PANEL_AXIS_LABEL_SIZE = 19
PANEL_TICK_SIZE = 15
PANEL_KNEE_KEY_SIZE = 22
PANEL_KNEE_VALUE_SIZE = 30

PANEL_TITLE = "CADENCE"
PANEL_UNIT = "spm"
PANEL_STEPS_LABEL = "{steps} steps"
PANEL_STEPS_LABEL_ONE = "{steps} step"
PANEL_GRAPH_TITLE = "AVG CADENCE OVER TIME"
PANEL_Y_LABEL = "steps per minute (spm)"
PANEL_X_LABEL = "time (s)"

# ── The average knee at foot strike ──────────────────────────────────────────
KNEE_SEARCH_SECONDS = 0.10     # the knee's own most-extended point is read this far either side of the strike
KNEE_STANCE_LIMIT_DEG = 85.0   # a planted knee bent past this is the pose, not the runner
PANEL_KNEE_FEET = ("right",)   # the near leg alone by default; ("left", "right") shows both
PANEL_KNEE_SHARE = 0.40
PANEL_KNEE_FILL = 0.92
PANEL_KNEE_THICKNESS = 6
PANEL_KNEE_FAN_SD = 1.0
PANEL_KNEE_FAN_OPACITY = 0.30
PANEL_KNEE_FAN_STEP_OPACITY = 0.05
PANEL_KNEE_ENVELOPE_STEPS = 11
PANEL_KNEE_EASING = 0.05       # per-frame chase toward the current average
PANEL_KNEE_WINDOW = 8          # strikes averaged for the drawn shape (0 = all)
PANEL_KNEE_DECIMALS = 1
PANEL_KNEE_LABEL_HOLD_SECONDS = 1.0
PANEL_KNEE_TITLE = "AVG KNEE SHAPE AT FOOT STRIKE"
PANEL_KNEE_LEFT = "LEFT"
PANEL_KNEE_RIGHT = "RIGHT"
PANEL_KNEE_FORMAT = "{deg}° ± {sd}°"
PANEL_KNEE_FORMAT_NO_SD = "{deg}°"

# ── The ankles, live ─────────────────────────────────────────────────────────
PANEL_ANKLE_TITLE = "ANKLE PATH VISUALIZATION"
PANEL_ANKLE_FEET = ("left", "right")
PANEL_ANKLE_FRAME = "auto"     # "image" (fixed camera), "hip" (moving camera), "auto" follows FOOT_REFERENCE
PANEL_ANKLE_PAD = 0.10
PANEL_ANKLE_PATH_EASE = 0.45
PANEL_ANKLE_PATH_SETTLE_SECONDS = 0.30
PANEL_ANKLE_MAX_SHARE = 0.55
PANEL_ANKLE_GROUND = True
PANEL_ANKLE_HIP_LABEL = "under the hip"
PANEL_ANKLE_DOT = 7
PANEL_ANKLE_TRAIL_SECONDS = 0.55
PANEL_ANKLE_TRAIL_WIDTH = 7
PANEL_ANKLE_TRAIL_TAPER = 0.20
PANEL_ANKLE_TRAIL_OPACITY = 0.95
PANEL_ANKLE_TRAIL_FADE = 1.35
PANEL_ANKLE_GLOW = 2.6
PANEL_ANKLE_GLOW_OPACITY = 0.30
PANEL_ANKLE_SOFTNESS = 0.6
PANEL_ANKLE_TRAIL_SMOOTH = 5
PANEL_ANKLE_WIND_SPEED = 90
PANEL_ANKLE_MEMORY_OPACITY = 0.35
PANEL_ANKLE_MEMORY_HALFLIFE_SECONDS = 0.8
PANEL_ANKLE_MEMORY_WIDTH = 5

PANEL_SECTION_GAP = 48
PANEL_TOP_MARGIN = 60
PANEL_BOTTOM_MARGIN = 34
PANEL_FONT = "auto"            # "auto", "opencv", or a path to a .ttf/.ttc
PANEL_FONT_INDEX = None

ATTRIBUTION = "Running Analysis"   # credit line, bottom-right of the panel; "" disables it
ATTRIBUTION_SIZE = 22
ATTRIBUTION_MARGIN = 22
ATTRIBUTION_OPACITY = 0.9

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"
SAVE_CLEARANCE_PLOT = True
SAVE_STEPS_PLOT = True
SAVE_KNEE_PLOT = True
