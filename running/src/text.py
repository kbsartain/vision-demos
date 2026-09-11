"""TrueType text on OpenCV frames, via Pillow.

OpenCV only ships the Hershey stroke fonts, which look like a plotter drew them.
Pillow can render any installed face onto the same frames. If no face is found,
callers fall back to Hershey rather than failing the run.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

_WIN = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

# (path, face index inside a .ttc, display name), in order of preference.
FONT_CANDIDATES: tuple[tuple[str, int, str], ...] = (
    (str(_WIN / "segoeuib.ttf"), 0, "Segoe UI Bold"),
    (str(_WIN / "arialbd.ttf"), 0, "Arial Bold"),
    ("/System/Library/Fonts/Avenir Next.ttc", 0, "Avenir Next Bold"),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 1, "Helvetica Neue Bold"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0, "DejaVu Sans Bold"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 0, "Liberation Sans Bold"),
)


def _matplotlib_dejavu():
    try:
        import matplotlib
    except ImportError:
        return None
    path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
    return (str(path), 0, "DejaVu Sans Bold (bundled)") if path.is_file() else None


def resolve_font(spec: str = "auto", index: int | None = None):
    """Pick a face. None means: use the OpenCV Hershey fallback."""
    try:
        from PIL import ImageFont  # noqa: F401
    except ImportError:
        return None
    if spec == "opencv":
        return None
    if spec != "auto":
        path = Path(spec).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"PANEL_FONT points at a missing file: {path}")
        return (str(path), index or 0, path.stem)
    for path, idx, name in FONT_CANDIDATES:
        if Path(path).is_file():
            return (path, index if index is not None else idx, name)
    return _matplotlib_dejavu()


@lru_cache(maxsize=4096)
def _patch(text: str, font_path: str, font_index: int, size: int, color) -> np.ndarray:
    """One string as a tight RGBA patch. Cached: a render draws few distinct strings."""
    from PIL import Image, ImageDraw, ImageFont

    face = ImageFont.truetype(font_path, size, index=font_index)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=face)
    w, h = max(1, right - left), max(1, bottom - top)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((-left, -top), text, font=face, fill=(*color[::-1], 255))
    return np.array(img)


def measure(text: str, font, size: int) -> tuple[int, int]:
    """Ink width and height in px."""
    if font is None:
        import cv2
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 30, 2)
        return w, h
    patch = _patch(text, font[0], font[1], int(size), (255, 255, 255))
    return patch.shape[1], patch.shape[0]


def blend(img: np.ndarray, patch: np.ndarray, x: int, y: int, *, opacity: float = 1.0):
    """Alpha-composite an RGBA patch onto a BGR frame in place, clipped at the edges."""
    ph, pw = patch.shape[:2]
    fh, fw = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + pw), min(fh, y + ph)
    if x0 >= x1 or y0 >= y1:
        return img
    sub = patch[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = img[y0:y1, x0:x1]
    alpha = (sub[..., 3:4].astype(np.float32) / 255.0) * opacity
    rgb = sub[..., 2::-1].astype(np.float32)
    img[y0:y1, x0:x1] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
    return img


def draw(img, text: str, font, *, size: int, xy, color=(255, 255, 255),
         anchor: str = "lt", opacity: float = 1.0, rotate: int = 0):
    """Draw *text* with its *anchor* (l/c/r + t/m/b) corner at *xy*, off the ink box."""
    if font is None:
        import cv2
        scale = size / 30
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x, y = xy
        x -= {"l": 0, "c": tw // 2, "r": tw}[anchor[0]]
        y += {"t": th, "m": th // 2, "b": 0}[anchor[1]]
        cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        return img
    patch = _patch(text, font[0], font[1], int(size), tuple(int(c) for c in color))
    if rotate:
        patch = np.rot90(patch, k=(rotate // 90) % 4)
    ph, pw = patch.shape[:2]
    x, y = xy
    x -= {"l": 0, "c": pw // 2, "r": pw}[anchor[0]]
    y -= {"t": 0, "m": ph // 2, "b": ph}[anchor[1]]
    return blend(img, patch, int(x), int(y), opacity=opacity)
