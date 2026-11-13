"""How damaged a picture is: how faded, how dusty and how scratched.

Two steps, each a process of its own (`python -m ems.tools.damage STEP …`, the parameters on
stdin), answering as ems.tools.inpaint's do:

- `look IMAGE SHOWN` measures the fading and writes the picture the defect detector is shown, as
  `image.defects` shows it at `scale`. Fading is the mean change `image.unfade` would make to the
  picture shrunk to FADE_EDGE pixels, in 8-bit steps: a faded print's channels are squeezed and
  cast, so stretching and balancing them moves every pixel; a picture whose colours are whole
  barely moves.
- `count FOUND SHOWN` reads the detector's answer (each pixel's probability, a PNG of 16 bits) at
  `threshold` and sorts what it marks into connected parts: a part at least SCRATCH_LENGTH of the
  shorter side long and at most SCRATCH_WIDTH pixels wide on average is a scratch, any other a
  speck. Only a part on smooth picture counts: the detector also marks a picture's own thin light
  details (branches against the sky, a window's frame, highlights on a rim), which lie among
  others, where damage shows least. A part touching the picture's edge is left out, since the
  detector marks the edge of a print or plate, and so is one of fewer than SPECK_PIXELS, which is
  the detector's noise.

Each kind is scored from 0 (none) to 1 (as much as a picture worth restoring for it has), linearly
between the two bounds below and clamped.
"""

from __future__ import annotations

import json
import sys

from ..errors import UNSUPPORTED
from . import defects as defects_tool
from . import unfade as unfade_tool
from .inpaint import REFUSED, Refusal, read

MODULE = "ems.tools.damage"

# Fading is measured on the picture shrunk to this longer side.
FADE_EDGE = 1024
# The mean change, in 8-bit steps, at which fading scores 0 and 1.
FADED = (6.0, 22.0)
# A part this share of the shorter side long, or longer, may be a scratch.
SCRATCH_LENGTH = 0.04
# The most a scratch is wide, on average, in pixels of the picture the detector is shown.
SCRATCH_WIDTH = 4.0
# Parts of fewer pixels are not counted.
SPECK_PIXELS = 3
# A part counts when the picture around it, a band from RING[0] to RING[1] pixels away, is smooth:
# the median of its gradient (Sobel, over the picture's contrast) below SMOOTH.
RING = (2, 6)
SMOOTH = 0.10
# Specks, per million pixels the detector is shown, at which dust scores 0 and 1.
DUSTY = (40.0, 200.0)
# The scratches' length together, in shorter sides, at which scratches score 0 and 1.
SCRATCHED = (0.4, 1.6)


def score(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return round(min(1.0, max(0.0, (value - lo) / (hi - lo))), 3)


def fading(picture) -> float:
    """The mean change `image.unfade` makes to `picture` shrunk to FADE_EDGE, in 8-bit steps."""
    import cv2
    import numpy as np

    h, w = picture.shape[:2]
    factor = min(1.0, FADE_EDGE / max(h, w))
    small = cv2.resize(picture, (max(1, round(w * factor)), max(1, round(h * factor))), interpolation=cv2.INTER_AREA) if factor < 1 else picture.copy()
    before = [plane.astype(np.float32) for plane in unfade_tool.colour_planes(small)]
    unfade_tool.unfade(small, 0.5, True, 1.0)
    top = float(np.iinfo(small.dtype).max)
    change = np.mean([np.abs(plane.astype(np.float32) - old).mean() for plane, old in zip(unfade_tool.colour_planes(small), before)])
    return float(change) / top * 255


def gradient(grey):
    """Each pixel's gradient in `grey` (floats from 0 to 1), over the picture's contrast."""
    import cv2
    import numpy as np

    lo, hi = np.percentile(grey, (1, 99))
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy) / max(float(hi - lo), 0.05)


def kinds(marked, grey) -> tuple:
    """The connected parts of a boolean mask, labelled, and what each is: 0 not counted, 1 a speck,
    2 a scratch; with each part's extent in pixels. `grey` is the picture the mask was made of."""
    import cv2
    import numpy as np

    count, labels, stats, _ = cv2.connectedComponentsWithStats(marked.astype(np.uint8), connectivity=8)
    h, w = marked.shape
    shorter = min(h, w)
    slope = gradient(grey)
    kind = np.zeros(count, np.uint8)
    extents = np.zeros(count, np.float64)
    pad = RING[1] + 1
    kernel = np.ones((3, 3), np.uint8)
    for i, (x, y, width, height, area) in enumerate(stats[1:count], start=1):
        if area < SPECK_PIXELS or x == 0 or y == 0 or x + width == w or y + height == h:
            continue
        y0, x0, y1, x1 = max(y - pad, 0), max(x - pad, 0), min(y + height + pad, h), min(x + width + pad, w)
        part = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(part, kernel, iterations=RING[1]) > 0) & (cv2.dilate(part, kernel, iterations=RING[0]) == 0) & ~marked[y0:y1, x0:x1]
        if not ring.any() or float(np.median(slope[y0:y1, x0:x1][ring])) >= SMOOTH:
            continue
        extents[i] = float(np.hypot(width, height))
        kind[i] = 2 if extents[i] >= SCRATCH_LENGTH * shorter and area / extents[i] <= SCRATCH_WIDTH else 1
    return labels, kind, extents


def parts(marked, grey) -> dict:
    """The specks and scratches of a boolean mask of `grey`, and the scratches' length together in shorter sides."""
    import numpy as np

    _, kind, extents = kinds(marked, grey)
    shorter = min(marked.shape)
    return {"specks": int(np.count_nonzero(kind == 1)), "scratches": int(np.count_nonzero(kind == 2)),
            "length": round(float(extents[kind == 2].sum()) / shorter, 3)}


def counted(found, shown, threshold: float) -> dict:
    """The parts the detector's answer `found` (16 bits) marks at `threshold` on the picture it was
    `shown` (grey, 16 bits), with their scores."""
    import numpy as np

    marked = found.astype(np.float32) / 65535 >= threshold
    h, w = marked.shape
    found_parts = parts(marked, shown.astype(np.float32) / 65535)
    per_million = found_parts["specks"] / (h * w) * 1e6
    return {
        **found_parts,
        "marked": round(float(marked.mean()), 6),
        "dusty": score(per_million, DUSTY),
        "scratched": score(found_parts["length"], SCRATCHED),
    }


def main(argv: list[str]) -> int:
    import cv2
    import numpy as np

    # One core, as every tool in the lane is sized.
    cv2.setNumThreads(1)
    name = argv[1]
    params = json.load(sys.stdin)
    if name == "look":
        try:
            picture = read(argv[2], "file")
            if picture.dtype not in (np.uint8, np.uint16):
                raise Refusal(UNSUPPORTED.name, f"the file holds {picture.dtype} samples; only 8 and 16 bits are looked at")
        except Refusal as refusal:
            print(json.dumps({"error": refusal.code, "message": str(refusal)}))
            return REFUSED
        change = fading(picture)
        if not cv2.imwrite(argv[3], defects_tool.shrink(picture, int(params["scale"]))):
            raise RuntimeError("OpenCV wrote no picture to look at")
        print(json.dumps({
            "width": picture.shape[1],
            "height": picture.shape[0],
            "change": round(change, 2),
            "faded": score(change, FADED),
            "opencv": cv2.__version__,
        }))
    elif name == "count":
        found = cv2.imread(argv[2], cv2.IMREAD_UNCHANGED)
        shown = cv2.imread(argv[3], cv2.IMREAD_UNCHANGED)
        if found is None or found.ndim != 2 or shown is None or shown.shape != found.shape:
            raise RuntimeError("the detector's answer or the picture it was shown cannot be read")
        print(json.dumps(counted(found, shown, float(params["threshold"]))))
    else:
        raise SystemExit(f"no step {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
