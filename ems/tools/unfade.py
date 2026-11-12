"""A faded picture's colours stretched back out, at the picture's own depth.

A process of its own (`python -m ems.tools.unfade unfade IMAGE OUT`, the parameters on stdin), answering
one line of JSON on stdout as `ems.tools.inpaint` does.

Fading leaves each colour channel squeezed into a narrower band of its own, so each is stretched
between its own darkest and lightest pixels, the share `clip` of each let go to black and white.
That restores the contrast and most of the cast. What is left is balanced on the picture's most
neutral pixels: of the mid-tones after the stretch, the third with the least colour are taken to
be grey, and each channel gets the gamma that makes their mean so. A picture of one colour (a red
brick square) keeps it, since its neutral pixels are the few that are grey.

Both are a curve per channel, so the picture is changed through one lookup table a channel,
whatever its depth; `strength` blends the curve with the identity.
"""

from __future__ import annotations

import json
import sys

from ..errors import UNSUPPORTED
from .inpaint import REFUSED, Refusal, colour_planes, read, write

# The statistics are taken on the picture shrunk to this longer side.
SAMPLE_EDGE = 1024
# Mid-tones: the luminance band, after the stretch, the neutral pixels are looked for in.
MID = (0.15, 0.85)
# The share of the mid-tones, those with the least colour, taken to be grey.
NEUTRAL_SHARE = 0.3
# The most a channel's balance may bend it, as a gamma and its inverse.
GAMMA_LIMIT = 2.0
# Fewer mid-tone pixels than this in the sample and the picture is not balanced.
MIN_PIXELS = 100


def sample(planes, edge: int = SAMPLE_EDGE):
    """The colour planes as floats from 0 to 1, shrunk so their longer side is at most `edge`."""
    import cv2
    import numpy as np

    top = float(np.iinfo(planes[0].dtype).max)
    h, w = planes[0].shape
    scale = min(1.0, edge / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return [cv2.resize(p, size, interpolation=cv2.INTER_AREA).astype(np.float32) / top if scale < 1 else p.astype(np.float32) / top for p in planes]


def levels(small, clip: float) -> list[tuple[float, float]]:
    """Each plane's black and white point: the `clip` percentiles from either end."""
    import numpy as np

    points = []
    for plane in small:
        lo, hi = (float(v) for v in np.percentile(plane, (clip, 100 - clip)))
        points.append((lo, hi) if hi - lo >= 1e-3 else (0.0, 1.0))
    return points


def stretch(x, point: tuple[float, float]):
    import numpy as np

    lo, hi = point
    return np.clip((x - lo) / (hi - lo), 0, 1)


def balance(small, points) -> list[float]:
    """The gamma per plane that makes the most neutral mid-tones grey; all 1 when there are too few."""
    import numpy as np

    if len(small) != 3:
        return [1.0] * len(small)
    rgb = np.stack([stretch(plane, point) for plane, point in zip(small, points)], axis=-1).reshape(-1, 3)
    # OpenCV's planes are blue, green, red.
    lum = rgb @ np.array([0.114, 0.587, 0.299], np.float32)
    mid = rgb[(lum > MID[0]) & (lum < MID[1])]
    if len(mid) < MIN_PIXELS:
        return [1.0, 1.0, 1.0]
    chroma = mid.max(axis=1) - mid.min(axis=1)
    neutral = mid[chroma <= np.quantile(chroma, NEUTRAL_SHARE)]
    means = np.clip(neutral.mean(axis=0), 1e-3, 1 - 1e-3)
    grey = float(means.mean())
    return [float(np.clip(np.log(grey) / np.log(m), 1 / GAMMA_LIMIT, GAMMA_LIMIT)) for m in means]


def table(dtype, point: tuple[float, float], gamma: float, strength: float):
    """The lookup table of one plane: every stored value to its unfaded one."""
    import numpy as np

    top = np.iinfo(dtype).max
    x = np.arange(top + 1, dtype=np.float64) / top
    y = stretch(x, point) ** gamma
    return np.rint((x + strength * (y - x)) * top).astype(dtype)


def unfade(picture, clip: float, balanced: bool, strength: float) -> dict:
    """`picture` unfaded in place; the black and white points and gammas it was given, per plane, red first."""
    planes = colour_planes(picture)
    small = sample(planes)
    points = levels(small, clip)
    gammas = balance(small, points) if balanced else [1.0] * len(planes)
    for plane, point, gamma in zip(planes, points, gammas):
        plane[...] = table(plane.dtype, point, gamma, strength)[plane]
    order = slice(None, None, -1) if len(planes) == 3 else slice(None)
    return {"levels": [[round(lo, 4), round(hi, 4)] for lo, hi in points[order]], "gammas": [round(g, 3) for g in gammas[order]]}


def main(argv: list[str]) -> int:
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    name, image, out = argv[1], argv[2], argv[3]
    if name != "unfade":
        raise SystemExit(f"no step {name}")
    params = json.load(sys.stdin)
    try:
        picture = read(image, "file")
        if picture.dtype not in (np.uint8, np.uint16):
            raise Refusal(UNSUPPORTED.name, f"the file holds {picture.dtype} samples; only 8 and 16 bits are unfaded")
    except Refusal as refusal:
        print(json.dumps({"error": refusal.code, "message": str(refusal)}))
        return REFUSED
    done = unfade(picture, float(params["clip"]), bool(params["balance"]), float(params["strength"]))
    write(picture, out, params["format"])
    print(json.dumps({
        "width": picture.shape[1],
        "height": picture.shape[0],
        "channels": 1 if picture.ndim == 2 else picture.shape[2],
        "depth": picture.dtype.itemsize * 8,
        **done,
        "opencv": cv2.__version__,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
