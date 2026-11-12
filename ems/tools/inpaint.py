"""A picture's masked pixels filled from their surroundings, at the picture's own depth.

Three steps, each a process of its own (`python -m ems.tools.inpaint STEP IMAGE MASK …`, the
parameters on stdin), so each is a tool like any other: niced, stopped when it stalls, killed with
its caller. Each answers one line of JSON on stdout: what it did, or with exit status 2 the error
code and message of a picture it refuses.

- `opencv IMAGE MASK OUT` fills with OpenCV. OpenCV inpaints a 16-bit picture only one channel at a
  time, so every picture is filled channel by channel; for an 8-bit one that gives exactly what
  OpenCV's three-channel fill does.
- `prepare IMAGE MASK DIR` cuts the crops a model fills (`plan`) into DIR, as RGB PNGs at the
  picture's depth with a mask beside each, and names the file each fill is to be written to.
- `compose IMAGE MASK DIR OUT` puts those fills back into the picture.

Either way an alpha channel is kept as it is, and a fill is put back only where the mask marks, so
every other pixel is the input's, bit for bit, whatever the method does around the hole.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

from ..errors import INVALID_REQUEST, TOOL_FAILED, UNDECODABLE, UNSUPPORTED, EmsError
from .run import run

# Where `-m ems.tools.inpaint` resolves, whatever directory the server was started in.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FORMATS = ("png", "tiff")
REFUSED = 2
CODES = {code.name: code for code in (INVALID_REQUEST, UNSUPPORTED, UNDECODABLE)}

# How much of the picture around its defects a model is shown, in pixels on each side.
MARGIN = 128
# The longest side of a crop a model is shown. A larger group of defects is cut into tiles of
# this size, or shrunk to it when its holes are too wide for a tile to hold what surrounds them.
LIMIT = 1024


async def step(name: str, args: list[str], params: dict, *, timeout: float) -> dict:
    """Run one step of the tool; what it answered."""
    done = await run([sys.executable, "-m", "ems.tools.inpaint", name, *args], stdin=json.dumps(params).encode(), cwd=PACKAGE_ROOT, timeout=timeout)
    lines = done.stdout.decode(errors="replace").strip().splitlines()
    try:
        answer = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        answer = None
    if done.code == REFUSED and isinstance(answer, dict) and answer.get("error") in CODES:
        raise EmsError(CODES[answer["error"]], str(answer.get("message")))
    if done.code != 0 or not isinstance(answer, dict):
        raise EmsError(TOOL_FAILED, f"inpaint {name} failed: {done.tail() or f'exit {done.code}'}")
    return answer


class Refusal(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def read(path: str, what: str):
    import cv2

    if not cv2.haveImageReader(path):
        raise Refusal(UNSUPPORTED.name, f"the {what} is in a format OpenCV has no reader for")
    picture = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if picture is None:
        raise Refusal(UNDECODABLE.name, f"the {what} cannot be decoded")
    return picture


def read_both(image: str, mask_path: str):
    """The picture as stored and which of its pixels the mask marks, or a Refusal."""
    import numpy as np

    picture = read(image, "file")
    if picture.dtype not in (np.uint8, np.uint16):
        raise Refusal(UNSUPPORTED.name, f"the file holds {picture.dtype} samples; only 8 and 16 bits are inpainted")
    mask = read(mask_path, "mask")
    if mask.ndim != 2:
        raise Refusal(INVALID_REQUEST.name, f"the mask has {mask.shape[2]} channels; a mask has one")
    if mask.shape != picture.shape[:2]:
        raise Refusal(INVALID_REQUEST.name, f"the mask is {mask.shape[1]}x{mask.shape[0]} and the file {picture.shape[1]}x{picture.shape[0]}")
    return picture, mask != 0


def colour_planes(picture) -> list:
    """The picture's colour channels as views into it: one for grey, three otherwise, never the alpha."""
    return [picture] if picture.ndim == 2 else [picture[:, :, c] for c in range(min(picture.shape[2], 3))]


def fill(picture, marked, method: str, radius: int):
    """`picture` with the pixels `marked` (a boolean array of its height and width) filled by OpenCV, in place."""
    import cv2
    import numpy as np

    flag = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}[method]
    mask = marked.astype(np.uint8)
    for plane in colour_planes(picture):
        filled = cv2.inpaint(np.ascontiguousarray(plane), mask, radius, flag)
        plane[marked] = filled[marked]
    return picture


@dataclass(frozen=True)
class Crop:
    """What a model is shown, and which of the pixels it fills are taken from it.

    `box` is the part of the picture shown, `(x0, y0, x1, y1)`, shrunk by `scale` when that is
    under 1. Of what the model fills there, only the marked pixels of `group` (a label of `plan`'s
    groups) inside `core`, another box of the picture, go back into it.
    """

    box: tuple[int, int, int, int]
    group: int
    core: tuple[int, int, int, int]
    scale: float = 1.0


def plan(marked, margin: int = MARGIN, limit: int = LIMIT):
    """The crops that fill every marked pixel once, and the groups they are labelled by.

    Defects closer than two margins apart are one group, shown together with a margin of the
    picture around them. A group whose box fits `limit` is one crop. A larger one is cut into
    tiles: their cores share out the box, each shown with a margin around it, so every marked
    pixel is filled where the model sees what lies around it. Unless its holes are wider than a
    margin, which no tile could show the surroundings of: then the whole group is one crop, shrunk
    to `limit`. Every crop's mask marks every defect it shows, its own group's or not, so no model
    takes an unfilled defect for picture.
    """
    import cv2
    import numpy as np

    height, width = marked.shape
    side = 2 * margin + 1
    grown = cv2.dilate(marked.astype(np.uint8), np.ones((side, side), np.uint8))
    count, groups, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    crops: list[Crop] = []
    core_side = limit - 2 * margin
    for group in range(1, count):
        x, y, w, h = (int(v) for v in stats[group][:4])
        box = (x, y, x + w, y + h)
        if max(w, h) <= limit:
            crops.append(Crop(box, group, box))
            continue
        own = (marked[y : y + h, x : x + w] & (groups[y : y + h, x : x + w] == group)).astype(np.uint8)
        widest = float(cv2.distanceTransform(own, cv2.DIST_L2, 3).max())
        if widest > margin / 2:
            crops.append(Crop(box, group, box, limit / max(w, h)))
            continue
        for ty in range(y, y + h, core_side):
            for tx in range(x, x + w, core_side):
                core = (tx, ty, min(tx + core_side, x + w), min(ty + core_side, y + h))
                if not own[core[1] - y : core[3] - y, core[0] - x : core[2] - x].any():
                    continue
                tile = (max(core[0] - margin, 0), max(core[1] - margin, 0), min(core[2] + margin, width), min(core[3] + margin, height))
                crops.append(Crop(tile, group, core))
    return groups, crops


def shown_size(crop: Crop) -> tuple[int, int]:
    """The width and height of what the model is shown of `crop`."""
    x0, y0, x1, y1 = crop.box
    return max(1, round((x1 - x0) * crop.scale)), max(1, round((y1 - y0) * crop.scale))


def as_rgb(picture):
    """The colour channels of a picture as cv2 orders RGB: grey repeated three times, an alpha left out."""
    import cv2

    if picture.ndim == 2:
        return cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR)
    return picture[:, :, :3]


def crop_files(directory: str, index: int) -> dict:
    return {name: os.path.join(directory, f"{name}-{index:04d}.png") for name in ("image", "mask", "out")}


def prepare(picture, marked, directory: str) -> list[dict]:
    """Write each crop of `plan` and its mask into `directory`; the files of each, in order."""
    import cv2
    import numpy as np

    _groups, crops = plan(marked)
    rgb = as_rgb(picture)
    files = []
    for index, crop in enumerate(crops):
        x0, y0, x1, y1 = crop.box
        shown, mask = rgb[y0:y1, x0:x1], marked[y0:y1, x0:x1].astype(np.uint8) * 255
        if crop.scale < 1:
            size = shown_size(crop)
            shown = cv2.resize(shown, size, interpolation=cv2.INTER_AREA)
            mask = np.where(cv2.resize(mask, size, interpolation=cv2.INTER_AREA) > 0, 255, 0).astype(np.uint8)
        named = crop_files(directory, index)
        if not (cv2.imwrite(named["image"], np.ascontiguousarray(shown)) and cv2.imwrite(named["mask"], mask)):
            raise RuntimeError(f"OpenCV wrote no crop {index}")
        files.append(named)
    return files


def compose(picture, marked, directory: str):
    """`picture` with the fills of `prepare`'s crops put back where they belong, in place.

    A fill is an RGB PNG of the size the crop was shown at, 8 or 16 bits, in 0 to its top value.
    """
    import cv2
    import numpy as np

    groups, crops = plan(marked)
    top = np.iinfo(picture.dtype).max
    for index, crop in enumerate(crops):
        filled = cv2.imread(crop_files(directory, index)["out"], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        if filled is None or filled.shape[:2][::-1] != shown_size(crop):
            raise RuntimeError(f"no fill of the size shown for crop {index}")
        fraction = filled.astype(np.float32) / np.iinfo(filled.dtype).max
        x0, y0, x1, y1 = crop.box
        if crop.scale < 1:
            fraction = cv2.resize(fraction, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
        cx0, cy0, cx1, cy1 = crop.core
        taken = np.zeros((y1 - y0, x1 - x0), bool)
        taken[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = True
        taken &= marked[y0:y1, x0:x1] & (groups[y0:y1, x0:x1] == crop.group)
        values = np.clip(np.rint(fraction * top), 0, top).astype(picture.dtype)
        if picture.ndim == 2:
            picture[y0:y1, x0:x1][taken] = cv2.cvtColor(values, cv2.COLOR_BGR2GRAY)[taken]
        else:
            for c in range(3):
                picture[y0:y1, x0:x1, c][taken] = values[:, :, c][taken]
    return picture


def write(picture, out: str, fmt: str) -> None:
    import cv2

    if fmt == "tiff":
        params = [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_ADOBE_DEFLATE, cv2.IMWRITE_TIFF_PREDICTOR, cv2.IMWRITE_TIFF_PREDICTOR_HORIZONTAL]
    else:
        params = []
    if not cv2.imwrite(out, picture, params):
        raise RuntimeError(f"OpenCV wrote no {fmt}")


def described(picture, marked) -> dict:
    import cv2
    import numpy as np

    return {
        "width": picture.shape[1],
        "height": picture.shape[0],
        "channels": 1 if picture.ndim == 2 else picture.shape[2],
        "depth": picture.dtype.itemsize * 8,
        "filled": int(np.count_nonzero(marked)),
        "opencv": cv2.__version__,
    }


def main(argv: list[str]) -> int:
    import cv2

    # One core, as every tool in the lane is sized.
    cv2.setNumThreads(1)
    name, image, mask = argv[1], argv[2], argv[3]
    params = json.load(sys.stdin)
    try:
        picture, marked = read_both(image, mask)
    except Refusal as refusal:
        print(json.dumps({"error": refusal.code, "message": str(refusal)}))
        return REFUSED
    answer = described(picture, marked)
    if name == "opencv":
        if answer["filled"]:
            fill(picture, marked, params["method"], int(params["radius"]))
        write(picture, argv[4], params["format"])
    elif name == "prepare":
        answer["crops"] = prepare(picture, marked, argv[4]) if answer["filled"] else []
    elif name == "compose":
        if answer["filled"]:
            compose(picture, marked, argv[4])
        write(picture, argv[5], params["format"])
    else:
        raise SystemExit(f"no step {name}")
    print(json.dumps(answer))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
