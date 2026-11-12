"""A picture's masked pixels filled from their surroundings by OpenCV, at the picture's own depth.

OpenCV inpaints a 16-bit picture only one channel at a time, so every picture is filled channel by
channel; for an 8-bit one that gives exactly what OpenCV's three-channel fill does. An alpha
channel is kept as it is. The fill is then put back only where the mask marks, so every other
pixel is the input's, bit for bit, whatever the method does around the hole.

It runs in a process of its own (`python -m ems.tools.inpaint IMAGE MASK OUT`, the parameters on
stdin), so a fill is a tool like any other: niced, stopped when it stalls, killed with its caller.
It answers one line of JSON on stdout: what it wrote, or with exit status 2 the error code and
message of a picture it refuses.
"""

from __future__ import annotations

import json
import os
import sys

from ..errors import INVALID_REQUEST, TOOL_FAILED, UNDECODABLE, UNSUPPORTED, EmsError
from .run import run

# Where `-m ems.tools.inpaint` resolves, whatever directory the server was started in.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHODS = ("telea", "ns")
FORMATS = ("png", "tiff")
REFUSED = 2
CODES = {code.name: code for code in (INVALID_REQUEST, UNSUPPORTED, UNDECODABLE)}


async def inpaint(image: str, mask: str, out: str, *, method: str, radius: int, fmt: str, timeout: float) -> dict:
    """Fill `image` where `mask` marks and write it to `out` as `fmt`; what the tool answered."""
    params = {"method": method, "radius": radius, "format": fmt}
    done = await run([sys.executable, "-m", "ems.tools.inpaint", image, mask, out], stdin=json.dumps(params).encode(), cwd=PACKAGE_ROOT, timeout=timeout)
    lines = done.stdout.decode(errors="replace").strip().splitlines()
    try:
        answer = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        answer = None
    if done.code == REFUSED and isinstance(answer, dict) and answer.get("error") in CODES:
        raise EmsError(CODES[answer["error"]], str(answer.get("message")))
    if done.code != 0 or not isinstance(answer, dict):
        raise EmsError(TOOL_FAILED, f"inpaint failed: {done.tail() or f'exit {done.code}'}")
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


def fill(picture, marked, method: str, radius: int):
    """`picture` with the pixels `marked` (a boolean array of its height and width) filled, in place."""
    import cv2
    import numpy as np

    flag = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}[method]
    mask = marked.astype(np.uint8)
    planes = [picture] if picture.ndim == 2 else [picture[:, :, c] for c in range(min(picture.shape[2], 3))]
    for plane in planes:
        filled = cv2.inpaint(np.ascontiguousarray(plane), mask, radius, flag)
        plane[marked] = filled[marked]
    return picture


def write(picture, out: str, fmt: str) -> None:
    import cv2

    if fmt == "tiff":
        params = [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_ADOBE_DEFLATE, cv2.IMWRITE_TIFF_PREDICTOR, cv2.IMWRITE_TIFF_PREDICTOR_HORIZONTAL]
    else:
        params = []
    if not cv2.imwrite(out, picture, params):
        raise RuntimeError(f"OpenCV wrote no {fmt}")


def main(argv: list[str]) -> int:
    import cv2
    import numpy as np

    # One core, as every tool in the lane is sized.
    cv2.setNumThreads(1)
    image, mask_path, out = argv[1], argv[2], argv[3]
    params = json.load(sys.stdin)
    try:
        picture = read(image, "file")
        if picture.dtype not in (np.uint8, np.uint16):
            raise Refusal(UNSUPPORTED.name, f"the file holds {picture.dtype} samples; only 8 and 16 bits are inpainted")
        mask = read(mask_path, "mask")
        if mask.ndim != 2:
            raise Refusal(INVALID_REQUEST.name, f"the mask has {mask.shape[2]} channels; a mask has one")
        if mask.shape != picture.shape[:2]:
            raise Refusal(INVALID_REQUEST.name, f"the mask is {mask.shape[1]}x{mask.shape[0]} and the file {picture.shape[1]}x{picture.shape[0]}")
    except Refusal as refusal:
        print(json.dumps({"error": refusal.code, "message": str(refusal)}))
        return REFUSED
    marked = mask != 0
    filled = int(np.count_nonzero(marked))
    if filled:
        fill(picture, marked, params["method"], int(params["radius"]))
    write(picture, out, params["format"])
    print(json.dumps({
        "width": picture.shape[1],
        "height": picture.shape[0],
        "channels": 1 if picture.ndim == 2 else picture.shape[2],
        "depth": picture.dtype.itemsize * 8,
        "filled": filled,
        "opencv": cv2.__version__,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
