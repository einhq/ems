"""The picture the defect detector is shown, and the mask made of what it found.

Two steps, each a process of its own (`python -m ems.tools.defects STEP …`, the parameters on
stdin), answering as ems.tools.inpaint's do:

- `shrink IMAGE OUT` writes the picture in grey with its shorter side `scale` pixels, a PNG of 16
  bits, for the detector to look at.
- `mask FOUND OUT` enlarges the detector's answer, each pixel's probability as a PNG of 16 bits, to
  the picture's `width` and `height`, and writes a PNG of one 8-bit channel, 255 where it reaches
  `threshold` and 0 elsewhere.
"""

from __future__ import annotations

import json
import sys

from ..errors import UNSUPPORTED
from .inpaint import REFUSED, Refusal, read

MODULE = "ems.tools.defects"


def shown_size(width: int, height: int, scale: int) -> tuple[int, int]:
    """The size the picture is shown at: its shorter side `scale` pixels."""
    factor = scale / min(width, height)
    return max(1, round(width * factor)), max(1, round(height * factor))


def shrink(picture, scale: int):
    """`picture` in grey as 16 bits, its shorter side `scale` pixels."""
    import cv2
    import numpy as np

    if picture.ndim == 3:
        picture = cv2.cvtColor(picture, cv2.COLOR_BGRA2GRAY if picture.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    grey = picture.astype(np.float32) / np.iinfo(picture.dtype).max
    height, width = grey.shape
    size = shown_size(width, height, scale)
    grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA if size[0] < width else cv2.INTER_CUBIC)
    return np.clip(np.rint(grey * 65535), 0, 65535).astype(np.uint16)


def mask(found, width: int, height: int, threshold: float):
    """The 8-bit mask of the picture's size where the probabilities `found` (16 bits) reach `threshold`."""
    import cv2
    import numpy as np

    probability = cv2.resize(found.astype(np.float32) / 65535, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.where(probability >= threshold, 255, 0).astype(np.uint8)


def main(argv: list[str]) -> int:
    import cv2
    import numpy as np

    # One core, as every tool in the lane is sized.
    cv2.setNumThreads(1)
    name = argv[1]
    params = json.load(sys.stdin)
    if name == "shrink":
        try:
            picture = read(argv[2], "file")
            if picture.dtype not in (np.uint8, np.uint16):
                raise Refusal(UNSUPPORTED.name, f"the file holds {picture.dtype} samples; only 8 and 16 bits are looked at")
        except Refusal as refusal:
            print(json.dumps({"error": refusal.code, "message": str(refusal)}))
            return REFUSED
        shown = shrink(picture, int(params["scale"]))
        if not cv2.imwrite(argv[3], shown):
            raise RuntimeError("OpenCV wrote no picture to look at")
        print(json.dumps({"width": picture.shape[1], "height": picture.shape[0], "opencv": cv2.__version__}))
    elif name == "mask":
        found = cv2.imread(argv[2], cv2.IMREAD_UNCHANGED)
        if found is None or found.ndim != 2:
            raise RuntimeError("the detector's answer cannot be read")
        marked = mask(found, int(params["width"]), int(params["height"]), float(params["threshold"]))
        if not cv2.imwrite(argv[3], marked):
            raise RuntimeError("OpenCV wrote no mask")
        count = int(np.count_nonzero(marked))
        print(json.dumps({"marked": count, "fraction": count / marked.size, "opencv": cv2.__version__}))
    else:
        raise SystemExit(f"no step {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
