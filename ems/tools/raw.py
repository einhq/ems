"""A camera RAW demosaiced by LibRaw into a plain raster, for ImageMagick to cut renditions from.

ImageMagick reads a RAW through LibRaw as well, but always at full size: a 24 Mpx X-Trans RAF
takes ~9 s to demosaic, for a ladder whose largest output is 2560 px and a face crop of at most
384 (ein/core#969). LibRaw's half-size mode skips the demosaic, taking each 2x2 block of the
sensor as one pixel, and reads the same file in ~1 s, at exactly half the sensor frame. A
stored box is a fraction of that frame, so it lands on the same place in either; the read is
half-size whenever every output can still be cut from it, and full otherwise (a small face in a
large frame).

A full read takes the cheapest demosaic that still looks like the photo: PPG on a Bayer sensor,
and on an X-Trans the generic linear one, where Fujifilm's own (which LibRaw runs for anything
from PPG up, 3-5x the CPU) differs only by a slight softness at 1:1 that no rendition, scaled
down or cut at most at full size, shows. ImageMagick runs that 3-pass X-Trans demosaic, ~25
CPU-seconds a RAF against ~5 for either read here.

Otherwise the raster is the one ImageMagick's own read gives: the sensor frame unrotated
(`user_flip=0`; nothing in the RAW path rotates, so the caller's whole angle is still owed) and
the camera's white balance. It goes to a PPM, which costs no encode and loses nothing.

LibRaw runs in a process of its own (`python -m ems.tools.raw SOURCE OUT`, the outputs' needs on
stdin), so a decode is a tool like any other: niced, stopped when it stalls, killed with its caller.
It prints `frame W H`, the full sensor frame, which is what the renditions report as the source's.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

from ..errors import EmsError, TOOL_FAILED
from . import magick
from .run import BACKGROUND_NICE, run

# Where `-m ems.tools.raw` resolves, whatever directory the server was started in.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Decoded:
    path: str
    # The full sensor frame, unrotated, whatever size `path` was read at.
    frame: tuple[int, int]


def half_size(frame: tuple[int, int], angle: int, plans: list[magick.Plan]) -> bool:
    """Whether every output can be cut from a half-size read of a sensor frame."""
    display = (frame[1], frame[0]) if angle % 180 == 90 else frame
    return max((magick.plan_scale(plan, display) for plan in plans), default=1.0) <= 0.5


def needs(angle: int, plans: list[magick.Plan]) -> dict:
    """What the decode needs to know of the outputs to pick its size."""
    return {
        "angle": angle,
        "plans": [
            {"width": p.width, "height": p.height, "fit": p.fit, "crop": [p.crop.x, p.crop.y, p.crop.width, p.crop.height] if p.crop else None}
            for p in plans
        ],
    }


def plans_of(needed: dict) -> list[magick.Plan]:
    return [
        magick.Plan(temp="", format="", quality=None, width=p["width"], height=p["height"], fit=p["fit"], crop=magick.Rect(*p["crop"]) if p["crop"] else None)
        for p in needed["plans"]
    ]


async def decode(source: str, out: str, angle: int, plans: list[magick.Plan], *, timeout: float) -> Decoded:
    """Decode `source` to `out` at the size the outputs need."""
    done = await run([sys.executable, "-m", "ems.tools.raw", source, out], stdin=json.dumps(needs(angle, plans)).encode(), cwd=PACKAGE_ROOT, timeout=timeout, niceness=BACKGROUND_NICE)
    line = done.stdout.decode(errors="replace").strip()
    if done.code != 0 or not line.startswith("frame "):
        raise EmsError(TOOL_FAILED, f"raw decode failed: {done.tail() or f'exit {done.code}'}")
    width, height = (int(v) for v in line.split()[1:3])
    return Decoded(out, (width, height))


def main(argv: list[str]) -> int:
    import rawpy
    from PIL import Image

    source, out = argv[1], argv[2]
    needed = json.load(sys.stdin)
    with rawpy.imread(source) as raw:
        frame = (raw.sizes.width, raw.sizes.height)
        half = half_size(frame, needed["angle"], plans_of(needed))
        xtrans = raw.raw_pattern is not None and raw.raw_pattern.shape == (6, 6)
        demosaic = rawpy.DemosaicAlgorithm.LINEAR if xtrans else rawpy.DemosaicAlgorithm.PPG
        rgb = raw.postprocess(half_size=half, demosaic_algorithm=demosaic, user_flip=0, use_camera_wb=True, output_bps=8)
    Image.fromarray(rgb).save(out, format="PPM")
    print(f"frame {frame[0]} {frame[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
