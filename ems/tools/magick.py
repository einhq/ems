"""ImageMagick for renditions: one decode, every output cloned from it.

The pipeline:

    magick SOURCE -set option:filter:blur 0.8 -filter Lagrange -strip -orient Undefined
      [-rotate -A +repage] [-flop +repage]
      ( +clone [-crop …] <fit> -quality Q -print "N %w %h\\n" -write FMT:TEMP +delete ) …
      -print "frame %w %h\\n" null:

Every output is cloned from the decoded, upright source; no rendition is ever resized from
another rendition. The source is decoded once, which is the point of taking a list.

`-strip` drops the EXIF block but not ImageMagick's own orientation property — it even carries
it through the rotation. The AVIF encoder then writes that as a HEIF `irot` and every decoder
applies it, silently undoing the turn just made. `-orient Undefined` beside `-strip` is what
actually clears it.

Crops are cut with `%[fx:…]` geometry, which is why this is `magick` and not the legacy
`convert`: a stored box is a fraction of the display frame, and the fraction is resolved
against the raster ImageMagick actually decoded — the sensor image for a RAW, the turned
raster for a HEIF — rather than against dimensions a caller declared, which for those two
families are not the raster at all. The legacy CLI does not expand escapes in `-crop`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..errors import EmsError, TOOL_FAILED
from .run import require, run

FORMATS = ("avif", "webp", "jpeg", "png")

# Lagrange with a touch of blur: the downsampling filter, tuned for WebP and AVIF output.
FILTER_ARGS = ["-set", "option:filter:blur", "0.8", "-filter", "Lagrange"]
STRIP_ARGS = ["-strip", "-orient", "Undefined"]

FRAME_LINE = re.compile(r"^frame (\d+) (\d+)$")
TARGET_LINE = re.compile(r"^target (\d+) (\d+) (\d+)$")


@dataclass(frozen=True)
class Rect:
    """A crop, as fractions of the display frame, already padded and clamped to it."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Plan:
    """One output of the pipeline."""

    temp: str
    format: str
    quality: int | None
    width: int | None
    height: int | None
    fit: str
    crop: Rect | None


def padded_rect(x: float, y: float, width: float, height: float, pad: float) -> Rect:
    """Grow a box by `pad` of its own size on each side, clamped to the frame."""
    px, py = width * pad, height * pad
    x0, y0 = max(0.0, x - px), max(0.0, y - py)
    x1, y1 = min(1.0, x + width + px), min(1.0, y + height + py)
    return Rect(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def crop_args(rect: Rect) -> list[str]:
    """`-crop` against the raster in hand: at least one pixel, never past the edge."""
    geometry = (
        f"%[fx:max(1,round(w*{rect.width:.6f}))]x%[fx:max(1,round(h*{rect.height:.6f}))]"
        f"+%[fx:round(w*{rect.x:.6f})]+%[fx:round(h*{rect.y:.6f})]"
    )
    # `+gravity` first: a `cover` target earlier in the same pipeline set `-gravity Center`, and
    # a gravity makes `-crop` place its offset from the centre instead of the corner.
    return ["+gravity", "-crop", geometry, "+repage"]


def fit_args(width: int | None, height: int | None, fit: str) -> list[str]:
    """The resize for a bounding box.

    `>` on every aspect-preserving resize: shrink only, never enlarge — a source smaller than
    the box lands at its own size. `cover` is the exception on purpose: it owes the caller a
    rendition that fills the box exactly, so it enlarges a small source rather than hand back
    something that is not the shape asked for.
    """
    if width and height:
        if fit == "contain":
            return ["-resize", f"{width}x{height}>"]
        if fit == "fill":
            return ["-resize", f"{width}x{height}!"]
        return ["-resize", f"{width}x{height}^", "-gravity", "Center", "-crop", f"{width}x{height}+0+0", "+repage"]
    if width:
        return ["-resize", f"{width}x>"]
    if height:
        return ["-resize", f"x{height}>"]
    return []


def pipeline_args(source: str, angle: int, mirror: bool, plans: list[Plan]) -> list[str]:
    """Everything after `magick`. Rotation is counter-clockwise, which ImageMagick spells negative."""
    args = [source, *FILTER_ARGS, *STRIP_ARGS]
    if angle:
        args += ["-rotate", str(-angle), "+repage"]
    if mirror:
        args += ["-flop", "+repage"]
    for index, plan in enumerate(plans):
        args += ["(", "+clone"]
        if plan.crop is not None:
            args += crop_args(plan.crop)
        args += fit_args(plan.width, plan.height, plan.fit)
        args += ["-quality", str(plan.quality if plan.quality is not None else 60)]
        args += ["-print", f"target {index} %w %h\\n", "-write", f"{plan.format}:{plan.temp}", "+delete", ")"]
    args += ["-print", "frame %w %h\\n", "null:"]
    return args


@dataclass
class Rendered:
    frame: tuple[int, int]
    sizes: list[tuple[int, int]]


def parse_output(stdout: str, count: int) -> Rendered:
    frame: tuple[int, int] | None = None
    sizes: dict[int, tuple[int, int]] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if m := FRAME_LINE.match(line):
            frame = (int(m.group(1)), int(m.group(2)))
        elif m := TARGET_LINE.match(line):
            sizes[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    if frame is None or any(i not in sizes for i in range(count)):
        raise EmsError(TOOL_FAILED, "magick did not report every output it was asked for")
    return Rendered(frame, [sizes[i] for i in range(count)])


async def render(source: str, angle: int, mirror: bool, plans: list[Plan], *, timeout: float) -> Rendered:
    """Run the pipeline. The caller checks the files it asked for and publishes them."""
    done = await run([require("magick"), *pipeline_args(source, angle, mirror, plans)], timeout=timeout)
    if done.code != 0:
        raise EmsError(TOOL_FAILED, f"magick failed: {done.tail() or f'exit {done.code}'}")
    return parse_output(done.stdout.decode(errors="replace"), len(plans))
