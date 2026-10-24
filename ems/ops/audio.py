"""`audio.waveform` — a rendered picture of a recording's envelope, fitted like a rendition.

A waveform is drawn, not resized into shape, so each target picks its own drawing box: a
`cover` target is drawn square (it owes its caller a square), any other target is drawn as a
banner `aspect` times wider than tall that fits inside the box. Every picture is drawn at
twice its size and scaled down once: `showwavespic` writes hard per-column pixels with no
antialiasing of its own, and that downscale is the only thing that smooths the envelope.

One ffmpeg decode draws every picture — the audio is split inside the filter graph — and the
decode is the cost: the picture is about a thousand columns wide however long the track is.
`aresample=8000` ahead of the drawing is what makes a thirty-minute recording cheap; at 8 kHz a
column still averages thousands of samples.

`filter=average` with `scale=sqrt`, and deliberately no `compand`: on a long recording compand
fills the frame with a solid mass (at a second per column the peak is near maximum
everywhere), a linear average barely registers a quiet tape, and a peak filter comes out
hairy. sqrt lifts the quiet passages in the *drawing*, which is where the problem is,
and leaves the loud/quiet contrast that makes a recording recognisable.
"""

from __future__ import annotations

import os
import shutil

from fastapi import APIRouter, Request
from pydantic import Field

from ..context import Context
from ..errors import EmsError, TOOL_FAILED, invalid
from ..schemas import FileRef, Strict, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools.run import require, run
from ..tools.versions import producer
from .image import MAX_TARGETS, RenditionTarget, render_targets

# The envelope, the ground and the baseline, for a tile on a dark ground. A caller with other
# colours states them.
DEFAULT_FOREGROUND = "#4a90f0"
DEFAULT_BACKGROUND = "#0a0b0e"
DEFAULT_BASELINE = "#2b70d6"
DEFAULT_ASPECT = 3.0
SAMPLE_RATE = 8000
OVERSAMPLE = 2


class WaveformStyle(Strict):
    foreground: str = DEFAULT_FOREGROUND
    background: str = DEFAULT_BACKGROUND
    baseline: str | None = DEFAULT_BASELINE
    aspect: float = Field(default=DEFAULT_ASPECT, gt=0)


class WaveformRequest(WithOptions):
    file: FileRef
    targets: list[RenditionTarget] = Field(min_length=1, max_length=MAX_TARGETS)
    style: WaveformStyle = Field(default_factory=WaveformStyle)


def drawing_box(target: RenditionTarget, aspect: float) -> tuple[int, int]:
    """The picture a target is drawn as, before the 2x oversampling."""
    if target.box is None or not target.box.width or not target.box.height:
        raise invalid(f"{target.output.path}: a waveform target needs a box with width and height")
    width, height = target.box.width, target.box.height
    if target.fit == "cover":
        return width, height
    return width, min(height, max(1, int(width / aspect)))


def _colour(value: str) -> str:
    if not value.startswith("#") or len(value) not in (7, 9) or any(c not in "0123456789abcdefABCDEF" for c in value[1:]):
        raise invalid(f"{value!r}: colours are #rrggbb")
    return value


def filter_graph(boxes: list[tuple[int, int]], style: WaveformStyle) -> str:
    fg, bg = _colour(style.foreground), _colour(style.background)
    baseline = _colour(style.baseline) if style.baseline else None
    n = len(boxes)
    split = "".join(f"[a{i}]" for i in range(n))
    parts = [f"[0:a]aformat=channel_layouts=mono,aresample={SAMPLE_RATE},asplit={n}{split}"]
    for i, (w, h) in enumerate(boxes):
        size = f"{w * OVERSAMPLE}x{h * OVERSAMPLE}"
        parts.append(f"[a{i}]showwavespic=s={size}:colors={fg}:filter=average:scale=sqrt[fg{i}]")
        parts.append(f"color=s={size}:color={bg}[bg{i}]")
        tail = f",drawbox=y=(ih-1)/2:w=iw:h=1:color={baseline}" if baseline else ""
        parts.append(f"[bg{i}][fg{i}]overlay=format=rgb{tail}[out{i}]")
    return ";".join(parts)


async def draw(ctx: Context, path: str, scratch: str, boxes: list[tuple[int, int]], style: WaveformStyle) -> list[str]:
    """Every picture from one decode, as PNGs in `scratch`."""
    outs = [os.path.join(scratch, f"wave-{i}.png") for i in range(len(boxes))]
    args = [require("ffmpeg"), "-y", "-v", "error", "-i", path, "-filter_complex", filter_graph(boxes, style)]
    for i, out in enumerate(outs):
        args += ["-map", f"[out{i}]", "-frames:v", "1", out]
    done = await run(args, timeout=ctx.settings.tool_stall_seconds)
    if done.code != 0:
        raise EmsError(TOOL_FAILED, f"ffmpeg waveform rendering failed: {done.tail() or f'exit {done.code}'}")
    for out in outs:
        if not os.path.isfile(out) or os.path.getsize(out) == 0:
            raise EmsError(TOOL_FAILED, "ffmpeg exited 0 but drew no waveform", permanent=True)
    return outs


async def waveform(ctx: Context, req: WaveformRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    boxes = [drawing_box(target, req.style.aspect) for target in req.targets]
    async with ctx.admission.slot("tools", interactive=req.options.interactive):
        scratch = ctx.scratch_dir("waveform")
        try:
            await progress.emit("draw")
            pictures = await draw(ctx, path, scratch, boxes, req.style)
            await progress.emit("render")
            targets = []
            for picture, target, (w, h) in zip(pictures, req.targets, boxes):
                # The drawing already has the target's shape: this only scales it down 2x -> 1x.
                fitted = target.model_copy(update={"fit": "contain", "box": target.box.model_copy(update={"width": w, "height": h})})
                rendered = await render_targets(ctx, picture, 0, False, [fitted])
                targets.append(rendered["targets"][0])
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    return Outcome({"targets": targets}, producer=await producer("ffmpeg", "magick"))


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/audio/waveform")
    async def audio_waveform(request: Request, body: WaveformRequest):
        return await run_op(request, lambda progress: waveform(ctx, body, progress))
