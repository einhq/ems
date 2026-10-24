"""`video.storyboard` — small frames of a video at an even interval, tiled into one sheet: what a
player's timeline shows under the pointer.

One sheet per video, whatever its length: the interval grows with the length rather than the
sheet, so a three-minute film gets a frame a second and a two-hour tape one every half minute,
which is still a few pixels of a timeline per frame.

The whole picture is decoded on the card, where the frames never touch the CPU's memory; without
one, only keyframes are, which is coarse — a frame repeats until the next keyframe — but bounded.
The sheet is encoded like a rendition (`render_targets`), through a temporary file beside it.
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass

from fastapi import APIRouter, Request
from pydantic import Field

from ..context import Context
from ..errors import EmsError, UNDECODABLE, invalid
from ..schemas import FileRef, OutputSpec, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import ffmpeg as ffmpeg_tool
from ..tools import ffprobe as ffprobe_tool
from ..tools.run import require, run
from ..tools.versions import producer
from .image import RenditionTarget, render_targets
from .video import SHOWINFO_TIME

# The largest a sheet may be on either side: what every browser decodes without a second thought.
MAX_SHEET_EDGE = 4096


class StoryboardRequest(WithOptions):
    file: FileRef
    tile: int = Field(default=160, ge=16, le=512)
    columns: int = Field(default=12, ge=1, le=64)
    max_tiles: int = Field(default=240, ge=1, le=4096)
    min_interval: float = Field(default=1.0, gt=0)
    output: OutputSpec


@dataclass(frozen=True)
class SheetPlan:
    tile_width: int
    tile_height: int
    interval: float
    count: int
    columns: int
    rows: int


def _even(value: float) -> int:
    """At least 2 and even, as a yuv420 frame's sides have to be."""
    return max(2, int(round(value / 2)) * 2)


def storyboard_plan(duration: float, width: int, height: int, sar: float | None, *, tile: int, columns: int, max_tiles: int, min_interval: float) -> SheetPlan:
    """The sheet for a picture of `width`×`height` stored pixels running `duration` seconds.

    A tile is the picture's display shape — stored pixels widened by the sample aspect ratio —
    with its longer side `tile`. Tile `i` is the frame at `i × interval`, and the interval is
    the longer of `min_interval` and the length shared out over `max_tiles`.
    """
    aspect = (width * (sar or 1.0)) / height
    if aspect >= 1:
        tile_width, tile_height = _even(tile), _even(tile / aspect)
    else:
        tile_width, tile_height = _even(tile * aspect), _even(tile)
    interval = max(min_interval, duration / max_tiles)
    count = max(1, min(max_tiles, math.ceil(duration / interval - 1e-9)))
    used_columns = min(columns, count)
    return SheetPlan(tile_width, tile_height, round(interval, 6), count, used_columns, math.ceil(count / used_columns))


def decode_attempts(path: str, sheet: str, plan: SheetPlan, *, gpu: bool) -> list[tuple[str, list[str]]]:
    """The ffmpeg runs to try in order, each named for the producer: the card's, then keyframes on the CPU.

    `fps` keeps the frame nearest each tick; the scale squares the pixels, since the tile is
    already the display shape; `showinfo` logs every frame that reaches the sheet; `tile` lays
    them out and emits the sheet when the grid is full, or at the end of the picture.
    """
    ffmpeg = require("ffmpeg")
    tick = f"fps=fps=1/{plan.interval:g}:round=near"
    size = f"{plan.tile_width}:{plan.tile_height}"
    grid = f"showinfo,tile={plan.columns}x{plan.rows}"
    # showinfo logs at info, so the level is info and the banner and progress are silenced instead.
    base = [ffmpeg, "-y", "-hide_banner", "-nostats", "-v", "info", "-display_rotation", "0"]
    out = ["-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", "1", "-update", "1", sheet]
    attempts = []
    if gpu:
        attempts.append(("cuda", base + ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", path, "-vf", f"{tick},scale_cuda={size}:interp_algo=bicubic:format=nv12,hwdownload,format=nv12,setsar=1,{grid}", *out]))
    attempts.append(("keyframes", base + ["-skip_frame", "nokey", "-i", path, "-vf", f"{tick},scale={size}:flags=area,setsar=1,{grid}", *out]))
    return attempts


async def storyboard(ctx: Context, req: StoryboardRequest, progress: Progress) -> Outcome:
    if req.file.angle or req.file.mirror:
        raise invalid("a storyboard is made from an upright file, such as a rendition: angle 0, no mirror")
    columns_px = req.columns * req.tile
    rows_px = math.ceil(req.max_tiles / req.columns) * req.tile
    if max(columns_px, rows_px) > MAX_SHEET_EDGE:
        raise invalid(f"a sheet of {req.max_tiles} tiles of {req.tile} px in {req.columns} columns can be {columns_px}x{rows_px} px; at most {MAX_SHEET_EDGE} a side")
    path = ctx.roots.input(req.file.path)
    timeout = ctx.settings.tool_stall_seconds

    async with ctx.admission.slot("encodes", interactive=req.options.interactive):
        await progress.emit("probe")
        probed = await ffprobe_tool.probe(path, timeout=timeout)
        if probed.refusal:
            raise EmsError(UNDECODABLE, f"source cannot be read: {probed.refusal}")
        stream = probed.first("video") or {}
        width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
        duration = ffprobe_tool.number(stream.get("duration")) or ffprobe_tool.number(probed.format.get("duration"))
        if not width or not height or not duration:
            raise EmsError(UNDECODABLE, "no video picture with a length")
        plan = storyboard_plan(duration, width, height, ffprobe_tool.ratio(stream.get("sample_aspect_ratio")), tile=req.tile, columns=req.columns, max_tiles=req.max_tiles, min_interval=req.min_interval)

        scratch = ctx.scratch_dir("storyboard")
        try:
            sheet = os.path.join(scratch, "sheet.png")
            await progress.emit("decode")
            chain, count, last = "", 0, ""
            for name, args in decode_attempts(path, sheet, plan, gpu=ffmpeg_tool.gpu_present()):
                done = await run(args, timeout=timeout)
                if done.code == 0 and os.path.isfile(sheet) and os.path.getsize(sheet) > 0:
                    chain, count = name, len(SHOWINFO_TIME.findall(done.stderr_text))
                    break
                last = done.tail() or f"exit {done.code}"
            else:
                raise await ffprobe_tool.decode_failure(path, f"ffmpeg storyboard failed: {last}", timeout=timeout)
            await progress.emit("render")
            rendered = await render_targets(ctx, sheet, 0, False, [RenditionTarget(output=req.output)])
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    written = rendered["targets"][0]
    return Outcome({
        **written,
        "tile": {"width": plan.tile_width, "height": plan.tile_height},
        "columns": plan.columns,
        "rows": plan.rows,
        # What reached the sheet: the plan's count, or fewer when the picture ends before the last tick.
        "count": min(count or plan.count, plan.count),
        "interval": plan.interval,
        "duration": duration,
    }, producer=f"{await producer('ffmpeg')}/{chain}+{await producer('magick')}")


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/video/storyboard")
    async def video_storyboard(request: Request, body: StoryboardRequest):
        return await run_op(request, lambda progress: storyboard(ctx, body, progress))
