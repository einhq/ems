"""`video.shots` — where a video cuts from one shot to the next: the starts of its chapters.

The picture is decoded once, on the card when there is one, to TransNetV2's own input — every
frame at 48×27, rgb24, in a scratch file — with each frame's second written beside it by ffmpeg's
`metadata` filter. The model worker scores the frames and turns the scores into cuts by explicit
rules whose defaults were measured on the archive (shot_rules.py); a caller passes its own to
tune them. Without a card, the CPU decodes the file only when the whole decode fits the CPU
budget a transcode is held to; a file over it is `refused`.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil

from fastapi import APIRouter, Request
from pydantic import Field

from ..context import Context
from ..errors import EmsError, REFUSED, UNDECODABLE, invalid
from ..schemas import FileRef, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import encode
from ..tools import ffmpeg as ffmpeg_tool
from ..tools import ffprobe as ffprobe_tool
from ..tools.run import require, run
from ..tools.versions import producer
from ._models import model_slot

FRAME_BYTES = 48 * 27 * 3

# The `metadata` filter's line for each frame it prints.
FRAME_TIME = re.compile(r"^frame:\s*\d+\s+pts:\s*\S+\s+pts_time:\s*(-?[0-9.]+)")


class ShotsRequest(WithOptions):
    file: FileRef
    threshold: float = Field(default=0.6, gt=0, lt=1)
    nms: float = Field(default=0.5, ge=0)
    merge: float = Field(default=1.0, ge=0)
    min_shot: float = Field(default=3.0, ge=0)
    min_change: float = Field(default=0.10, ge=0, le=1)
    debug: bool = False


def decode_attempts(path: str, frames: str, times: str, *, gpu: bool) -> list[tuple[str, list[str]]]:
    """The decodes to try, each named for the producer: the card's, then the CPU's.

    The frame's second is printed by `metadata` into a file of its own rather than by `showinfo`
    into the log: a tape's 180 000 lines would sit in memory with the rest of ffmpeg's stderr.
    `signalstats` is there to give every frame a key for `metadata` to print, and only the one key
    is printed. The card scales to yuv444p: 27 rows are odd, and nv12 garbles the last of them.
    """
    ffmpeg = require("ffmpeg")
    stamp = f"format=yuv420p,signalstats,metadata=mode=print:key=lavfi.signalstats.YAVG:file={times},format=rgb24"
    base = [ffmpeg, "-y", "-hide_banner", "-nostats", "-v", "error", "-display_rotation", "0"]
    out = ["-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough", "-f", "rawvideo", frames]
    attempts = []
    if gpu:
        attempts.append(("cuda", base + ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", path, "-vf", f"scale_cuda=48:27:interp_algo=bicubic:format=yuv444p,hwdownload,format=yuv444p,{stamp}", *out]))
    attempts.append(("cpu", base + ["-i", path, "-vf", f"scale=48:27,{stamp}", *out]))
    return attempts


def frame_times(path: str) -> list[float]:
    """Each frame's second, from the file the `metadata` filter printed, read a line at a time."""
    with open(path) as handle:
        return [float(m.group(1)) for m in map(FRAME_TIME.match, handle) if m]


async def decode(ctx: Context, path: str, scratch: str, *, gpu: bool, cpu_allowed) -> tuple[str, list[float], str]:
    """The video's frames at 48×27 in a file, each frame's second, and the chain that decoded them."""
    frames, times = os.path.join(scratch, "frames.rgb"), os.path.join(scratch, "times.txt")
    last = ""
    for chain, args in decode_attempts(path, frames, times, gpu=gpu):
        if chain == "cpu":
            refusal = cpu_allowed()
            if refusal is not None:
                raise EmsError(REFUSED, refusal, permanent=True)
        done = await run(args, timeout=ctx.settings.tool_stall_seconds)
        if done.code == 0 and os.path.isfile(frames) and os.path.getsize(frames) >= FRAME_BYTES and os.path.isfile(times):
            return frames, await asyncio.to_thread(frame_times, times), chain
        last = done.tail() or f"exit {done.code}"
    raise await ffprobe_tool.decode_failure(path, f"ffmpeg shot decode failed: {last}", timeout=ctx.settings.tool_stall_seconds)


async def shots(ctx: Context, req: ShotsRequest, progress: Progress) -> Outcome:
    if req.file.angle or req.file.mirror:
        raise invalid("shots are found in an upright file, such as a rendition: angle 0, no mirror")
    path = ctx.roots.input(req.file.path)
    timeout = ctx.settings.tool_stall_seconds
    settings = ctx.settings
    budget = encode.Budget(settings.cpu_encode_max_mb, settings.cpu_encode_mb_per_1k_frames, settings.cpu_encode_max_segments)

    scratch = ctx.scratch_dir("shots")
    try:
        async with ctx.admission.slot("encodes", interactive=req.options.interactive):
            await progress.emit("probe")
            probed = await ffprobe_tool.probe(path, timeout=timeout)
            if probed.refusal:
                raise EmsError(UNDECODABLE, f"source cannot be read: {probed.refusal}")
            stream = probed.first("video") or {}
            duration = ffprobe_tool.number(stream.get("duration")) or ffprobe_tool.number(probed.format.get("duration"))
            if not duration:
                raise EmsError(UNDECODABLE, "no video picture with a length")
            rate = ffprobe_tool.fraction(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "")) or 30.0

            def cpu_allowed() -> str | None:
                # The decode is not split into windows here: it fits whole, or the file is refused.
                projected = encode.project_cpu_decode(int(duration) + 1, round(rate * 1000), 1000, budget)
                if isinstance(projected, encode.Refusal):
                    return projected.reason
                count, mb = projected
                if mb > budget.max_mb:
                    return f"a CPU decode of {count} frames projects to {mb} MB, over the {budget.max_mb} MB budget, and shots are found in one decode, not windows"
                return None

            await progress.emit("decode")
            frames, times, chain = await decode(ctx, path, scratch, gpu=ffmpeg_tool.gpu_present(), cpu_allowed=cpu_allowed)
        if not times:
            raise EmsError(UNDECODABLE, "no video frames")

        await progress.emit("score", message=f"{len(times)} frames")
        async with model_slot(ctx, req.options):
            raw = await ctx.worker.call("shot_boundaries", {
                "path": frames,
                "times": times,
                "duration": duration,
                "threshold": req.threshold,
                "nms": req.nms,
                "merge": req.merge,
                "min_shot": req.min_shot,
                "min_change": req.min_change,
                "debug": req.debug,
            }, deadline=req.options.deadline)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    result = {
        "duration": duration,
        "frames": raw.get("frames", len(times)),
        "cuts": raw.get("cuts", []),
        "gaps": raw.get("gaps", []),
        # What the cuts were found with, so a caller can tell a result made by other rules.
        "params": {"threshold": req.threshold, "nms": req.nms, "merge": req.merge, "min_shot": req.min_shot, "min_change": req.min_change},
    }
    if req.debug:
        result["candidates"] = raw.get("candidates", [])
        result["scores"] = raw.get("scores", [])
    return Outcome(result, producer=f"{raw.get('model') or 'transnetv2'}+{await producer('ffmpeg')}/{chain}")


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/video/shots")
    async def video_shots(request: Request, body: ShotsRequest):
        return await run_op(request, lambda progress: shots(ctx, body, progress))
