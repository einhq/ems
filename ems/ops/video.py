"""`video.frames` — frames as images, at given times or spread across the file — and
`video.poster`, one frame fitted into rendition targets like a photo."""

from __future__ import annotations

import os
import re
import shutil
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import Field

from ..context import Context
from ..errors import EmsError, UNDECODABLE, invalid
from ..schemas import FileRef, Strict, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import encode
from ..tools import ffmpeg as ffmpeg_tool
from ..tools import ffprobe as ffprobe_tool
from ..tools.run import require, run
from ..tools.versions import producer
from .image import MAX_TARGETS, RenditionTarget, render_targets

# The pixel aspect ratio is the half of the display frame a JPEG cannot carry, so it is baked
# into the frame rather than left to the reader: a model given an anamorphic frame sees a squeezed
# picture. Commas inside the expressions are escaped for
# ffmpeg's filtergraph parser.
SQUARE_PIXELS_FILTER = r"scale=w=if(gt(sar\,1)\,trunc(iw*sar/2)*2\,iw):h=if(lt(sar\,1)\,trunc(ih/sar/2)*2\,ih),setsar=1"

# What ffmpeg says when a seek lands past the last frame and the output gets nothing.
NO_FRAMES_MARKER = "Nothing was written into output file"

# The time of the first frame showinfo logs, in seconds from the seek point.
SHOWINFO_TIME = re.compile(r"Parsed_showinfo\S* @ [^\]]*\].*?pts_time:\s*(-?[0-9.]+)")

# How many frames `representative` weighs for each pick: the thumbnail filter keeps the one
# nearest their average colour histogram, which passes over a black leader, a flash or a cut.
REPRESENTATIVE_BATCH = 300


class FrameOutput(Strict):
    dir: str
    format: Literal["jpeg", "png"] = "jpeg"
    quality: int = Field(default=90, ge=1, le=100)
    max: int | None = Field(default=None, gt=0)


class FramesRequest(WithOptions):
    file: FileRef
    at: list[float] | None = None
    count: int | None = Field(default=None, ge=1, le=64)
    strategy: Literal["representative", "even"] = "representative"
    output: FrameOutput


async def extract_frames(ctx: Context, path: str, out_dir: str, *, at: list[float] | None, count: int | None, strategy: str, fmt: str, quality: int, max_edge: int | None, progress: Progress, angle: int = 0, mirror: bool = False) -> list[dict]:
    """Frames with square pixels, turned by `angle`/`mirror` and nothing else: the container's
    display matrix is never applied, so the defaults give the raw frame. `max_edge` bounds the
    longer side."""
    timeout = ctx.settings.tool_stall_seconds
    ffmpeg = require("ffmpeg")
    ext = "jpg" if fmt == "jpeg" else "png"
    # JPEG quality is ffmpeg's 2..31 scale, 2 best; map 1..100 onto it.
    qscale = str(max(2, min(31, round(31 - (quality / 100) * 29))))
    finish = f",scale={encode.scale_spec(max_edge, max_edge)}" if max_edge else ""
    finish += "".join(f",{part}" for part in encode.turn_filters(angle, mirror))
    frames: list[dict] = []
    if at:
        for index, t in enumerate(at):
            out = os.path.join(out_dir, f"frame-{index:03d}.{ext}")
            args = [ffmpeg, "-y", "-v", "error", "-display_rotation", "0", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", "-vf", SQUARE_PIXELS_FILTER + finish]
            if fmt == "jpeg":
                args += ["-q:v", qscale]
            done = await run(args + [out], timeout=timeout)
            if done.code == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
                frames.append({"path": out, "t": t})
            await progress.emit("frames", fraction=(index + 1) / len(at))
        return frames

    n = count or 4
    seconds = await picture_seconds(path, timeout=timeout)
    # One span per frame; a picture of unknown length is one span from the start.
    span = seconds / n if seconds else None
    quality_args = ["-q:v", qscale] if fmt == "jpeg" else []
    last = ""
    for index in range(n if span else 1):
        out = os.path.join(out_dir, f"frame-{index:03d}.{ext}")
        if strategy == "representative":
            start, length, pick = index * (span or 0), span, f"thumbnail={REPRESENTATIVE_BATCH},"
        else:
            start, length, pick = (index + 0.5) * (span or 0), None, ""
        t, error = await _frame_from(path, out, start=start, length=length, filters=f"{pick}showinfo,{SQUARE_PIXELS_FILTER}{finish}", tail=quality_args, timeout=timeout)
        if t is not None:
            frames.append({"path": out, "t": t})
        last = error or last
        await progress.emit("frames", fraction=(index + 1) / (n if span else 1))
    if frames:
        return frames
    raise await ffprobe_tool.decode_failure(path, f"ffmpeg frame extraction failed: {last or 'no frames produced'}", timeout=timeout)


async def picture_seconds(path: str, *, timeout: float) -> float | None:
    """How long the picture runs: the video stream's duration, else the container's.

    A phone clip's picture can stop short of its sound (ein/core#618), so the stream's
    own duration comes first.
    """
    probed = await ffprobe_tool.probe(path, timeout=timeout)
    stream = probed.first("video") or {}
    return ffprobe_tool.number(stream.get("duration")) or ffprobe_tool.number(probed.format.get("duration"))


async def _frame_from(path: str, out: str, *, start: float, length: float | None, filters: str, tail: list[str], timeout: float) -> tuple[float | None, str]:
    """The first frame `filters` passes from `start` on (for at most `length` seconds) at `out`,
    and its second in the file; None when there was none, with what ffmpeg said.

    The seek is on the input, so ffmpeg decodes from the keyframe before `start` and drops what
    lies before it. Output times count from `start`; `showinfo` reports the frame's, which is
    how a pick within the span knows where it came from.
    """
    ffmpeg = require("ffmpeg")
    window = ["-t", f"{length:.3f}"] if length else []
    # showinfo logs at info, so the level is info and the banner and progress are silenced instead.
    base = [ffmpeg, "-y", "-hide_banner", "-nostats", "-v", "info", "-display_rotation", "0"]
    rest = ["-ss", f"{start:.3f}", *window, "-i", path, "-vf", filters, "-frames:v", "1", *tail, "-update", "1", out]
    attempts = []
    if ffmpeg_tool.gpu_present():
        attempts.append(base + ["-hwaccel", "cuda"] + rest)
    attempts.append(base + rest)
    error = ""
    for args in attempts:
        done = await run(args, timeout=timeout)
        if done.code == 0 and os.path.isfile(out) and os.path.getsize(out) > 0:
            shown = SHOWINFO_TIME.search(done.stderr_text)
            return round(start + (float(shown.group(1)) if shown else 0.0), 3), ""
        # A span past the end of the picture gives nothing, and will on the CPU too.
        if NO_FRAMES_MARKER in done.stderr_text or done.code == 0:
            return None, ""
        error = done.tail()
    return None, error or "no frame produced"


async def frames(ctx: Context, req: FramesRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    if not req.at and not req.count:
        raise invalid("frames needs `at` (seconds) or `count`")
    out_dir = ctx.roots.output_dir(req.output.dir)
    async with ctx.admission.slot("tools", interactive=req.options.interactive):
        await progress.emit("frames")
        got = await extract_frames(ctx, path, out_dir, at=req.at, count=req.count, strategy=req.strategy, fmt=req.output.format, quality=req.output.quality, max_edge=req.output.max, progress=progress, angle=req.file.angle, mirror=req.file.mirror)
    for frame in got:
        frame["path"] = os.path.join(req.output.dir, os.path.basename(frame["path"]))
    return Outcome({"frames": got}, producer=await producer("ffmpeg"))


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/video/frames")
    async def video_frames(request: Request, body: FramesRequest):
        return await run_op(request, lambda progress: frames(ctx, body, progress))

    register_poster(router, ctx)


class PosterRequest(WithOptions):
    file: FileRef
    at: float = Field(default=1.0, ge=0)
    deinterlace: bool = False
    targets: list[RenditionTarget] = Field(min_length=1, max_length=MAX_TARGETS)


class NoFrame(Exception):
    """The decode reached the end of the picture without a frame at the second asked for."""


async def extract_poster(ctx: Context, path: str, out: str, *, at: float, deinterlace: bool) -> None:
    """One frame at `at` seconds as a lossless-as-JPEG-gets still, in the raw frame with square pixels.

    `-ss` after `-i`: decoding from the start and dropping frames to the timestamp is
    frame-accurate, and the poster is taken a second in, so there is nothing to skip past.
    `-display_rotation 0` keeps ffmpeg from applying the container's matrix on its own — the
    caller's angle is the only thing that turns pixels, the same rule every op keeps.
    """
    timeout = ctx.settings.tool_stall_seconds
    ffmpeg = require("ffmpeg")
    filters = ("yadif," if deinterlace else "") + SQUARE_PIXELS_FILTER
    tail = ["-i", path, "-vf", filters, "-ss", f"{at:.3f}", "-vframes", "1", "-qscale:v", "0", "-y", "-f", "image2", "-update", "1", out]
    attempts = []
    if ffmpeg_tool.gpu_present():
        attempts.append([ffmpeg, "-v", "error", "-display_rotation", "0", "-hwaccel", "cuda"] + tail)
    attempts.append([ffmpeg, "-v", "error", "-display_rotation", "0"] + tail)
    last = ""
    for args in attempts:
        done = await run(args, timeout=timeout)
        if done.code == 0:
            break
        # A seek past the end: every frame is decoded and dropped, the encoder sees none, and
        # ffmpeg exits with an error saying so (or exits 0 with an empty file, checked below).
        # The same request will do the same again, so it is not worth a second decode on the CPU.
        if NO_FRAMES_MARKER in done.stderr_text:
            raise NoFrame()
        last = done.tail() or f"exit {done.code}"
    else:
        raise await ffprobe_tool.decode_failure(path, f"ffmpeg frame extraction failed: {last}", timeout=timeout)
    if not os.path.isfile(out) or os.path.getsize(out) == 0:
        raise NoFrame()


async def extract_poster_within(ctx: Context, path: str, out: str, *, at: float, deinterlace: bool) -> float:
    """The frame at `at`, or the first frame when the picture ends before it; the second taken.

    A caller cannot know where the picture ends: a container's duration counts its longest
    stream, and a phone clip's picture can stop short of its sound (a 1.6 s file with no frame
    at 1 s, ein/core#618). A file with no frame at all is `undecodable`.
    """
    for second in ([at, 0.0] if at > 0 else [at]):
        try:
            await extract_poster(ctx, path, out, at=second, deinterlace=deinterlace)
            return second
        except NoFrame:
            pass
    raise EmsError(UNDECODABLE, "no video frame")


async def poster(ctx: Context, req: PosterRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    async with ctx.admission.slot("tools", interactive=req.options.interactive):
        scratch = ctx.scratch_dir("poster")
        try:
            await progress.emit("frame")
            frame = os.path.join(scratch, "frame.jpg")
            t = await extract_poster_within(ctx, path, frame, at=req.at, deinterlace=req.deinterlace)
            await progress.emit("render")
            # The frame is a JPEG now: the caller's angle applies whole, no reader-applied turn to subtract.
            result = await render_targets(ctx, frame, req.file.angle, req.file.mirror, req.targets)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    result["t"] = t
    return Outcome(result, producer=await producer("ffmpeg", "magick"))


def register_poster(router: APIRouter, ctx: Context) -> None:
    @router.post("/video/poster")
    async def video_poster(request: Request, body: PosterRequest):
        return await run_op(request, lambda progress: poster(ctx, body, progress))
