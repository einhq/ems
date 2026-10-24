"""`image.renditions` — sized copies of one image, from one decode.

Every target is fitted from the same decoded, upright source: a bounding box with `contain`
(fits inside, never upscales), `cover` (fills exactly, cropping the centre) or `fill`
(stretches), optionally after a `crop` cut out of the display frame and padded — which is what
a face crop is. Outputs carry no metadata and no orientation tag; the pixels are upright.

The op writes every output through a private temporary file beside it and publishes all of
them only once every one has bytes, so a caller never sees a partial rendition and a failed
call leaves nothing behind. The renditions are the caller's to name, sync and swap.
"""

from __future__ import annotations

import os
import secrets
import shutil
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import Field

from ..context import Context
from ..errors import EmsError, TOOL_FAILED, invalid
from ..schemas import Box, FileRef, OutputSpec, Strict, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import decode as decode_tool
from ..tools import magick
from ..tools.versions import producer

MAX_TARGETS = 32


class BoxSize(Strict):
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


class RenditionTarget(Strict):
    output: OutputSpec
    box: BoxSize | None = None
    fit: Literal["contain", "cover", "fill"] = "contain"
    crop: Box | None = None
    pad: float = Field(default=0.0, ge=0, le=2)


class RenditionsRequest(WithOptions):
    file: FileRef
    targets: list[RenditionTarget] = Field(min_length=1, max_length=MAX_TARGETS)


def output_format(spec: OutputSpec) -> str:
    """The encoder for an output: what was asked, else `avif`; a contradicting extension is refused."""
    fmt = (spec.format or "avif").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in magick.FORMATS:
        raise invalid(f"{spec.path}: unsupported image format {spec.format!r} (one of {', '.join(magick.FORMATS)})")
    ext = os.path.splitext(spec.path)[1].lower().lstrip(".")
    named = "jpeg" if ext == "jpg" else ext
    if named in magick.FORMATS and named != fmt:
        raise invalid(f"{spec.path}: named .{ext} but asked to hold {fmt}")
    return fmt


def plan_targets(ctx: Context, targets: list[RenditionTarget]) -> tuple[list[magick.Plan], list[str]]:
    """Resolve every output under a writable root and pair it with a private temp path."""
    plans: list[magick.Plan] = []
    finals: list[str] = []
    for target in targets:
        final = ctx.roots.output(target.output.path)
        fmt = output_format(target.output)
        if target.box is not None and target.box.width is None and target.box.height is None:
            raise invalid(f"{target.output.path}: box names neither width nor height")
        crop = magick.padded_rect(target.crop.x, target.crop.y, target.crop.width, target.crop.height, target.pad) if target.crop else None
        if crop is not None and (crop.width <= 0 or crop.height <= 0):
            raise invalid(f"{target.output.path}: crop lies outside the frame")
        temp = os.path.join(os.path.dirname(final), f".ems-{secrets.token_hex(8)}")
        plans.append(magick.Plan(
            temp=temp,
            format=fmt,
            quality=target.output.quality,
            width=target.box.width if target.box else None,
            height=target.box.height if target.box else None,
            fit=target.fit,
            crop=crop,
        ))
        finals.append(final)
    return plans, finals


def discard(plans: list[magick.Plan]) -> None:
    for plan in plans:
        try:
            os.remove(plan.temp)
        except FileNotFoundError:
            pass


async def render_targets(ctx: Context, source: str, angle: int, mirror: bool, targets: list[RenditionTarget], requested: list[str] | None = None) -> dict:
    """Render every target from `source` and publish them all, or none.

    `requested` is what each output path was called in the request, so the result names the
    paths the caller knows rather than the resolved ones.
    """
    plans, finals = plan_targets(ctx, targets)
    names = requested or [t.output.path for t in targets]
    try:
        rendered = await magick.render(source, angle, mirror, plans, timeout=ctx.settings.tool_stall_seconds)
        for plan in plans:
            if not os.path.isfile(plan.temp) or os.path.getsize(plan.temp) == 0:
                raise EmsError(TOOL_FAILED, f"magick exited 0 but wrote no bytes for {plan.format}:{os.path.basename(plan.temp)}")
        results = []
        for plan, final, name, (width, height) in zip(plans, finals, names, rendered.sizes):
            size = os.path.getsize(plan.temp)
            os.replace(plan.temp, final)
            results.append({"path": name, "width": width, "height": height, "bytes": size})
    finally:
        discard(plans)
    return {"frame": {"width": rendered.frame[0], "height": rendered.frame[1]}, "targets": results}


async def owed_angle(ctx: Context, path: str, file: FileRef) -> int:
    """The turn ImageMagick still has to make: the caller's, minus what libheif already did.

    A HEIF is the one family whose reader applies the container's rotation on its own, so a
    caller's full `angle` would turn it a second time. The residual is what is left — zero in
    the ordinary case, and exactly the hand correction when someone has overridden the tag.
    """
    angle = decode_tool.normalize_angle(file.angle)
    mimetype = file.mimetype
    if mimetype is None and ctx.exiftool is not None:
        try:
            exif = await ctx.exiftool.extract(path)
            mimetype = str(exif.get("MIMEType") or "") or None
        except EmsError:
            mimetype = None
    if decode_tool.is_heif(mimetype):
        applied = await decode_tool.applied_angle(path, ctx.settings.tool_stall_seconds)
        return decode_tool.normalize_angle(angle - applied)
    return angle


async def renditions(ctx: Context, req: RenditionsRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    async with ctx.admission.slot("tools", interactive=req.options.interactive):
        await progress.emit("render")
        angle = await owed_angle(ctx, path, req.file)
        try:
            result = await render_targets(ctx, path, angle, req.file.mirror, req.targets)
        except EmsError as exc:
            # A RAW ImageMagick cannot demosaic — a truncated raw section reads as an I/O error
            # in libraw while the file itself opens fine — usually still carries its embedded
            # JPEG, which is what the model ops fall back to as well. Same raster orientation as
            # the sensor image (nothing in the RAW path rotates), so the caller's angle applies
            # whole. Only a `tool_failed` from the render, and only for a RAW: anything else
            # is answered as it was.
            if exc.code is not TOOL_FAILED or not decode_tool.is_raw(req.file.mimetype):
                raise
            scratch = ctx.scratch_dir("preview")
            try:
                preview = await decode_tool._preview(path, scratch, ctx.settings.tool_stall_seconds)
                if preview is None:
                    raise
                await progress.log("warning", f"rendering from the embedded preview: {exc.message}")
                result = await render_targets(ctx, preview, angle, req.file.mirror, req.targets)
                result["source"] = "preview"
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
    return Outcome(result, producer=await producer("magick"))


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/image/renditions")
    async def image_renditions(request: Request, body: RenditionsRequest):
        return await run_op(request, lambda progress: renditions(ctx, body, progress))
