"""`image.defects` — where a scanned print or negative is damaged, as a mask `image.inpaint` takes.

The tool shrinks the picture to the scale asked for, in grey, the model worker gives each pixel of
that a probability of being damage (model family `defects`), and the tool enlarges the answer to
the picture's size and marks what reaches the threshold (ems.tools.defects). The mask is written
through a private temporary file beside the output, as a rendition is.
"""

from __future__ import annotations

import os
import secrets
import shutil

from fastapi import APIRouter, Request
from pydantic import Field

from .. import remote
from ..context import Context
from ..errors import EmsError, TOOL_FAILED, invalid
from ..schemas import FileRef, OutputSpec, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import defects as defects_tool
from ..tools import inpaint as inpaint_tool
from ._models import model_slot


class DefectsRequest(WithOptions):
    file: FileRef
    scale: int = Field(default=640, ge=256, le=1536)
    threshold: float = Field(default=0.4, ge=0, le=1)
    dark: bool = False
    output: OutputSpec


async def defects(ctx: Context, req: DefectsRequest, progress: Progress) -> Outcome:
    if req.file.angle or req.file.mirror:
        raise invalid(f"{req.file.path}: a picture is looked at as it is stored: angle 0, no mirror")
    if (req.output.format or "png").lower() != "png" or req.output.quality is not None:
        raise invalid(f"{req.output.path}: a mask is written as png, which takes no quality")
    if os.path.splitext(req.output.path)[1].lower() not in ("", ".png"):
        raise invalid(f"{req.output.path}: a mask is written as png")
    image = remote.input(req.file.path)
    final = remote.output(req.output.path)
    temp = os.path.join(os.path.dirname(final), f".ems-{secrets.token_hex(8)}.png")
    timeout = ctx.settings.tool_stall_seconds
    scratch = ctx.scratch_dir("defects")
    shown, found = os.path.join(scratch, "shown.png"), os.path.join(scratch, "found.png")
    try:
        async with ctx.admission.slot("tools", interactive=req.options.interactive):
            await progress.emit("shrink")
            picture = await inpaint_tool.step("shrink", [image, shown], {"scale": req.scale}, timeout=timeout, module=defects_tool.MODULE)
        async with model_slot(ctx, req.options):
            await progress.emit("detect")
            detected = await ctx.worker.call("detect_defects", {"path": shown, "out": found, "dark": req.dark}, deadline=req.options.deadline)
        async with ctx.admission.slot("tools", interactive=req.options.interactive):
            await progress.emit("mask")
            masked = await inpaint_tool.step("mask", [found, temp], {"width": picture["width"], "height": picture["height"], "threshold": req.threshold},
                                             timeout=timeout, module=defects_tool.MODULE)
        if not os.path.isfile(temp) or os.path.getsize(temp) == 0:
            raise EmsError(TOOL_FAILED, "the mask step exited 0 but wrote no png")
        os.replace(temp, final)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if os.path.exists(temp):
            os.remove(temp)
    result = {"path": req.output.path, "width": picture["width"], "height": picture["height"], "marked": masked["marked"], "fraction": masked["fraction"]}
    return Outcome(result, producer=f"{detected.get('model') or 'defects'}+opencv/{masked.get('opencv', '?')}")


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/image/defects")
    async def image_defects(request: Request, body: DefectsRequest):
        return await run_op(request, lambda progress: defects(ctx, body, progress))
