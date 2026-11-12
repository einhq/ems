"""`image.inpaint` — a picture with the pixels a mask marks filled in from what surrounds them.

For the defects of a scanned print or negative: dust, scratches, tears. The output keeps the
picture's depth and channels, and every pixel the mask does not mark is the picture's own, so a
restored copy never differs from the original outside the damage. OpenCV fills on the CPU, in the
tool lane. LaMa (model family `lama`) is shown crops around the damage that the tool cuts and puts
back (ems.tools.inpaint), so a large scan never goes to the card whole.

The output is written through a private temporary file beside it and put in place only once it
has bytes, as a rendition is.
"""

from __future__ import annotations

import os
import secrets
import shutil
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import Field

from .. import remote
from ..context import Context
from ..errors import EmsError, TOOL_FAILED, invalid
from ..schemas import FileRef, OutputSpec, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import inpaint as inpaint_tool
from ._models import model_slot

EXTENSIONS = {"png": "png", "tif": "tiff", "tiff": "tiff"}


class InpaintRequest(WithOptions):
    file: FileRef
    mask: FileRef
    method: Literal["telea", "ns", "lama"] = "telea"
    radius: int = Field(default=3, ge=1, le=64)
    output: OutputSpec


def output_format(spec: OutputSpec) -> str:
    """The lossless format of an output: what was asked, else `png`; a contradicting extension is refused."""
    fmt = (spec.format or "png").lower()
    fmt = EXTENSIONS.get(fmt, fmt)
    if fmt not in inpaint_tool.FORMATS:
        raise invalid(f"{spec.path}: a restored picture is written as {' or '.join(inpaint_tool.FORMATS)}, not {spec.format!r}")
    if spec.quality is not None:
        raise invalid(f"{spec.path}: {fmt} is lossless and takes no quality")
    named = EXTENSIONS.get(os.path.splitext(spec.path)[1].lower().lstrip("."))
    if named is not None and named != fmt:
        raise invalid(f"{spec.path}: named as {named} but asked to hold {fmt}")
    return fmt


async def inpaint(ctx: Context, req: InpaintRequest, progress: Progress) -> Outcome:
    for ref in (req.file, req.mask):
        if ref.angle or ref.mirror:
            raise invalid(f"{ref.path}: a picture is inpainted as it is stored: angle 0, no mirror")
    if req.method == "lama" and "radius" in req.model_fields_set:
        raise invalid("radius is OpenCV's; lama takes none")
    fmt = output_format(req.output)
    image, mask = remote.input(req.file.path), remote.input(req.mask.path)
    final = remote.output(req.output.path)
    temp = os.path.join(os.path.dirname(final), f".ems-{secrets.token_hex(8)}.{fmt}")
    try:
        if req.method == "lama":
            answer, model = await lama(ctx, req, image, mask, temp, fmt, progress)
        else:
            async with ctx.admission.slot("tools", interactive=req.options.interactive):
                await progress.emit("inpaint")
                answer = await inpaint_tool.step("opencv", [image, mask, temp], {"method": req.method, "radius": req.radius, "format": fmt}, timeout=ctx.settings.tool_stall_seconds)
            model = None
        if not os.path.isfile(temp) or os.path.getsize(temp) == 0:
            raise EmsError(TOOL_FAILED, f"inpaint exited 0 but wrote no {fmt}")
        size = os.path.getsize(temp)
        os.replace(temp, final)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    result = {key: answer[key] for key in ("width", "height", "channels", "depth", "filled")}
    opencv = f"opencv/{answer.get('opencv', '?')}"
    return Outcome({"path": req.output.path, **result, "bytes": size}, producer=f"{model}+{opencv}" if model else opencv)


async def lama(ctx: Context, req: InpaintRequest, image: str, mask: str, out: str, fmt: str, progress: Progress) -> tuple[dict, str]:
    """Fill with LaMa: the tool cuts the crops, the model worker paints them, the tool puts them
    back and writes `out`. What the tool answered, and the model's name."""
    timeout = ctx.settings.tool_stall_seconds
    scratch = ctx.scratch_dir("inpaint")
    try:
        async with ctx.admission.slot("tools", interactive=req.options.interactive):
            await progress.emit("crop")
            prepared = await inpaint_tool.step("prepare", [image, mask, scratch], {}, timeout=timeout)
        crops = prepared["crops"]
        async with model_slot(ctx, req.options):
            await progress.emit("inpaint", message=f"{len(crops)} crops")
            painted = await ctx.worker.call("inpaint", {"crops": crops}, deadline=req.options.deadline, on_progress=lambda fraction: progress.emit("inpaint", fraction))
        async with ctx.admission.slot("tools", interactive=req.options.interactive):
            await progress.emit("compose")
            answer = await inpaint_tool.step("compose", [image, mask, scratch, out], {"format": fmt}, timeout=timeout)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return answer, str(painted.get("model") or "lama")


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/image/inpaint")
    async def image_inpaint(request: Request, body: InpaintRequest):
        return await run_op(request, lambda progress: inpaint(ctx, body, progress))
