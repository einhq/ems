"""`image.inpaint` — a picture with the pixels a mask marks filled in from what surrounds them.

For the defects of a scanned print or negative: dust, scratches, small tears. The output keeps
the picture's depth and channels, and every pixel the mask does not mark is the picture's own, so
a restored copy never differs from the original outside the damage. OpenCV fills on the CPU, in
the tool lane (ems.tools.inpaint).

The output is written through a private temporary file beside it and put in place only once it
has bytes, as a rendition is.
"""

from __future__ import annotations

import os
import secrets
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import Field

from .. import remote
from ..context import Context
from ..errors import EmsError, TOOL_FAILED, invalid
from ..schemas import FileRef, OutputSpec, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import inpaint as inpaint_tool

EXTENSIONS = {"png": "png", "tif": "tiff", "tiff": "tiff"}


class InpaintRequest(WithOptions):
    file: FileRef
    mask: FileRef
    method: Literal["telea", "ns"] = "telea"
    radius: int = Field(default=3, ge=1, le=64)
    output: OutputSpec


def output_format(spec: OutputSpec) -> str:
    """The lossless format of an output: what was asked, else `png`; a contradicting extension is refused."""
    fmt = (spec.format or "png").lower()
    fmt = EXTENSIONS.get(fmt, fmt)
    if fmt not in inpaint_tool.FORMATS:
        raise invalid(f"{spec.path}: an inpainted picture is written as {' or '.join(inpaint_tool.FORMATS)}, not {spec.format!r}")
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
    fmt = output_format(req.output)
    image, mask = remote.input(req.file.path), remote.input(req.mask.path)
    final = remote.output(req.output.path)
    temp = os.path.join(os.path.dirname(final), f".ems-{secrets.token_hex(8)}.{fmt}")
    try:
        async with ctx.admission.slot("tools", interactive=req.options.interactive):
            await progress.emit("inpaint")
            answer = await inpaint_tool.inpaint(image, mask, temp, method=req.method, radius=req.radius, fmt=fmt, timeout=ctx.settings.tool_stall_seconds)
        if not os.path.isfile(temp) or os.path.getsize(temp) == 0:
            raise EmsError(TOOL_FAILED, f"inpaint exited 0 but wrote no {fmt}")
        size = os.path.getsize(temp)
        os.replace(temp, final)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    result = {key: answer[key] for key in ("width", "height", "channels", "depth", "filled")}
    return Outcome({"path": req.output.path, **result, "bytes": size}, producer=f"opencv/{answer.get('opencv', '?')}")


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/image/inpaint")
    async def image_inpaint(request: Request, body: InpaintRequest):
        return await run_op(request, lambda progress: inpaint(ctx, body, progress))
