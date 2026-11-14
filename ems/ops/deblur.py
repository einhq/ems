"""`image.deblur` — a picture with its blur taken out, at its own depth and channels.

For a photograph out of focus or blurred by a moving camera. Restormer (model family `deblur`)
answers, tile by tile in the model worker, which reads the picture and writes the output; the front
checks the request and puts the output in place through a private temporary file beside it, as a
rendition is.
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
from ._models import model_slot
from .inpaint import output_format


class DeblurRequest(WithOptions):
    file: FileRef
    blur: Literal["focus", "motion"] = "focus"
    strength: float = Field(default=1.0, ge=0, le=1)
    edge: int = Field(default=2048, ge=256, le=8192)
    output: OutputSpec


async def deblur(ctx: Context, req: DeblurRequest, progress: Progress) -> Outcome:
    if req.file.angle or req.file.mirror:
        raise invalid(f"{req.file.path}: a picture is deblurred as it is stored: angle 0, no mirror")
    fmt = output_format(req.output)
    image = remote.input(req.file.path)
    final = remote.output(req.output.path)
    temp = os.path.join(os.path.dirname(final), f".ems-{secrets.token_hex(8)}.{fmt}")
    try:
        async with model_slot(ctx, req.options):
            await progress.emit("deblur")
            answer = await ctx.worker.call(
                "deblur", {"path": image, "out": temp, "blur": req.blur, "strength": req.strength, "edge": req.edge, "format": fmt},
                deadline=req.options.deadline, on_progress=lambda fraction: progress.emit("deblur", fraction),
            )
        if not os.path.isfile(temp) or os.path.getsize(temp) == 0:
            raise EmsError(TOOL_FAILED, f"the deblurrer wrote no {fmt}")
        size = os.path.getsize(temp)
        os.replace(temp, final)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    result = {"path": req.output.path, **{key: answer[key] for key in ("width", "height", "channels", "depth")}, "bytes": size}
    return Outcome(result, producer=str(answer.get("model") or "restormer"))


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/image/deblur")
    async def image_deblur(request: Request, body: DeblurRequest):
        return await run_op(request, lambda progress: deblur(ctx, body, progress))
