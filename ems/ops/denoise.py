"""`image.denoise` — a picture with less noise, at its own depth and channels.

For the grain and the scanner's noise of a scanned print or negative. SCUNet (model family
`denoise`) answers, tile by tile in the model worker, which reads the picture and writes the
output; the front checks the request and puts the output in place through a private temporary
file beside it, as a rendition is.
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


class DenoiseRequest(WithOptions):
    file: FileRef
    noise: Literal["15", "25", "real"] = "15"
    strength: float = Field(default=0.5, ge=0, le=1)
    output: OutputSpec


async def denoise(ctx: Context, req: DenoiseRequest, progress: Progress) -> Outcome:
    if req.file.angle or req.file.mirror:
        raise invalid(f"{req.file.path}: a picture is denoised as it is stored: angle 0, no mirror")
    fmt = output_format(req.output)
    image = remote.input(req.file.path)
    final = remote.output(req.output.path)
    temp = os.path.join(os.path.dirname(final), f".ems-{secrets.token_hex(8)}.{fmt}")
    try:
        async with model_slot(ctx, req.options):
            await progress.emit("denoise")
            answer = await ctx.worker.call(
                "denoise", {"path": image, "out": temp, "noise": req.noise, "strength": req.strength, "format": fmt},
                deadline=req.options.deadline, on_progress=lambda fraction: progress.emit("denoise", fraction),
            )
        if not os.path.isfile(temp) or os.path.getsize(temp) == 0:
            raise EmsError(TOOL_FAILED, f"the denoiser wrote no {fmt}")
        size = os.path.getsize(temp)
        os.replace(temp, final)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    result = {"path": req.output.path, **{key: answer[key] for key in ("width", "height", "channels", "depth")}, "bytes": size}
    return Outcome(result, producer=str(answer.get("model") or "scunet"))


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/image/denoise")
    async def image_denoise(request: Request, body: DenoiseRequest):
        return await run_op(request, lambda progress: denoise(ctx, body, progress))
