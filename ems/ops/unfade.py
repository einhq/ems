"""`image.unfade` — a faded picture's colours stretched back out, at its own depth and channels.

For colour prints and slides whose dyes have faded unevenly. Deterministic, without a model: the
tool (ems.tools.unfade) runs on the CPU in the tool lane. The output is put in place through a
private temporary file beside it, as a rendition is.
"""

from __future__ import annotations

import os
import secrets

from fastapi import APIRouter, Request
from pydantic import Field

from .. import remote
from ..context import Context
from ..errors import EmsError, TOOL_FAILED, invalid
from ..schemas import FileRef, OutputSpec, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import inpaint as inpaint_tool
from .inpaint import output_format

# The version of the tool's method, in meta.producer: raised whenever the same request would give
# another picture.
METHOD = "unfade/1"


class UnfadeRequest(WithOptions):
    file: FileRef
    clip: float = Field(default=0.5, ge=0, le=5)
    balance: bool = True
    strength: float = Field(default=1.0, ge=0, le=1)
    output: OutputSpec


async def unfade(ctx: Context, req: UnfadeRequest, progress: Progress) -> Outcome:
    if req.file.angle or req.file.mirror:
        raise invalid(f"{req.file.path}: a picture is unfaded as it is stored: angle 0, no mirror")
    fmt = output_format(req.output)
    image = remote.input(req.file.path)
    final = remote.output(req.output.path)
    temp = os.path.join(os.path.dirname(final), f".ems-{secrets.token_hex(8)}.{fmt}")
    try:
        async with ctx.admission.slot("tools", interactive=req.options.interactive):
            await progress.emit("unfade")
            answer = await inpaint_tool.step(
                "unfade", [image, temp], {"clip": req.clip, "balance": req.balance, "strength": req.strength, "format": fmt},
                timeout=ctx.settings.tool_stall_seconds, module="ems.tools.unfade",
            )
        if not os.path.isfile(temp) or os.path.getsize(temp) == 0:
            raise EmsError(TOOL_FAILED, f"unfade exited 0 but wrote no {fmt}")
        size = os.path.getsize(temp)
        os.replace(temp, final)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    result = {key: answer[key] for key in ("width", "height", "channels", "depth", "levels", "gammas")}
    return Outcome({"path": req.output.path, **result, "bytes": size}, producer=f"{METHOD}+opencv/{answer.get('opencv', '?')}")


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/image/unfade")
    async def image_unfade(request: Request, body: UnfadeRequest):
        return await run_op(request, lambda progress: unfade(ctx, body, progress))
