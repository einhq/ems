"""`image.damage` — how faded, dusty, scratched, blurred and shaken a picture is, to tell which are
worth restoring.

The tool measures the fading and the blur and shrinks the picture for the defect detector (ems.tools.damage),
the model worker gives each pixel of that a probability of being damage (model family `defects`,
as for `image.defects`), and the tool sorts what it marks into specks and scratches. Nothing is
written but scratch files.
"""

from __future__ import annotations

import os
import shutil

from fastapi import APIRouter, Request

from .. import remote
from ..context import Context
from ..errors import invalid
from ..schemas import FileRef, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import damage as damage_tool
from ..tools import inpaint as inpaint_tool
from ._models import model_slot

# The version of the method, in meta.producer: raised whenever the same request would give
# other scores.
METHOD = "damage/2"
# The shorter side the detector is shown, as image.defects shows it by default, and the
# probability from which a pixel is marked.
SCALE = 640
THRESHOLD = 0.4


def producer(model: str = "bopb-detection") -> str:
    return f"{METHOD}+{model}"


class DamageRequest(WithOptions):
    file: FileRef


async def damage(ctx: Context, req: DamageRequest, progress: Progress) -> Outcome:
    if req.file.angle or req.file.mirror:
        raise invalid(f"{req.file.path}: a picture is looked at as it is stored: angle 0, no mirror")
    image = remote.input(req.file.path)
    timeout = ctx.settings.tool_stall_seconds
    scratch = ctx.scratch_dir("damage")
    shown, found = os.path.join(scratch, "shown.png"), os.path.join(scratch, "found.png")
    try:
        async with ctx.admission.slot("tools", interactive=req.options.interactive):
            await progress.emit("look")
            looked = await inpaint_tool.step("look", [image, shown], {"scale": SCALE}, timeout=timeout, module=damage_tool.MODULE)
        async with model_slot(ctx, req.options):
            await progress.emit("detect")
            detected = await ctx.worker.call("detect_defects", {"path": shown, "out": found, "dark": False}, deadline=req.options.deadline)
        async with ctx.admission.slot("tools", interactive=req.options.interactive):
            await progress.emit("count")
            counted = await inpaint_tool.step("count", [found, shown], {"threshold": THRESHOLD}, timeout=timeout, module=damage_tool.MODULE)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    result = {
        "width": looked["width"],
        "height": looked["height"],
        "faded": looked["faded"],
        "dusty": counted["dusty"],
        "scratched": counted["scratched"],
        "blurred": looked["blurred"],
        "shaken": looked["shaken"],
        "measures": {
            "change": looked["change"],
            "edge": looked["edge"],
            "shake": looked["shake"],
            "specks": counted["specks"],
            "scratches": counted["scratches"],
            "length": counted["length"],
            "marked": counted["marked"],
        },
    }
    return Outcome(result, producer=producer(detected.get("model") or "bopb-detection"))


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/image/damage")
    async def image_damage(request: Request, body: DamageRequest):
        return await run_op(request, lambda progress: damage(ctx, body, progress))
