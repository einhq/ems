"""`image.restore_faces` — the faces in a picture redrawn by GFPGAN, which invents their detail.

The faces are found as `faces.detect` finds them, those too small or outside the boxes asked for
are left alone, and the model worker redraws the rest and puts them back at the picture's depth
(model family `gfpgan`). Every pixel outside the faces is the picture's own. The output is written
through a private temporary file beside it, as a rendition is.
"""

from __future__ import annotations

import os
import secrets

from fastapi import APIRouter, Request
from pydantic import Field

from .. import remote
from ..context import Context
from ..errors import EmsError, TOOL_FAILED, invalid
from ..schemas import Box, FileRef, OutputSpec, WithOptions
from ..streaming import Outcome, Progress, run_op
from . import faces as faces_op
from ._models import model_slot
from .inpaint import output_format

DETECTOR = faces_op.PRODUCER.split("+")[0]
# A detection below this may be no face at all, and redrawing it would put one there.
MIN_CONFIDENCE = 0.7


class RestoreFacesRequest(WithOptions):
    file: FileRef
    faces: list[Box] | None = Field(default=None, max_length=256)
    min_size: int = Field(default=128, ge=16, le=4096)
    strength: float = Field(default=0.5, ge=0, le=1)
    output: OutputSpec


def chosen(raw: dict, req: RestoreFacesRequest) -> tuple[list[dict], int]:
    """The faces of the detector's answer to redraw, and how many were left alone."""
    frame = raw.get("frame") or {}
    width, height = float(frame.get("width") or 1), float(frame.get("height") or 1)
    picked, left = [], 0
    for face in raw.get("faces") or []:
        box = face.get("box") or {}
        x, y, w, h = (float(box.get(key, 0)) for key in ("x", "y", "width", "height"))
        cx, cy = (x + w / 2) / width, (y + h / 2) / height
        inside = req.faces is None or any(b.x <= cx <= b.x + b.width and b.y <= cy <= b.y + b.height for b in req.faces)
        if not inside or min(w, h) < req.min_size or float(face.get("confidence", 0)) < MIN_CONFIDENCE or not face.get("landmarks"):
            left += 1
            continue
        picked.append({
            "landmarks": face["landmarks"],
            "box": {"x": x / width, "y": y / height, "width": w / width, "height": h / height},
            "confidence": float(face.get("confidence", 0)),
        })
    return picked, left


async def restore_faces(ctx: Context, req: RestoreFacesRequest, progress: Progress) -> Outcome:
    if req.file.angle or req.file.mirror:
        raise invalid(f"{req.file.path}: a picture is restored as it is stored: angle 0, no mirror")
    fmt = output_format(req.output)
    image = remote.input(req.file.path)
    final = remote.output(req.output.path)
    temp = os.path.join(os.path.dirname(final), f".ems-{secrets.token_hex(8)}.{fmt}")
    try:
        async with model_slot(ctx, req.options):
            await progress.emit("detect")
            raw = await ctx.worker.call("detect_faces", {"path": image, "angle": 0, "mirror": False}, deadline=req.options.deadline)
            picked, left = chosen(raw or {}, req)
            await progress.emit("restore", message=f"{len(picked)} faces")
            answer = await ctx.worker.call(
                "restore_faces",
                {"path": image, "out": temp, "faces": [{"landmarks": f["landmarks"]} for f in picked], "strength": req.strength, "format": fmt},
                deadline=req.options.deadline, on_progress=lambda fraction: progress.emit("restore", fraction),
            )
        if not os.path.isfile(temp) or os.path.getsize(temp) == 0:
            raise EmsError(TOOL_FAILED, f"the face restorer wrote no {fmt}")
        size = os.path.getsize(temp)
        os.replace(temp, final)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    result = {
        "path": req.output.path,
        **{key: answer[key] for key in ("width", "height", "channels", "depth")},
        "bytes": size,
        "generative": True,
        "faces": [{"box": f["box"], "confidence": f["confidence"]} for f in picked],
        "left": left,
    }
    return Outcome(result, producer=f"{DETECTOR}+{answer.get('model') or 'gfpgan'}")


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/image/restore_faces")
    async def image_restore_faces(request: Request, body: RestoreFacesRequest):
        return await run_op(request, lambda progress: restore_faces(ctx, body, progress))
