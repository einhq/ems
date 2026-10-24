"""`ocr.read` — the lines of text in a picture, each with its corners and its score.

`engine: photo` is PP-OCRv6 (model family `ocr`, ems/modelworker/text_reader.py): scene text in
photographs, read in the display frame the file reference names. A RAW or HEIF the readers
cannot open is decoded out of process, as for `vision.embed`. The language of the text is told
from the reading, or given (`language`), and comes back in `meta.language`.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import Field

from ..context import Context
from ..errors import EmsError, UNSUPPORTED
from ..schemas import FileRef, WithOptions
from ..streaming import Outcome, Progress, run_op
from ._models import call_on_pictures, mimetype_of, model_slot
from .media import kind_of


class ReadRequest(WithOptions):
    file: FileRef
    engine: Literal["photo"] = "photo"
    # ISO 639-1, as a person set it: the reading is not asked which language it is in.
    language: str | None = Field(default=None, pattern=r"^[a-z]{2}$")


async def read(ctx: Context, req: ReadRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    mimetype = await mimetype_of(ctx, req.file, path)
    kind = kind_of(mimetype)
    if kind != "image":
        raise EmsError(UNSUPPORTED, f"{req.file.path}: cannot read text in a {kind} file")
    async with model_slot(ctx, req.options):
        await progress.emit("read")
        raw = await call_on_pictures(ctx, "read_text", [path], req.file.angle, req.file.mirror, mimetype=mimetype, deadline=req.options.deadline,
                                     extra={"language": req.language} if req.language else None)
    lines = raw.get("lines") or []
    result = {
        "text": str(raw.get("text") or ""),
        "lines": lines,
        "width": raw.get("width"),
        "height": raw.get("height"),
    }
    return Outcome(result, producer=str(raw.get("model") or "ocr"), meta={"language": raw.get("language")})


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/ocr/read")
    async def ocr_read(request: Request, body: ReadRequest):
        return await run_op(request, lambda progress: read(ctx, body, progress))
