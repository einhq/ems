"""`image.copy_tags` — a picture with another file's metadata: EXIF, XMP, IPTC and the colour profile.

For a picture made from another, such as a restored copy, whose pixels are new but whose date,
camera, place and description are the original's. The file's pixels are not touched: exiftool
writes the output in the file's own format. Run on the CPU in the tool lane.
"""

from __future__ import annotations

import json
import os
import secrets

from fastapi import APIRouter, Request
from pydantic import Field

from .. import remote
from ..context import Context
from ..errors import EmsError, TOOL_FAILED, invalid
from ..schemas import FileRef, OutputSpec, Strict, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools.run import require, run
from ..tools.versions import tool_version

DIGITAL_SOURCE_TYPES = "http://cv.iptc.org/newscodes/digitalsourcetype/"

# The file's turn (counter-clockwise, then flopped) → the EXIF Orientation that shows it upright.
ORIENTATION = {(0, False): 1, (90, False): 8, (180, False): 3, (270, False): 6, (0, True): 2, (90, True): 7, (180, True): 4, (270, True): 5}

# The source's tags that are not the output's: its embedded small pictures, which show it as it
# was, its orientation, which the file's turn replaces, and its instance id, since the output is
# another instance. exiftool never copies the tags describing the source's own layout.
LEFT_OUT = ["--Orientation", "--ThumbnailImage", "--PreviewImage", "--XMP-xmpMM:InstanceID"]

# One spelling a format: the output is written in the file's format, so its name ends the same.
SAME_EXTENSION = {"jpeg": "jpg", "tiff": "tif"}


class HistoryEvent(Strict):
    """An XMP `stEvt` ResourceEvent: what was done to the picture."""

    action: str = Field(min_length=1, max_length=64)
    when: str | None = Field(default=None, max_length=64)
    software_agent: str | None = Field(default=None, max_length=256)
    parameters: str | None = Field(default=None, max_length=4096)
    changed: str | None = Field(default=None, max_length=256)


class CopyTagsRequest(WithOptions):
    file: FileRef
    source: FileRef
    history: list[HistoryEvent] = Field(default_factory=list, max_length=64)
    digital_source_type: str | None = Field(default=None, pattern=r"^[A-Za-z]+$")
    output: OutputSpec


def _extension(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return SAME_EXTENSION.get(ext, ext)


async def _exiftool(args: list[str], timeout: float) -> bytes:
    done = await run([require("exiftool"), "-m", "-q", "-q", *args], timeout=timeout)
    if done.code != 0:
        raise EmsError(TOOL_FAILED, f"exiftool failed: {done.tail()}", permanent=True)
    return done.stdout


async def copy_tags(ctx: Context, req: CopyTagsRequest, progress: Progress) -> Outcome:
    if req.source.angle or req.source.mirror:
        raise invalid(f"{req.source.path}: the source's tags are copied, never turned: angle 0, no mirror")
    if req.output.format or req.output.quality:
        raise invalid("the output is the file in its own format: no format or quality")
    if _extension(req.output.path) != _extension(req.file.path):
        raise invalid(f"{req.output.path}: the output is written in the file's format, so its name ends as {req.file.path}'s does")
    image = remote.input(req.file.path)
    source = remote.input(req.source.path)
    final = remote.output(req.output.path)
    scratch = os.path.dirname(final)
    token = secrets.token_hex(8)
    temp = os.path.join(scratch, f".ems-{token}{os.path.splitext(final)[1]}")
    tags = os.path.join(scratch, f".ems-{token}.json")
    timeout = ctx.settings.tool_stall_seconds
    try:
        async with ctx.admission.slot("tools", interactive=req.options.interactive):
            await progress.emit("copy")
            await _exiftool(["-all=", "-tagsFromFile", source, "-all:all", "-icc_profile", *LEFT_OUT, "-o", temp, image], timeout)
            if not os.path.isfile(temp):
                raise EmsError(TOOL_FAILED, "exiftool exited 0 but wrote no output")
            written: dict = {"SourceFile": "*"}
            if req.history:
                had = json.loads(await _exiftool(["-j", "-struct", "-XMP-xmpMM:History", temp], timeout) or b"[{}]")
                events = had[0].get("History") or []
                for event in req.history:
                    events.append({key: value for key, value in (
                        ("Action", event.action), ("When", event.when), ("SoftwareAgent", event.software_agent),
                        ("Parameters", event.parameters), ("Changed", event.changed),
                    ) if value is not None})
                written["XMP-xmpMM:History"] = events
            if req.digital_source_type:
                written["XMP-iptcExt:DigitalSourceType"] = DIGITAL_SOURCE_TYPES + req.digital_source_type
            turn = ORIENTATION[(req.file.angle, req.file.mirror)]
            args = ["-overwrite_original", "-Orientation="]
            if turn != 1:
                args.append(f"-EXIF:Orientation#={turn}")
            if len(written) > 1:
                with open(tags, "w", encoding="utf-8") as out:
                    json.dump([written], out)
                args.append(f"-j={tags}")
            await _exiftool([*args, temp], timeout)
        size = os.path.getsize(temp)
        os.replace(temp, final)
    finally:
        for path in (temp, tags):
            if os.path.exists(path):
                os.remove(path)
    return Outcome({"path": req.output.path, "bytes": size}, producer=f"exiftool/{await tool_version('exiftool')}")


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/image/copy_tags")
    async def image_copy_tags(request: Request, body: CopyTagsRequest):
        return await run_op(request, lambda progress: copy_tags(ctx, body, progress))
