"""Shared plumbing for the ops that forward to the model worker."""

from __future__ import annotations

from contextlib import asynccontextmanager

from ..context import Context
from ..errors import EmsError, UNSUPPORTED
from ..schemas import FileRef, Options
from ..tools import decode as decode_tool


@asynccontextmanager
async def model_slot(ctx: Context, options: Options):
    async with ctx.admission.slot("models", interactive=options.interactive):
        yield


async def mimetype_of(ctx: Context, file: FileRef, path: str) -> str | None:
    """The file's mimetype: as the caller named it, else as exiftool reads it, else by its extension."""
    if file.mimetype:
        return file.mimetype
    if ctx.exiftool is not None:
        try:
            exif = await ctx.exiftool.extract(path)
            if exif.get("MIMEType"):
                return str(exif["MIMEType"])
        except EmsError:
            pass
    import mimetypes

    return mimetypes.guess_type(path)[0]


async def call_on_pictures(ctx: Context, op: str, paths: list[str], angle: int, mirror: bool, *, mimetype: str | None, deadline: float | None,
                           extra: dict | None = None) -> dict:
    """A worker op over pictures, decoding a single RAW or HEIF out of process when its readers cannot.
    `extra` goes to the worker beside the pictures."""
    payload = {"paths": paths, "angle": angle, "mirror": mirror, **(extra or {})}
    try:
        return await ctx.worker.call(op, payload, deadline=deadline)
    except EmsError as exc:
        if exc.code is not UNSUPPORTED or len(paths) != 1 or not decode_tool.is_unreadable(exc.message):
            raise
    decoded = await decode_tool.decode_for_models(paths[0], mimetype, angle, mirror, scratch=ctx.settings.scratch, timeout=ctx.settings.tool_stall_seconds)
    try:
        return await ctx.worker.call(op, {**payload, "paths": [decoded.path], "angle": decoded.angle, "mirror": decoded.mirror}, deadline=deadline)
    finally:
        decoded.cleanup()
