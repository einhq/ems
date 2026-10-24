"""`faces.detect` and `faces.embed` — InsightFace, boxes as fractions of the display frame."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import Field

from ..context import Context
from ..errors import EmsError, UNSUPPORTED
from ..schemas import Box, FileRef, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import decode as decode_tool
from ._models import model_slot

PRODUCER = "insightface/buffalo_l"


class DetectRequest(WithOptions):
    file: FileRef
    # Smallest face to report, as a fraction of the frame's shorter side.
    min_size: float = Field(default=0.0, ge=0, le=1)


class EmbedRequest(WithOptions):
    file: FileRef
    box: Box


def pose_of(face: dict) -> dict | None:
    """Head pose in degrees — `pitch`, `yaw`, `roll`, all 0 facing the camera — or None when the
    worker did not report one. Angles, not coordinates, so nothing to normalize."""
    pose = face.get("pose")
    if not isinstance(pose, dict):
        return None
    try:
        return {key: float(pose[key]) for key in ("pitch", "yaw", "roll")}
    except (KeyError, TypeError, ValueError):
        return None


def to_fractions(raw: dict, min_size: float) -> dict:
    """The worker measures in pixels of the display frame; the API speaks in fractions."""
    frame = raw.get("frame") or {}
    width, height = float(frame.get("width") or 0), float(frame.get("height") or 0)
    faces = []
    if width > 0 and height > 0:
        shorter = min(width, height)
        for face in raw.get("faces") or []:
            box = face.get("box") or {}
            bw, bh = float(box.get("width", 0)), float(box.get("height", 0))
            if min_size > 0 and min(bw, bh) < min_size * shorter:
                continue
            entry = {
                "box": {
                    "x": max(0.0, float(box.get("x", 0)) / width),
                    "y": max(0.0, float(box.get("y", 0)) / height),
                    "width": min(1.0, bw / width),
                    "height": min(1.0, bh / height),
                },
                "confidence": float(face.get("confidence", 0)),
                "embedding": face.get("embedding") or [],
            }
            landmarks = face.get("landmarks")
            if landmarks:
                entry["landmarks"] = [[float(x) / width, float(y) / height] for x, y in landmarks]
            pose = pose_of(face)
            if pose:
                entry["pose"] = pose
            faces.append(entry)
    return {"frame": {"width": int(width), "height": int(height)}, "faces": faces}


async def _with_fallback(ctx: Context, op: str, path: str, file: FileRef, extra: dict, deadline: float | None) -> dict | None:
    """The worker's answer, decoding the file out of process when its readers cannot open it."""
    payload = {"path": path, "angle": file.angle, "mirror": file.mirror, **extra}
    try:
        return await ctx.worker.call(op, payload, deadline=deadline)
    except EmsError as exc:
        if exc.code is not UNSUPPORTED or not decode_tool.is_unreadable(exc.message):
            raise
    decoded = await decode_tool.decode_for_models(path, file.mimetype, file.angle, file.mirror, scratch=ctx.settings.scratch, timeout=ctx.settings.tool_stall_seconds)
    try:
        return await ctx.worker.call(op, {**payload, "path": decoded.path, "angle": decoded.angle, "mirror": decoded.mirror}, deadline=deadline)
    finally:
        decoded.cleanup()


async def detect(ctx: Context, req: DetectRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    async with model_slot(ctx, req.options):
        await progress.emit("detect")
        raw = await _with_fallback(ctx, "detect_faces", path, req.file, {}, req.options.deadline)
    return Outcome(to_fractions(raw or {}, req.min_size), producer=PRODUCER)


async def embed(ctx: Context, req: EmbedRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    async with model_slot(ctx, req.options):
        await progress.emit("embed")
        raw = await _with_fallback(ctx, "embed_face", path, req.file, {"box": req.box.model_dump()}, req.options.deadline)
    if raw is None:
        return Outcome(None, producer=PRODUCER)
    result = {"embedding": raw.get("embedding") or [], "confidence": float(raw.get("confidence", 0))}
    pose = pose_of(raw)
    if pose:
        result["pose"] = pose
    return Outcome(result, producer=PRODUCER)


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/faces/detect")
    async def faces_detect(request: Request, body: DetectRequest):
        return await run_op(request, lambda progress: detect(ctx, body, progress))

    @router.post("/faces/embed")
    async def faces_embed(request: Request, body: EmbedRequest):
        return await run_op(request, lambda progress: embed(ctx, body, progress))
