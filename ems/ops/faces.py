"""`faces.detect`, `faces.detect_video` and `faces.embed` — InsightFace, boxes as fractions of the
display frame."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import Field

from ..context import Context
from ..errors import EmsError, UNDECODABLE, UNSUPPORTED
from ..schemas import Box, FileRef, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import decode as decode_tool
from ..tools import ffprobe as ffprobe_tool
from ._models import model_slot

PRODUCER = "insightface/buffalo_l"


class DetectRequest(WithOptions):
    file: FileRef
    # Smallest face to report, as a fraction of the frame's shorter side.
    min_size: float = Field(default=0.0, ge=0, le=1)


class DetectVideoRequest(WithOptions):
    file: FileRef
    # Seconds between the frames looked at; a long video is sampled more sparsely (_4.11a_).
    interval: float = Field(default=1.0, ge=0.2, le=60)
    min_size: float = Field(default=0.0, ge=0, le=1)
    # Detections below this confidence are left out before faces are followed.
    min_confidence: float = Field(default=0.0, ge=0, le=1)


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


async def detect_video(ctx: Context, req: DetectVideoRequest, progress: Progress) -> Outcome:
    """The faces a video shows, each with when it is seen — one call, one decode of the video
    (`inference_ops.detect_video_faces`). A file with no picture shows none."""
    path = ctx.roots.input(req.file.path)
    probed = await ffprobe_tool.probe(path, timeout=ctx.settings.tool_stall_seconds)
    if probed.refusal:
        raise EmsError(UNDECODABLE, f"faces.detect_video: {probed.refusal}")
    stream = probed.first("video")
    if not stream:
        return Outcome({"frame": {"width": 0, "height": 0}, "duration": None, "interval": req.interval, "samples": 0, "faces": []}, producer=PRODUCER)
    # The picture's own length, which a phone clip's sound can outlast (ein/core#618).
    duration = ffprobe_tool.number(stream.get("duration")) or ffprobe_tool.number(probed.format.get("duration"))
    async with model_slot(ctx, req.options):
        await progress.emit("detect")
        raw = await ctx.worker.call(
            "detect_video_faces",
            {
                "path": path,
                "angle": req.file.angle,
                "mirror": req.file.mirror,
                "interval": req.interval,
                "duration": duration,
                "min_size": req.min_size,
                "min_confidence": req.min_confidence,
            },
            deadline=req.options.deadline,
        )
    faces = []
    for face in raw.get("faces") or []:
        entry = {
            "appearances": face.get("appearances") or [],
            "detections": int(face.get("detections") or 0),
            "t": float(face.get("t") or 0),
            "box": face.get("box") or {},
            "confidence": float(face.get("confidence") or 0),
            "embedding": face.get("embedding") or [],
            "track": face.get("track") or [],
        }
        if face.get("landmarks"):
            entry["landmarks"] = face["landmarks"]
        pose = pose_of(face)
        if pose:
            entry["pose"] = pose
        faces.append(entry)
    return Outcome(
        {
            "frame": raw.get("frame") or {"width": 0, "height": 0},
            "duration": duration,
            "interval": float(raw.get("interval") or req.interval),
            "samples": int(raw.get("samples") or 0),
            "faces": faces,
        },
        producer=PRODUCER,
    )


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

    @router.post("/faces/detect_video")
    async def faces_detect_video(request: Request, body: DetectVideoRequest):
        return await run_op(request, lambda progress: detect_video(ctx, body, progress))

    @router.post("/faces/embed")
    async def faces_embed(request: Request, body: EmbedRequest):
        return await run_op(request, lambda progress: embed(ctx, body, progress))
