"""`vision.embed` and `vision.embed_text`: pictures and search text in one vector space (SigLIP2)."""

from __future__ import annotations

import shutil

from fastapi import APIRouter, Request
from pydantic import Field

from ..context import Context
from ..errors import EmsError, UNSUPPORTED
from ..schemas import FileRef, WithOptions
from ..streaming import Outcome, Progress, run_op
from ._models import call_on_pictures, mimetype_of, model_slot
from .media import kind_of
from .video import extract_frames


# The keyframes a video's vectors come from, spread over the clip: enough to catch what someone is
# holding in a short clip.
VIDEO_KEYFRAMES = 8


class EmbedRequest(WithOptions):
    file: FileRef


class EmbedTextRequest(WithOptions):
    text: str = Field(min_length=1)


async def embed(ctx: Context, req: EmbedRequest, progress: Progress) -> Outcome:
    """A picture's vector in the image-text space; a video's, one per keyframe."""
    path = ctx.roots.input(req.file.path)
    mimetype = await mimetype_of(ctx, req.file, path)
    kind = kind_of(mimetype)
    deadline = req.options.deadline

    if kind == "image":
        async with model_slot(ctx, req.options):
            await progress.emit("embed")
            raw = await call_on_pictures(ctx, "embed_images", [path], req.file.angle, req.file.mirror, mimetype=mimetype, deadline=deadline)
        vectors = raw.get("embeddings") or []
        if not vectors:
            raise EmsError(UNSUPPORTED, f"{req.file.path}: the image embedder returned nothing", permanent=False)
        return Outcome({"embedding": vectors[0], "dimension": len(vectors[0])}, producer=str(raw.get("model") or "image-embed"))

    if kind != "video":
        raise EmsError(UNSUPPORTED, f"{req.file.path}: cannot embed a {kind} file as a picture")

    tmp = ctx.scratch_dir("embed")
    try:
        async with model_slot(ctx, req.options):
            await progress.emit("keyframes")
            frames = await extract_frames(ctx, path, tmp, at=None, count=VIDEO_KEYFRAMES, strategy="representative", fmt="jpeg", quality=93, max_edge=None, progress=progress)
            if not frames:
                raise EmsError(UNSUPPORTED, f"{req.file.path}: no frames to embed")
            await progress.emit("embed", message=f"{len(frames)} keyframes")
            raw = await call_on_pictures(ctx, "embed_images", [f["path"] for f in frames], req.file.angle, req.file.mirror, mimetype="image/jpeg", deadline=deadline)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    vectors = raw.get("embeddings") or []
    result = {
        "frames": [{"at": f["t"], "embedding": v} for f, v in zip(frames, vectors)],
        "dimension": len(vectors[0]) if vectors else 0,
    }
    return Outcome(result, producer=str(raw.get("model") or "image-embed"))


async def embed_text(ctx: Context, req: EmbedTextRequest, progress: Progress) -> Outcome:
    """Search text as a vector in the pictures' space."""
    async with model_slot(ctx, req.options):
        raw = await ctx.for_queries("siglip-text").call("embed_image_texts", {"texts": [req.text]}, deadline=req.options.deadline)
    vector = (raw.get("embeddings") or [[]])[0]
    return Outcome({"embedding": vector, "dimension": len(vector)}, producer=str(raw.get("model") or "image-embed"))


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/vision/embed")
    async def vision_embed(request: Request, body: EmbedRequest):
        return await run_op(request, lambda progress: embed(ctx, body, progress))

    @router.post("/vision/embed_text")
    async def vision_embed_text(request: Request, body: EmbedTextRequest):
        return await run_op(request, lambda progress: embed_text(ctx, body, progress))
