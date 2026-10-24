"""`fingerprint.compute` — fingerprints for finding copies: SSCD for pictures and frames, chromaprint for sound.

An SSCD descriptor is a unit vector, so a vector store answers near-duplicate queries by cosine
similarity; a picture's is taken in its display frame. A chromaprint is the whole recording's
sequence of 32-bit values, one per ~0.124 s: two copies match only once aligned, which is the
caller's to do.
"""

from __future__ import annotations

import shutil
from typing import Literal

from fastapi import APIRouter, Request

from ..context import Context
from ..errors import EmsError, TOOL_FAILED, UNSUPPORTED, invalid
from ..schemas import FileRef, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools.run import require, run
from ..tools.versions import producer
from ._models import call_on_pictures, mimetype_of, model_slot
from .media import kind_of
from .video import extract_frames

Kind = Literal["sscd-image", "sscd-video", "chromaprint"]
DEFAULT_KINDS = {"image": ["sscd-image"], "video": ["sscd-video"], "audio": ["chromaprint"]}
VIDEO_KEYFRAMES = 8

# Chromaprint stacks FFT frames before it emits anything, so a clip below its window, about
# two and a half seconds, has no acoustic fingerprint at all and never will: fpcalc exits with
# "Empty fingerprint" or answers no values. Both mean nothing to store, and neither is an error.
FPCALC_EMPTY_MARKER = "Empty fingerprint"


class FingerprintRequest(WithOptions):
    file: FileRef
    kinds: list[Kind] | None = None


def _entry(kind: str, vector: list[float], label: str) -> dict:
    return {"kind": kind, "dimension": len(vector), "vector": vector, "label": label}


async def sscd_image(ctx: Context, req: FingerprintRequest, path: str, mimetype: str | None) -> tuple[list[dict], str]:
    async with model_slot(ctx, req.options):
        raw = await call_on_pictures(ctx, "copy_descriptors", [path], req.file.angle, req.file.mirror, mimetype=mimetype, deadline=req.options.deadline)
    vectors = raw.get("descriptors") or []
    if not vectors:
        raise EmsError(UNSUPPORTED, f"{req.file.path}: the copy detector returned nothing", permanent=False)
    return [_entry("sscd-image", vectors[0], "primary")], str(raw.get("model") or "copy-detect")


async def sscd_video(ctx: Context, req: FingerprintRequest, path: str, progress: Progress) -> tuple[list[dict], str]:
    tmp = ctx.scratch_dir("fingerprint")
    try:
        await progress.emit("keyframes")
        async with ctx.admission.slot("tools", interactive=req.options.interactive):
            frames = [f["path"] for f in await extract_frames(ctx, path, tmp, at=None, count=VIDEO_KEYFRAMES, strategy="representative", fmt="jpeg", quality=100, max_edge=None, progress=progress)]
        await progress.emit("describe", message=f"{len(frames)} keyframes")
        async with model_slot(ctx, req.options):
            raw = await ctx.worker.call("copy_descriptors", {"paths": frames, "angle": req.file.angle, "mirror": req.file.mirror}, deadline=req.options.deadline)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    vectors = raw.get("descriptors") or []
    return [_entry("sscd-video", v, f"keyframe-{i}") for i, v in enumerate(vectors)], str(raw.get("model") or "copy-detect")


def parse_fpcalc(stdout: str) -> list[int]:
    for line in stdout.splitlines():
        if line.startswith("FINGERPRINT="):
            return [int(x) for x in line[len("FINGERPRINT="):].split(",") if x]
    return []


async def chromaprint(ctx: Context, path: str) -> list[dict]:
    # `-length 0` takes the whole recording rather than fpcalc's first two minutes.
    done = await run([require("fpcalc"), "-raw", "-length", "0", path], timeout=ctx.settings.tool_stall_seconds)
    if done.code != 0:
        if FPCALC_EMPTY_MARKER in done.stderr_text:
            return []
        raise EmsError(TOOL_FAILED, f"fpcalc failed: {done.tail()}")
    values = parse_fpcalc(done.stdout.decode(errors="replace"))
    if not values:
        return []
    return [{"kind": "chromaprint", "label": "primary", "fingerprint": values}]


async def compute(ctx: Context, req: FingerprintRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    mimetype = await mimetype_of(ctx, req.file, path)
    kind = kind_of(mimetype)
    kinds = req.kinds if req.kinds is not None else DEFAULT_KINDS.get(kind)
    if not kinds:
        raise EmsError(UNSUPPORTED, f"{req.file.path}: no fingerprint applies to a {kind} file")
    fingerprints: list[dict] = []
    models: list[str] = []
    tools: list[str] = []
    for wanted in kinds:
        if wanted == "sscd-image":
            if kind != "image":
                raise invalid(f"sscd-image needs an image, got {kind}")
            await progress.emit("describe")
            prints, model = await sscd_image(ctx, req, path, mimetype)
        elif wanted == "sscd-video":
            if kind != "video":
                raise invalid(f"sscd-video needs a video, got {kind}")
            prints, model = await sscd_video(ctx, req, path, progress)
            tools.append("ffmpeg")
        else:
            await progress.emit("chromaprint")
            async with ctx.admission.slot("tools", interactive=req.options.interactive):
                prints = await chromaprint(ctx, path)
            model = None
            tools.append("fpcalc")
        fingerprints += prints
        if model and model not in models:
            models.append(model)
    parts = [*models, await producer(*dict.fromkeys(tools))] if tools else models
    return Outcome({"fingerprints": fingerprints}, producer="+".join(parts))


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/fingerprint/compute")
    async def fingerprint_compute(request: Request, body: FingerprintRequest):
        return await run_op(request, lambda progress: compute(ctx, body, progress))
