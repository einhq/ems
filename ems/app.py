"""The HTTP front. `create_app()` builds it; `python -m ems` serves it."""

from __future__ import annotations

import fnmatch
import os
import sys
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import API_VERSION
from .admission import Admission, current_caller, current_op
from .auth import require_key
from .card import CardArbiter, CardBudget
from .config import Settings
from .context import Context
from .errors import INTERNAL, INVALID_REQUEST, OP_NOT_SERVED, EmsError
from .ops import audio, document, faces, fingerprint, image, inpaint, media, ocr, shots, speech, storyboard, system, text, transcode, video, vision, watches
from .remote import Mounts
from .tools.audio import AudioCache
from .tools.exiftool import ExifTool
from .tools.run import which
from .clips import ffmpeg_join, ffmpeg_segments
from .watches import Watches, ffmpeg_crop, ffmpeg_frames
from .worker import QUERY_FAMILIES, Worker

OP_MODULES = (media, image, inpaint, video, storyboard, shots, audio, transcode, fingerprint, document, ocr, faces, speech, vision, text, watches)


def log(message: str) -> None:
    print(f"ems: {message}", file=sys.stderr, flush=True)


class NameTheOp:
    """Sets `current_op` (`/v0/vision/embed` -> `vision.embed`) and `current_caller` (the name
    its key has, `Caller`) for the request's task. An unknown key names no one; `require_key`
    refuses it.

    Plain ASGI rather than `@app.middleware("http")`: that one wraps the response in a stream of
    its own, and the event-stream ops are the long ones this exists to name. Not in `require_key`:
    a dependency that is not async runs in a thread, and what it sets stays there.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            current_op.set(scope["path"].removeprefix("/v0/").replace("/", "."))
            current_caller.set(caller_of(scope))
        await self.app(scope, receive, send)


def caller_of(scope) -> str:
    """The name of the caller whose key the request carries: `anonymous` with authentication off."""
    app = scope.get("app")
    keys = app.state.settings.keys if app is not None else {}
    if not keys:
        return "anonymous"
    for name, value in scope.get("headers") or ():
        if name == b"authorization":
            scheme, _, token = value.decode("latin-1").partition(" ")
            caller = keys.get(token.strip()) if scheme.lower() == "bearer" else None
            return caller.name if caller is not None else "unknown"
    return "unknown"


def require_served(request: Request) -> None:
    """Refuses an op outside EMS_OPS before its body is read, so a caller learns it from any request."""
    op = request.url.path.removeprefix("/v0/").replace("/", ".")
    if request.method == "POST" and not request.app.state.settings.serves(op):
        raise EmsError(OP_NOT_SERVED, f"{op} is not served by this instance ({request.app.state.settings.name})")


def make_workers(settings: Settings, env: dict[str, str] | None = None) -> tuple[Worker, Worker | None]:
    """The batch worker, and the query worker when EMS_QUERY_WORKER is on (Context.for_queries).

    The query worker holds the query families and nothing else, on the CPU, with no GPU visible
    — a CUDA context alone costs it a few hundred MB of the card for models that never use it.
    The batch worker then leaves those families out of its pins and runs at EMS_WORKER_NICE.
    """
    batch = Worker(settings.worker_script, settings.worker_socket, env=env, log=log)
    if not settings.query_worker:
        return batch, None
    pinned = [name.strip() for name in batch.env.get("CFG_MODEL_PINNED", "").split(",") if name.strip()]
    batch.env["CFG_MODEL_PINNED"] = ",".join(name for name in pinned if name not in QUERY_FAMILIES)
    batch.env["CFG_MODEL_NICE"] = str(settings.worker_nice)
    query = Worker(
        settings.worker_script,
        settings.query_worker_socket,
        env=env,
        log=lambda message: log(f"query {message}"),
        overrides={
            "CFG_MODEL_PINNED": ",".join(QUERY_FAMILIES),
            "CFG_MODEL_ROLE": "query",
            "CFG_MODEL_MAX_RSS": settings.query_worker_max_rss,
            "CFG_TEXT_EMBED_QUERY_DEVICE": "cpu",
            "CFG_IMAGE_EMBED_TEXT_DEVICE": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            # torch reads these at import, onnxruntime through CFG_MODEL_THREADS (image_embed).
            "OMP_NUM_THREADS": str(settings.query_worker_threads),
            "MKL_NUM_THREADS": str(settings.query_worker_threads),
            "CFG_MODEL_THREADS": str(settings.query_worker_threads),
        },
    )
    return batch, query


def make_card_worker(settings: Settings, env: dict[str, str] | None = None) -> Worker:
    """The query worker's twin on the card, which the CardArbiter runs: with nothing loaded while
    the card is busy, and with the query families pinned on the card while it is quiet. Which of
    them is the arbiter's choice at each switch, by what fits.
    """
    return Worker(
        settings.worker_script,
        settings.card_worker_socket,
        env=env,
        log=lambda message: log(f"card {message}"),
        overrides={
            "CFG_MODEL_PINNED": "",
            "CFG_MODEL_ROLE": "card",
            "CFG_MODEL_MAX_RSS": settings.card_worker_max_rss,
            "CFG_TEXT_EMBED_QUERY_DEVICE": "cuda",
            "CFG_IMAGE_EMBED_TEXT_DEVICE": "cuda",
            "OMP_NUM_THREADS": str(settings.query_worker_threads),
            "MKL_NUM_THREADS": str(settings.query_worker_threads),
            "CFG_MODEL_THREADS": str(settings.query_worker_threads),
        },
    )


def create_app(
    settings: Settings | None = None,
    *,
    worker: Worker | None = None,
    query_worker: Worker | None = None,
    start_worker: bool = True,
    card_worker: Worker | None = None,
    card_stats=None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if worker is None:
        worker, made_query_worker = make_workers(settings)
        query_worker = query_worker or made_query_worker
        if query_worker is not None and settings.card_quiet_after > 0:
            card_worker = card_worker or make_card_worker(settings)
    ctx = Context(
        settings=settings,
        admission=Admission(
            limit_models=settings.limit_models,
            limit_tools=settings.limit_tools,
            queue_depth=settings.queue_depth,
            interactive_reserve=settings.interactive_reserve,
            limit_encodes=settings.limit_encodes,
            limit_watches=settings.watches,
            weights={caller.name: caller.weight for caller in settings.keys.values()},
            affinity_wait=settings.affinity_wait,
        ),
        worker=worker,
        query_worker=query_worker,
        exiftool=None,
        card=CardArbiter(
            worker, query_worker, card_worker, settings.card_quiet_after, card_stats or system.card_stats,
            CardBudget(
                family_mb=(("text-query", settings.card_text_query_mb), ("siglip-text", settings.card_siglip_text_mb)),
                context_mb=settings.card_context_mb,
                margin_mb=settings.card_margin_mb,
            ),
            share_mb=settings.card_share_mb,
            log=log,
        ),
        audio=AudioCache(settings.scratch, ttl_seconds=settings.audio_cache_seconds, budget_mb=settings.audio_cache_mb, timeout=settings.tool_stall_seconds),
        files=Mounts(),
    )
    kept = watches.FacesKept(ctx, log)
    ctx.watches = Watches(
        settings.watches, source=ffmpeg_frames, find=watches.frame_finder(ctx, kept), crop=ffmpeg_crop, log=log, idle=kept.release,
        folder=os.path.join(settings.scratch, "watches"), segments=ffmpeg_segments, join=ffmpeg_join,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        os.makedirs(settings.scratch, exist_ok=True)
        if not settings.keys:
            ctx.warnings.append("no keys configured: authentication is off")
            log("no keys configured (EMS_KEYS_FILE / EMS_KEY): authentication is OFF")
        if not ctx.files.possible:
            ctx.warnings.append("no /dev/fuse or sshfs: every request that names files will be refused")
            log("this host cannot mount its callers' files (no /dev/fuse or sshfs): every request that names files will be refused")
        if which("exiftool"):
            ctx.exiftool = ExifTool(settings.exiftool_workers, settings.tool_stall_seconds)
        else:
            ctx.warnings.append("exiftool is not installed: media.probe answers from ffprobe only")
        await ctx.audio.start()
        ctx.files.start()
        if start_worker:
            await worker.start()
            if query_worker is not None:
                await query_worker.start()
        await ctx.card.start()
        try:
            yield
        finally:
            await ctx.watches.stop()
            await ctx.card.stop()
            if start_worker:
                if query_worker is not None:
                    await query_worker.stop()
                await worker.stop()
            if ctx.exiftool is not None:
                await ctx.exiftool.stop()
            await ctx.audio.stop()
            await ctx.files.stop()

    app = FastAPI(title="EMS", version=API_VERSION, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.ctx = ctx

    app.add_middleware(NameTheOp)

    router = APIRouter(prefix="/v0", dependencies=[Depends(require_key), Depends(require_served)])
    for module in OP_MODULES:
        module.register(router, ctx)
    system.register(router, ctx)
    app.include_router(router)
    known = system.all_ops(router)
    for glob in settings.ops:
        if not any(fnmatch.fnmatchcase(op, glob) for op in known):
            ctx.warnings.append(f"EMS_OPS names {glob}, which matches no op")

    @app.exception_handler(EmsError)
    async def ems_error(_request: Request, exc: EmsError):
        headers = {"Retry-After": str(int(exc.retry_after))} if exc.retry_after else None
        return JSONResponse(exc.envelope(), status_code=exc.code.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        detail = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ()) if p != 'body')}: {e.get('msg')}" for e in exc.errors())
        return JSONResponse(EmsError(INVALID_REQUEST, detail or "invalid request").envelope(), status_code=INVALID_REQUEST.status)

    @app.exception_handler(Exception)
    async def unexpected(_request: Request, exc: Exception):
        log(f"internal error: {type(exc).__name__}: {exc}")
        return JSONResponse(EmsError(INTERNAL, f"{type(exc).__name__}: {exc}").envelope(), status_code=INTERNAL.status)

    return app
