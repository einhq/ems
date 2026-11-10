"""`PUT|GET|DELETE /v0/watches/<id>`, `GET /v0/watches` and `GET /v0/watches/<id>/events`: a live
stream read for its faces and motion, told as server-sent events (ems/watches.py, _4.24_)."""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .. import API_VERSION
from ..context import Context
from ..schemas import Options
from ..watches import Frame, WatchSpec
from ._models import model_slot

PRODUCER = "retinaface-r34+transface"

# A comment on an event stream this often keeps a proxy from closing it while nothing happens.
KEEP_ALIVE_SECONDS = 15.0


def frame_finder(ctx: Context):
    """How a watch's frame is looked at for faces: written where the worker reads it, and given
    to `detect_faces` as an interactive request, ahead of batch work."""

    async def find(watch_id: str, frame: Frame) -> dict:
        os.makedirs(ctx.settings.scratch, exist_ok=True)
        path = os.path.join(ctx.settings.scratch, f"watch-{watch_id}.ppm")
        with open(path, "wb") as handle:
            handle.write(frame.ppm)
        async with model_slot(ctx, Options(priority="interactive")):
            return await ctx.worker.call("detect_faces", {"path": path, "angle": 0, "mirror": False}) or {}

    return find


def _ok(result) -> dict:
    return {"ok": True, "result": result, "meta": {"api": API_VERSION, "producer": PRODUCER}}


def _sse(kind: str, data: dict) -> bytes:
    return f"event: {kind}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def register(router: APIRouter, ctx: Context) -> None:
    @router.get("/watches")
    async def watches_list():
        return _ok([watch.status() for watch in ctx.watches.watches.values()])

    @router.put("/watches/{watch_id}")
    async def watches_put(watch_id: str, body: WatchSpec):
        watch, made = await ctx.watches.put(watch_id, body)
        return _ok({**watch.status(), "made": made})

    @router.get("/watches/{watch_id}")
    async def watches_get(watch_id: str):
        return _ok(ctx.watches.get(watch_id).status())

    @router.delete("/watches/{watch_id}")
    async def watches_delete(watch_id: str):
        await ctx.watches.delete(watch_id)
        return _ok(None)

    @router.get("/watches/{watch_id}/events")
    async def watches_events(request: Request, watch_id: str):
        watch, queue = ctx.watches.subscribe(watch_id)

        async def generate():
            try:
                yield _sse("state", watch.status())
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), KEEP_ALIVE_SECONDS)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            return
                        yield b": keep-alive\n\n"
                        continue
                    if event is None:
                        yield _sse("end", {"watch": watch.id})
                        return
                    yield _sse(event["type"], event)
            finally:
                watch.subscribers.discard(queue)

        return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
