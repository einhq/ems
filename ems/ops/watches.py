"""`PUT|GET|DELETE /v0/watches/<id>`, `GET /v0/watches` and `GET /v0/watches/<id>/events`: a live
stream read for its faces and motion, told as server-sent events (ems/watches.py, _4.24_)."""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .. import API_VERSION
from ..card import watching
from ..context import Context
from ..errors import EmsError
from ..watches import Frame, WatchSpec

PRODUCER = "retinaface-r34+transface"
# The model family a watch looks with.
FAMILY = "faces"

# A comment on an event stream this often keeps a proxy from closing it while nothing happens.
KEEP_ALIVE_SECONDS = 15.0


class FacesKept:
    """The faces family pinned on the batch worker while any watch runs, so batch work never
    unloads it between two frames (ems/card.py, `kept`). Pinned at a watch's frame when it is not
    yet, which is again after the worker restarted and forgot its pins; left to eviction once the
    last watch stops. A family the configuration pins (EMS_MODEL_PINNED) is left alone."""

    def __init__(self, ctx: Context, log):
        self.ctx = ctx
        self.log = log
        # The worker's restart count when it was pinned; None while it is not.
        self.pinned_at: int | None = None
        self.lock = asyncio.Lock()

    def _configured(self) -> bool:
        return FAMILY in self.ctx.worker.env.get("CFG_MODEL_PINNED", "").split(",")

    def _mark(self, kept: bool) -> None:
        if self.ctx.card is not None:
            self.ctx.card.kept = frozenset({FAMILY}) if kept else frozenset()

    async def keep(self) -> None:
        worker = self.ctx.worker
        if self.pinned_at == worker.restarts or self._configured():
            return
        async with self.lock:
            if self.pinned_at == worker.restarts:
                return
            # Marked before the answer: a worker that cannot pin is not asked again each frame.
            self.pinned_at = worker.restarts
            self._mark(True)
            try:
                await worker.call("pin", {"families": [FAMILY]})
            except EmsError as exc:
                self.log(f"watches: the worker did not pin {FAMILY}: {exc}")

    async def release(self) -> None:
        async with self.lock:
            if self.pinned_at is None:
                return
            self.pinned_at = None
            self._mark(False)
            try:
                await self.ctx.worker.call("unpin", {"families": [FAMILY]})
            except EmsError as exc:
                self.log(f"watches: the worker did not unpin {FAMILY}: {exc}")


def frame_finder(ctx: Context, kept: FacesKept):
    """How a watch's frame is looked at for faces: written where the worker reads it, and given to
    `detect_faces` in the `watches` lane, with its family kept loaded and the card left as it is."""

    async def find(watch_id: str, frame: Frame) -> dict:
        os.makedirs(ctx.settings.scratch, exist_ok=True)
        path = os.path.join(ctx.settings.scratch, f"watch-{watch_id}.ppm")
        with open(path, "wb") as handle:
            handle.write(frame.ppm)
        token = watching.set(True)
        try:
            await kept.keep()
            async with ctx.admission.slot("watches", interactive=True):
                return await ctx.worker.call("detect_faces", {"path": path, "angle": 0, "mirror": False}) or {}
        finally:
            watching.reset(token)

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
