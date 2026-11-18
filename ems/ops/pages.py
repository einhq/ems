"""`PUT|GET|DELETE /v0/pages/<id>`, `GET /v0/pages`, `GET /v0/pages/<id>/live.mp4[?duration=s]|live.ts`,
`GET /v0/pages/<id>/hls/<name>`, `POST /v0/pages/<id>/keys`, `GET /v0/pages/<id>/state` and
`POST /v0/pages/<id>/message`: a page rendered into live H.264 and AAC for a screen, keys pressed in it,
what it says it shows and what it is told (ems/pages.py, _4.25_)."""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, Query
from fastapi.responses import Response, StreamingResponse

from .. import API_VERSION
from ..context import Context
from ..pages import PageKeys, PageMessage, PageSpec

MEDIA_TYPES = {"mp4": "video/mp4", "ts": "video/mp2t"}
# A screen's player fetches the stream from a page of its own.
HEADERS = {"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"}


def _ok(result) -> dict:
    return {"ok": True, "result": result, "meta": {"api": API_VERSION}}


class LiveResponse(StreamingResponse):
    """A streaming response that runs `close` however it ends, its reader gone before or while it
    is sent."""

    def __init__(self, body: AsyncIterator[bytes], close, media_type: str):
        super().__init__(body, media_type=media_type, headers=HEADERS)
        self.close = close

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.close()


def register(router: APIRouter, ctx: Context) -> None:
    @router.get("/pages")
    async def pages_list():
        return _ok([page.status() for page in ctx.pages.list()])

    @router.put("/pages/{page_id}")
    async def pages_put(page_id: str, body: PageSpec):
        page, made = await ctx.pages.put(page_id, body)
        return _ok({**page.status(), "made": made})

    @router.get("/pages/{page_id}")
    async def pages_get(page_id: str):
        return _ok(ctx.pages.get(page_id).status())

    @router.delete("/pages/{page_id}")
    async def pages_delete(page_id: str):
        await ctx.pages.delete(page_id)
        return _ok(None)

    @router.post("/pages/{page_id}/keys")
    async def pages_keys(page_id: str, body: PageKeys):
        await ctx.pages.get(page_id).press(body.keys)
        return _ok(None)

    @router.get("/pages/{page_id}/state")
    async def pages_state(page_id: str):
        return _ok(await ctx.pages.get(page_id).shown())

    @router.post("/pages/{page_id}/message")
    async def pages_message(page_id: str, body: PageMessage):
        return _ok({"taken": await ctx.pages.get(page_id).tell(body.message)})

    @router.get("/pages/{page_id}/live.mp4")
    async def pages_mp4(page_id: str, duration: int | None = Query(default=None, ge=16, le=7 * 86400)):
        page = ctx.pages.get(page_id)
        channel, reader = page.join("mp4")
        return LiveResponse(channel.chunks(reader, duration), lambda: page.leave(channel, reader), MEDIA_TYPES["mp4"])

    @router.get("/pages/{page_id}/live.ts")
    async def pages_ts(page_id: str):
        page = ctx.pages.get(page_id)
        channel, reader = page.join("ts")
        return LiveResponse(channel.chunks(reader), lambda: page.leave(channel, reader), MEDIA_TYPES["ts"])

    @router.get("/pages/{page_id}/hls/{name}")
    async def pages_hls(page_id: str, name: str):
        data, media_type = await ctx.pages.get(page_id).hls_file(name)
        return Response(data, media_type=media_type, headers=HEADERS)
