"""`PUT|GET|DELETE /v0/pages/<id>`, `GET /v0/pages`, `POST /v0/pages/<id>/keys`, `GET
/v0/pages/<id>/state` and `POST /v0/pages/<id>/message`: a page rendered into live H.264 and AAC for
a screen, written into live streams its caller grants under `EMS-Files`, keys pressed in it, what it
says it shows and what it is told (ems/pages.py, _4.25_)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import API_VERSION, remote
from ..context import Context
from ..errors import invalid
from ..pages import PageKeys, PageMessage, PageSpec


def _ok(result) -> dict:
    return {"ok": True, "result": result, "meta": {"api": API_VERSION}}


def register(router: APIRouter, ctx: Context) -> None:
    @router.get("/pages")
    async def pages_list():
        return _ok([page.status() for page in ctx.pages.list()])

    @router.put("/pages/{page_id}")
    async def pages_put(request: Request, page_id: str, body: PageSpec):
        header = request.headers.get(remote.HEADER)
        if not header:
            raise invalid("a page writes into names its caller grants: name their endpoint in EMS-Files")
        page, made = await ctx.pages.put(page_id, body, remote.parse_header(header))
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
