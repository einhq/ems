"""`PUT|GET|DELETE /v0/transcodes/<id>` and `GET /v0/transcodes`: a live stream its caller grants
under `EMS-Files`, transcoded as H.264 into another it grants for writing (ems/transcodes.py,
_4.26_)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import API_VERSION, remote
from ..context import Context
from ..errors import invalid
from ..transcodes import TranscodeSpec


def _ok(result) -> dict:
    return {"ok": True, "result": result, "meta": {"api": API_VERSION}}


def register(router: APIRouter, ctx: Context) -> None:
    @router.get("/transcodes")
    async def transcodes_list():
        return _ok([transcode.status() for transcode in ctx.transcodes.list()])

    @router.put("/transcodes/{transcode_id}")
    async def transcodes_put(request: Request, transcode_id: str, body: TranscodeSpec):
        header = request.headers.get(remote.HEADER)
        if not header:
            raise invalid("a transcode reads and writes names its caller grants: name their endpoint in EMS-Files")
        transcode, made = await ctx.transcodes.put(transcode_id, body, remote.parse_header(header))
        return _ok({**transcode.status(), "made": made})

    @router.get("/transcodes/{transcode_id}")
    async def transcodes_get(transcode_id: str):
        return _ok(ctx.transcodes.get(transcode_id).status())

    @router.delete("/transcodes/{transcode_id}")
    async def transcodes_delete(transcode_id: str):
        await ctx.transcodes.delete(transcode_id)
        return _ok(None)
