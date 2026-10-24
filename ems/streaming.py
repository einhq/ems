"""Running one operation: blocking or streamed, and cancelled when the caller goes away.

An op is an async function taking a `Progress` and returning an `Outcome`. `run_op` decides
the wire form from the request's `Accept` header:

- `text/event-stream`: a server-sent event stream — `progress` and `log` events as the op
  reports them, then exactly one `result` or `error` event with the same envelope the blocking
  form would have returned as its body.
- anything else: the op runs to completion and the envelope is the body.

In both forms a caller that closes the connection cancels the op: Starlette cancels a streaming
response's task when the client disconnects, and the blocking form watches for the disconnect
itself. Cancellation reaches the op as `asyncio.CancelledError`, which is what kills a
subprocess or drops a worker socket in the code below it. There are no job ids; the connection
is the job.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import API_VERSION
from .errors import INTERNAL, EmsError


@dataclass
class Outcome:
    result: Any
    producer: str | None = None
    meta: dict = field(default_factory=dict)


class Progress:
    """What an op reports while it runs. A no-op unless someone is streaming."""

    def __init__(self, queue: asyncio.Queue | None = None):
        self._queue = queue

    async def emit(self, phase: str, fraction: float | None = None, message: str | None = None) -> None:
        if self._queue is None:
            return
        event: dict = {"phase": phase}
        if fraction is not None:
            event["fraction"] = max(0.0, min(1.0, float(fraction)))
        if message:
            event["message"] = message
        await self._queue.put(("progress", event))

    async def log(self, level: str, message: str) -> None:
        if self._queue is None:
            return
        await self._queue.put(("log", {"level": level, "message": message}))


Op = Callable[[Progress], Awaitable[Outcome]]

DISCONNECT_POLL_SECONDS = 0.5


def _envelope(outcome: Outcome, started: float) -> dict:
    meta = {"took_ms": round((time.monotonic() - started) * 1000), "api": API_VERSION, **outcome.meta}
    if outcome.producer:
        meta["producer"] = outcome.producer
    return {"ok": True, "result": outcome.result, "meta": meta}


def _failure(exc: BaseException) -> tuple[int, dict, dict]:
    if isinstance(exc, EmsError):
        headers = {"Retry-After": str(int(exc.retry_after))} if exc.retry_after else {}
        return exc.code.status, exc.envelope(), headers
    return INTERNAL.status, EmsError(INTERNAL, f"{type(exc).__name__}: {exc}").envelope(), {}


def wants_stream(request: Request) -> bool:
    return "text/event-stream" in request.headers.get("accept", "")


async def run_op(request: Request, op: Op) -> Response:
    if wants_stream(request):
        return _stream(op)
    return await _block(request, op)


async def _block(request: Request, op: Op) -> Response:
    started = time.monotonic()
    task = asyncio.create_task(op(Progress()))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=DISCONNECT_POLL_SECONDS)
            if done:
                break
            if await request.is_disconnected():
                task.cancel()
                # Nobody is listening; the status is for the access log only.
                await asyncio.gather(task, return_exceptions=True)
                return Response(status_code=499)
        outcome = task.result()
    except EmsError as exc:
        status, body, headers = _failure(exc)
        return JSONResponse(body, status_code=status, headers=headers)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - every failure becomes the envelope
        status, body, headers = _failure(exc)
        return JSONResponse(body, status_code=status, headers=headers)
    return JSONResponse(_envelope(outcome, started))


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def _stream(op: Op) -> Response:
    started = time.monotonic()
    queue: asyncio.Queue = asyncio.Queue()
    progress = Progress(queue)

    async def generate():
        task = asyncio.create_task(op(progress))
        getter: asyncio.Task | None = None
        try:
            # Forward events until the op finishes, then drain what it left behind.
            while not task.done():
                getter = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    kind, data = getter.result()
                    getter = None
                    yield _sse(kind, data)
                else:
                    getter.cancel()
                    getter = None
            while not queue.empty():
                kind, data = queue.get_nowait()
                yield _sse(kind, data)
            try:
                outcome = task.result()
            except Exception as exc:  # noqa: BLE001
                _status, body, _headers = _failure(exc)
                yield _sse("error", body)
                return
            yield _sse("result", _envelope(outcome, started))
        finally:
            # A client that went away cancels this generator; take the op down with it.
            if getter is not None and not getter.done():
                getter.cancel()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    # The status line is committed before the op runs, so a failure arrives as an `error`
    # event on a 200 stream — which is why the terminal event carries the whole envelope.
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
