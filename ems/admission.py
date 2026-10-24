"""Bounded concurrency, and a queue that is allowed to say no.

Three lanes: `models` (requests forwarded to the model worker, which serialises per model
family behind this), `tools` (CPU and subprocess work in this process) and `encodes` (the
transcodes, which hold a slot for minutes and must not sit ahead of every probe and rendition
queued behind them). Each lane runs at most
`limit` requests at once; an interactive request may also use `reserve` extra slots, so a crawl
that keeps the lane full never makes a person wait behind it. Past `queue_depth` waiters across
both lanes a new request is answered `busy` with a `Retry-After` — the caller's own queue is
the right place for that work to sit, not a socket held open here.
"""

from __future__ import annotations

import asyncio
import contextvars
import itertools
import time
from contextlib import asynccontextmanager

from .errors import BUSY, EmsError

BUSY_RETRY_AFTER_SECONDS = 5

# The op the current request is for (`speech.transcribe`), set by the app's middleware, so a slot
# can say what it is held for without every op passing its own name down.
current_op: contextvars.ContextVar[str] = contextvars.ContextVar("current_op", default="?")

# The slot the current request holds, so the model worker client can say which model family the
# request is using right now (`note_family`) — a voiceprint is sortformer first, then titanet.
current_request: contextvars.ContextVar[dict | None] = contextvars.ContextVar("current_request", default=None)


def note_family(family: str) -> None:
    """Record the model family the current request has just asked the worker for."""
    entry = current_request.get()
    if entry is not None:
        entry["family"] = family


class Lane:
    def __init__(self, name: str, limit: int, reserve: int):
        self.name = name
        self.limit = max(1, limit)
        self.reserve = max(0, reserve)
        self.running = 0
        self.waiting = 0
        self._cond = asyncio.Condition()

    def _may_run(self, interactive: bool) -> bool:
        cap = self.limit + (self.reserve if interactive else 0)
        return self.running < cap

    def snapshot(self) -> dict:
        return {"running": self.running, "waiting": self.waiting, "limit": self.limit, "reserve": self.reserve}


class Admission:
    def __init__(self, *, limit_models: int, limit_tools: int, queue_depth: int, interactive_reserve: int, limit_encodes: int = 2):
        self.lanes = {
            "models": Lane("models", limit_models, interactive_reserve),
            "tools": Lane("tools", limit_tools, interactive_reserve),
            "encodes": Lane("encodes", limit_encodes, 0),
        }
        self.queue_depth = max(0, queue_depth)
        # Every slot asked for and not yet given back, for `health`'s `requests`: which op, in
        # which lane, since when it queued and since when it runs. The lane counters say how
        # full a lane is; this says with what.
        self._requests: dict[int, dict] = {}
        self._ids = itertools.count()

    def snapshot(self) -> dict:
        return {name: lane.snapshot() for name, lane in self.lanes.items()}

    def requests(self) -> list[dict]:
        """The slots held or waited for right now, oldest first."""
        return sorted((dict(entry) for entry in self._requests.values()), key=lambda entry: entry["queued"])

    def _waiting_total(self) -> int:
        return sum(lane.waiting for lane in self.lanes.values())

    @asynccontextmanager
    async def slot(self, lane_name: str, *, interactive: bool = False):
        lane = self.lanes[lane_name]
        request_id = next(self._ids)
        entry = {"op": current_op.get(), "lane": lane.name, "interactive": interactive, "queued": round(time.time(), 1)}
        try:
            async with lane._cond:
                if not lane._may_run(interactive):
                    if self._waiting_total() >= self.queue_depth:
                        raise EmsError(BUSY, f"{lane.name}: queue is full", retry_after=BUSY_RETRY_AFTER_SECONDS)
                    lane.waiting += 1
                    self._requests[request_id] = entry
                    try:
                        await lane._cond.wait_for(lambda: lane._may_run(interactive))
                    finally:
                        lane.waiting -= 1
                lane.running += 1
                entry["started"] = round(time.time(), 1)
                self._requests[request_id] = entry
            token = current_request.set(entry)
            try:
                yield
            finally:
                current_request.reset(token)
                async with lane._cond:
                    lane.running -= 1
                    lane._cond.notify_all()
        finally:
            self._requests.pop(request_id, None)
