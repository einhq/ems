"""Bounded concurrency, a queue that is allowed to say no, and the choice of what runs next.

Four lanes: `models` (requests forwarded to the model worker, which serialises per model
family behind this), `tools` (CPU and subprocess work in this process), `encodes` (the
transcodes, which hold a slot for minutes and must not sit ahead of every probe and rendition
queued behind them) and `watches` (a live stream's frames on their way to the model worker, one
at a time a watch, so a crawl filling `models` never holds a frame back). Each lane runs at most `limit` requests at once; an interactive request may
also use `reserve` extra slots, so a crawl that keeps the lane full never makes a person wait
behind it. Past `queue_depth` waiters of one caller, across the lanes, a new request of that
caller's is answered `busy` with a `Retry-After` — the caller's own queue is the right place for
that work to sit, not a socket held open here — while other callers are still admitted.

A freed slot goes to the first of:

1. the oldest interactive waiter, within the limit and the reserve;
2. else, of the callers with a batch waiter, the one that has used the least: the slot-seconds it
   has held, the slots it holds now included, divided by its weight (`config.Caller`). A caller that
   starts waiting after holding and wanting nothing starts at the least use of the callers that
   do, so being idle saves up no credit;
3. within that caller, in a lane that keeps families together (`models`), its oldest waiter for a
   model family the lane is running or ran last, so a loaded family is used before another is
   loaded beside or in place of it, unless the caller's oldest waiter has waited `affinity_wait`
   seconds; else its oldest waiter.

The queue lives in memory only. A request waiting here is an open connection; when EMS restarts,
its caller sees the connection close and tries again later, which is all a persistent queue would
have bought.
"""

from __future__ import annotations

import asyncio
import contextvars
import itertools
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .errors import BUSY, EmsError

BUSY_RETRY_AFTER_SECONDS = 5

# The op the current request is for (`speech.transcribe`), set by the app's middleware, so a slot
# can say what it is held for without every op passing its own name down.
current_op: contextvars.ContextVar[str] = contextvars.ContextVar("current_op", default="?")

# Who the current request is from, the name its key has in the keys file, set by the app's middleware.
current_caller: contextvars.ContextVar[str] = contextvars.ContextVar("current_caller", default="anonymous")

# The slot the current request holds, so the model worker client can say which model family the
# request is using right now (`note_family`) — a voiceprint is sortformer first, then titanet.
current_request: contextvars.ContextVar[dict | None] = contextvars.ContextVar("current_request", default=None)

# The model family an op needs first, so the scheduler can keep a family's requests together before
# any of them reaches the worker. An op not named here is its own family.
OP_FAMILIES = {
    "faces.detect": "faces",
    "faces.detect_video": "faces",
    "faces.embed": "faces",
    "speech.transcribe": "whisper",
    "speech.diarize": "sortformer",
    "speech.voiceprint": "sortformer",
    "speech.active_speakers": "asd",
    "speech.active_speaker": "asd",
    "vision.embed": "siglip-visual",
    "vision.embed_text": "siglip-text",
    "text.embed": "text-embed",
    "fingerprint.compute": "sscd",
    "video.shots": "transnet",
    "image.inpaint": "lama",
    "image.defects": "defects",
    "image.denoise": "denoise",
    "ocr.read": "ocr",
}


def note_family(family: str) -> None:
    """Record the model family the current request has just asked the worker for."""
    entry = current_request.get()
    if entry is not None:
        entry["family"] = family


@dataclass
class Share:
    """A caller as the scheduler weighs it: its weight, and the slot-seconds it has given back."""

    weight: float = 1.0
    used: float = 0.0


@dataclass(eq=False)
class Waiter:
    caller: str
    family: str
    interactive: bool
    queued: float
    granted: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())


class Lane:
    def __init__(self, name: str, limit: int, reserve: int, *, affinity: bool = False):
        self.name = name
        self.limit = max(1, limit)
        self.reserve = max(0, reserve)
        self.affinity = affinity
        self.running = 0
        self.waiters: list[Waiter] = []
        # The family of the request that gave a slot back last, for affinity once nothing runs.
        self.last_family: str | None = None

    @property
    def waiting(self) -> int:
        return len(self.waiters)

    def may_run(self, interactive: bool) -> bool:
        return self.running < self.limit + (self.reserve if interactive else 0)

    def snapshot(self) -> dict:
        return {"running": self.running, "waiting": self.waiting, "limit": self.limit, "reserve": self.reserve}


class Admission:
    def __init__(self, *, limit_models: int, limit_tools: int, queue_depth: int, interactive_reserve: int, limit_encodes: int = 2,
                 limit_watches: int = 4, weights: dict[str, float] | None = None, affinity_wait: float = 300.0):
        self.lanes = {
            "models": Lane("models", limit_models, interactive_reserve, affinity=True),
            "tools": Lane("tools", limit_tools, interactive_reserve),
            "encodes": Lane("encodes", limit_encodes, 0),
            "watches": Lane("watches", limit_watches, 0),
        }
        self.queue_depth = max(0, queue_depth)
        self.affinity_wait = max(0.0, affinity_wait)
        self.callers: dict[str, Share] = {name: Share(weight=weight) for name, weight in (weights or {}).items()}
        # Every slot asked for and not yet given back, for `health`'s `requests`: which op, in
        # which lane, from whom, since when it queued and since when it runs. The lane counters say
        # how full a lane is; this says with what.
        self._requests: dict[int, dict] = {}
        self._ids = itertools.count()

    def snapshot(self) -> dict:
        return {name: lane.snapshot() for name, lane in self.lanes.items()}

    def requests(self) -> list[dict]:
        """The slots held or waited for right now, oldest first."""
        listed = [{**entry, "started": round(entry["started"], 1)} if "started" in entry else dict(entry) for entry in self._requests.values()]
        return sorted(listed, key=lambda entry: entry["queued"])

    def caller_states(self) -> dict[str, dict]:
        """Each caller seen or configured: its weight, its use so far and what it holds and waits for now."""
        now = time.time()
        states: dict[str, dict] = {}
        for name, caller in self.callers.items():
            mine = [entry for entry in self._requests.values() if entry["caller"] == name]
            states[name] = {
                "weight": caller.weight,
                "used_seconds": round(self._use(name, now) * caller.weight, 1),
                "running": sum(1 for entry in mine if "started" in entry),
                "waiting": sum(1 for entry in mine if "started" not in entry),
            }
        return states

    def _caller(self, name: str) -> Share:
        return self.callers.setdefault(name, Share())

    def _use(self, name: str, now: float) -> float:
        """The caller's slot-seconds, those of the slots it holds now included, over its weight."""
        caller = self._caller(name)
        held = sum(now - entry["started"] for entry in self._requests.values() if entry["caller"] == name and "started" in entry)
        return (caller.used + held) / caller.weight

    def _waiting_of(self, name: str) -> int:
        return sum(1 for lane in self.lanes.values() for waiter in lane.waiters if waiter.caller == name)

    def _active(self, name: str) -> bool:
        return any(entry["caller"] == name for entry in self._requests.values())

    def _catch_up(self, name: str, now: float) -> None:
        """A caller that wanted nothing until now starts level with the least use of those that do."""
        others = {entry["caller"] for entry in self._requests.values()} - {name}
        if not others:
            return
        floor = min(self._use(other, now) for other in others)
        caller = self._caller(name)
        caller.used = max(caller.used, floor * caller.weight)

    def _choose(self, lane: Lane) -> Waiter | None:
        """The waiter the lane's next slot goes to, or None when none may run now."""
        interactive = [waiter for waiter in lane.waiters if waiter.interactive]
        if interactive and lane.may_run(True):
            return min(interactive, key=lambda waiter: waiter.queued)
        batch = [waiter for waiter in lane.waiters if not waiter.interactive]
        if not batch or not lane.may_run(False):
            return None
        now = time.time()
        oldest_of: dict[str, Waiter] = {}
        for waiter in batch:
            if waiter.caller not in oldest_of or waiter.queued < oldest_of[waiter.caller].queued:
                oldest_of[waiter.caller] = waiter
        name = min(oldest_of, key=lambda caller: (self._use(caller, now), oldest_of[caller].queued))
        oldest = oldest_of[name]
        if not lane.affinity or now - oldest.queued >= self.affinity_wait:
            return oldest
        hot = {entry.get("family") for entry in self._requests.values() if entry["lane"] == lane.name and "started" in entry}
        hot.add(lane.last_family)
        warm = [waiter for waiter in batch if waiter.caller == name and waiter.family in hot]
        return min(warm, key=lambda waiter: waiter.queued) if warm else oldest

    def _dispatch(self, lane: Lane) -> None:
        while (waiter := self._choose(lane)) is not None:
            lane.waiters.remove(waiter)
            lane.running += 1
            waiter.granted.set_result(None)

    def _give_back(self, lane: Lane, entry: dict) -> None:
        if "started" in entry:
            self._caller(entry["caller"]).used += time.time() - entry["started"]
        lane.last_family = entry.get("family") or lane.last_family
        lane.running -= 1
        self._dispatch(lane)

    @asynccontextmanager
    async def slot(self, lane_name: str, *, interactive: bool = False):
        lane = self.lanes[lane_name]
        request_id = next(self._ids)
        op, caller = current_op.get(), current_caller.get()
        family = OP_FAMILIES.get(op, op)
        now = time.time()
        entry = {"op": op, "caller": caller, "lane": lane.name, "interactive": interactive, "queued": round(now, 1)}
        if lane.affinity:
            entry["family"] = family
        try:
            # A slot no one waits ahead of is taken at once; one others wait for is the scheduler's.
            if lane.may_run(interactive) and not any(waiter.interactive or not interactive for waiter in lane.waiters):
                lane.running += 1
            else:
                if self._waiting_of(caller) >= self.queue_depth:
                    raise EmsError(BUSY, f"{lane.name}: {caller}'s queue is full", retry_after=BUSY_RETRY_AFTER_SECONDS)
                if not self._active(caller):
                    self._catch_up(caller, now)
                waiter = Waiter(caller, family, interactive, now)
                lane.waiters.append(waiter)
                self._requests[request_id] = entry
                try:
                    await waiter.granted
                except asyncio.CancelledError:
                    if waiter in lane.waiters:
                        lane.waiters.remove(waiter)
                    elif waiter.granted.done() and not waiter.granted.cancelled():
                        # Granted as the caller went away: the slot goes straight to the next waiter.
                        self._give_back(lane, entry)
                    raise
            entry["started"] = time.time()
            self._requests[request_id] = entry
            token = current_request.set(entry)
            try:
                yield
            finally:
                current_request.reset(token)
                self._requests.pop(request_id, None)
                self._give_back(lane, entry)
        finally:
            self._requests.pop(request_id, None)
