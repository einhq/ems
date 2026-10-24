"""Who has the card: batch work while there is some, search while there is none.

Two modes, decided here from the requests the front forwards to the batch worker:

- **busy** while batch model work arrives. The batch worker's families own the card under their
  own rules (claims, room-making, idle eviction); search queries embed on the query worker, on the
  CPU.
- **quiet** after `quiet_after` seconds without any. The batch worker unloads what it holds, and
  a card worker starts with the query families that fit what is then free: a third model worker,
  the query worker's twin on the card. Queries go to it once it has loaded them; documents embed
  on the card too (`document_device`).

The card copies live in a process of their own because a process that can see the card touches
it: libraries probe it at import or load onto it whenever it is there (peft), and either leaves a
CUDA context behind. Stopping the card worker gives back everything it held.

A batch request entering busy mode waits until the card worker has stopped, so a batch family never
loads beside it; a search never waits, because the query worker answers throughout. Text embedding of
documents does not count as batch work: it is small, and runs where the mode says.

With no card, no query worker or `quiet_after` 0 there is nothing to arbitrate, and every call
here is a pass-through.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .errors import EmsError
from .worker import Worker

# Worker ops that are not batch model work: they never make the card busy. The query ops are here
# for a query worker that is down, when search falls back to the batch worker.
_NOT_BATCH = frozenset({"health", "memstats", "transcribe_signature", "unload", "embed_texts", "embed_image_texts"})
TICK_SECONDS = 30.0
# How long the card worker may take to load its families before the card is left to the CPU.
READY_TIMEOUT_SECONDS = 300.0
# How long going busy waits for searches already on the card worker.
DRAIN_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class CardBudget:
    """What each query family's card copy needs free, measured on fry (RTX 5060 Ti, 2026-10-09):
    text-query 1.7 GB in float16, siglip-text 3.0 GB with its onnxruntime arena; plus the card
    worker's CUDA context and a margin. Families are taken in order, most gained per MB first."""

    family_mb: tuple[tuple[str, float], ...] = (("text-query", 1800.0), ("siglip-text", 3100.0))
    context_mb: float = 400.0
    margin_mb: float = 1024.0

    def fitting(self, free_mb: float) -> tuple[list[str], dict[str, str]]:
        """The families whose copies fit `free_mb`, and why each other one does not."""
        chosen: list[str] = []
        skipped: dict[str, str] = {}
        need = self.context_mb + self.margin_mb
        for family, mb in self.family_mb:
            if need + mb <= free_mb:
                chosen.append(family)
                need += mb
            else:
                skipped[family] = f"{free_mb:.0f} MB free, needs {need + mb:.0f} MB"
        return chosen, skipped


def room_mb(stats: dict | None, share_mb: float = 0.0, held_mb: float = 0.0) -> float:
    """What EMS may still load onto the card, in MB, from an `nvidia-smi` reading.

    The card's free memory plus `held_mb` (what is about to be given back), and with a share no
    more than the share less what EMS's own processes hold — `processes` lists only this
    container's, so what the card's other tenants hold never counts against it.
    """
    if not stats or "mem_total_mb" not in stats or "mem_used_mb" not in stats:
        return 0.0
    free = stats["mem_total_mb"] - stats["mem_used_mb"] + held_mb
    if share_mb > 0 and "processes" in stats:
        own = sum(p.get("used_mb", 0) for p in stats["processes"])
        free = min(free, share_mb - own + held_mb)
    return max(0.0, free)


class CardArbiter:
    def __init__(
        self,
        batch: Worker,
        query: Worker | None,
        card: Worker | None,
        quiet_after: float,
        card_stats: Callable[[], Awaitable[dict | None]],
        budget: CardBudget = CardBudget(),
        share_mb: float = 0.0,
        log: Callable[[str], None] = print,
    ):
        self.batch = batch
        self.query = query
        self.card = card
        self.quiet_after = quiet_after
        self.card_stats = card_stats
        self.budget = budget
        self.share_mb = share_mb
        self.log = log
        self.enabled = False
        # Busy at start: after a restart, batch work is what usually arrives first, and quiet is
        # one `quiet_after` away if it does not.
        self.mode = "busy"
        self.since = time.time()
        self.last_batch = time.time()
        self.inflight = 0
        # The families queries go to the card worker for; empty while it is down or loading.
        self.on_card: frozenset[str] = frozenset()
        self.skipped: dict[str, str] = {}
        self._card_inflight = 0
        self._ready_restarts = 0
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self.enabled = self.query is not None and self.card is not None and self.quiet_after > 0 and await self.card_stats() is not None
        if self.enabled:
            self.batch.gate = self.gate
            self.card.gate = self._card_gate
            self._task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if self.card is not None and self.card.running:
            await self.card.stop()

    def for_queries(self, family: str) -> Worker | None:
        """The card worker when it holds `family` ready, else None (the query worker answers)."""
        card = self.card
        if family in self.on_card and card is not None and card.running and card.restarts == self._ready_restarts:
            return card
        return None

    def document_device(self) -> str | None:
        """What documents ask the batch worker for: the card while quiet, else its setting."""
        return "card" if self.enabled and self.mode == "quiet" else None

    def gate(self, op: str, _payload: dict[str, Any]):
        """The batch worker's hook around each call: batch model work holds the card busy."""
        return nullcontext() if op in _NOT_BATCH else self._batch_work()

    @asynccontextmanager
    async def _batch_work(self):
        self.inflight += 1
        self.last_batch = time.time()
        try:
            if self.mode != "busy":
                async with self._lock:
                    if self.mode != "busy":
                        await self._to_busy()
            yield
        finally:
            self.inflight -= 1
            self.last_batch = time.time()

    @asynccontextmanager
    async def _counted(self):
        self._card_inflight += 1
        try:
            yield
        finally:
            self._card_inflight -= 1

    def _card_gate(self, op: str, _payload: dict[str, Any]):
        """The card worker's hook: counts the searches on it, which going busy waits out."""
        return nullcontext() if op == "health" else self._counted()

    async def tick(self) -> None:
        """Go quiet when no batch work has come for `quiet_after`; while quiet, use a card worker that came back."""
        if not self.enabled:
            return
        async with self._lock:
            if self.mode == "busy" and self.inflight == 0 and time.time() - self.last_batch >= self.quiet_after:
                await self._to_quiet()
            elif self.mode == "quiet" and self.card.running and self.card.restarts != self._ready_restarts:
                # The card worker died and its supervisor restarted it: queries wait on the CPU until it has loaded again.
                await self._await_ready(list(self.on_card) or self._pinned())

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(TICK_SECONDS)
            try:
                await self.tick()
            except Exception as exc:  # noqa: BLE001 - the loop must outlive one bad tick
                self.log(f"card: tick failed: {exc}")

    def _pinned(self) -> list[str]:
        return [name for name in self.card.env.get("CFG_MODEL_PINNED", "").split(",") if name]

    async def _await_ready(self, families: list[str]) -> None:
        """Route `families` to the card worker once its health shows them loaded on the card."""
        restarts = self.card.restarts
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        want = {f"{family}:cuda" for family in families}
        while time.monotonic() < deadline:
            health = await self.card.health() if self.card.running else None
            if health is not None and want <= set(health.get("loaded") or []):
                self._ready_restarts = restarts
                self.on_card = frozenset(families)
                return
            await asyncio.sleep(1.0)
        self.on_card = frozenset()
        self.log(f"card: the card worker did not load {', '.join(families)} in {READY_TIMEOUT_SECONDS:.0f}s; search stays on the CPU")

    async def _to_quiet(self) -> None:
        self.mode = "switching"
        try:
            unloaded = (await self.batch.call("unload", {})).get("unloaded") or []
        except EmsError as exc:
            self.log(f"card: batch worker could not unload: {exc}")
            unloaded = []
        if self.inflight:
            # Batch work arrived while the batch worker unloaded: it keeps the card.
            self.mode, self.since = "busy", time.time()
            return
        families, self.skipped = self.budget.fitting(room_mb(await self.card_stats(), self.share_mb))
        if families:
            self.card.env["CFG_MODEL_PINNED"] = ",".join(families)
            await self.card.start()
            await self._await_ready(families)
        self.mode, self.since = "quiet", time.time()
        self.log(f"card: quiet after {self.quiet_after:.0f}s without batch work; unloaded {', '.join(unloaded) or 'nothing'}; "
                 f"search on the card: {', '.join(sorted(self.on_card)) or 'nothing'}"
                 + (f" ({'; '.join(f'{k}: {v}' for k, v in self.skipped.items())})" if self.skipped else ""))

    async def _to_busy(self) -> None:
        self.mode = "switching"
        self.on_card = frozenset()
        deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS
        while self._card_inflight and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if self.card.running:
            await self.card.stop()
        try:
            await self.batch.call("unload", {"families": ["text-embed"], "device": "cuda", "wait": True})
        except EmsError as exc:
            self.log(f"card: batch worker could not unload text-embed from the card: {exc}")
        self.mode, self.since = "busy", time.time()
        self.log("card: busy, batch work arrived; search back on the CPU")

    def plan(self, families: dict[str, dict], gpu: dict) -> dict[str, dict]:
        """Where each family runs in each mode — `card`, `cpu` or `unloaded` — as the arbiter places it.

        Batch families run where their device setting says while busy and are unloaded while quiet
        (batch work switches to busy before it runs); the document embedder follows its setting
        while busy and goes on the card while quiet, room permitting; the query families answer on
        the CPU while busy and on the card while quiet if they fit. While busy that last is a
        prediction from the card now, less what the batch and card workers hold; while quiet it is
        what happened, with `why` for a family that did not fit.
        """
        if not self.enabled:
            return {}
        if self.mode == "quiet":
            on_card, skipped = set(self.on_card), dict(self.skipped)
        else:
            held = sum(p.get("used_mb", 0) for p in gpu.get("processes") or [] if p.get("worker") in ("batch", "card"))
            chosen, skipped = self.budget.fitting(room_mb(gpu, self.share_mb, held))
            on_card = set(chosen)
        query = {family for family, _ in self.budget.family_mb}
        plan: dict[str, dict] = {}
        for name, entry in families.items():
            setting = "cpu" if entry.get("device") == "cpu" else "card"
            if name in query:
                plan[name] = {"busy": "cpu", "quiet": "card" if name in on_card else "cpu", **({"why": skipped[name]} if name in skipped else {})}
            elif name == "text-embed":
                plan[name] = {"busy": setting, "quiet": "card"}
            else:
                plan[name] = {"busy": setting, "quiet": "unloaded"}
        return plan

    def snapshot(self) -> dict:
        """For `health`."""
        if not self.enabled:
            return {"mode": "off"}
        return {
            "mode": self.mode,
            "since": round(self.since, 1),
            "quiet_after": self.quiet_after,
            "last_batch": round(self.last_batch, 1),
            "batch_in_flight": self.inflight,
            "search_on_card": sorted(self.on_card),
            **({"skipped": self.skipped} if self.skipped else {}),
        }
