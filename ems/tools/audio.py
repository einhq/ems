"""The sound of a file as 16 kHz mono PCM, extracted once and shared for a while.

Every speech op wants the same WAV, and a caller working through one recording asks for it many
times: a diarization, then a voiceprint per turn, then an active-speaker score per turn — a
long recording is hundreds of requests. Extracting per request would decode the whole file
each time, so extractions are cached in scratch, keyed on the file's identity (real path, size,
mtime), shared between concurrent requests for the same file, and dropped after `ttl` seconds
of disuse or when the cache outgrows its budget.

A file with no audio stream is cached as `None` too: asking ffmpeg again would not change the
answer, and silent clips are a large share of any home archive.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import dataclass, field

from . import ffmpeg as ffmpeg_tool


@dataclass
class Entry:
    wav: str | None
    size: int
    last_used: float = field(default_factory=time.monotonic)
    users: int = 0


class AudioCache:
    def __init__(self, scratch: str, *, ttl_seconds: float = 1800, budget_mb: int = 2048, timeout: float = 900):
        self.dir = os.path.join(scratch, "audio-cache")
        self.ttl = ttl_seconds
        self.budget = budget_mb * 1024 * 1024
        self.timeout = timeout
        self.entries: dict[tuple, Entry] = {}
        self.locks: dict[tuple, asyncio.Lock] = {}
        self._sweeper: asyncio.Task | None = None

    @staticmethod
    def key_of(path: str) -> tuple:
        st = os.stat(path)
        return (os.path.realpath(path), st.st_size, st.st_mtime_ns)

    async def start(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir, exist_ok=True)
        self._sweeper = asyncio.create_task(self._sweep_loop())

    async def stop(self) -> None:
        if self._sweeper:
            self._sweeper.cancel()
            await asyncio.gather(self._sweeper, return_exceptions=True)
        shutil.rmtree(self.dir, ignore_errors=True)

    async def acquire(self, path: str) -> str | None:
        """The WAV for `path`, or None when it has no audio. Pair with `release`."""
        key = self.key_of(path)
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            entry = self.entries.get(key)
            if entry is None:
                entry = await self._extract(key, path)
                self.entries[key] = entry
            entry.users += 1
            entry.last_used = time.monotonic()
            return entry.wav

    def release(self, path: str) -> None:
        try:
            key = self.key_of(path)
        except FileNotFoundError:
            return
        entry = self.entries.get(key)
        if entry is not None:
            entry.users = max(0, entry.users - 1)
            entry.last_used = time.monotonic()

    async def _extract(self, key: tuple, path: str) -> Entry:
        os.makedirs(self.dir, exist_ok=True)
        name = f"{abs(hash(key)):x}-{os.getpid()}-{int(time.time() * 1000)}.wav"
        wav = os.path.join(self.dir, name)
        has_audio = await ffmpeg_tool.extract_audio(path, wav, timeout=self.timeout)
        if not has_audio:
            try:
                os.remove(wav)
            except FileNotFoundError:
                pass
            return Entry(None, 0)
        return Entry(wav, os.path.getsize(wav))

    def _evict(self, key: tuple) -> None:
        entry = self.entries.pop(key, None)
        self.locks.pop(key, None)
        if entry and entry.wav:
            try:
                os.remove(entry.wav)
            except FileNotFoundError:
                pass

    def sweep(self) -> None:
        """Drop idle entries past the ttl, then the least recently used until under budget."""
        now = time.monotonic()
        for key, entry in list(self.entries.items()):
            if entry.users == 0 and now - entry.last_used > self.ttl:
                self._evict(key)
        total = sum(e.size for e in self.entries.values())
        if total > self.budget:
            for key, entry in sorted(self.entries.items(), key=lambda kv: kv[1].last_used):
                if total <= self.budget:
                    break
                if entry.users == 0:
                    total -= entry.size
                    self._evict(key)

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                self.sweep()
            except Exception:  # noqa: BLE001 - a sweep must never kill the loop
                pass

    def snapshot(self) -> dict:
        return {"entries": len(self.entries), "bytes": sum(e.size for e in self.entries.values())}
