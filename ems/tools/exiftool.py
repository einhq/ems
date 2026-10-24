"""exiftool, kept open.

Perl startup and tag-table loading cost more than reading a file's metadata, so `-stay_open`
keeps a few exiftool processes alive and feeds each one request at a time through stdin, the
answer delimited by `{ready<n>}`. Same arguments as one-shot `exiftool -n -j`, minus the tags
nobody wants (binary dumps, file-system dates that say nothing about the media).

A path containing a newline cannot cross the line protocol; that name gets a one-shot run.
"""

from __future__ import annotations

import asyncio
import json
import os

from ..errors import EmsError, TOOL_FAILED
from .run import require, run

READ_ARGS = (
    "-x", "DataDump",
    "-x", "ThumbnailImage",
    "-x", "Directory",
    "-x", "FilePermissions",
    "-x", "TextJunk",
    "-x", "ColorBalanceUnknown",
    "-x", "Warning",
    "-x", "FileModifyDate",
    "-x", "FileAccessDate",
    "-x", "FileInodeChangeDate",
    "-x", "FileName",
    "-x", "SourceFile",
    "-api", "LargeFileSupport=1",
    "-n",
    "-j",
)


class _StayOpen:
    def __init__(self, executable: str, timeout: float):
        self.executable = executable
        self.timeout = timeout
        self.proc: asyncio.subprocess.Process | None = None
        self.sequence = 0
        self.lock = asyncio.Lock()

    async def _start(self) -> None:
        env = dict(os.environ, LC_ALL="C")
        self.proc = await asyncio.create_subprocess_exec(
            self.executable, "-stay_open", "True", "-@", "-",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
        )

    async def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.stdin.write(b"-stay_open\nFalse\n")
            await proc.stdin.drain()
            await asyncio.wait_for(proc.wait(), 5)
        except Exception:  # noqa: BLE001
            proc.kill()
            await proc.wait()

    async def extract(self, path: str) -> dict:
        async with self.lock:
            if self.proc is None or self.proc.returncode is not None:
                await self._start()
            assert self.proc is not None
            self.sequence += 1
            marker = f"{{ready{self.sequence}}}".encode()
            payload = "\n".join([*READ_ARGS, path, f"-execute{self.sequence}"]) + "\n"
            try:
                self.proc.stdin.write(payload.encode())
                await self.proc.stdin.drain()
                raw = await asyncio.wait_for(self._read_through(marker), self.timeout)
            except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError) as exc:
                await self.stop()
                raise EmsError(TOOL_FAILED, f"exiftool did not answer ({type(exc).__name__})") from exc
            except asyncio.CancelledError:
                await self.stop()
                raise
        return _parse(raw, path)

    async def _read_through(self, marker: bytes) -> bytes:
        assert self.proc is not None
        chunks = bytearray()
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                raise BrokenPipeError("exiftool exited")
            if line.strip() == marker:
                return bytes(chunks)
            chunks += line


def _parse(raw: bytes, path: str) -> dict:
    text = raw.decode(errors="replace").strip()
    if not text:
        # exiftool prints nothing (and an error to stderr) for a file it cannot read at all.
        raise EmsError(TOOL_FAILED, f"exiftool read nothing from {path}", permanent=True)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EmsError(TOOL_FAILED, f"exiftool returned unparseable JSON for {path}") from exc
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if not isinstance(parsed, dict):
        raise EmsError(TOOL_FAILED, f"exiftool returned an unexpected shape for {path}")
    # exiftool writes `SourceFile` into every `-j` object whatever `-x` says. It names the path
    # this process was handed — the resolved one, under whatever root EMS mounts — which is a
    # fact about this service and not about the file, so it does not reach the caller.
    parsed.pop("SourceFile", None)
    return parsed


class ExifTool:
    """A small pool of stay-open processes; requests spread over whichever is free."""

    def __init__(self, workers: int, timeout: float):
        self.executable = require("exiftool")
        self.timeout = timeout
        self.pool = [_StayOpen(self.executable, timeout) for _ in range(max(1, workers))]
        self._next = 0

    async def stop(self) -> None:
        await asyncio.gather(*(w.stop() for w in self.pool), return_exceptions=True)

    async def extract(self, path: str) -> dict:
        if "\n" in path or "\r" in path:
            done = await run([self.executable, *READ_ARGS, path], timeout=self.timeout)
            if done.code != 0 and not done.stdout.strip():
                raise EmsError(TOOL_FAILED, f"exiftool failed: {done.tail()}")
            return _parse(done.stdout, path)
        # Prefer an idle worker; otherwise round-robin, and the lock queues the request.
        for worker in self.pool:
            if not worker.lock.locked():
                return await worker.extract(path)
        worker = self.pool[self._next % len(self.pool)]
        self._next += 1
        return await worker.extract(path)
