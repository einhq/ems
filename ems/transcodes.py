"""Live streams transcoded into others (`/v0/transcodes`, docs/api.md 4.26).

A transcode reads a live stream its caller grants under `EMS-Files` (a camera's, which a browser or
a screen cannot play as it comes) and writes it, as H.264 and AAC with a keyframe every second in
MPEG-TS or fragmented MP4, into another the caller grants for writing, as the bytes come. The source is read as a
watch reads one (_4.24_): a read that ends is a source that dropped, opened again after a pause that
doubles from a second to a minute, and the encode starts anew on what comes then, written on into
the same stream. EMS serves nothing of it; a transcode runs until its caller deletes it, or its
output cannot be written.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Literal

from pydantic import Field

from .errors import BUSY, INVALID_REQUEST, NOT_FOUND, TOOL_FAILED, EmsError
from .remote import Endpoint
from .schemas import Strict, StreamRef

TRANSCODE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# How long an ended transcode stays listed, with why, before it is forgotten.
ENDED_KEPT_SECONDS = 60.0
# The pause before a source that dropped is opened again: doubling, from the first to the last.
RETRY_SECONDS = (1.0, 60.0)
AUDIO_RATE = 48000
AUDIO_KBPS = 128


class TranscodeVideo(Strict):
    # The picture at most this high, scaled down keeping its shape; as it comes without.
    height: int | None = Field(default=None, ge=144, le=2160)
    # NVENC's `cq` (0–51).
    quality: int = Field(default=23, ge=0, le=51)


class TranscodeSpec(Strict):
    source: StreamRef
    output: StreamRef
    # What the output is written as: MPEG-TS, or fragmented MP4 with a fragment at each keyframe.
    format: Literal["mpegts", "mp4"] = "mpegts"
    video: TranscodeVideo = Field(default_factory=TranscodeVideo)


Opener = Callable[[Endpoint, str], Awaitable[int]]
Encoders = Callable[[], Awaitable[set[str]]]


def encoder_args(spec: TranscodeSpec, encoder: str, source_fd: int) -> list[str]:
    """ffmpeg reading the source from `source_fd` as it comes and writing the spec's format to stdout:
    the picture as H.264 with a keyframe every second (in MPEG-TS its parameters at each, in MP4 a
    fragment starting at each), and the first sound, when there is one, as AAC."""
    v = spec.video
    if encoder == "h264_nvenc":
        codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll", "-rc", "vbr", "-cq", str(v.quality), "-maxrate", "8M",
                 "-bufsize", "16M", "-profile:v", "high", "-forced-idr", "1", "-bf", "0"]
    else:
        codec = ["-c:v", encoder, "-b:v", "5M", "-maxrate", "8M"]
    scale = ["-vf", f"scale=-2:'min(ih,{v.height})',format=yuv420p"] if v.height else ["-vf", "format=yuv420p"]
    if spec.format == "mp4":
        container = ["-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4"]
    else:
        container = ["-bsf:v", "dump_extra=freq=keyframe", "-f", "mpegts"]
    return [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-fflags", "+genpts+discardcorrupt", "-i", f"pipe:{source_fd}",
        "-map", "0:v:0", "-map", "0:a:0?", *scale, *codec, "-force_key_frames", "expr:gte(t,n_forced*1)",
        "-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k", "-ar", str(AUDIO_RATE), "-ac", "2",
        *container, "pipe:1",
    ]


Command = Callable[[TranscodeSpec, str, int], list[str]]


def _iso(at: float) -> str:
    return datetime.fromtimestamp(at, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _encoder_failed(error: str, encoder: str) -> bool:
    """Whether what ffmpeg said as it ended is the encoder's failing to start, rather than the
    source's: a source that cannot be read leaves the encoder as it is."""
    return encoder in error or "encoder" in error.lower()


def _kill(process: asyncio.subprocess.Process) -> None:
    """End the encode and whatever it started: its group, which it leads."""
    if process.returncode is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


@dataclass
class Ending:
    reason: str
    at: float


class Transcode:
    """One transcode: its output opened for its life, and the source read and encoded, again after
    it drops."""

    def __init__(self, transcode_id: str, spec: TranscodeSpec, endpoint: Endpoint, *, open_source: Opener, open_output: Opener,
                 encoders: list[str], command: Command, log: Callable[[str], None]):
        self.id = transcode_id
        self.spec = spec
        self.endpoint = endpoint
        self.open_source = open_source
        self.open_output = open_output
        # The encoders to use, the card's first: one that could not start gives way to the next.
        self.encoders = encoders
        self.encoder = encoders[0]
        self.command = command
        self.log = log
        self.since = time.time()
        self.state = "connecting"
        self.error: str | None = None
        self.made = 0
        self.ending: Ending | None = None
        self.output: int | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.task: asyncio.Task | None = None

    def status(self) -> dict:
        found = {
            "id": self.id,
            "source": self.spec.source.path,
            "output": self.spec.output.path,
            "state": "ended" if self.ending is not None else self.state,
            "since": _iso(self.since),
            "format": self.spec.format,
            "video": self.spec.video.model_dump(),
            "encoder": self.encoder,
            "bytes": self.made,
        }
        if self.error and self.ending is None:
            found["error"] = self.error
        if self.ending is not None:
            found["reason"] = self.ending.reason
        return found

    async def start(self) -> None:
        self.output = await self.open_output(self.endpoint, self.spec.output.path)
        self.task = asyncio.ensure_future(self._run())

    async def _run(self) -> None:
        pause = RETRY_SECONDS[0]
        while self.ending is None:
            try:
                source = await self.open_source(self.endpoint, self.spec.source.path)
            except EmsError as exc:
                self.state, self.error = "lost", exc.message
            else:
                try:
                    written = await self._encode(source)
                finally:
                    with suppress(OSError):
                        os.close(source)
                if written is None:
                    return
                if written:
                    pause = RETRY_SECONDS[0]
                elif self.encoder != self.encoders[-1] and _encoder_failed(self.error or "", self.encoder):
                    self.encoder = self.encoders[self.encoders.index(self.encoder) + 1]
                if self.ending is None:
                    self.state = "lost"
            await asyncio.sleep(pause)
            pause = min(pause * 2, RETRY_SECONDS[1])

    async def _encode(self, source: int) -> int | None:
        """Encode what the source gives into the output until it ends: how much was written, or
        None once the output could not be written, which ends the transcode."""
        process = self.process = await asyncio.create_subprocess_exec(
            *self.command(self.spec, self.encoder, source), stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, pass_fds=(source,), start_new_session=True)
        assert process.stdout is not None and process.stderr is not None
        errors = asyncio.ensure_future(process.stderr.read())
        written = 0
        try:
            while chunk := await process.stdout.read(1 << 16):
                await asyncio.to_thread(_write_all, self.output, chunk)
                written += len(chunk)
                self.made += len(chunk)
                self.state, self.error = "running", None
        except OSError as exc:
            await self.end(f"the output could not be written: {exc.strerror or exc}")
            return None
        finally:
            _kill(process)
            await process.wait()
            self.process = None
        tail = (await errors).decode(errors="replace").strip().splitlines()
        self.error = f"the encode ended{f': {tail[-1][-200:]}' if tail else ''}"
        return written

    async def end(self, reason: str) -> None:
        if self.ending is not None:
            return
        self.ending = Ending(reason, time.monotonic())
        self.log(f"transcodes: {self.id} ended: {reason}")
        if self.process is not None:
            _kill(self.process)
        if self.task is not None and self.task is not asyncio.current_task():
            self.task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self.task
        # Closed, the caller's stream ends for its readers.
        if self.output is not None:
            with suppress(OSError):
                os.close(self.output)
            self.output = None


class Transcodes:
    """Every transcode this EMS runs, by id; past `limit` running a new one is busy."""

    def __init__(self, limit: int, *, encoders: Encoders, open_source: Opener, open_output: Opener, log: Callable[[str], None],
                 command: Command = encoder_args):
        self.limit = limit
        self.encoders = encoders
        self.open_source = open_source
        self.open_output = open_output
        self.command = command
        self.log = log
        self.transcodes: dict[str, Transcode] = {}

    def _forget_ended(self) -> None:
        now = time.monotonic()
        for transcode_id, transcode in list(self.transcodes.items()):
            if transcode.ending is not None and now - transcode.ending.at > ENDED_KEPT_SECONDS:
                del self.transcodes[transcode_id]

    def list(self) -> list[Transcode]:
        self._forget_ended()
        return list(self.transcodes.values())

    def get(self, transcode_id: str) -> Transcode:
        self._forget_ended()
        transcode = self.transcodes.get(transcode_id)
        if transcode is None:
            raise EmsError(NOT_FOUND, f"no transcode {transcode_id}")
        return transcode

    async def put(self, transcode_id: str, spec: TranscodeSpec, endpoint: Endpoint) -> tuple[Transcode, bool]:
        """The transcode `transcode_id` of `spec` through `endpoint`: started, or started again
        when its spec or endpoint changed or it ended. Answers it and whether it was started now."""
        if not TRANSCODE_ID.match(transcode_id):
            raise EmsError(INVALID_REQUEST, "a transcode's id is 1 to 128 letters, digits, '.', '_' or '-'")
        self._forget_ended()
        known = self.transcodes.get(transcode_id)
        if known is not None and known.ending is None and known.spec == spec and known.endpoint == endpoint:
            return known, False
        running = sum(1 for each in self.transcodes.values() if each.ending is None and each.id != transcode_id)
        if running >= self.limit:
            raise EmsError(BUSY, f"this EMS transcodes at most {self.limit} streams (EMS_TRANSCODES)", retry_after=5)
        available = await self.encoders()
        order = [name for name in ("h264_nvenc", "libopenh264") if name in available]
        if not order:
            raise EmsError(TOOL_FAILED, "this EMS's ffmpeg has no H.264 encoder", permanent=True)
        if known is not None:
            await known.end("put again")
        transcode = Transcode(transcode_id, spec, endpoint, open_source=self.open_source, open_output=self.open_output,
                              encoders=order, command=self.command, log=self.log)
        self.transcodes[transcode_id] = transcode
        try:
            await transcode.start()
        except BaseException as exc:
            await transcode.end(exc.message if isinstance(exc, EmsError) else "did not start")
            raise
        return transcode, True

    async def delete(self, transcode_id: str) -> None:
        transcode = self.get(transcode_id)
        await transcode.end("deleted")
        del self.transcodes[transcode_id]

    async def stop(self) -> None:
        for transcode in list(self.transcodes.values()):
            await transcode.end("EMS stopped")
        self.transcodes.clear()
