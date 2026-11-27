"""`video.stream` — a video as H.264 and AAC in fragmented MP4 or MPEG-TS, made while it is read.

For a player that cannot decode the AV1 renditions, such as a Cast or DLNA screen. Nothing is
written anywhere: the response body is the stream, and closing the connection kills the encoder.
A stream of a set length on the source's clock is one segment of an HLS playlist its caller makes:
each is its own encode, so its timestamps are made to run on from the one before it.

The encoder is chosen before the first byte goes out: everything in VRAM, then a CPU decode into
NVENC, then the CPU alone, each started in turn until one writes. Once bytes have gone out a
failure can only end the stream.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
from collections import OrderedDict
from typing import AsyncIterator, Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import Field

from .. import remote
from ..context import Context
from ..errors import BUSY, REFUSED, TOOL_FAILED, UNDECODABLE, EmsError, invalid
from ..schemas import FileRef, Strict, WithOptions
from ..tools import encode
from ..tools import ffmpeg as ffmpeg_tool
from ..tools import ffprobe as ffprobe_tool
from ..tools.run import require, run, which
from ..tools.versions import producer
from .transcode import BoxSize, Hints

GPU_ENCODER = "h264_nvenc"
CPU_ENCODER = "libopenh264"
# High profile at level 4.1 is what the oldest screens still in use decode (a 2nd-generation
# Chromecast: 1080p at 30 frames a second); the bitrate is held under what Wi-Fi carries.
PROFILE = ["-profile:v", "high", "-level:v", "4.1"]
MAX_RATE = "8M"
BUFFER = "16M"
CPU_RATE = "5M"
# A keyframe every two seconds, so a reader can start, or a player seek, without waiting long.
KEYFRAMES = ["-force_key_frames", "expr:gte(t,n_forced*2)"]
# The header goes out at once (empty_moov), and then a fragment at least every second.
FRAGMENTS = ["-movflags", "frag_keyframe+empty_moov+default_base_moof", "-frag_duration", "1000000", "-f", "mp4"]
# MPEG-TS with no mux delay, so a stream on the source's clock starts where it says.
TRANSPORT = ["-muxdelay", "0", "-f", "mpegts"]
# A stream on the source's clock runs this far ahead of it, every segment alike. A muxer moves a
# stream whose first decode time is below 0 (its first picture's less the B-frames' reorder delay)
# up to 0, which would put the first segment alone off the clock, its decode times past the next's.
CLOCK_LEAD = 10
# The AAC encoder's first packet is its priming, before the stream's start; in a segment it would
# fall inside the one before, and a player that checks the times refuses the playlist.
NO_PRIMING = ["-bsf:a", "noise=drop=not(n)"]
CONTENT_TYPES = {"mp4": "video/mp4", "mpegts": "video/mp2t"}
CHUNK = 64 * 1024
# The probes kept of the files streamed last: an HLS player reads a video as a request a segment.
PROBES = 16


class StreamVideo(Strict):
    box: BoxSize = Field(default_factory=lambda: BoxSize(width=1920, height=1080))
    fps: int = Field(default=30, ge=1, le=60)
    quality: int = Field(default=23, ge=0, le=51)


class StreamAudio(Strict):
    bitrate: str = "128k"


class VideoStreamRequest(WithOptions):
    file: FileRef
    video: StreamVideo = Field(default_factory=StreamVideo)
    audio: StreamAudio | None = Field(default_factory=StreamAudio)
    start: float = Field(default=0, ge=0)
    duration: float | None = Field(default=None, gt=0)
    source_clock: bool = False
    format: Literal["mp4", "mpegts"] = "mp4"
    hints: Hints = Field(default_factory=Hints)


_encoders: dict[str, set[str]] = {}
_probes: OrderedDict[tuple[str, str, int, int], ffprobe_tool.Probe] = OrderedDict()


async def probed(mount: str, src: str, timeout: float) -> ffprobe_tool.Probe:
    """The probe of `src`, a file in the caller's `mount`, kept while its size and change time stay
    as they were; one that could not read the file is asked again the next time. The file is known
    by its name: the caller grants it in a directory of its own at each request."""
    try:
        stat = await asyncio.to_thread(os.stat, src)
    except OSError:
        return await ffprobe_tool.probe(src, timeout=timeout)
    key = (mount, os.path.basename(src), stat.st_size, stat.st_mtime_ns)
    probe = _probes.get(key)
    if probe is not None:
        _probes.move_to_end(key)
        return probe
    probe = await ffprobe_tool.probe(src, timeout=timeout)
    if not probe.refusal:
        _probes[key] = probe
        while len(_probes) > PROBES:
            _probes.popitem(last=False)
    return probe


async def h264_encoders() -> set[str]:
    """The H.264 encoders this ffmpeg offers: NVENC's where a card is, and the CPU's; none without
    ffmpeg. Read once."""
    if "found" not in _encoders:
        found: set[str] = set()
        ffmpeg = which("ffmpeg")
        done = await run([ffmpeg, "-hide_banner", "-encoders"], timeout=30) if ffmpeg else None
        if done is not None and done.code == 0:
            listed = set(re.findall(r"^\s*V\S*\s+(\S+)", done.stdout.decode(errors="replace"), re.MULTILINE))
            if GPU_ENCODER in listed and ffmpeg_tool.gpu_present():
                found.add(GPU_ENCODER)
            if CPU_ENCODER in listed:
                found.add(CPU_ENCODER)
        _encoders["found"] = found
    return _encoders["found"]


class Streams:
    """How many streams run, against `EMS_STREAMS`: each holds an encoder, and the card's encoder
    sessions are few and shared with the transcodes."""

    def __init__(self, limit: int):
        self.limit = limit
        self.running = 0

    def take(self) -> None:
        if self.running >= self.limit:
            raise EmsError(BUSY, f"this EMS runs at most {self.limit} streams (EMS_STREAMS)", retry_after=5)
        self.running += 1

    def give(self) -> None:
        self.running -= 1


class Encoding:
    """One ffmpeg writing the stream to its stdout, in its own process group."""

    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process
        self.errors = bytearray()
        self.drain = asyncio.ensure_future(self._drain())

    async def _drain(self) -> None:
        assert self.process.stderr is not None
        while chunk := await self.process.stderr.read(CHUNK):
            self.errors = (self.errors + chunk)[-4096:]

    async def read(self) -> bytes:
        assert self.process.stdout is not None
        return await self.process.stdout.read(CHUNK)

    def tail(self) -> str:
        lines = [line for line in self.errors.decode(errors="replace").splitlines() if line.strip()]
        return " | ".join(lines[-2:])[-300:]

    async def stop(self) -> None:
        if self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await self.process.wait()
        await asyncio.gather(self.drain, return_exceptions=True)


async def spawn(args: list[str]) -> Encoding:
    # Not niced, unlike EMS's other tools: someone is watching, and a stream that falls behind
    # stalls on the screen.
    process = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
    return Encoding(process)


def stages(req: VideoStreamRequest, probe: ffprobe_tool.Probe, src: str, encoders: set[str]) -> list[tuple[str, list[str]]]:
    """Each way to make the stream, best first, as (what ran, ffmpeg's arguments)."""
    video = probe.first("video")
    if video is None:
        raise EmsError(REFUSED, "the file has no picture to stream", permanent=True)
    shape = encode.Shape(width=req.video.box.width, height=req.video.box.height, angle=req.file.angle, mirror=req.file.mirror, deinterlace=False)
    seek = ["-ss", f"{req.start:g}"] if req.start > 0 else []
    maps = ["-map", f"0:{video['index']}"] + (ffprobe_tool.audio_map(probe) if req.audio is not None else [])
    sound = ["-an"] if req.audio is None else ["-c:a", "aac", "-b:a", req.audio.bitrate, "-ac", "2"]
    length = ["-t", f"{req.duration:g}"] if req.duration else []
    clock = ["-output_ts_offset", f"{req.start + CLOCK_LEAD:g}", *NO_PRIMING] if req.source_clock else []
    container = TRANSPORT if req.format == "mpegts" else FRAGMENTS
    tail = [*length, "-fpsmax", str(req.video.fps), *KEYFRAMES, *sound, *clock, *container, "pipe:1"]
    head = [require("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error", "-display_rotation", "0"]
    gpu = ["-c:v", GPU_ENCODER, "-preset", "p4", "-rc", "vbr", "-cq", str(req.video.quality), "-maxrate", MAX_RATE, "-bufsize", BUFFER, *PROFILE, "-forced-idr", "1"]
    cpu = ["-c:v", CPU_ENCODER, "-b:v", CPU_RATE, "-maxrate", MAX_RATE]
    found: list[tuple[str, list[str]]] = []
    if GPU_ENCODER in encoders:
        cuda_chain = encode.cuda_filters(shape)
        codec = req.hints.source_codec or video.get("codec_name")
        if cuda_chain and codec not in encode.NVDEC_CANNOT_DECODE:
            found.append((f"{GPU_ENCODER}/cuda", [*head, "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", *seek, *ffmpeg_tool.source(src), *maps, "-vf", cuda_chain, *gpu, *tail]))
        found.append((f"{GPU_ENCODER}/upload", [*head, *seek, *ffmpeg_tool.source(src), *maps, "-vf", encode.upload_filters(shape), *gpu, *tail]))
    if CPU_ENCODER in encoders:
        found.append((CPU_ENCODER, [*head, *seek, *ffmpeg_tool.source(src), *maps, "-vf", f"{encode.cpu_filters(shape)},format=yuv420p", *cpu, *tail]))
    if not found:
        raise EmsError(TOOL_FAILED, "this EMS's ffmpeg has no H.264 encoder", permanent=True)
    return found


async def start(ctx: Context, req: VideoStreamRequest, mount: str, src: str) -> tuple[str, Encoding, bytes]:
    """The first stage that writes: what ran, its encoder, and its first bytes."""
    probe = await probed(mount, src, ctx.settings.tool_stall_seconds)
    if probe.refusal:
        raise EmsError(UNDECODABLE, f"source cannot be read: {probe.refusal}")
    failures: list[str] = []
    for ran, args in stages(req, probe, src, await h264_encoders()):
        encoding = await spawn(args)
        try:
            first = await encoding.read()
        except BaseException:
            await encoding.stop()
            raise
        if first:
            return ran, encoding, first
        await encoding.stop()
        failures.append(f"{ran}: {encoding.tail() or f'exited {encoding.process.returncode} with nothing written'}")
    raise EmsError(TOOL_FAILED, "no encoder could start the stream: " + " | ".join(failures))


class StreamResponse(StreamingResponse):
    """A streaming response that runs `close` however it ends: read to the end, or the reader gone
    before or while it is sent."""

    def __init__(self, body: AsyncIterator[bytes], close, media_type: str, headers: dict[str, str]):
        super().__init__(body, media_type=media_type, headers=headers)
        self.close = close

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.close()


def register(router: APIRouter, ctx: Context) -> None:
    streams = Streams(ctx.settings.streams)

    @router.post("/video/stream")
    async def video_stream(request: Request, body: VideoStreamRequest):
        header = request.headers.get(remote.HEADER)
        if not header:
            raise invalid(f"{body.file.path}: a request that names files names their endpoint in {remote.HEADER}")
        if ctx.files is None:
            raise invalid("this EMS reaches no caller's files")
        staging = remote.Staging(ctx.files, remote.parse_header(header), ctx.scratch_dir("stream"))
        streams.take()
        encoding: Encoding | None = None
        closed = False

        async def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            if encoding is not None:
                await encoding.stop()
            streams.give()
            staging.close()

        # Until the response holds `close`, a reader that goes away (a cancel at any await here) is
        # let go of here.
        try:
            await staging.prepare(body.model_dump())
            ran, encoding, first = await start(ctx, body, staging.point, staging.input(body.file.path))
            made_by = f"{await producer('ffmpeg')}/{ran}"
        except BaseException:
            await close()
            raise

        async def chunks() -> AsyncIterator[bytes]:
            yield first
            while chunk := await encoding.read():
                yield chunk

        return StreamResponse(chunks(), close, CONTENT_TYPES[body.format], {"Cache-Control": "no-store", "EMS-Producer": made_by})
