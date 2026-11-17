"""Pages rendered into live video for screens (`/v0/pages`, docs/api.md 4.25).

A page is opened in Chromium on a virtual X display of its own (Xvfb), which ffmpeg grabs at a
steady frame rate and encodes once as H.264. The one encode is muxed three times: as fragmented MP4
(what a Cast player takes when it is told the stream is live) and as MPEG-TS (what a DLNA screen
takes), whose every reader is served from it, one that joins getting the format's header and
everything since the latest keyframe, then what comes; and as live HLS, a playlist of the latest
seconds in files of a second each, which ffmpeg writes to the page's folder (what a browser hands a
screen near it, and AirPlay plays). A stream nobody has read for its `idle` seconds stops itself.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import struct
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable, Literal

from pydantic import Field, field_validator

from .errors import BUSY, INVALID_REQUEST, NOT_FOUND, TOOL_FAILED, EmsError
from .schemas import Strict

PAGE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
FORMATS = ("mp4", "ts")
# A fragment at least this often, in microseconds, so a player is never more than this behind what
# was encoded.
FRAGMENT_US = 200_000
# Chrome reports a stream's metadata, which it needs before it offers a screen near it the stream,
# once it has read about a second of media, and it reads in blocks of this many bytes: a new
# page's fragments, a few hundred bytes of blank page each, take seconds to fill one
# (ein/core#1420). A reader that asks for a declared duration is given both at once.
QUICK_START_SECONDS = 1.0
CHROME_READ = 32 * 1024
# Pieces a reader may fall behind before it is let go: about a minute of fragments.
READER_BACKLOG = 300
# How long a stream that ended stays listed, with why, before it is forgotten.
ENDED_KEPT_SECONDS = 60.0
# How long a new stream may take to give its first keyframe in both formats.
START_SECONDS = 30.0
# How often the processes, the memory and the readers are looked at.
WATCH_SECONDS = 1.0
TS_PACKET = 188
# The HLS form: segments of a keyframe's second each, the playlist naming the latest of them.
HLS_PLAYLIST = "index.m3u8"
HLS_SEGMENT_SECONDS = 1
HLS_LISTED = 10
# The files a reader may ask for, as ffmpeg names them beside the playlist.
HLS_NAME = re.compile(r"^index(\.m3u8|\d{1,9}\.ts)$")
# How long a playlist asked for before ffmpeg has written it is waited for.
HLS_START_SECONDS = 10.0
# An HLS player holds no answer open; it is counted a reader this long after it asked for a file.
HLS_READ_SECONDS = 5.0


class PageVideo(Strict):
    width: int = Field(default=1920, ge=320, le=1920)
    height: int = Field(default=1080, ge=240, le=1080)
    fps: int = Field(default=30, ge=5, le=30)
    quality: int = Field(default=23, ge=0, le=51)


class PageSpec(Strict):
    url: str
    video: PageVideo = Field(default_factory=PageVideo)
    idle: float = Field(default=20, ge=5, le=600)

    @field_validator("url")
    @classmethod
    def _http(cls, url: str) -> str:
        if not re.match(r"^https?://[^\s/]+", url):
            raise ValueError("a page's url is http:// or https://")
        return url


# ── The two formats, cut where a reader may start ─────────────────────────────────────────────


def _boxes(data: bytes, start: int, end: int):
    """(type, offset, size) of each box in data[start:end]."""
    at = start
    while at + 8 <= end:
        size, kind = struct.unpack(">I4s", data[at:at + 8])
        if size == 1:
            size = struct.unpack(">Q", data[at + 8:at + 16])[0]
        if size < 8 or at + size > end:
            return
        yield kind, at, size
        at += size


def _child(data: bytes, offset: int, size: int, kind: bytes) -> tuple[int, int] | None:
    for found, at, length in _boxes(data, offset + 8, offset + size):
        if found == kind:
            return at, length
    return None


NON_SYNC = 0x10000


def fragment_is_key(moof: bytes) -> bool:
    """Whether a fragment's first sample is a keyframe: its `trun`'s first-sample or first sample's
    flags, else its `tfhd`'s default, without the non-sync bit."""
    traf = _child(moof, 0, len(moof), b"traf")
    if traf is None:
        return False
    default: int | None = None
    tfhd = _child(moof, *traf, b"tfhd")
    if tfhd is not None:
        flags = struct.unpack(">I", moof[tfhd[0] + 8:tfhd[0] + 12])[0] & 0xFFFFFF
        at = tfhd[0] + 16
        for bit, width in ((0x1, 8), (0x2, 4), (0x8, 4), (0x10, 4)):
            if flags & bit:
                at += width
        if flags & 0x20:
            default = struct.unpack(">I", moof[at:at + 4])[0]
    trun = _child(moof, *traf, b"trun")
    if trun is None:
        return False
    flags = struct.unpack(">I", moof[trun[0] + 8:trun[0] + 12])[0] & 0xFFFFFF
    at = trun[0] + 16
    if flags & 0x1:
        at += 4
    if flags & 0x4:
        return not struct.unpack(">I", moof[at:at + 4])[0] & NON_SYNC
    if flags & 0x400:
        at += (4 if flags & 0x100 else 0) + (4 if flags & 0x200 else 0)
        return not struct.unpack(">I", moof[at:at + 4])[0] & NON_SYNC
    return default is not None and not default & NON_SYNC


def _tfdt(fragment: bytes) -> tuple[int, int] | None:
    """Where a fragment's decode time is, and its width in bytes."""
    moof = next(((at, size) for kind, at, size in _boxes(fragment, 0, len(fragment)) if kind == b"moof"), None)
    if moof is None:
        return None
    traf = _child(fragment, *moof, b"traf")
    tfdt = _child(fragment, *traf, b"tfdt") if traf is not None else None
    if tfdt is None:
        return None
    return (tfdt[0] + 12, 8 if fragment[tfdt[0] + 8] == 1 else 4)


def decode_time(fragment: bytes) -> int:
    found = _tfdt(fragment)
    if found is None:
        return 0
    at, width = found
    return int.from_bytes(fragment[at:at + width], "big")


def at_decode_time(fragment: bytes, value: int) -> bytes:
    """The fragment with its decode time `value`."""
    found = _tfdt(fragment)
    if found is None:
        return fragment
    at, width = found
    return fragment[:at] + value.to_bytes(width, "big") + fragment[at + width:]


def _flags(box: bytes, at: int) -> int:
    return struct.unpack(">I", box[at + 8:at + 12])[0] & 0xFFFFFF


def fragment_duration(fragment: bytes, header: bytes) -> int | None:
    """How long a fragment lasts in its track's timescale: its samples' durations from its `trun`,
    or their count times its `tfhd`'s default, or the header's `trex`'s; `None` when it says none."""
    moof = next(((at, size) for kind, at, size in _boxes(fragment, 0, len(fragment)) if kind == b"moof"), None)
    traf = _child(fragment, *moof, b"traf") if moof is not None else None
    tfhd = _child(fragment, *traf, b"tfhd") if traf is not None else None
    trun = _child(fragment, *traf, b"trun") if traf is not None else None
    if tfhd is None or trun is None:
        return None
    default: int | None = None
    flags = _flags(fragment, tfhd[0])
    at = tfhd[0] + 16 + (8 if flags & 0x1 else 0) + (4 if flags & 0x2 else 0)
    if flags & 0x8:
        default = struct.unpack(">I", fragment[at:at + 4])[0]
    else:
        moov = next(((at, size) for kind, at, size in _boxes(header, 0, len(header)) if kind == b"moov"), None)
        mvex = _child(header, *moov, b"mvex") if moov is not None else None
        trex = _child(header, *mvex, b"trex") if mvex is not None else None
        if trex is not None:
            default = struct.unpack(">I", header[trex[0] + 20:trex[0] + 24])[0]
    flags = _flags(fragment, trun[0])
    count = struct.unpack(">I", fragment[trun[0] + 12:trun[0] + 16])[0]
    if not flags & 0x100:
        return count * default if default else None
    at = trun[0] + 16 + (4 if flags & 0x1 else 0) + (4 if flags & 0x4 else 0)
    width = 4 * sum(1 for bit in (0x100, 0x200, 0x400, 0x800) if flags & bit)
    return sum(struct.unpack(">I", fragment[at + n * width:at + n * width + 4])[0] for n in range(count))


def quick_start(header: bytes, fragments: list[bytes]) -> tuple[list[bytes], int]:
    """What a reader is sent at once so that Chrome reads its metadata at once: the header and the
    fragments since the keyframe (their clock at 0), the keyframe's fragment repeated after itself
    until they hold `QUICK_START_SECONDS`, and a `free` box filling the bytes out to Chrome's next
    read. Repeated straight after itself, the fragment leaves the decoder as it found it, so the
    next one follows on. Also how much later the fragments that follow come: what was repeated."""
    moov = next(((at, size) for kind, at, size in _boxes(header, 0, len(header)) if kind == b"moov"), None)
    trak = _child(header, *moov, b"trak") if moov is not None else None
    mdia = _child(header, *trak, b"mdia") if trak is not None else None
    mdhd = _child(header, *mdia, b"mdhd") if mdia is not None else None
    lengths = [fragment_duration(fragment, header) for fragment in fragments]
    if mdhd is None or not fragments or None in lengths or not lengths[0]:
        return [header, *fragments], 0
    scale = struct.unpack(">I", header[mdhd[0] + (28 if header[mdhd[0] + 8] == 1 else 20):][:4])[0]
    first = lengths[0]
    repeats = max(0, -(-(int(QUICK_START_SECONDS * scale) - sum(lengths)) // first))
    shift = repeats * first
    out = [header, fragments[0], *(at_decode_time(fragments[0], n * first) for n in range(1, repeats + 1)),
           *(at_decode_time(fragment, decode_time(fragment) + shift) for fragment in fragments[1:])]
    gap = -sum(len(piece) for piece in out) % CHROME_READ
    if gap:
        gap += CHROME_READ if gap < 8 else 0
        out.append(struct.pack(">I4s", gap, b"free") + bytes(gap - 8))
    return out, shift


def rebased(fragment: bytes, base: int) -> bytes:
    """The fragment with its decode time `base` earlier, so a reader's stream starts at 0."""
    found = _tfdt(fragment)
    if found is None or base == 0:
        return fragment
    at, width = found
    value = max(0, int.from_bytes(fragment[at:at + width], "big") - base)
    return fragment[:at] + value.to_bytes(width, "big") + fragment[at + width:]


def declared(header: bytes, seconds: int) -> bytes:
    """The MP4 header (`ftyp` and `moov`) saying the stream lasts `seconds`: in `mvhd`, each track's
    `tkhd` and `mdhd`, and an `mehd` in `mvex`. ffmpeg's live header says nothing (0), so a browser
    takes what it has read for the duration, and Chrome offers a screen near it nothing of 15 s or
    less; one that reads this one counts the stream that long."""
    b = bytearray(header)
    moov = next(((at, size) for kind, at, size in _boxes(b, 0, len(b)) if kind == b"moov"), None)
    if moov is None:
        return header

    def put(box: int, value: int, track: bool = False) -> None:
        # A full box's duration follows its version and flags, two times and a word: mvhd's and
        # mdhd's timescale, tkhd's track id, then for tkhd a reserved word too.
        at = box + (32 if b[box + 8] == 1 else 24) + (4 if track else 0)
        if b[box + 8] == 1:
            struct.pack_into(">Q", b, at, value)
        else:
            struct.pack_into(">I", b, at, value)

    def timescale(box: int) -> int:
        return struct.unpack_from(">I", b, box + (28 if b[box + 8] == 1 else 20))[0]

    mvhd = _child(b, *moov, b"mvhd")
    mvex = _child(b, *moov, b"mvex")
    if mvhd is None or mvex is None:
        return header
    scale = timescale(mvhd[0])
    put(mvhd[0], seconds * scale)
    for kind, at, size in _boxes(b, moov[0] + 8, moov[0] + moov[1]):
        if kind != b"trak":
            continue
        tkhd = _child(b, at, size, b"tkhd")
        if tkhd is not None:
            put(tkhd[0], seconds * scale, track=True)
        mdia = _child(b, at, size, b"mdia")
        mdhd = _child(b, *mdia, b"mdhd") if mdia is not None else None
        if mdhd is not None:
            put(mdhd[0], seconds * timescale(mdhd[0]))
    if _child(b, *mvex, b"mehd") is None:
        b[mvex[0] + 8:mvex[0] + 8] = struct.pack(">I4sII", 16, b"mehd", 0, seconds * scale)
        struct.pack_into(">I", b, mvex[0], mvex[1] + 16)
        struct.pack_into(">I", b, moov[0], moov[1] + 16)
    return bytes(b)


class Mp4Splitter:
    """ffmpeg's fragmented MP4 cut into its header (`ftyp` and `moov`) and its fragments (`moof`
    and `mdat` together), each told whether it starts at a keyframe."""

    def __init__(self):
        self.buffer = bytearray()
        self.head = bytearray()
        self.header: bytes | None = None
        self.moof: bytes | None = None

    def feed(self, data: bytes) -> list[tuple[bool, bytes]]:
        self.buffer += data
        out: list[tuple[bool, bytes]] = []
        while len(self.buffer) >= 8:
            size, kind = struct.unpack(">I4s", self.buffer[:8])
            if size == 1:
                if len(self.buffer) < 16:
                    break
                size = struct.unpack(">Q", self.buffer[8:16])[0]
            if size < 8:
                raise ValueError(f"an MP4 box of {size} bytes")
            if len(self.buffer) < size:
                break
            box = bytes(self.buffer[:size])
            del self.buffer[:size]
            if kind in (b"ftyp", b"moov"):
                self.head += box
                if kind == b"moov":
                    self.header = bytes(self.head)
            elif kind == b"moof":
                self.moof = box
            elif kind == b"mdat" and self.moof is not None:
                out.append((fragment_is_key(self.moof), self.moof + box))
                self.moof = None
        return out


class TsSplitter:
    """ffmpeg's MPEG-TS cut into pieces of whole packets, a piece starting at each packet that marks
    a keyframe (`random_access_indicator`). The header a reader starts with is the latest programme
    tables: the PAT and the PMT it names."""

    def __init__(self):
        self.buffer = bytearray()
        self.pmt_pid: int | None = None
        self.pat: bytes | None = None
        self.pmt: bytes | None = None

    @property
    def header(self) -> bytes | None:
        return self.pat + self.pmt if self.pat is not None and self.pmt is not None else None

    def feed(self, data: bytes) -> list[tuple[bool, bytes]]:
        self.buffer += data
        out: list[tuple[bool, bytes]] = []
        piece = bytearray()
        key = False
        while len(self.buffer) >= TS_PACKET:
            if self.buffer[0] != 0x47:
                skip = self.buffer.find(b"\x47", 1)
                del self.buffer[:skip if skip > 0 else len(self.buffer)]
                continue
            packet = bytes(self.buffer[:TS_PACKET])
            del self.buffer[:TS_PACKET]
            pid = ((packet[1] & 0x1F) << 8) | packet[2]
            adaptation = (packet[3] >> 4) & 0x3
            if pid == 0 and packet[1] & 0x40:
                self.pat = packet
                self.pmt_pid = _pmt_pid(packet, adaptation)
            elif pid == self.pmt_pid and packet[1] & 0x40:
                self.pmt = packet
            if adaptation & 0x2 and packet[4] > 0 and packet[5] & 0x40:
                if piece:
                    out.append((key, bytes(piece)))
                piece = bytearray()
                key = True
            piece += packet
        if piece:
            out.append((key, bytes(piece)))
        return out


def _pmt_pid(packet: bytes, adaptation: int) -> int | None:
    at = 4 + (1 + packet[4] if adaptation & 0x2 else 0)
    at += 1 + packet[at]  # the pointer field
    length = ((packet[at + 1] & 0x0F) << 8) | packet[at + 2]
    end = min(at + 3 + length - 4, TS_PACKET)
    for entry in range(at + 8, end - 3, 4):
        program = (packet[entry] << 8) | packet[entry + 1]
        if program != 0:
            return ((packet[entry + 2] & 0x1F) << 8) | packet[entry + 3]
    return None


# ── Readers ────────────────────────────────────────────────────────────────────────────────────


class Reader:
    """One reader's pieces, waiting to be sent."""

    def __init__(self, first: list[bytes]):
        self.pieces: deque[bytes] = deque(first)
        # How many of the pieces it was given on joining, all there at once.
        self.joined = len(first)
        self.event = asyncio.Event()
        self.ended = False

    def offer(self, piece: bytes) -> bool:
        if len(self.pieces) >= READER_BACKLOG:
            return False
        self.pieces.append(piece)
        self.event.set()
        return True

    def end(self) -> None:
        self.ended = True
        self.event.set()

    async def next(self) -> bytes | None:
        while not self.pieces:
            if self.ended:
                return None
            self.event.clear()
            await self.event.wait()
        return self.pieces.popleft()


class Channel:
    """One format of a stream: its header, the pieces since the latest keyframe, and its readers."""

    def __init__(self, kind: Literal["mp4", "ts"], splitter: Mp4Splitter | TsSplitter):
        self.kind = kind
        self.splitter = splitter
        self.gop: list[bytes] = []
        self.readers: set[Reader] = set()

    @property
    def playable(self) -> bool:
        return self.splitter.header is not None and bool(self.gop)

    def feed(self, data: bytes) -> None:
        for key, piece in self.splitter.feed(data):
            if key:
                self.gop = [piece]
            elif self.gop:
                self.gop.append(piece)
            else:
                continue  # nothing before the first keyframe reaches anyone
            for reader in list(self.readers):
                if not reader.offer(piece):
                    self.leave(reader)
                    reader.end()

    def join(self) -> Reader:
        header = self.splitter.header
        assert header is not None and self.gop
        reader = Reader([header, *self.gop])
        self.readers.add(reader)
        return reader

    def leave(self, reader: Reader) -> None:
        self.readers.discard(reader)

    def end(self) -> None:
        for reader in list(self.readers):
            reader.end()
        self.readers.clear()

    async def chunks(self, reader: Reader, duration: int | None = None) -> AsyncIterator[bytes]:
        """What `reader` is sent: an MP4 reader's fragments with their clock starting at 0, and
        with `duration`, its header saying the stream lasts that many seconds (`declared`) and its
        start made for Chrome to read at once (`quick_start`)."""
        base: int | None = None
        if self.kind == "mp4" and duration:
            joined = [piece for _ in range(reader.joined) if (piece := await reader.next()) is not None]
            if not joined:
                return
            header, *fragments = joined
            base = decode_time(fragments[0]) if fragments else None
            start, shift = quick_start(declared(header, duration), [rebased(fragment, base or 0) for fragment in fragments])
            for piece in start:
                yield piece
            if base is not None:
                base -= shift
        while (piece := await reader.next()) is not None:
            if self.kind == "mp4" and piece[4:8] == b"moof":
                if base is None:
                    base = decode_time(piece)
                piece = rebased(piece, base)
            yield piece


# ── Rendering ──────────────────────────────────────────────────────────────────────────────────


class Rendering:
    """What draws and encodes a page: its processes, the encoder's two outputs, and the profile."""

    def __init__(self, processes: dict[str, asyncio.subprocess.Process], outputs: dict[str, asyncio.StreamReader], encoder: str,
                 folder: str | None = None, hls: str | None = None):
        self.processes = processes
        self.outputs = outputs
        self.encoder = encoder
        self.folder = folder
        self.hls = hls
        self.errors = bytearray()

    def exited(self) -> str | None:
        """The first of its processes that has ended, as `<name> exited with <code>`."""
        for name, process in self.processes.items():
            if process.returncode is not None:
                return f"{name} exited with {process.returncode}"
        return None

    def memory_mb(self) -> float:
        """What the browser holds, every process of its group together."""
        browser = self.processes.get("chromium")
        return group_pss_mb(browser.pid) if browser is not None else 0.0

    async def stop(self) -> None:
        for process in reversed(list(self.processes.values())):
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for process in self.processes.values():
            await process.wait()
        if self.folder:
            shutil.rmtree(self.folder, ignore_errors=True)


def group_pss_mb(pgid: int) -> float:
    """The memory every process in group `pgid` holds, in MB: each one's proportional share, so
    the pages Chromium's processes share are counted once."""
    total = 0
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as handle:
                fields = handle.read().rsplit(b")", 1)[1].split()
            if int(fields[2]) != pgid:
                continue
            with open(f"/proc/{name}/smaps_rollup", "rb") as handle:
                for line in handle:
                    if line.startswith(b"Pss:"):
                        total += int(line.split()[1]) * 1024
                        break
        except (OSError, IndexError, ValueError):
            continue
    return total / 1e6


Launch = Callable[[PageSpec, str], Awaitable[Rendering]]


async def _pipe_reader(fd: int) -> asyncio.StreamReader:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=1 << 22)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(fd, "rb", buffering=0))
    return reader


def chromium_args(spec: PageSpec, profile: str, memory_mb: int) -> list[str]:
    v = spec.video
    args = [
        "--kiosk", "--no-first-run", "--no-default-browser-check", f"--user-data-dir={profile}",
        "--window-position=0,0", f"--window-size={v.width},{v.height}", "--force-device-scale-factor=1",
        "--noerrdialogs", "--disable-infobars", "--disable-session-crashed-bubble", "--disable-extensions",
        "--disable-component-update", "--disable-features=Translate,MediaRouter", "--password-store=basic",
        "--autoplay-policy=no-user-gesture-required", "--hide-scrollbars", "--disable-dev-shm-usage",
        # The page's heap: half the cap, the rest is the browser's own.
        f"--js-flags=--max-old-space-size={max(128, memory_mb // 2)}",
    ]
    # Chromium's sandbox does not run as root, which EMS's container is.
    if os.geteuid() == 0:
        args.append("--no-sandbox")
    return [*args, spec.url]


def _tee_path(path: str) -> str:
    """`path` as a file of ffmpeg's tee, whose own characters are escaped."""
    return re.sub(r"([\\:|\[\]'])", r"\\\1", path)


def encoder_args(spec: PageSpec, encoder: str, ts_fd: int, hls: str) -> list[str]:
    """ffmpeg's output side: the encode, the fragmented MP4 to stdout, the MPEG-TS to `ts_fd` and
    the HLS to the folder `hls`."""
    v = spec.video
    if encoder == "h264_nvenc":
        codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll", "-rc", "vbr", "-cq", str(v.quality), "-maxrate", "8M",
                 "-bufsize", "16M", "-profile:v", "high", "-level:v", "4.1", "-forced-idr", "1", "-bf", "0"]
    else:
        codec = ["-c:v", encoder, "-b:v", "5M", "-maxrate", "8M"]
    # One encode, muxed three times. Each format's header is written once, so the MPEG-TS and the
    # HLS segments repeat the picture's parameters at every keyframe for a reader that starts there.
    tee = (f"[f=mp4:movflags=frag_keyframe+empty_moov+default_base_moof:frag_duration={FRAGMENT_US}]pipe\\:1"
           f"|[f=mpegts:bsfs/v=dump_extra=freq=keyframe]pipe\\:{ts_fd}"
           f"|[f=hls:bsfs/v=dump_extra=freq=keyframe:hls_time={HLS_SEGMENT_SECONDS}:hls_list_size={HLS_LISTED}"
           f":hls_flags=delete_segments+independent_segments]{_tee_path(os.path.join(hls, HLS_PLAYLIST))}")
    return [*codec, "-g", str(v.fps), "-flags", "+global_header", "-map", "0:v", "-f", "tee", tee]


def renderer(memory_mb: int, encoders: Callable[[], Awaitable[set[str]]], scratch: Callable[[], str]) -> Launch:
    """How a page is rendered here: Xvfb, Chromium on it, and ffmpeg grabbing it with the first
    encoder of NVENC and the CPU's that writes."""

    async def launch(spec: PageSpec, page_id: str) -> Rendering:
        for tool in ("Xvfb", "chromium", "ffmpeg"):
            if shutil.which(tool) is None:
                raise EmsError(TOOL_FAILED, f"this EMS has no {tool}, which renders pages", permanent=True)
        available = await encoders()
        order = [name for name in ("h264_nvenc", "libopenh264") if name in available]
        if not order:
            raise EmsError(TOOL_FAILED, "this EMS's ffmpeg has no H.264 encoder", permanent=True)
        folder = scratch()
        v = spec.video
        processes: dict[str, asyncio.subprocess.Process] = {}
        try:
            read, write = os.pipe()
            processes["xvfb"] = await asyncio.create_subprocess_exec(
                "Xvfb", "-displayfd", str(write), "-screen", "0", f"{v.width}x{v.height}x24", "-nolisten", "tcp",
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                pass_fds=(write,), start_new_session=True)
            os.close(write)
            told = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(None, os.read, read, 16), 10)
            os.close(read)
            if not told.strip():
                raise EmsError(TOOL_FAILED, "Xvfb gave no display")
            display = f":{int(told.strip())}"
            env = {**os.environ, "DISPLAY": display, "HOME": folder, "XDG_CONFIG_HOME": folder, "XDG_CACHE_HOME": folder}
            processes["chromium"] = await asyncio.create_subprocess_exec(
                "chromium", *chromium_args(spec, os.path.join(folder, "profile"), memory_mb), env=env,
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
            failures: list[str] = []
            hls = os.path.join(folder, "hls")
            for encoder in order:
                # What an encoder that could not start wrote is not the next one's.
                shutil.rmtree(hls, ignore_errors=True)
                os.makedirs(hls)
                grab = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "x11grab", "-framerate", str(v.fps),
                        "-video_size", f"{v.width}x{v.height}", "-draw_mouse", "0", "-i", display, "-vf", "format=yuv420p"]
                ts_read, ts_write = os.pipe()
                process = await asyncio.create_subprocess_exec(
                    *grab, *encoder_args(spec, encoder, ts_write, hls), stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, pass_fds=(ts_write,), start_new_session=True)
                os.close(ts_write)
                ts = await _pipe_reader(ts_read)
                assert process.stdout is not None and process.stderr is not None
                first = await process.stdout.read(1 << 16)
                if first:
                    processes["ffmpeg"] = process
                    mp4 = _prepended(process.stdout, first)
                    rendering = Rendering(processes, {"mp4": mp4, "ts": ts}, encoder, folder, hls)
                    asyncio.ensure_future(_drain(process.stderr, rendering.errors))
                    return rendering
                errors = (await process.stderr.read()).decode(errors="replace").strip().splitlines()
                await process.wait()
                failures.append(f"{encoder}: {(errors[-1] if errors else f'exited {process.returncode}')[-200:]}")
            raise EmsError(TOOL_FAILED, "no encoder could start the page's stream: " + " | ".join(failures))
        except BaseException:
            await Rendering(processes, {}, "", folder).stop()
            raise

    return launch


def _prepended(reader: asyncio.StreamReader, first: bytes) -> asyncio.StreamReader:
    """`reader` with `first` put back in front of what it has still to give."""
    joined = asyncio.StreamReader(limit=1 << 22)
    joined.feed_data(first)

    async def carry() -> None:
        while chunk := await reader.read(1 << 16):
            joined.feed_data(chunk)
        joined.feed_eof()

    asyncio.ensure_future(carry())
    return joined


async def _drain(stream: asyncio.StreamReader, into: bytearray) -> None:
    while chunk := await stream.read(1 << 14):
        into[:] = (into + chunk)[-4096:]


# ── Streams ────────────────────────────────────────────────────────────────────────────────────


def _iso(at: float) -> str:
    return datetime.fromtimestamp(at, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class Ending:
    reason: str
    at: float


class Page:
    """One page streamed: its rendering, its formats, and when it last had a reader."""

    def __init__(self, page_id: str, spec: PageSpec, memory_mb: int, log: Callable[[str], None]):
        self.id = page_id
        self.spec = spec
        self.memory_mb = memory_mb
        self.log = log
        self.since = time.time()
        self.channels = {"mp4": Channel("mp4", Mp4Splitter()), "ts": Channel("ts", TsSplitter())}
        self.rendering: Rendering | None = None
        self.ending: Ending | None = None
        self.made = 0
        self.unread_since = time.monotonic()
        self.hls_read: float | None = None
        self.playable = asyncio.Event()
        self.tasks: list[asyncio.Task] = []
        self.ended = asyncio.Event()

    @property
    def state(self) -> str:
        if self.ending is not None:
            return "ended"
        return "running" if self.playable.is_set() else "starting"

    def readers(self) -> int:
        return sum(len(channel.readers) for channel in self.channels.values()) + self.hls_readers()

    def hls_readers(self) -> int:
        """1 while the HLS form is read: a file of it was asked for within `HLS_READ_SECONDS`."""
        return int(self.hls_read is not None and time.monotonic() - self.hls_read < HLS_READ_SECONDS)

    def status(self) -> dict:
        found = {
            "id": self.id,
            "url": self.spec.url,
            "state": self.state,
            "since": _iso(self.since),
            "video": self.spec.video.model_dump(),
            "idle": self.spec.idle,
            "encoder": self.rendering.encoder if self.rendering is not None else None,
            "readers": {**{kind: len(channel.readers) for kind, channel in self.channels.items()}, "hls": self.hls_readers()},
            "bytes": self.made,
        }
        if self.rendering is not None and self.ending is None:
            found["memory_mb"] = round(self.rendering.memory_mb())
        if self.ending is not None:
            found["reason"] = self.ending.reason
        return found

    async def start(self, launch: Launch) -> None:
        self.rendering = await launch(self.spec, self.id)
        for kind in FORMATS:
            self.tasks.append(asyncio.ensure_future(self._read(kind)))
        self.tasks.append(asyncio.ensure_future(self._watch()))
        try:
            await asyncio.wait_for(self._first(), START_SECONDS)
        except asyncio.TimeoutError:
            await self.end(f"no picture within {START_SECONDS:g} s")
        if self.ending is not None:
            tail = self.rendering.errors.decode(errors="replace").strip().splitlines()
            raise EmsError(TOOL_FAILED, f"the page's stream did not start: {self.ending.reason}" + (f" ({tail[-1][-200:]})" if tail else ""))

    async def _first(self) -> None:
        playable = asyncio.ensure_future(self.playable.wait())
        ended = asyncio.ensure_future(self.ended.wait())
        await asyncio.wait([playable, ended], return_when=asyncio.FIRST_COMPLETED)
        playable.cancel()
        ended.cancel()

    async def _read(self, kind: str) -> None:
        assert self.rendering is not None
        output = self.rendering.outputs[kind]
        channel = self.channels[kind]
        try:
            while chunk := await output.read(1 << 16):
                channel.feed(chunk)
                if kind == "mp4":
                    self.made += len(chunk)
                if all(each.playable for each in self.channels.values()):
                    self.playable.set()
        except ValueError as exc:
            await self.end(f"the encoder's {kind} could not be read: {exc}")
            return
        await self.end(self.rendering.exited() or "the encoder stopped")

    async def _watch(self) -> None:
        assert self.rendering is not None
        while self.ending is None:
            await asyncio.sleep(WATCH_SECONDS)
            exited = self.rendering.exited()
            if exited:
                await self.end(exited)
                return
            used = self.rendering.memory_mb()
            if used > self.memory_mb:
                await self.end(f"the browser held {used:.0f} MB, over EMS_PAGE_MEMORY_MB ({self.memory_mb})")
                return
            if self.readers():
                self.unread_since = time.monotonic()
            elif self.playable.is_set() and time.monotonic() - self.unread_since > self.spec.idle:
                await self.end("idle")
                return

    def join(self, kind: str) -> tuple[Channel, Reader]:
        if self.ending is not None:
            raise EmsError(NOT_FOUND, f"page {self.id} has ended: {self.ending.reason}")
        channel = self.channels[kind]
        if not channel.playable:
            raise EmsError(BUSY, f"page {self.id} has no picture yet", retry_after=1)
        return channel, channel.join()

    async def hls_file(self, name: str) -> tuple[bytes, str]:
        """The file `name` of the HLS form, and its media type. The playlist is waited for while
        ffmpeg has still to write it; a segment it has deleted is `not_found`."""
        if not HLS_NAME.match(name):
            raise EmsError(NOT_FOUND, f"page {self.id} has no {name}")
        if self.ending is not None:
            raise EmsError(NOT_FOUND, f"page {self.id} has ended: {self.ending.reason}")
        if self.rendering is None or self.rendering.hls is None:
            raise EmsError(BUSY, f"page {self.id} has no picture yet", retry_after=1)
        self.hls_read = self.unread_since = time.monotonic()
        path = os.path.join(self.rendering.hls, name)
        playlist = name == HLS_PLAYLIST
        waited = 0.0
        while playlist and not os.path.exists(path) and waited < HLS_START_SECONDS and self.ending is None:
            await asyncio.sleep(0.1)
            waited += 0.1
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except FileNotFoundError:
            if playlist and self.ending is None:
                raise EmsError(BUSY, f"page {self.id} has no playlist yet", retry_after=1) from None
            raise EmsError(NOT_FOUND, f"page {self.id} has no {name}" + (f" (ended: {self.ending.reason})" if self.ending else "")) from None
        return data, "application/vnd.apple.mpegurl" if playlist else "video/mp2t"

    def leave(self, channel: Channel, reader: Reader) -> None:
        channel.leave(reader)
        reader.end()
        if not self.readers():
            self.unread_since = time.monotonic()

    async def end(self, reason: str) -> None:
        if self.ending is not None:
            return
        self.ending = Ending(reason, time.monotonic())
        self.ended.set()
        self.log(f"pages: {self.id} ended: {reason}")
        for channel in self.channels.values():
            channel.end()
        current = asyncio.current_task()
        for task in self.tasks:
            if task is not current:
                task.cancel()
        if self.rendering is not None:
            await self.rendering.stop()


class Pages:
    """Every page streamed by this EMS, by id; past `limit` running a new one is busy."""

    def __init__(self, limit: int, memory_mb: int, launch: Launch, log: Callable[[str], None]):
        self.limit = limit
        self.memory_mb = memory_mb
        self.launch = launch
        self.log = log
        self.pages: dict[str, Page] = {}

    def _forget_ended(self) -> None:
        now = time.monotonic()
        for page_id, page in list(self.pages.items()):
            if page.ending is not None and now - page.ending.at > ENDED_KEPT_SECONDS:
                del self.pages[page_id]

    def list(self) -> list[Page]:
        self._forget_ended()
        return list(self.pages.values())

    def get(self, page_id: str) -> Page:
        self._forget_ended()
        page = self.pages.get(page_id)
        if page is None:
            raise EmsError(NOT_FOUND, f"no page {page_id}")
        return page

    async def put(self, page_id: str, spec: PageSpec) -> tuple[Page, bool]:
        """The page `page_id` streaming `spec`: started, or started again when its spec changed or
        it ended. Answers it and whether it was started now."""
        if not PAGE_ID.match(page_id):
            raise EmsError(INVALID_REQUEST, "a page's id is 1 to 128 letters, digits, '.', '_' or '-'")
        self._forget_ended()
        known = self.pages.get(page_id)
        if known is not None and known.ending is None and known.spec == spec:
            return known, False
        running = sum(1 for page in self.pages.values() if page.ending is None and page.id != page_id)
        if running >= self.limit:
            raise EmsError(BUSY, f"this EMS streams at most {self.limit} pages (EMS_PAGES)", retry_after=5)
        if known is not None:
            await known.end("put again")
        page = Page(page_id, spec, self.memory_mb, self.log)
        self.pages[page_id] = page
        try:
            await page.start(self.launch)
        except BaseException as exc:
            # A page that ended while it started keeps why; one whose rendering never started, the error.
            await page.end(exc.message if isinstance(exc, EmsError) else "did not start")
            raise
        return page, True

    async def delete(self, page_id: str) -> None:
        page = self.get(page_id)
        await page.end("deleted")
        del self.pages[page_id]

    async def stop(self) -> None:
        for page in list(self.pages.values()):
            await page.end("EMS stopped")
        self.pages.clear()
