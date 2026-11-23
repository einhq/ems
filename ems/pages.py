"""Pages rendered into live video for screens (`/v0/pages`, docs/api.md 4.25).

A page is opened in Chromium on a virtual X display of its own (Xvfb), which ffmpeg grabs at a
steady frame rate and encodes once as H.264. The one encode is muxed twice, as fragmented MP4 (what
a Cast player takes when it is told the stream is live) and as MPEG-TS (what a DLNA screen takes),
and each is written as it comes into a live stream its caller grants under `EMS-Files`, which serves
it to its readers. EMS serves nothing of it, and a page runs until its caller deletes it or it fails.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import struct
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable

from pydantic import Field, field_validator

from .errors import BUSY, INVALID_REQUEST, NOT_FOUND, TOOL_FAILED, EmsError
from .page_proxy import PageProxy, origin_of
from .remote import Endpoint
from .schemas import Strict, StreamRef

PAGE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
FORMATS = ("mp4", "ts")
# A fragment at least this often, in microseconds, so a player is never more than this behind what
# was encoded.
FRAGMENT_US = 200_000
# How long a stream that ended stays listed, with why, before it is forgotten.
ENDED_KEPT_SECONDS = 60.0
# How long a new stream may take to give its first keyframe in both formats.
START_SECONDS = 30.0
# How often the processes and the memory are looked at.
WATCH_SECONDS = 1.0
TS_PACKET = 188
# The page's sound: what its browser plays, as AAC.
AUDIO_RATE = 48000
AUDIO_KBPS = 128
# How long the sound server may take to open its socket.
PULSE_START_SECONDS = 5.0
# How long a new page is waited for to load before it is answered as it is.
LOAD_SECONDS = 15.0


class PageVideo(Strict):
    width: int = Field(default=1920, ge=320, le=1920)
    height: int = Field(default=1080, ge=240, le=1080)
    fps: int = Field(default=30, ge=5, le=30)
    quality: int = Field(default=23, ge=0, le=51)


class PageKeys(Strict):
    keys: list[str] = Field(min_length=1, max_length=32)


class PageMessage(Strict):
    message: dict


class PageOutput(Strict):
    """The live streams the page is written into, one for each format, as the caller grants them."""

    mp4: StreamRef
    ts: StreamRef


class PageSpec(Strict):
    url: str
    video: PageVideo = Field(default_factory=PageVideo)
    output: PageOutput

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


@dataclass(frozen=True)
class Track:
    """A track of the MP4 header: its timescale, whether it is the picture, and its `trex`'s default
    sample duration."""
    timescale: int
    video: bool
    default_duration: int | None


def tracks(header: bytes) -> dict[int, Track]:
    """The header's tracks, by track id."""
    moov = next(((at, size) for kind, at, size in _boxes(header, 0, len(header)) if kind == b"moov"), None)
    if moov is None:
        return {}
    defaults: dict[int, int] = {}
    mvex = _child(header, *moov, b"mvex")
    if mvex is not None:
        for kind, at, _ in _boxes(header, mvex[0] + 8, mvex[0] + mvex[1]):
            if kind == b"trex":
                track_id, _, duration = struct.unpack(">III", header[at + 12:at + 24])
                defaults[track_id] = duration
    found: dict[int, Track] = {}
    for kind, at, size in _boxes(header, moov[0] + 8, moov[0] + moov[1]):
        if kind != b"trak":
            continue
        tkhd = _child(header, at, size, b"tkhd")
        mdia = _child(header, at, size, b"mdia")
        mdhd = _child(header, *mdia, b"mdhd") if mdia is not None else None
        hdlr = _child(header, *mdia, b"hdlr") if mdia is not None else None
        if tkhd is None or mdhd is None:
            continue
        track_id = struct.unpack(">I", header[tkhd[0] + (28 if header[tkhd[0] + 8] == 1 else 20):][:4])[0]
        scale = struct.unpack(">I", header[mdhd[0] + (28 if header[mdhd[0] + 8] == 1 else 20):][:4])[0]
        video = hdlr is not None and header[hdlr[0] + 16:hdlr[0] + 20] == b"vide"
        found[track_id] = Track(scale, video, defaults.get(track_id))
    return found


def _trafs(fragment: bytes) -> list[tuple[int, int, int]]:
    """Each `traf` of a fragment's `moof`: its track id, offset and size."""
    moof = next(((at, size) for kind, at, size in _boxes(fragment, 0, len(fragment)) if kind == b"moof"), None)
    if moof is None:
        return []
    found = []
    for kind, at, size in _boxes(fragment, moof[0] + 8, moof[0] + moof[1]):
        tfhd = _child(fragment, at, size, b"tfhd") if kind == b"traf" else None
        if tfhd is not None:
            found.append((struct.unpack(">I", fragment[tfhd[0] + 12:tfhd[0] + 16])[0], at, size))
    return found


def _tfdts(fragment: bytes) -> dict[int, tuple[int, int]]:
    """Where each track's decode time is in a fragment, and its width in bytes."""
    found = {}
    for track_id, at, size in _trafs(fragment):
        tfdt = _child(fragment, at, size, b"tfdt")
        if tfdt is not None:
            found[track_id] = (tfdt[0] + 12, 8 if fragment[tfdt[0] + 8] == 1 else 4)
    return found


def decode_times(fragment: bytes) -> dict[int, int]:
    """Each track's decode time in a fragment, in the track's timescale."""
    return {track_id: int.from_bytes(fragment[at:at + width], "big") for track_id, (at, width) in _tfdts(fragment).items()}


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
    """ffmpeg's MPEG-TS cut into pieces of whole packets, a piece starting at each packet of the
    picture that marks a keyframe (`random_access_indicator`; ffmpeg marks every packet of sound
    too). The header a reader starts with is the latest programme tables: the PAT and the PMT it
    names, which says which packets are the picture's."""

    def __init__(self):
        self.buffer = bytearray()
        self.pmt_pid: int | None = None
        self.pat: bytes | None = None
        self.pmt: bytes | None = None
        self.video_pid: int | None = None

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
                self.video_pid = _video_pid(packet, adaptation)
            if pid == self.video_pid and adaptation & 0x2 and packet[4] > 0 and packet[5] & 0x40:
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


# The PMT's stream types of a picture: MPEG-2, H.264, HEVC.
VIDEO_STREAM_TYPES = (0x02, 0x1B, 0x24)


def _video_pid(packet: bytes, adaptation: int) -> int | None:
    """The PID of the first picture the PMT in `packet` lists."""
    at = 4 + (1 + packet[4] if adaptation & 0x2 else 0)
    at += 1 + packet[at]  # the pointer field
    length = ((packet[at + 1] & 0x0F) << 8) | packet[at + 2]
    end = min(at + 3 + length - 4, TS_PACKET)
    entry = at + 12 + (((packet[at + 10] & 0x0F) << 8) | packet[at + 11])
    while entry + 5 <= end:
        if packet[entry] in VIDEO_STREAM_TYPES:
            return ((packet[entry + 1] & 0x1F) << 8) | packet[entry + 2]
        entry += 5 + (((packet[entry + 3] & 0x0F) << 8) | packet[entry + 4])
    return None


# ── Outputs ────────────────────────────────────────────────────────────────────────────────────


class Output:
    """One format of the page's stream as it is written: cut where a reader may start, so that its
    latest keyframe, and how many came, are known."""

    def __init__(self, splitter: Mp4Splitter | TsSplitter):
        self.splitter = splitter
        # The fragment, or the TS's run of packets, that starts at the latest keyframe.
        self.key: bytes | None = None
        # Keyframes seen, so one that came after a moment is told apart.
        self.keys = 0

    @property
    def playable(self) -> bool:
        return self.splitter.header is not None and self.key is not None

    def feed(self, data: bytes) -> None:
        for key, piece in self.splitter.feed(data):
            if key:
                self.keys += 1
                self.key = piece


# ── Keys ───────────────────────────────────────────────────────────────────────────────────────

# The keys a page may be sent, by name: the DOM's `key`, `code` and the virtual key code Chromium
# reports as `keyCode`, and the text a key types. Letters and digits are their own names.
NAMED_KEYS: dict[str, tuple[str, str, int, str]] = {
    "ArrowLeft": ("ArrowLeft", "ArrowLeft", 37, ""),
    "ArrowUp": ("ArrowUp", "ArrowUp", 38, ""),
    "ArrowRight": ("ArrowRight", "ArrowRight", 39, ""),
    "ArrowDown": ("ArrowDown", "ArrowDown", 40, ""),
    "Space": (" ", "Space", 32, " "),
    "Enter": ("Enter", "Enter", 13, "\r"),
    "Escape": ("Escape", "Escape", 27, ""),
    "Tab": ("Tab", "Tab", 9, ""),
    "Backspace": ("Backspace", "Backspace", 8, ""),
    "Home": ("Home", "Home", 36, ""),
    "End": ("End", "End", 35, ""),
    "PageUp": ("PageUp", "PageUp", 33, ""),
    "PageDown": ("PageDown", "PageDown", 34, ""),
}
# How long the browser may take to answer a DevTools call.
DEVTOOLS_SECONDS = 5.0


def key_event(name: str) -> tuple[str, str, int, str]:
    """The `key`, `code`, key code and text of the key `name`; `invalid_request` for any other."""
    if name in NAMED_KEYS:
        return NAMED_KEYS[name]
    if len(name) == 1 and name.isascii() and name.isalnum():
        upper = name.upper()
        return name, f"Key{upper}" if name.isalpha() else f"Digit{name}", ord(upper), name
    raise EmsError(INVALID_REQUEST, f"no key {name!r}: a letter, a digit or one of {', '.join(NAMED_KEYS)}")


class Devtools:
    """Chromium's DevTools protocol over `--remote-debugging-pipe`: JSON messages, each ended by a
    NUL, the browser reading ours on its fd 3 and writing its own on its fd 4. Commands go to the
    page through a session attached to it once."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.WriteTransport):
        self.reader = reader
        self.writer = writer
        self.pending: dict[int, asyncio.Future] = {}
        self.next_id = 0
        self.session: str | None = None
        self.lock = asyncio.Lock()
        self.task = asyncio.ensure_future(self._read())

    async def _read(self) -> None:
        try:
            while True:
                message = json.loads((await self.reader.readuntil(b"\0"))[:-1])
                waiting = self.pending.pop(message.get("id"), None) if "id" in message else None
                if waiting is not None and not waiting.done():
                    waiting.set_result(message)
        except (asyncio.IncompleteReadError, ValueError):
            pass
        finally:
            for waiting in self.pending.values():
                if not waiting.done():
                    waiting.set_exception(EmsError(TOOL_FAILED, "the page's browser closed its DevTools pipe"))
            self.pending.clear()

    async def call(self, method: str, params: dict | None = None, session: str | None = None) -> dict:
        if self.task.done():
            raise EmsError(TOOL_FAILED, "the page's browser closed its DevTools pipe")
        self.next_id += 1
        message: dict = {"id": self.next_id, "method": method, "params": params or {}}
        if session is not None:
            message["sessionId"] = session
        waiting = asyncio.get_running_loop().create_future()
        self.pending[self.next_id] = waiting
        self.writer.write(json.dumps(message).encode() + b"\0")
        try:
            answer = await asyncio.wait_for(waiting, DEVTOOLS_SECONDS)
        except asyncio.TimeoutError:
            self.pending.pop(message["id"], None)
            raise EmsError(TOOL_FAILED, f"the page's browser did not answer {method} within {DEVTOOLS_SECONDS:g} s") from None
        if "error" in answer:
            raise EmsError(TOOL_FAILED, f"the page's browser refused {method}: {answer['error'].get('message', answer['error'])}")
        return answer.get("result", {})

    async def _page(self) -> str:
        if self.session is None:
            targets = (await self.call("Target.getTargets"))["targetInfos"]
            page = next((target for target in targets if target["type"] == "page"), None)
            if page is None:
                raise EmsError(TOOL_FAILED, "the page's browser shows no page")
            self.session = (await self.call("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}))["sessionId"]
        return self.session

    async def loaded(self) -> None:
        """Wait until the page the browser opened has loaded and painted: a document other than the
        blank one it starts on, whose `readyState` is complete, and two frames of it drawn."""
        async with self.lock:
            while True:
                try:
                    found = await self.call("Runtime.evaluate", {"expression": "[document.readyState, location.href]", "returnByValue": True},
                                            await self._page())
                    state, href = found["result"]["value"]
                    if state == "complete" and href != "about:blank":
                        await self.call("Runtime.evaluate", {
                            "expression": "new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(() => done(true))))",
                            "awaitPromise": True,
                        }, await self._page())
                        return
                except (EmsError, KeyError, TypeError, ValueError):
                    # Between documents, as a redirect or a navigation replaces one.
                    if self.task.done():
                        raise
                await asyncio.sleep(0.1)

    async def screen(self, call: str, argument: object = None) -> object:
        """What the page's `globalThis.einScreen[call](argument)` answers, as JSON: how a page says
        what it shows (`state`) and takes what it is told (`message`). `None` when the page has no
        such function."""
        expression = (f"(() => {{ const screen = globalThis.einScreen; return screen && typeof screen.{call} === 'function' "
                      f"? screen.{call}({json.dumps(argument)}) ?? null : null; }})()")
        async with self.lock:
            for attempt in (0, 1):
                try:
                    found = await self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True},
                                            await self._page())
                    if "exceptionDetails" in found:
                        raise EmsError(TOOL_FAILED, f"the page's einScreen.{call} failed: {found['exceptionDetails'].get('text', '')}")
                    return found.get("result", {}).get("value")
                except EmsError:
                    self.session = None
                    if attempt:
                        raise
        return None

    async def press(self, names: list[str]) -> None:
        """Press and release each key of `names` in the page, in turn, as someone at it would."""
        events = [key_event(name) for name in names]
        async with self.lock:
            for attempt in (0, 1):
                try:
                    session = await self._page()
                    for key, code, number, text in events:
                        typed = {"key": key, "code": code, "windowsVirtualKeyCode": number}
                        await self.call("Input.dispatchKeyEvent", {"type": "keyDown", **typed, **({"text": text} if text else {})}, session)
                        await self.call("Input.dispatchKeyEvent", {"type": "keyUp", **typed}, session)
                    return
                except EmsError:
                    # The page's session ends with its target, which a crashed renderer replaces.
                    self.session = None
                    if attempt:
                        raise

    def close(self) -> None:
        self.task.cancel()
        self.writer.close()


# ── Rendering ──────────────────────────────────────────────────────────────────────────────────


class Rendering:
    """What draws and encodes a page: its processes, the encoder's two outputs, the profile, the
    browser's DevTools, which keys reach the page by, and the proxy it reaches the network through."""

    def __init__(self, processes: dict[str, asyncio.subprocess.Process], outputs: dict[str, asyncio.StreamReader], encoder: str,
                 folder: str | None = None, devtools: Devtools | None = None, proxy: PageProxy | None = None):
        self.processes = processes
        self.outputs = outputs
        self.encoder = encoder
        self.folder = folder
        self.devtools = devtools
        self.proxy = proxy
        self.errors = bytearray()
        # When the grab started, which the stream's clock counts from, at the latest.
        self.grabbed_from = time.monotonic()

    async def loaded(self) -> bool:
        """Wait until the page has loaded; false at once where the browser has no DevTools to say."""
        if self.devtools is None:
            return False
        await self.devtools.loaded()
        return True

    async def press(self, names: list[str]) -> None:
        if self.devtools is None:
            raise EmsError(TOOL_FAILED, "this page's browser takes no keys")
        await self.devtools.press(names)

    async def screen(self, call: str, argument: object = None) -> object:
        if self.devtools is None:
            return None
        return await self.devtools.screen(call, argument)

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
        if self.devtools is not None:
            self.devtools.close()
        for process in reversed(list(self.processes.values())):
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for process in self.processes.values():
            await process.wait()
        if self.proxy is not None:
            self.proxy.close()
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


def chromium_args(spec: PageSpec, profile: str, memory_mb: int, proxy: int, sandbox: bool) -> list[str]:
    """Chromium for the page, every connection of it through the SOCKS5 proxy on 127.0.0.1's port
    `proxy`, which resolves its names: loopback is not exempt, and WebRTC sends nothing past it.
    Without `sandbox` (`sandbox_possible`), Chromium runs without its sandbox."""
    v = spec.video
    args = [
        "--kiosk", "--no-first-run", "--no-default-browser-check", f"--user-data-dir={profile}",
        "--window-position=0,0", f"--window-size={v.width},{v.height}", "--force-device-scale-factor=1",
        "--noerrdialogs", "--disable-infobars", "--disable-session-crashed-bubble", "--disable-extensions",
        "--disable-component-update", "--disable-features=Translate,MediaRouter", "--password-store=basic",
        "--autoplay-policy=no-user-gesture-required", "--hide-scrollbars", "--disable-dev-shm-usage", "--remote-debugging-pipe",
        # The window is dark rather than white until the page paints, as a screen shows it from
        # the start; a page that follows the scheme is drawn dark.
        "--force-dark-mode",
        # The page's heap: half the cap, the rest is the browser's own.
        f"--js-flags=--max-old-space-size={max(128, memory_mb // 2)}",
        f"--proxy-server=socks5://127.0.0.1:{proxy}", "--proxy-bypass-list=<-loopback>",
        "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    ]
    if not sandbox:
        args.append("--no-sandbox")
    return [*args, spec.url]


async def sandbox_possible() -> str | None:
    """Why Chromium's sandbox cannot start here, or None when it can. The sandbox puts each of its
    processes in user, PID and network namespaces of their own and chroots it there, which a host
    can forbid: Ubuntu's kernel.apparmor_restrict_unprivileged_userns, or a container without
    SYS_CHROOT in its bounding set. It does not run as root at all."""
    if os.geteuid() == 0:
        return "EMS runs as root"
    probe = await asyncio.create_subprocess_exec(
        "unshare", "--user", "--map-root-user", "--pid", "--net", "--fork", "chroot", "/", "true",
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, error = await probe.communicate()
    return None if probe.returncode == 0 else error.decode(errors="replace").strip() or f"unshare exited {probe.returncode}"


def _display_told(fd: int) -> bytes:
    """What Xvfb writes to its `-displayfd`, read up to the newline it ends with. The number and the
    newline are two writes, and an Xvfb whose newline finds the pipe closed ends itself
    (ein/core#1422)."""
    told = b""
    while not told.endswith(b"\n"):
        piece = os.read(fd, 16)
        if not piece:
            break
        told += piece
    return told


def encoder_args(spec: PageSpec, encoder: str, ts_fd: int) -> list[str]:
    """ffmpeg's output side: the encode of the sound (input 0) and the picture (input 1), the
    fragmented MP4 to stdout and the MPEG-TS to `ts_fd`. The picture is the first track, which the
    cuts where a reader may start go by."""
    v = spec.video
    if encoder == "h264_nvenc":
        codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll", "-rc", "vbr", "-cq", str(v.quality), "-maxrate", "8M",
                 "-bufsize", "16M", "-profile:v", "high", "-level:v", "4.1", "-forced-idr", "1", "-bf", "0"]
    else:
        codec = ["-c:v", encoder, "-b:v", "5M", "-maxrate", "8M"]
    # One encode, muxed twice. Each format's header is written once, so the MPEG-TS repeats the
    # picture's parameters at every keyframe for a reader that starts there.
    tee = (f"[f=mp4:movflags=frag_keyframe+empty_moov+default_base_moof:frag_duration={FRAGMENT_US}]pipe\\:1"
           f"|[f=mpegts:bsfs/v=dump_extra=freq=keyframe]pipe\\:{ts_fd}")
    sound = ["-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k", "-ar", str(AUDIO_RATE), "-ac", "2"]
    return [*codec, *sound, "-g", str(v.fps), "-flags", "+global_header", "-map", "1:v", "-map", "0:a", "-f", "tee", tee]


def renderer(memory_mb: int, encoders: Callable[[], Awaitable[set[str]]], scratch: Callable[[], str],
             log: Callable[[str], None]) -> Launch:
    """How a page is rendered here: Xvfb, a PulseAudio server whose one sink takes nothing but the
    page's sound, Chromium on the display playing into that sink with its DevTools on a pipe and the
    network through a proxy of its own (`PageProxy`: the page's origin and the internet, nothing
    else, each refusal logged once), and ffmpeg grabbing the display and the sink with the first
    encoder of NVENC and the CPU's that writes. Chromium runs in its sandbox where the host lets it
    start (`sandbox_possible`, asked once), and the log says it once where not."""
    sandbox: list[bool] = []

    async def launch(spec: PageSpec, page_id: str) -> Rendering:
        for tool in ("Xvfb", "chromium", "ffmpeg", "pulseaudio", "bash"):
            if shutil.which(tool) is None:
                raise EmsError(TOOL_FAILED, f"this EMS has no {tool}, which renders pages", permanent=True)
        available = await encoders()
        order = [name for name in ("h264_nvenc", "libopenh264") if name in available]
        if not order:
            raise EmsError(TOOL_FAILED, "this EMS's ffmpeg has no H.264 encoder", permanent=True)
        if not sandbox:
            why = await sandbox_possible()
            if why is not None:
                log(f"pages: Chromium's sandbox cannot start here ({why}): pages run without it")
            sandbox.append(why is None)
        folder = scratch()
        v = spec.video
        processes: dict[str, asyncio.subprocess.Process] = {}
        devtools: Devtools | None = None
        proxy = PageProxy(origin_of(spec.url), lambda where: log(f"pages: {page_id}: its browser may not reach {where}, which is not on the internet"))
        try:
            proxy_port = await proxy.start()
            read, write = os.pipe()
            processes["xvfb"] = await asyncio.create_subprocess_exec(
                "Xvfb", "-displayfd", str(write), "-screen", "0", f"{v.width}x{v.height}x24", "-nolisten", "tcp",
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                pass_fds=(write,), start_new_session=True)
            os.close(write)
            told = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(None, _display_told, read), 10)
            os.close(read)
            if not told.strip():
                raise EmsError(TOOL_FAILED, "Xvfb gave no display")
            display = f":{int(told.strip())}"
            env = {**os.environ, "DISPLAY": display, "HOME": folder, "XDG_CONFIG_HOME": folder, "XDG_CACHE_HOME": folder,
                   "PULSE_RUNTIME_PATH": os.path.join(folder, "pulse")}
            socket = os.path.join(folder, "pulse", "native")
            os.makedirs(os.path.dirname(socket))
            processes["pulseaudio"] = await asyncio.create_subprocess_exec(
                "pulseaudio", "--daemonize=no", "--exit-idle-time=-1", "-n", "--use-pid-file=no", "--disable-shm=yes",
                "-L", f"module-null-sink sink_name=page rate={AUDIO_RATE}", "-L", f"module-native-protocol-unix socket={socket} auth-anonymous=1",
                env=env, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True)
            waited = 0.0
            while not os.path.exists(socket):
                if processes["pulseaudio"].returncode is not None or waited >= PULSE_START_SECONDS:
                    raise EmsError(TOOL_FAILED, "PulseAudio did not start for the page's sound")
                await asyncio.sleep(0.05)
                waited += 0.05
            env = {**env, "PULSE_SERVER": f"unix:{socket}", "PULSE_SINK": "page"}
            # DevTools on the pipe Chromium reads at fd 3 and writes at fd 4, where bash puts ours.
            ours_in, its_out = os.pipe()
            its_in, ours_out = os.pipe()
            processes["chromium"] = await asyncio.create_subprocess_exec(
                "bash", "-c", f'exec "$0" "$@" 3<&{its_in} 4>&{its_out} {its_in}<&- {its_out}>&-',
                "chromium", *chromium_args(spec, os.path.join(folder, "profile"), memory_mb, proxy_port, sandbox[0]), env=env, pass_fds=(its_in, its_out),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
            os.close(its_in)
            os.close(its_out)
            loop = asyncio.get_running_loop()
            writer, _ = await loop.connect_write_pipe(asyncio.Protocol, os.fdopen(ours_out, "wb", buffering=0))
            devtools = Devtools(await _pipe_reader(ours_in), writer)
            failures: list[str] = []
            for encoder in order:
                # The sound is opened first: PulseAudio takes about two seconds to start recording,
                # and a display grab opened before it starts that much ahead of it, which ffmpeg
                # then holds back to the sound and so grabs a frame or two a second (ein/core#1428).
                grab = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
                        "-thread_queue_size", "1024", "-f", "pulse", "-server", f"unix:{socket}", "-i", "page.monitor",
                        "-f", "x11grab", "-framerate", str(v.fps), "-video_size", f"{v.width}x{v.height}", "-draw_mouse", "0", "-i", display,
                        "-vf", "format=yuv420p"]
                ts_read, ts_write = os.pipe()
                grabbed_from = time.monotonic()
                process = await asyncio.create_subprocess_exec(
                    *grab, *encoder_args(spec, encoder, ts_write), stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, pass_fds=(ts_write,), start_new_session=True)
                os.close(ts_write)
                ts = await _pipe_reader(ts_read)
                assert process.stdout is not None and process.stderr is not None
                first = await process.stdout.read(1 << 16)
                if first:
                    processes["ffmpeg"] = process
                    mp4 = _prepended(process.stdout, first)
                    rendering = Rendering(processes, {"mp4": mp4, "ts": ts}, encoder, folder, devtools, proxy)
                    rendering.grabbed_from = grabbed_from
                    asyncio.ensure_future(_drain(process.stderr, rendering.errors))
                    return rendering
                errors = (await process.stderr.read()).decode(errors="replace").strip().splitlines()
                await process.wait()
                failures.append(f"{encoder}: {(errors[-1] if errors else f'exited {process.returncode}')[-200:]}")
            raise EmsError(TOOL_FAILED, "no encoder could start the page's stream: " + " | ".join(failures))
        except BaseException:
            await Rendering(processes, {}, "", folder, devtools=devtools, proxy=proxy).stop()
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


Opener = Callable[[Endpoint, str], Awaitable[int]]


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


class Page:
    """One page streamed: its rendering, and its formats as they are written into the caller's
    streams."""

    def __init__(self, page_id: str, spec: PageSpec, endpoint: Endpoint, memory_mb: int, log: Callable[[str], None]):
        self.id = page_id
        self.spec = spec
        self.endpoint = endpoint
        self.memory_mb = memory_mb
        self.log = log
        self.since = time.time()
        self.outputs = {"mp4": Output(Mp4Splitter()), "ts": Output(TsSplitter())}
        # The descriptor each format is written to, the caller's name opened to write.
        self.written: dict[str, int] = {}
        self.rendering: Rendering | None = None
        self.ending: Ending | None = None
        self.made = 0
        self.playable = asyncio.Event()
        self.tasks: list[asyncio.Task] = []
        self.ended = asyncio.Event()

    @property
    def state(self) -> str:
        if self.ending is not None:
            return "ended"
        return "running" if self.playable.is_set() else "starting"

    def status(self) -> dict:
        found = {
            "id": self.id,
            "url": self.spec.url,
            "state": self.state,
            "since": _iso(self.since),
            "video": self.spec.video.model_dump(),
            "encoder": self.rendering.encoder if self.rendering is not None else None,
            "bytes": self.made,
        }
        if self.rendering is not None and self.ending is None:
            found["memory_mb"] = round(self.rendering.memory_mb())
        if self.ending is not None:
            found["reason"] = self.ending.reason
        return found

    async def start(self, launch: Launch, open_name: Opener) -> None:
        for kind in FORMATS:
            self.written[kind] = await open_name(self.endpoint, getattr(self.spec.output, kind).path)
        self.rendering = await launch(self.spec, self.id)
        for kind in FORMATS:
            self.tasks.append(asyncio.ensure_future(self._read(kind)))
        self.tasks.append(asyncio.ensure_future(self._watch()))
        try:
            await asyncio.wait_for(self._first(), START_SECONDS)
        except asyncio.TimeoutError:
            await self.end(f"no picture within {START_SECONDS:g} s")
        if self.ending is None:
            try:
                await asyncio.wait_for(self._shown(), LOAD_SECONDS)
            except (asyncio.TimeoutError, EmsError):
                pass  # a page slow to load, or whose browser says nothing, is streamed as it is
        if self.ending is not None:
            tail = self.rendering.errors.decode(errors="replace").strip().splitlines()
            raise EmsError(TOOL_FAILED, f"the page's stream did not start: {self.ending.reason}" + (f" ({tail[-1][-200:]})" if tail else ""))

    async def _shown(self) -> None:
        """Wait until the page has painted and the latest keyframe was grabbed after that: a reader
        starts at the latest keyframe, so it starts at the page rather than at the browser opening
        it. The encoder runs behind the grab, so a keyframe that arrives after the paint may have
        been grabbed before it; its time in the stream, which counts from the grab's start, says."""
        assert self.rendering is not None
        if not await self.rendering.loaded():
            return
        painted = time.monotonic() - self.rendering.grabbed_from
        ts = self.outputs["ts"]
        seen = ts.keys
        while self.ending is None and not (self._keyed_after(painted) and ts.keys > seen):
            await asyncio.sleep(0.05)

    def _keyed_after(self, seconds: float) -> bool:
        """Whether the MP4's latest keyframe is `seconds` or later into the stream."""
        output = self.outputs["mp4"]
        header = output.splitter.header
        if header is None or output.key is None:
            return False
        video = next((track_id for track_id, track in tracks(header).items() if track.video), None)
        times = decode_times(output.key)
        return video is not None and video in times and times[video] >= seconds * tracks(header)[video].timescale

    async def _first(self) -> None:
        playable = asyncio.ensure_future(self.playable.wait())
        ended = asyncio.ensure_future(self.ended.wait())
        await asyncio.wait([playable, ended], return_when=asyncio.FIRST_COMPLETED)
        playable.cancel()
        ended.cancel()

    async def _read(self, kind: str) -> None:
        """The encoder's `kind` as it comes, written into the caller's stream in the same order."""
        assert self.rendering is not None
        source = self.rendering.outputs[kind]
        output = self.outputs[kind]
        try:
            while chunk := await source.read(1 << 16):
                output.feed(chunk)
                await asyncio.to_thread(_write_all, self.written[kind], chunk)
                if kind == "mp4":
                    self.made += len(chunk)
                if all(each.playable for each in self.outputs.values()):
                    self.playable.set()
        except ValueError as exc:
            await self.end(f"the encoder's {kind} could not be read: {exc}")
            return
        except OSError as exc:
            await self.end(f"the {kind} stream could not be written: {exc.strerror or exc}")
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

    def _running(self) -> Rendering:
        if self.ending is not None:
            raise EmsError(NOT_FOUND, f"page {self.id} has ended: {self.ending.reason}")
        if self.rendering is None:
            raise EmsError(BUSY, f"page {self.id} has no browser yet", retry_after=1)
        return self.rendering

    async def shown(self) -> object:
        """What the page says it shows (`einScreen.state()`), or `None` when it says nothing."""
        return await self._running().screen("state")

    async def tell(self, message: dict) -> bool:
        """Hand the page `message` (`einScreen.message(message)`): whether it has one to take it."""
        return await self._running().screen("message", message) is not None

    async def press(self, names: list[str]) -> None:
        """Press the keys `names` in the page, in turn."""
        if self.ending is not None:
            raise EmsError(NOT_FOUND, f"page {self.id} has ended: {self.ending.reason}")
        if self.rendering is None:
            raise EmsError(BUSY, f"page {self.id} has no browser yet", retry_after=1)
        await self.rendering.press(names)

    async def end(self, reason: str) -> None:
        if self.ending is not None:
            return
        self.ending = Ending(reason, time.monotonic())
        self.ended.set()
        self.log(f"pages: {self.id} ended: {reason}")
        current = asyncio.current_task()
        for task in self.tasks:
            if task is not current:
                task.cancel()
        if self.rendering is not None:
            await self.rendering.stop()
        # Closed, the caller's streams end for their readers.
        for fd in self.written.values():
            with suppress(OSError):
                os.close(fd)
        self.written.clear()


class Pages:
    """Every page streamed by this EMS, by id; past `limit` running a new one is busy. `open_name`
    opens a caller's name to write a format into."""

    def __init__(self, limit: int, memory_mb: int, launch: Launch, open_name: Opener, log: Callable[[str], None]):
        self.limit = limit
        self.memory_mb = memory_mb
        self.launch = launch
        self.open_name = open_name
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

    async def put(self, page_id: str, spec: PageSpec, endpoint: Endpoint) -> tuple[Page, bool]:
        """The page `page_id` streaming `spec` into the names `endpoint` grants: started, or started
        again when its spec or endpoint changed or it ended. Answers it and whether it was started
        now."""
        if not PAGE_ID.match(page_id):
            raise EmsError(INVALID_REQUEST, "a page's id is 1 to 128 letters, digits, '.', '_' or '-'")
        self._forget_ended()
        known = self.pages.get(page_id)
        if known is not None and known.ending is None and known.spec == spec and known.endpoint == endpoint:
            return known, False
        running = sum(1 for page in self.pages.values() if page.ending is None and page.id != page_id)
        if running >= self.limit:
            raise EmsError(BUSY, f"this EMS streams at most {self.limit} pages (EMS_PAGES)", retry_after=5)
        if known is not None:
            await known.end("put again")
        page = Page(page_id, spec, endpoint, self.memory_mb, self.log)
        self.pages[page_id] = page
        try:
            await page.start(self.launch, self.open_name)
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
