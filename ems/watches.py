"""Watches: a live stream EMS reads for as long as its caller wants, and what it sees there told as
events, under an id the caller chose (_4.24_).

A watch reads its source with ffmpeg, a few frames a second in the display frame, each frame a PPM
on ffmpeg's output. A frame is looked at for faces while it differs from the one before (`motion`)
or a face is being followed; the model worker finds the faces (`detect_faces`), and they are
followed from frame to frame by `face_tracks`' rules: a face continues its track by overlapping
box and likeness, or by likeness alone, and a track ends when its face has been gone for
`face_tracks.max_gap`. A track is told once it has been seen twice (`face`), and again when it
ends (`face.end`), with its best detection, the mean of its best vectors and a JPEG of the face.

Nothing of a watch is kept: it lives in memory, and a caller states its watches again when EMS
starts. A source that drops is read again after a pause that doubles up to a minute, and the
watch says so (`source.lost`, `source.back`).
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field

from .errors import BUSY, INVALID_REQUEST, NOT_FOUND, EmsError
from .schemas import Box, Strict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelworker"))
import face_tracks  # noqa: E402

# The longest side a frame is looked at in, as `faces.detect_video` reads a video's.
MAX_EDGE = 1024
# Detections of one track before it is told: one sighting is often a detector's mistake.
TOLD_AFTER = 2
# The motion check: a grid of this many cells across and down, a cell changed when its grey moved
# by more than CELL_CHANGE of 255, and motion while more than MOVING of the cells did.
GRID = (32, 18)
CELL_CHANGE = 12
MOVING = 0.01
# Seconds without a changed frame before motion ends.
MOTION_ENDS_AFTER = 3.0
# A face's crop takes this much of its box's size again around it, on each side.
CROP_MARGIN = 0.3
MIN_BACKOFF, MAX_BACKOFF = 1.0, 60.0
# A subscriber that falls this many events behind is let go, not waited for.
SUBSCRIBER_QUEUE = 256
WATCH_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class FacesSpec(Strict):
    # Frames looked at a second.
    fps: float = Field(default=3.0, ge=0.2, le=10)
    # The smallest face told, as a fraction of the frame's shorter side.
    min_size: float = Field(default=0.03, ge=0, le=1)
    min_confidence: float = Field(default=0.7, ge=0, le=1)


class WatchSpec(Strict):
    """What a watch reads and how: `PUT /v0/watches/<id>`'s body."""

    source: str = Field(min_length=1)
    faces: FacesSpec = Field(default_factory=FacesSpec)
    # Look for faces only in frames that changed, and tell when there is motion.
    motion: bool = True
    # Regions left out: no motion is counted there, no face told whose centre is there.
    mask: list[Box] = Field(default_factory=list)
    angle: Literal[0, 90, 180, 270] = 0
    mirror: bool = False
    # A JPEG of each face told, in its events.
    crops: bool = True


@dataclass
class Frame:
    """One frame of a source, in the display frame: RGB, three bytes a pixel, rows top to bottom."""

    width: int
    height: int
    rgb: bytes
    # The PPM the worker reads: its header and `rgb`.
    ppm: bytes


# What reads a source: given the spec, yields its frames until the source ends or fails, when it
# raises with why. `ffmpeg_frames` in the service, a list of frames in tests.
FrameSource = Callable[[WatchSpec], AsyncIterator[Frame]]
# What makes a JPEG of a part of a frame: `(frame, x, y, width, height)` in pixels.
CropEncoder = Callable[[Frame, int, int, int, int], Awaitable[bytes]]
# What finds the faces in a frame, as the worker's `detect_faces` answers: pixel boxes.
FaceFinder = Callable[[str, Frame], Awaitable[dict]]


def masked_source(source: str) -> str:
    """The source as a status shows it: without a password."""
    try:
        parts = urlsplit(source)
    except ValueError:
        return source
    if not parts.password:
        return source
    host = parts.hostname or ""
    netloc = f"{parts.username}:***@{host}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def ppm_size(header: bytes) -> tuple[int, int] | None:
    """A whole binary PPM header's width and height, as ffmpeg writes it (`P6 640 360 255` and one
    whitespace); None while `header` is not yet all of one."""
    fields = header.split()
    if len(fields) < 4 or not header[-1:].isspace():
        return None
    if fields[0] != b"P6" or fields[3] != b"255":
        raise ValueError("not an 8-bit binary PPM")
    return int(fields[1]), int(fields[2])


def grey_grid(frame: Frame, mask: list[Box]) -> list[int | None]:
    """The frame's grey on GRID's cells, a sample at each cell's centre; None in a masked cell."""
    cols, rows = GRID
    out: list[int | None] = []
    for row in range(rows):
        fy = (row + 0.5) / rows
        y = min(frame.height - 1, int(fy * frame.height))
        for col in range(cols):
            fx = (col + 0.5) / cols
            if any(r.x <= fx <= r.x + r.width and r.y <= fy <= r.y + r.height for r in mask):
                out.append(None)
                continue
            x = min(frame.width - 1, int(fx * frame.width))
            at = (y * frame.width + x) * 3
            out.append((frame.rgb[at] + frame.rgb[at + 1] + frame.rgb[at + 2]) // 3)
    return out


def changed(before: list[int | None] | None, after: list[int | None]) -> bool:
    """Whether more than MOVING of the cells changed between two grids."""
    if before is None:
        return False
    cells = [(a, b) for a, b in zip(before, after) if a is not None and b is not None]
    if not cells:
        return False
    moved = sum(1 for a, b in cells if abs(a - b) > CELL_CHANGE)
    return moved / len(cells) > MOVING


def let_go(queue: asyncio.Queue) -> None:
    """Ends a subscriber: what it has not read is dropped, and None tells it the end."""
    while not queue.empty():
        queue.get_nowait()
    queue.put_nowait(None)


@dataclass
class Track:
    id: int
    detections: list[dict] = field(default_factory=list)
    told: bool = False
    # The frame of its best detection, for its crop.
    best: dict | None = None
    best_frame: Frame | None = None

    @property
    def first(self) -> float:
        return self.detections[0]["t"]

    @property
    def last(self) -> float:
        return self.detections[-1]["t"]


def follow(tracks: list[Track], found: list[dict]) -> list[tuple[Track | None, dict]]:
    """Each detection of a frame with the open track it continues, or None for a new face, by
    `face_tracks`' rules: likeness at least its floor, and an overlapping box or likeness enough
    alone. Best pairs first; a track takes one detection a frame."""
    pairs = []
    for t_index, track in enumerate(tracks):
        last = track.detections[-1]
        for d_index, detection in enumerate(found):
            similarity = face_tracks.dot(last["unit"], detection["unit"])
            overlap = face_tracks.iou(last["box"], detection["box"])
            if similarity < face_tracks.TRACK_FLOOR_SIMILARITY:
                continue
            if overlap >= face_tracks.TRACK_MIN_IOU or similarity >= face_tracks.TRACK_MIN_SIMILARITY:
                pairs.append((similarity + overlap, t_index, d_index))
    taken_tracks: set[int] = set()
    taken: dict[int, Track] = {}
    for _, t_index, d_index in sorted(pairs, reverse=True):
        if t_index in taken_tracks or d_index in taken:
            continue
        taken_tracks.add(t_index)
        taken[d_index] = tracks[t_index]
    return [(taken.get(i), detection) for i, detection in enumerate(found)]


class Watch:
    """One watch: its spec, its reading loop, what it has seen, and who listens."""

    def __init__(self, watch_id: str, spec: WatchSpec, *, source: FrameSource, find: FaceFinder, crop: CropEncoder, log: Callable[[str], None]):
        self.id = watch_id
        self.spec = spec
        self.source = source
        self.find = find
        self.crop = crop
        self.log = log
        self.state = "connecting"
        self.since = now_iso()
        self.started = time.monotonic()
        self.frames = 0
        self.looked = 0
        self.faces = 0
        self.error: str | None = None
        self.moving = False
        self.moved_at = 0.0
        self.tracks: list[Track] = []
        self.next_track = 1
        self.subscribers: set[asyncio.Queue] = set()
        self.task: asyncio.Task | None = None

    def status(self) -> dict:
        return {
            "id": self.id,
            "source": masked_source(self.spec.source),
            "state": self.state,
            "since": self.since,
            "frames": self.frames,
            "looked": self.looked,
            "faces": self.faces,
            "following": len(self.tracks),
            "motion": self.moving,
            **({"error": self.error} if self.error else {}),
        }

    def start(self) -> None:
        self.task = asyncio.create_task(self.run(), name=f"watch {self.id}")

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
        for queue in list(self.subscribers):
            let_go(queue)

    def tell(self, kind: str, data: dict) -> None:
        """An event to every subscriber; one that has fallen too far behind is let go."""
        event = {"type": kind, "watch": self.id, "at": now_iso(), "t": round(time.monotonic() - self.started, 3), **data}
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self.subscribers.discard(queue)
                let_go(queue)

    async def run(self) -> None:
        backoff = MIN_BACKOFF
        lost = False
        while True:
            self.state = "connecting"
            began = time.monotonic()
            try:
                grid = None
                async for frame in self.source(self.spec):
                    if lost:
                        lost = False
                        self.tell("source.back", {})
                    self.state = "running"
                    self.error = None
                    self.frames += 1
                    grid = await self.on_frame(frame, grid)
                raise ConnectionError("the source ended")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error = str(exc) or type(exc).__name__
                self.state = "lost"
                await self.end_all()
                if not lost:
                    lost = True
                    self.log(f"watch {self.id}: {self.error}")
                    self.tell("source.lost", {"reason": self.error})
            if time.monotonic() - began > MAX_BACKOFF:
                backoff = MIN_BACKOFF
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

    async def on_frame(self, frame: Frame, grid: list[int | None] | None) -> list[int | None]:
        now = time.monotonic() - self.started
        after = grey_grid(frame, self.spec.mask)
        if changed(grid, after):
            self.moved_at = now
            if self.spec.motion and not self.moving:
                self.tell("motion", {})
            self.moving = True
        elif self.moving and now - self.moved_at > MOTION_ENDS_AFTER:
            self.moving = False
            if self.spec.motion:
                self.tell("motion.end", {})
        if not self.spec.motion or self.moving or self.tracks:
            await self.look(frame, now)
        await self.end_gone(now)
        return after

    async def look(self, frame: Frame, now: float) -> None:
        """The faces in one frame, each continuing its track or starting one."""
        self.looked += 1
        raw = await self.find(self.id, frame)
        shorter = min(frame.width, frame.height)
        found = []
        for face in raw.get("faces") or []:
            box = face.get("box") or {}
            x0, y0 = float(box.get("x", 0)), float(box.get("y", 0))
            x1, y1 = x0 + float(box.get("width", 0)), y0 + float(box.get("height", 0))
            if min(x1 - x0, y1 - y0) < self.spec.faces.min_size * shorter or float(face.get("confidence", 0)) < self.spec.faces.min_confidence:
                continue
            cx, cy = (x0 + x1) / 2 / frame.width, (y0 + y1) / 2 / frame.height
            if any(r.x <= cx <= r.x + r.width and r.y <= cy <= r.y + r.height for r in self.spec.mask):
                continue
            detection = {
                "box": (x0, y0, x1, y1),
                "frame": (frame.width, frame.height),
                "confidence": float(face.get("confidence", 0)),
                "unit": face_tracks.unit([float(v) for v in face.get("embedding") or []]),
                "landmarks": face.get("landmarks") or [],
                "t": round(now, 3),
            }
            if isinstance(face.get("pose"), dict):
                detection["pose"] = face["pose"]
            found.append(detection)
        for track, detection in follow(self.tracks, found):
            if track is None:
                track = Track(self.next_track)
                self.next_track += 1
                self.tracks.append(track)
            track.detections.append(detection)
            if track.best is None or face_tracks.quality(detection) > face_tracks.quality(track.best):
                track.best, track.best_frame = detection, frame
            if not track.told and len(track.detections) >= TOLD_AFTER:
                track.told = True
                self.faces += 1
                self.tell("face", await self.face_of(track, detection, frame))

    async def end_gone(self, now: float) -> None:
        gap = face_tracks.max_gap(1 / self.spec.faces.fps)
        for track in [track for track in self.tracks if now - track.last > gap]:
            await self.end(track)

    async def end_all(self) -> None:
        for track in list(self.tracks):
            await self.end(track)

    async def end(self, track: Track) -> None:
        self.tracks.remove(track)
        if not track.told or track.best is None or track.best_frame is None:
            return
        half = 0.5 / self.spec.faces.fps
        face = await self.face_of(track, track.best, track.best_frame)
        self.tell("face.end", {**face, "appearances": [[round(max(0.0, track.first - half), 3), round(track.last + half, 3)]], "detections": len(track.detections)})

    async def face_of(self, track: Track, detection: dict, frame: Frame) -> dict:
        """What an event says of a face: the detection's box and the rest in fractions of the frame,
        and the track's vector so far."""
        x0, y0, x1, y1 = detection["box"]
        width, height = detection["frame"]
        face = {
            "track": track.id,
            "box": {"x": round(x0 / width, 4), "y": round(y0 / height, 4), "width": round((x1 - x0) / width, 4), "height": round((y1 - y0) / height, 4)},
            "confidence": round(detection["confidence"], 4),
            "frame": {"width": width, "height": height},
            "embedding": face_tracks.mean_vector(track.detections),
        }
        if detection.get("landmarks"):
            face["landmarks"] = [[round(float(x) / width, 4), round(float(y) / height, 4)] for x, y in detection["landmarks"]]
        if detection.get("pose"):
            face["pose"] = detection["pose"]
        if self.spec.crops:
            mx, my = (x1 - x0) * CROP_MARGIN, (y1 - y0) * CROP_MARGIN
            cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
            cx1, cy1 = min(frame.width, int(x1 + mx)), min(frame.height, int(y1 + my))
            try:
                jpeg = await self.crop(frame, cx0, cy0, cx1 - cx0, cy1 - cy0)
                face["crop"] = {"type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}
            except Exception as exc:
                self.log(f"watch {self.id}: a face's crop failed: {exc}")
        return face


class Watches:
    """Every watch of this EMS, by id."""

    def __init__(self, limit: int, *, source: FrameSource, find: FaceFinder, crop: CropEncoder, log: Callable[[str], None]):
        self.limit = limit
        self.source = source
        self.find = find
        self.crop = crop
        self.log = log
        self.watches: dict[str, Watch] = {}

    def get(self, watch_id: str) -> Watch:
        watch = self.watches.get(watch_id)
        if watch is None:
            raise EmsError(NOT_FOUND, f"no watch {watch_id}")
        return watch

    async def put(self, watch_id: str, spec: WatchSpec) -> tuple[Watch, bool]:
        """The watch `watch_id` reading `spec`: made, or made again when its spec changed. Answers
        it and whether it was made now; the same spec twice leaves it running."""
        if not WATCH_ID.match(watch_id):
            raise EmsError(INVALID_REQUEST, "a watch's id is 1 to 128 letters, digits, '.', '_' or '-'")
        known = self.watches.get(watch_id)
        if known is not None and known.spec == spec:
            return known, False
        if known is None and len(self.watches) >= self.limit:
            raise EmsError(BUSY, f"this EMS runs at most {self.limit} watches (EMS_WATCHES)")
        if known is not None:
            await self.delete(watch_id)
        watch = Watch(watch_id, spec, source=self.source, find=self.find, crop=self.crop, log=self.log)
        self.watches[watch_id] = watch
        watch.start()
        return watch, True

    async def delete(self, watch_id: str) -> None:
        watch = self.get(watch_id)
        del self.watches[watch_id]
        await watch.stop()

    async def stop(self) -> None:
        for watch_id in list(self.watches):
            await self.delete(watch_id)

    def subscribe(self, watch_id: str) -> tuple[Watch, asyncio.Queue]:
        watch = self.get(watch_id)
        queue: asyncio.Queue = asyncio.Queue(SUBSCRIBER_QUEUE)
        watch.subscribers.add(queue)
        return watch, queue


def filters(spec: WatchSpec) -> str:
    """ffmpeg's filters for a watch: its frames a second, turned into the display frame, no side
    longer than MAX_EDGE."""
    turn = {0: [], 90: ["transpose=2"], 180: ["hflip", "vflip"], 270: ["transpose=1"]}[spec.angle]
    chain = [f"fps={spec.faces.fps}", *turn, *(["hflip"] if spec.mirror else [])]
    chain.append(f"scale=w='min({MAX_EDGE},iw)':h='min({MAX_EDGE},ih)':force_original_aspect_ratio=decrease:force_divisible_by=2")
    return ",".join(chain)


async def ffmpeg_frames(spec: WatchSpec) -> AsyncIterator[Frame]:
    """The source's frames as ffmpeg decodes them: a network stream as it comes, over TCP for RTSP;
    a local file at its own pace, again and again, for a test."""
    source = spec.source
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if source.startswith(("rtsp://", "rtsps://")):
        args += ["-rtsp_transport", "tcp", "-timeout", "10000000"]
    elif "://" not in source:
        args += ["-re", "-stream_loop", "-1"]
    args += ["-i", source, "-an", "-vf", filters(spec), "-f", "image2pipe", "-vcodec", "ppm", "pipe:1"]
    process = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    try:
        while True:
            header = b""
            size = None
            while size is None:
                byte = await process.stdout.read(1)
                if not byte:
                    break
                header += byte
                if len(header) > 64:
                    raise ConnectionError("ffmpeg wrote something that is not a frame")
                size = ppm_size(header)
            if size is None:
                break
            width, height = size
            pixels = await process.stdout.readexactly(width * height * 3)
            yield Frame(width, height, pixels, header + pixels)
    except asyncio.IncompleteReadError:
        pass
    finally:
        if process.returncode is None:
            process.kill()
        stderr = (await process.stderr.read()).decode(errors="replace").strip()
        await process.wait()
        if stderr:
            raise ConnectionError(stderr.splitlines()[-1])


async def ffmpeg_crop(frame: Frame, x: int, y: int, width: int, height: int) -> bytes:
    """A JPEG of a part of a frame, by ffmpeg."""
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "image2pipe", "-vcodec", "ppm", "-i", "pipe:0",
        "-vf", f"crop={width}:{height}:{x}:{y}", "-frames:v", "1", "-q:v", "3", "-f", "image2", "-vcodec", "mjpeg", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await process.communicate(frame.ppm)
    if process.returncode != 0 or not out:
        raise RuntimeError(err.decode(errors="replace").strip() or f"ffmpeg exited {process.returncode}")
    return out
