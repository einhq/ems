"""Event clips: a watch's stream kept a little while, and cut into a video around what the watch
saw there (_4.24_).

A recorder copies its stream, not re-encoding the picture, into MPEG-TS segments of about
SEGMENT_SECONDS each (each starts at a key frame, so a camera's own key-frame interval can make
them longer) in a folder of its own, and keeps only as many as the latest `pre` seconds and the
clips still being cut need. A trigger opens a clip from `pre` seconds before it to `post` after,
and a trigger while it is open moves its end, so one visit is one clip, never longer than `max`.
Once a segment past its end has come, its segments are joined into an MP4 the caller fetches by id
and is told of; a clip nobody fetched is dropped after CLIP_KEEP_SECONDS. A source that drops is
read again after a pause that doubles up to a minute, and a clip open then is cut from what came.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Awaitable, Callable, Literal

from pydantic import Field

from .errors import NOT_FOUND, EmsError
from .schemas import Strict

# How long a segment is asked to be.
SEGMENT_SECONDS = 2
# How long a clip waits for its caller before it is dropped.
CLIP_KEEP_SECONDS = 3600.0
# How long past its end a clip waits for the segment that closes it before it is cut from what came.
LATE_SECONDS = 30.0
MIN_BACKOFF, MAX_BACKOFF = 1.0, 60.0


class ClipsSpec(Strict):
    """A watch's `clips`: what opens a clip, and how much of the stream it holds."""

    # The events that open a clip, and keep it open while they last: a face followed, frames moving.
    on: list[Literal["face", "motion"]] = Field(default_factory=lambda: ["face", "motion"], min_length=1)
    # Seconds before the event, and after it ends.
    pre: float = Field(default=5, ge=0, le=30)
    post: float = Field(default=10, ge=1, le=60)
    # The longest clip, in seconds.
    max: float = Field(default=120, ge=10, le=600)
    # The stream recorded, when not the one watched: the camera's main stream beside its sub-stream.
    source: str | None = Field(default=None, min_length=1)


class ClipRequest(Strict):
    """`POST /v0/watches/<id>/clips`' body: a clip around now, `post` seconds on unless the watch's."""

    post: float | None = Field(default=None, ge=1, le=60)


@dataclass
class Segment:
    path: str
    # Monotonic seconds: when it began and ended.
    start: float
    end: float


@dataclass
class Clip:
    id: str
    start: float
    end: float
    # The longest it may grow to: its start and `max`.
    limit: float
    reasons: list[str] = field(default_factory=list)
    # Once cut: the file, its length, and when it was made (monotonic).
    path: str | None = None
    duration: float = 0.0
    made: float = 0.0
    info: dict = field(default_factory=dict)


# What records a source: given it and a folder, writes segments there and yields each one's path
# and length, in seconds, as it is done, starting its file names at the number given;
# `ffmpeg_segments` in the service, a list in tests.
SegmentSource = Callable[[str, str, int], AsyncIterator[tuple[str, float]]]
# What joins segments into an MP4 at the path given, made at the moment given (ISO 8601).
Joiner = Callable[[list[str], str, str], Awaitable[None]]


def wall(at: float) -> str:
    """A monotonic moment as the wall clock's ISO 8601."""
    moment = datetime.now(timezone.utc) - timedelta(seconds=time.monotonic() - at)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Recorder:
    """One watch's recording: its segments, its clips, and the loop that reads its source."""

    def __init__(self, watch_id: str, spec: ClipsSpec, source: str, folder: str, *, segments: SegmentSource, join: Joiner, tell: Callable[[str, dict], None], log: Callable[[str], None]):
        self.watch_id = watch_id
        self.spec = spec
        self.source = source
        self.folder = folder
        self.segments_of = segments
        self.join = join
        self.tell = tell
        self.log = log
        self.state = "connecting"
        self.error: str | None = None
        self.segments: list[Segment] = []
        self.next_number = 0
        self.open: Clip | None = None
        # Clips cut or being cut, by id.
        self.clips: dict[str, Clip] = {}
        self.made = 0
        self.task: asyncio.Task | None = None
        self.cutting: set[asyncio.Task] = set()

    def status(self) -> dict:
        return {
            "state": self.state,
            "segments": len(self.segments),
            "made": self.made,
            "ready": sum(1 for clip in self.clips.values() if clip.path),
            **({"open": self.open.id} if self.open else {}),
            **({"error": self.error} if self.error else {}),
        }

    def start(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)
        os.makedirs(os.path.join(self.folder, "clips"), exist_ok=True)
        self.task = asyncio.create_task(self.run(), name=f"clips {self.watch_id}")

    async def stop(self) -> None:
        for task in [self.task, *self.cutting]:
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        shutil.rmtree(self.folder, ignore_errors=True)

    def trigger(self, reason: str, at: float, post: float | None = None) -> str:
        """A clip around `at` (monotonic): the open one, its end moved, or a new one. Answers its id."""
        post = self.spec.post if post is None else post
        if self.open is not None:
            clip = self.open
            clip.end = min(max(clip.end, at + post), clip.limit)
            if reason not in clip.reasons:
                clip.reasons.append(reason)
            return clip.id
        start = at - self.spec.pre
        clip = Clip(id=f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}", start=start, end=at + post, limit=start + self.spec.max, reasons=[reason])
        self.open = clip
        self.clips[clip.id] = clip
        return clip.id

    def extend(self, at: float) -> None:
        """The open clip held to `post` past `at`, while what opened it lasts."""
        if self.open is not None:
            self.open.end = min(max(self.open.end, at + self.spec.post), self.open.limit)

    def get(self, clip_id: str) -> Clip:
        clip = self.clips.get(clip_id)
        if clip is None or clip.path is None:
            raise EmsError(NOT_FOUND, f"no clip {clip_id} of watch {self.watch_id}")
        return clip

    def ready(self) -> list[dict]:
        return [clip.info for clip in self.clips.values() if clip.path]

    def drop(self, clip_id: str) -> None:
        clip = self.get(clip_id)
        del self.clips[clip_id]
        if clip.path:
            try:
                os.remove(clip.path)
            except FileNotFoundError:
                pass

    async def run(self) -> None:
        backoff = MIN_BACKOFF
        while True:
            self.state = "connecting"
            began = time.monotonic()
            try:
                async for path, length in self.segments_of(self.source, self.folder, self.next_number):
                    self.next_number += 1
                    end = time.monotonic()
                    self.segments.append(Segment(path, end - length, end))
                    self.state = "recording"
                    self.error = None
                    self.tick(end)
                raise ConnectionError("the source ended")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self.error is None:
                    self.log(f"watch {self.watch_id}: recording: {exc}")
                self.error = str(exc) or type(exc).__name__
                self.state = "lost"
                if self.open is not None:
                    self.cut(self.open)
            if time.monotonic() - began > MAX_BACKOFF:
                backoff = MIN_BACKOFF
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

    def tick(self, now: float) -> None:
        """After a segment came: the open clip cut once its end has been recorded, or has waited too
        long; segments no clip needs and clips nobody fetched dropped."""
        clip = self.open
        if clip is not None and (self.segments[-1].end >= clip.end or now > clip.end + LATE_SECONDS):
            self.cut(clip)
        needed = now - self.spec.pre
        for each in self.clips.values():
            if each.path is None:
                needed = min(needed, each.start)
        while self.segments and self.segments[0].end < needed:
            gone = self.segments.pop(0)
            try:
                os.remove(gone.path)
            except FileNotFoundError:
                pass
        for each in [each for each in self.clips.values() if each.path and now - each.made > CLIP_KEEP_SECONDS]:
            self.drop(each.id)

    def cut(self, clip: Clip) -> None:
        """The clip closed, and joined from its segments in the background."""
        if self.open is clip:
            self.open = None
        parts = [segment for segment in self.segments if segment.end > clip.start and segment.start < clip.end]
        task = asyncio.create_task(self.make(clip, parts), name=f"clip {clip.id}")
        self.cutting.add(task)
        task.add_done_callback(self.cutting.discard)

    async def make(self, clip: Clip, parts: list[Segment]) -> None:
        path = os.path.join(self.folder, "clips", f"{clip.id}.mp4")
        try:
            if not parts:
                raise RuntimeError("nothing of the stream was recorded then")
            start = wall(parts[0].start)
            await self.join([part.path for part in parts], path, start)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.clips.pop(clip.id, None)
            self.log(f"watch {self.watch_id}: clip {clip.id}: {exc}")
            self.tell("clip.failed", {"clip": clip.id, "reason": str(exc) or type(exc).__name__})
            return
        clip.path = path
        clip.made = time.monotonic()
        clip.duration = round(parts[-1].end - parts[0].start, 3)
        clip.info = {
            "clip": clip.id,
            "start": start,
            "end": wall(parts[-1].end),
            "duration": clip.duration,
            "size": os.path.getsize(path),
            "reasons": clip.reasons,
        }
        self.made += 1
        self.tell("clip", clip.info)


def input_args(source: str) -> list[str]:
    """How ffmpeg opens a source: a network stream as it comes, over TCP for RTSP; a local file at
    its own pace, again and again, for a test."""
    if source.startswith(("rtsp://", "rtsps://")):
        return ["-rtsp_transport", "tcp", "-timeout", "10000000", "-i", source]
    if "://" not in source:
        return ["-re", "-stream_loop", "-1", "-i", source]
    return ["-i", source]


async def ffmpeg_segments(source: str, folder: str, first: int) -> AsyncIterator[tuple[str, float]]:
    """The source copied into MPEG-TS segments by ffmpeg, its picture as it came and its sound as
    AAC, each segment told by ffmpeg's segment list as it closes. Each segment's timestamps start
    at zero: an RTSP stream with B-frames gives its first packets no time, which a segment's muxer
    refuses when it carries the stream's own on (`ffmpeg_join` puts them end to end again)."""
    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-fflags", "+genpts", *input_args(source),
        "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "copy", "-c:a", "aac", "-avoid_negative_ts", "make_zero",
        "-f", "segment", "-segment_time", str(SEGMENT_SECONDS), "-segment_format", "mpegts", "-reset_timestamps", "1", "-segment_start_number", str(first),
        "-segment_list", "pipe:1", "-segment_list_type", "csv", os.path.join(folder, "%010d.ts"),
    ]
    process = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    try:
        while line := await process.stdout.readline():
            name, start, end = line.decode().strip().rsplit(",", 2)
            yield os.path.join(folder, os.path.basename(name)), max(0.0, float(end) - float(start))
    finally:
        if process.returncode is None:
            process.kill()
        stderr = (await process.stderr.read()).decode(errors="replace").strip()
        await process.wait()
        if stderr:
            raise ConnectionError(stderr.splitlines()[-1])


async def ffmpeg_join(paths: list[str], out: str, made: str) -> None:
    """MPEG-TS segments joined end to end into an MP4 by ffmpeg's concat demuxer, which moves each
    segment's timestamps on past the one before; copied, its index at the front and its
    `creation_time` the moment it starts."""
    listing = f"{out}.txt"
    with open(listing, "w") as handle:
        handle.writelines(f"file '{path}'\n" for path in paths)
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", listing,
        "-map", "0", "-c", "copy", "-metadata", f"creation_time={made}", "-movflags", "+faststart", "-f", "mp4", out,
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, err = await process.communicate()
    finally:
        os.remove(listing)
    if process.returncode != 0:
        raise RuntimeError(err.decode(errors="replace").strip() or f"ffmpeg exited {process.returncode}")
