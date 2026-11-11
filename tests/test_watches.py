"""Watches: a stream's frames looked at for faces while they change, a face told once it was seen
twice and again when it is gone, motion, a source that drops and comes back, the same watch stated
twice, clips cut around a face and on a request, and the routes — with the frames, the worker, the
crops and the recording stubbed; and ffmpeg's recording and joining of a real stream."""

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import face_tracks  # noqa: E402

from conftest import AUTH, KEY, FakeModelWorker  # noqa: E402
from ems import watches as watches_module  # noqa: E402
from ems.app import create_app  # noqa: E402
from ems.card import CardArbiter, watching  # noqa: E402
from ems.clips import ffmpeg_join, ffmpeg_segments  # noqa: E402
from ems.config import Settings  # noqa: E402
from ems.ops import watches as watches_op  # noqa: E402
from ems.watches import Frame, Watches, WatchSpec, masked_source, ppm_size  # noqa: E402
from ems.worker import Worker  # noqa: E402

WIDTH, HEIGHT = 64, 36


def frame(grey: int, face: bool) -> Frame:
    """A frame all one grey, which the motion check sees change when the grey does; its first
    pixel says to the stub worker whether a face is in it."""
    pixels = bytearray(bytes([grey]) * (WIDTH * HEIGHT * 3))
    pixels[0] = 255 if face else 0
    header = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode()
    return Frame(WIDTH, HEIGHT, bytes(pixels), header + bytes(pixels))


def detect_faces(request: dict) -> dict:
    """The stub worker: one face where the frame's first pixel says there is one."""
    with open(request["path"], "rb") as handle:
        data = handle.read()
    start = data.index(b"255\n") + 4
    faces = []
    if data[start] == 255:
        faces.append({
            "box": {"x": 10, "y": 5, "width": 20, "height": 20},
            "confidence": 0.95,
            "embedding": [1.0, 0.0, 0.0, 0.0],
            "landmarks": [[15, 10], [25, 10], [20, 15], [16, 20], [24, 20]],
            "pose": {"pitch": 0.0, "yaw": 5.0, "roll": 0.0},
        })
    return {"frame": {"width": WIDTH, "height": HEIGHT}, "faces": faces}


class Script:
    """A source that gives its frames, a twentieth of a second apart, then ends; read again, it
    gives `again`'s, and then nothing more."""

    def __init__(self, frames: list[Frame], again: list[Frame] | None = None):
        self.rounds = [frames, again or []]
        self.reads = 0

    async def __call__(self, spec: WatchSpec):
        frames = self.rounds[min(self.reads, len(self.rounds) - 1)]
        self.reads += 1
        for each in frames:
            await asyncio.sleep(0.05)
            yield each
        if self.reads > len(self.rounds):
            await asyncio.sleep(3600)


async def crop(frame: Frame, x: int, y: int, width: int, height: int) -> bytes:
    return f"crop {x},{y} {width}x{height}".encode()


SEGMENT = 0.1


async def segments(source: str, folder: str, first: int):
    """A recording stub: a segment a tenth of a second long every tenth of a second, its bytes its
    number."""
    number = first
    while True:
        await asyncio.sleep(SEGMENT)
        path = os.path.join(folder, f"{number:010d}.ts")
        with open(path, "wb") as handle:
            handle.write(f"{number};".encode())
        yield path, SEGMENT
        number += 1


async def join(paths: list[str], out: str, made: str) -> None:
    with open(out, "wb") as handle:
        for path in paths:
            with open(path, "rb") as part:
                handle.write(part.read())


@pytest.fixture
async def ems(tmp_path, monkeypatch):
    # A face gone for a fifth of a second has left, rather than two: the test does not wait.
    monkeypatch.setattr(face_tracks, "TRACK_MAX_GAP_SECONDS", 0.2)
    monkeypatch.setattr("ems.watches.MIN_BACKOFF", 0.05)
    worker_stub = FakeModelWorker(str(tmp_path / "worker.sock"))
    worker_stub.on("detect_faces", detect_faces)
    worker_stub.on("pin", lambda req: {"pinning": req["families"]})
    worker_stub.on("unpin", lambda req: {"unpinned": req["families"]})
    await worker_stub.start()
    settings = Settings.from_env({"EMS_KEY": KEY, "EMS_SCRATCH": str(tmp_path / "scratch"), "EMS_WORKER_SOCKET": worker_stub.socket_path, "EMS_WATCHES": "1"})
    app = create_app(settings, worker=Worker("/nonexistent/model_server.py", worker_stub.socket_path, log=lambda m: None), start_worker=False)
    ctx = app.state.ctx

    def watches(source) -> Watches:
        kept = watches_op.FacesKept(ctx, lambda m: None)
        ctx.watches = Watches(
            settings.watches, source=source, find=watches_op.frame_finder(ctx, kept), crop=crop, log=lambda m: None, idle=kept.release,
            folder=str(tmp_path / "watches"), segments=segments, join=join,
        )
        return ctx.watches

    watches.ctx = ctx
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ems", headers=AUTH) as client:
            yield client, watches, worker_stub
    await worker_stub.stop()


async def events_until(queue: asyncio.Queue, last: str, timeout: float = 5.0) -> list[dict]:
    """What a subscriber hears, up to and with the first event of type `last`."""
    heard = []
    async with asyncio.timeout(timeout):
        while True:
            event = await queue.get()
            heard.append(event)
            if event["type"] == last:
                return heard


async def test_a_face_is_told_once_seen_twice_and_again_when_gone(ems):
    client, watches, worker_stub = ems
    # Still, then a change with a face, the face held through still frames, then gone.
    script = Script([frame(10, False), frame(100, True), frame(10, True), frame(10, True), *[frame(10, False)] * 8])
    watch_set = watches(script)
    put = await client.put("/v0/watches/front-door", json={"source": "rtsp://viewer:s3cret@192.0.2.30/sub", "faces": {"fps": 5}})
    assert put.status_code == 200
    assert put.json()["result"]["made"] is True
    _, queue = watch_set.subscribe("front-door")
    heard = await events_until(queue, "face.end")
    kinds = [event["type"] for event in heard]
    assert kinds == ["motion", "face", "face.end"] or kinds == ["motion", "face", "motion.end", "face.end"]
    face = next(event for event in heard if event["type"] == "face")
    assert face["watch"] == "front-door" and face["track"] == 1
    assert face["box"] == {"x": round(10 / WIDTH, 4), "y": round(5 / HEIGHT, 4), "width": round(20 / WIDTH, 4), "height": round(20 / HEIGHT, 4)}
    assert face["embedding"] == [1.0, 0.0, 0.0, 0.0]
    assert face["pose"] == {"pitch": 0.0, "yaw": 5.0, "roll": 0.0}
    # The crop takes three tenths of the face's size again on each side, inside the frame.
    assert face["crop"]["type"] == "image/jpeg"
    assert base64.b64decode(face["crop"]["data"]) == b"crop 4,0 32x31"
    end = heard[-1]
    assert end["track"] == 1 and end["detections"] == 3
    assert end["appearances"][0][0] < end["appearances"][0][1]


async def test_frames_that_do_not_change_are_not_looked_at(ems):
    client, watches, worker_stub = ems
    watch_set = watches(Script([frame(10, True)] * 10, again=[frame(10, True)]))
    await client.put("/v0/watches/front-door", json={"source": "rtsp://192.0.2.30/sub"})
    _, queue = watch_set.subscribe("front-door")
    heard = await events_until(queue, "source.back")
    assert [event["type"] for event in heard] == ["source.lost", "source.back"]
    assert [request for request in worker_stub.requests if request["op"] == "detect_faces"] == []
    assert (await client.get("/v0/watches/front-door")).json()["result"]["looked"] == 0


async def test_a_source_that_ends_is_read_again_and_said_to_be_back(ems):
    client, watches, _ = ems
    watch_set = watches(Script([frame(10, False)], again=[frame(10, False)]))
    await client.put("/v0/watches/porch", json={"source": "rtsp://192.0.2.31/sub"})
    _, queue = watch_set.subscribe("porch")
    heard = await events_until(queue, "source.back")
    assert [event["type"] for event in heard] == ["source.lost", "source.back"]
    assert heard[0]["reason"] == "the source ended"


async def test_the_same_watch_stated_twice_runs_on_and_another_is_busy(ems):
    client, watches, _ = ems
    watches(Script([frame(10, False)] * 100))
    spec = {"source": "rtsp://viewer:s3cret@192.0.2.30/sub"}
    first = (await client.put("/v0/watches/front-door", json=spec)).json()["result"]
    again = (await client.put("/v0/watches/front-door", json=spec)).json()["result"]
    assert (first["made"], again["made"]) == (True, False)
    assert again["source"] == "rtsp://viewer:***@192.0.2.30/sub"
    busy = await client.put("/v0/watches/porch", json={"source": "rtsp://192.0.2.31/sub"})
    assert busy.status_code == 503 and busy.json()["error"]["code"] == "busy"
    listed = (await client.get("/v0/watches")).json()["result"]
    assert [watch["id"] for watch in listed] == ["front-door"]
    assert (await client.put("/v0/watches/bad id", json=spec)).status_code == 400
    assert (await client.put("/v0/watches/front-door", json={**spec, "colour": "red"})).status_code == 400


async def test_the_event_stream_ends_when_its_watch_is_deleted(ems):
    client, watches, _ = ems
    watches(Script([frame(10, False)] * 100))
    await client.put("/v0/watches/front-door", json={"source": "rtsp://192.0.2.30/sub"})
    stream = asyncio.create_task(client.get("/v0/watches/front-door/events"))
    await asyncio.sleep(0.2)
    deleted = await client.delete("/v0/watches/front-door")
    assert deleted.status_code == 200
    response = await asyncio.wait_for(stream, 5)
    kinds = [line.removeprefix("event: ") for line in response.text.splitlines() if line.startswith("event: ")]
    assert kinds[0] == "state" and kinds[-1] == "end"
    state = json.loads(response.text.split("data: ", 1)[1].split("\n", 1)[0])
    assert state["id"] == "front-door" and state["state"] in ("connecting", "running")
    assert (await client.get("/v0/watches/front-door")).status_code == 404


async def test_the_faces_family_is_pinned_while_a_watch_runs_and_again_after_a_worker_restart(ems):
    client, watches, worker_stub = ems
    watches(Script([frame(10 + 90 * (n % 2), False) for n in range(40)]))
    pins = lambda: [(request["op"], request["families"]) for request in worker_stub.requests if request["op"] in ("pin", "unpin")]
    await client.put("/v0/watches/front-door", json={"source": "rtsp://192.0.2.30/sub"})
    async with asyncio.timeout(5):
        while not pins():
            await asyncio.sleep(0.05)
    assert pins() == [("pin", ["faces"])]
    ctx = watches.ctx
    assert ctx.card.kept == frozenset({"faces"})
    ctx.worker.restarts += 1
    async with asyncio.timeout(5):
        while len(pins()) < 2:
            await asyncio.sleep(0.05)
    # A watch made again with another body keeps the family; the last one stopped lets it go.
    await client.put("/v0/watches/front-door", json={"source": "rtsp://192.0.2.30/sub", "motion": False})
    assert pins() == [("pin", ["faces"])] * 2
    await client.delete("/v0/watches/front-door")
    assert pins() == [("pin", ["faces"])] * 2 + [("unpin", ["faces"])]
    assert ctx.card.kept == frozenset()


async def test_a_watchs_frame_leaves_a_quiet_card_quiet():
    arbiter = CardArbiter(None, None, None, 60, lambda: None)
    arbiter.enabled, arbiter.mode = True, "quiet"
    token = watching.set(True)
    try:
        async with arbiter.gate("detect_faces", {}):
            assert arbiter.inflight == 0
    finally:
        watching.reset(token)
    assert arbiter.mode == "quiet"


async def test_nvdec_decodes_and_the_cpu_takes_a_source_it_gives_nothing_of(monkeypatch):
    reads: list[tuple[str, bool]] = []

    async def decode(spec, nvdec):
        reads.append((spec.source, nvdec))
        if nvdec and spec.source.endswith("mjpeg"):
            raise ConnectionError("no NVDEC for this codec")
        yield frame(10, False)

    monkeypatch.setattr(watches_module, "_ffmpeg_frames", decode)
    monkeypatch.setattr(watches_module, "_cpu_only", {})
    monkeypatch.setattr("ems.tools.ffmpeg.gpu_present", lambda: True)
    for source in ("rtsp://192.0.2.30/h264", "rtsp://192.0.2.31/mjpeg", "rtsp://192.0.2.31/mjpeg"):
        assert len([each async for each in watches_module.ffmpeg_frames(WatchSpec(source=source))]) == 1
    assert reads == [("rtsp://192.0.2.30/h264", True), ("rtsp://192.0.2.31/mjpeg", True), ("rtsp://192.0.2.31/mjpeg", False), ("rtsp://192.0.2.31/mjpeg", False)]
    assert watches_module.filters(WatchSpec(source="x", angle=90), nvdec=True).startswith("fps=3.0,hwdownload,format=nv12,transpose=2,")


def test_a_source_is_shown_without_its_password_and_ffmpeg_headers_are_read():
    assert masked_source("rtsp://viewer:s3cret@192.0.2.30:554/sub") == "rtsp://viewer:***@192.0.2.30:554/sub"
    assert masked_source("rtsp://192.0.2.30/sub") == "rtsp://192.0.2.30/sub"
    assert ppm_size(b"P6\n640 360\n255\n") == (640, 360)
    assert ppm_size(b"P6\n640 360\n25") is None


def numbers(clip: bytes) -> list[int]:
    return [int(number) for number in clip.decode().split(";") if number]


async def test_a_face_cuts_a_clip_from_before_it_until_after_it_is_gone(ems):
    client, watches, _ = ems
    script = Script([frame(10, False), frame(100, True), frame(10, True), frame(10, True), *[frame(10, False)] * 8])
    watch_set = watches(script)
    clips = {"on": ["face"], "pre": 0.5, "post": 1}
    put = await client.put("/v0/watches/front-door", json={"source": "rtsp://192.0.2.30/sub", "faces": {"fps": 5}, "clips": clips})
    assert put.json()["result"]["clips"]["state"] in ("connecting", "recording")
    _, queue = watch_set.subscribe("front-door")
    heard = await events_until(queue, "clip")
    kinds = [event["type"] for event in heard]
    assert kinds.index("face") < kinds.index("clip")
    clip = heard[-1]
    assert clip["reasons"] == ["face"]
    assert clip["start"] < clip["end"]
    # Half a second before the face, and a second after it was last followed.
    assert clip["duration"] >= 1.4
    listed = (await client.get("/v0/watches/front-door/clips")).json()["result"]
    assert [each["clip"] for each in listed] == [clip["clip"]]
    got = await client.get(f"/v0/watches/front-door/clips/{clip['clip']}")
    assert got.status_code == 200 and got.headers["content-type"] == "video/mp4"
    parts = numbers(got.content)
    assert parts == list(range(parts[0], parts[-1] + 1)) and len(got.content) == clip["size"]
    assert (await client.delete(f"/v0/watches/front-door/clips/{clip['clip']}")).status_code == 200
    assert (await client.get(f"/v0/watches/front-door/clips/{clip['clip']}")).status_code == 404
    # Once a segment has come since, only what a clip may still need is kept: the latest half second.
    await asyncio.sleep(0.3)
    status = (await client.get("/v0/watches/front-door")).json()["result"]["clips"]
    assert status["made"] == 1 and status["segments"] <= 7


async def test_a_request_cuts_a_clip_and_a_watch_without_clips_refuses_one(ems):
    client, watches, _ = ems
    watch_set = watches(Script([frame(10, False)] * 200))
    await client.put("/v0/watches/porch", json={"source": "rtsp://192.0.2.31/sub", "clips": {"pre": 0.2, "post": 1}})
    _, queue = watch_set.subscribe("porch")
    asked = await client.post("/v0/watches/porch/clips", json={"post": 2})
    clip_id = asked.json()["result"]["clip"]
    again = (await client.post("/v0/watches/porch/clips", json={})).json()["result"]["clip"]
    assert again == clip_id
    heard = await events_until(queue, "clip")
    assert heard[-1]["clip"] == clip_id and heard[-1]["reasons"] == ["request"]
    assert 2 <= heard[-1]["duration"] < 3
    await client.delete("/v0/watches/porch")
    # A watch deleted takes its recording and its clips with it.
    assert os.path.isdir(watch_set.folder) and not os.path.exists(os.path.join(watch_set.folder, "porch"))
    await client.put("/v0/watches/porch", json={"source": "rtsp://192.0.2.31/sub"})
    refused = await client.post("/v0/watches/porch/clips", json={})
    assert refused.status_code == 400 and refused.json()["error"]["code"] == "invalid_request"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
async def test_ffmpeg_records_segments_and_joins_them_into_an_mp4(tmp_path):
    source = tmp_path / "camera.mkv"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25", "-f", "lavfi", "-i", "sine",
         "-t", "3", "-c:v", "libx264", "-g", "25", "-c:a", "pcm_alaw", str(source)],
        check=True,
    )
    folder = tmp_path / "segments"
    folder.mkdir()
    got = []
    async with asyncio.timeout(30):
        recording = ffmpeg_segments(str(source), str(folder), 7)
        async for path, length in recording:
            got.append((path, length))
            if len(got) == 2:
                break
        await recording.aclose()
    assert [os.path.basename(path) for path, _ in got] == ["0000000007.ts", "0000000008.ts"]
    assert all(1.5 < length < 2.5 for _, length in got)
    out = tmp_path / "clip.mp4"
    await ffmpeg_join([path for path, _ in got], str(out), "2026-11-11T07:53:00.250Z")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name:format=duration:format_tags=creation_time", "-of", "json", str(out)], check=True, capture_output=True, text=True)
    info = json.loads(probe.stdout)
    assert info["format"]["tags"]["creation_time"].startswith("2026-11-11T07:53:00")
    assert sorted(stream["codec_name"] for stream in info["streams"]) == ["aac", "h264"]
    assert 3.5 < float(info["format"]["duration"]) < 4.5
