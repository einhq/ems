"""Watches: a stream's frames looked at for faces while they change, a face told once it was seen
twice and again when it is gone, motion, a source that drops and comes back, the same watch stated
twice, and the routes — with the frames, the worker and the crops stubbed."""

import asyncio
import base64
import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import face_tracks  # noqa: E402

from conftest import AUTH, KEY, FakeModelWorker  # noqa: E402
from ems.app import create_app  # noqa: E402
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


@pytest.fixture
async def ems(tmp_path, monkeypatch):
    # A face gone for a fifth of a second has left, rather than two: the test does not wait.
    monkeypatch.setattr(face_tracks, "TRACK_MAX_GAP_SECONDS", 0.2)
    monkeypatch.setattr("ems.watches.MIN_BACKOFF", 0.05)
    worker_stub = FakeModelWorker(str(tmp_path / "worker.sock"))
    worker_stub.on("detect_faces", detect_faces)
    await worker_stub.start()
    settings = Settings.from_env({"EMS_KEY": KEY, "EMS_SCRATCH": str(tmp_path / "scratch"), "EMS_WORKER_SOCKET": worker_stub.socket_path, "EMS_WATCHES": "1"})
    app = create_app(settings, worker=Worker("/nonexistent/model_server.py", worker_stub.socket_path, log=lambda m: None), start_worker=False)
    ctx = app.state.ctx

    def watches(source) -> Watches:
        ctx.watches = Watches(settings.watches, source=source, find=watches_op.frame_finder(ctx), crop=crop, log=lambda m: None)
        return ctx.watches

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


def test_a_source_is_shown_without_its_password_and_ffmpeg_headers_are_read():
    assert masked_source("rtsp://viewer:s3cret@192.0.2.30:554/sub") == "rtsp://viewer:***@192.0.2.30:554/sub"
    assert masked_source("rtsp://192.0.2.30/sub") == "rtsp://192.0.2.30/sub"
    assert ppm_size(b"P6\n640 360\n255\n") == (640, 360)
    assert ppm_size(b"P6\n640 360\n25") is None
