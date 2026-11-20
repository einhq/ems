"""`pages`: ffmpeg's MP4 and TS cut where a reader may start, each written as it comes into the
name its caller grants, and a stream that ends when its browser fails, when a write fails or when
it is deleted, and keys pressed in it through the browser's DevTools. The
rendering is stood in for; the fixtures are what ffmpeg's tee writes (two seconds of a picture at 15
fps with a keyframe a second, and a tone as AAC, a fragment every 200 ms):

    ffmpeg -f lavfi -i testsrc2=s=160x90:r=15 -f lavfi -i sine=f=440:r=48000 -t 2 \
      -c:v libx264 -preset ultrafast -bf 0 -g 15 -c:a aac -b:a 64k -ar 48000 -ac 2 \
      -map 0:v -map 1:a -flags +global_header -f tee \
      "[f=mp4:movflags=frag_keyframe+empty_moov+default_base_moof:frag_duration=200000]page.mp4|[f=mpegts:bsfs/v=dump_extra=freq=keyframe]page.mpegts"
"""

import asyncio
import json
import os
import tempfile
import threading
import time

import pytest

from ems import pages
from ems.errors import EmsError
from ems.pages import Devtools, Mp4Splitter, PageSpec, Rendering, TsSplitter, decode_times, encoder_args, key_event, tracks

DATA = os.path.join(os.path.dirname(__file__), "data")
MP4 = open(os.path.join(DATA, "page.mp4"), "rb").read()
TS = open(os.path.join(DATA, "page.mpegts"), "rb").read()


def test_the_mp4_is_cut_into_its_header_and_fragments_marked_at_keyframes():
    splitter = Mp4Splitter()
    pieces = [piece for at in range(0, len(MP4), 1000) for piece in splitter.feed(MP4[at:at + 1000])]
    assert splitter.header is not None and splitter.header[4:8] == b"ftyp"
    assert all(piece[4:8] == b"moof" for _, piece in pieces)
    joined = b"".join([splitter.header, *(piece for _, piece in pieces)])
    # All but the index ffmpeg writes as it closes, which a live stream never reaches.
    assert MP4.startswith(joined) and MP4[len(joined) + 4:len(joined) + 8] == b"mfra"
    assert [key for key, _ in pieces].count(True) == 2
    assert pieces[0][0] is True
    # Every fragment holds the picture and the sound.
    assert all(len(decode_times(piece)) == 2 for _, piece in pieces)


def test_the_header_names_the_picture_and_the_sound_by_their_timescales():
    splitter = Mp4Splitter()
    splitter.feed(MP4)
    found = tracks(splitter.header)
    assert [track.video for track in found.values()] == [True, False]
    assert found[2].timescale == 48000


def test_the_ts_is_cut_at_its_keyframes_and_keeps_its_tables():
    splitter = TsSplitter()
    pieces = [piece for at in range(0, len(TS), 1000) for piece in splitter.feed(TS[at:at + 1000])]
    assert b"".join(piece for _, piece in pieces) == TS
    assert splitter.header is not None and len(splitter.header) == 2 * 188
    keys = [piece for key, piece in pieces if key]
    # The picture's two keyframes, not the sound's packets, which ffmpeg marks too.
    assert len(keys) == 2 and splitter.video_pid is not None
    assert all(piece[0] == 0x47 and piece[5] & 0x40 for piece in keys)
    assert all(((piece[1] & 0x1F) << 8 | piece[2]) == splitter.video_pid for piece in keys)
    marked = {((TS[at + 1] & 0x1F) << 8) | TS[at + 2] for at in range(0, len(TS), 188) if TS[at + 3] & 0x20 and TS[at + 4] and TS[at + 5] & 0x40}
    assert len(marked) == 2


class FakeProcess:
    def __init__(self):
        self.returncode = None


class FakeRendering(Rendering):
    """Outputs fed by the test; its processes are never real, and the keys it is sent are kept."""

    def __init__(self):
        self.mp4 = asyncio.StreamReader()
        self.ts = asyncio.StreamReader()
        super().__init__({"chromium": FakeProcess(), "ffmpeg": FakeProcess()}, {"mp4": self.mp4, "ts": self.ts}, "h264_nvenc")
        self.memory = 300.0
        self.stopped = False
        self.pressed: list[str] = []
        self.told: list[tuple[str, object]] = []
        self.state: object = None

    async def press(self, names: list[str]) -> None:
        for name in names:
            key_event(name)
        self.pressed += names

    async def screen(self, call: str, argument: object = None) -> object:
        self.told.append((call, argument))
        return self.state if call == "state" else (True if self.state is not None else None)

    def feed(self, mp4: bytes, ts: bytes) -> None:
        self.mp4.feed_data(mp4)
        self.ts.feed_data(ts)

    def memory_mb(self) -> float:
        return self.memory

    async def stop(self) -> None:
        self.stopped = True
        self.mp4.feed_eof()
        self.ts.feed_eof()


class Launches:
    def __init__(self, fail=False):
        self.renderings: list[FakeRendering] = []
        self.fail = fail

    async def __call__(self, spec, page_id):
        if self.fail:
            raise EmsError(pages.TOOL_FAILED, "this EMS has no chromium, which renders pages", permanent=True)
        rendering = FakeRendering()
        self.renderings.append(rendering)
        # A second and a half: two keyframes in each format, the second a second in.
        rendering.feed(MP4[: len(MP4) * 3 // 4], TS[: len(TS) * 3 // 4])
        return rendering


class Written:
    """The caller's names as a page opens them to write: a file of its own each, by name."""

    def __init__(self):
        self.folder = tempfile.mkdtemp(prefix="ems-page-written-")
        self.opened: list[tuple[str, str]] = []

    async def __call__(self, endpoint, name):
        self.opened.append((endpoint.host, name))
        return os.open(self.path(name), os.O_WRONLY | os.O_CREAT)

    def path(self, name: str) -> str:
        return os.path.join(self.folder, name.strip("/").replace("/", "-"))

    def bytes(self, name: str) -> bytes:
        return open(self.path(name), "rb").read()


@pytest.fixture
def launches(client, monkeypatch):
    monkeypatch.setattr(pages, "WATCH_SECONDS", 0.05)
    found = Launches()
    client.app.state.ctx.pages.launch = found
    found.written = Written()
    client.app.state.ctx.pages.open_name = found.written
    return found


SPEC = {"url": "http://ein.test:8080/__page/abc", "video": {"width": 1280, "height": 720, "fps": 30},
        "output": {"mp4": {"path": "/0f/0-page.mp4"}, "ts": {"path": "/0f/1-page.ts"}}}


async def _written(written: Written, name: str, size: int) -> bytes:
    """What `name` holds once it holds `size` bytes."""
    for _ in range(200):
        data = written.bytes(name)
        if len(data) >= size:
            return data
        await asyncio.sleep(0.01)
    return written.bytes(name)


async def test_a_page_runs_once_it_has_a_picture_and_the_same_body_again_leaves_it(client, launches):
    r = await client.put("/v0/pages/kitchen", json=SPEC)
    assert r.status_code == 200, r.text
    status = r.json()["result"]
    assert status["state"] == "running" and status["made"] is True
    assert status["encoder"] == "h264_nvenc"
    again = (await client.put("/v0/pages/kitchen", json=SPEC)).json()["result"]
    assert again["made"] is False and len(launches.renderings) == 1
    listed = (await client.get("/v0/pages")).json()["result"]
    assert [page["id"] for page in listed] == ["kitchen"]


async def test_each_format_is_written_into_its_name_as_it_comes_and_closed_when_the_page_ends(client, launches):
    await client.put("/v0/pages/kitchen", json=SPEC)
    written = launches.written
    assert written.opened == [("caller.test", "/0f/0-page.mp4"), ("caller.test", "/0f/1-page.ts")]
    fed = len(MP4) * 3 // 4
    assert await _written(written, "/0f/0-page.mp4", fed) == MP4[:fed]
    launches.renderings[0].feed(MP4[fed:], TS[len(TS) * 3 // 4:])
    assert await _written(written, "/0f/0-page.mp4", len(MP4)) == MP4
    assert await _written(written, "/0f/1-page.ts", len(TS)) == TS
    page = client.app.state.ctx.pages.get("kitchen")
    assert page.status()["bytes"] == len(MP4) and "readers" not in page.status()
    fds = list(page.written.values())
    await client.delete("/v0/pages/kitchen")
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


async def test_a_page_is_put_with_the_endpoint_its_names_are_granted_under(client, launches):
    r = await client.put("/v0/pages/kitchen", json=SPEC, headers={"ems-files": ""})
    assert r.status_code == 400 and "EMS-Files" in r.json()["error"]["message"]
    r = await client.put("/v0/pages/kitchen", json={key: value for key, value in SPEC.items() if key != "output"})
    assert r.status_code == 400
    await client.put("/v0/pages/kitchen", json=SPEC)
    # The same body through another endpoint is the caller started anew: the page starts anew.
    again = (await client.put("/v0/pages/kitchen", json=SPEC, headers={"ems-files": "address=caller.test:2224; secret=0th3r"})).json()["result"]
    assert again["made"] is True and len(launches.renderings) == 2


async def test_a_page_whose_stream_cannot_be_written_ends(client, launches):
    await client.put("/v0/pages/kitchen", json=SPEC)
    page = client.app.state.ctx.pages.get("kitchen")
    os.close(page.written["ts"])
    launches.renderings[0].feed(b"", TS[len(TS) * 3 // 4:])
    await asyncio.wait_for(page.ended.wait(), 2)
    assert page.status()["reason"].startswith("the ts stream could not be written")
    assert launches.renderings[0].stopped


async def test_a_browser_over_its_memory_or_gone_ends_the_page(client, launches):
    await client.put("/v0/pages/kitchen", json=SPEC)
    launches.renderings[0].memory = 10_000
    page = client.app.state.ctx.pages.get("kitchen")
    await asyncio.wait_for(page.ended.wait(), 2)
    assert "EMS_PAGE_MEMORY_MB" in page.status()["reason"]

    await client.put("/v0/pages/hall", json=SPEC)
    launches.renderings[1].processes["chromium"].returncode = 1
    hall = client.app.state.ctx.pages.get("hall")
    await asyncio.wait_for(hall.ended.wait(), 2)
    assert hall.status()["reason"] == "chromium exited with 1"


async def test_deleting_a_page_stops_it(client, launches):
    await client.put("/v0/pages/kitchen", json=SPEC)
    assert (await client.delete("/v0/pages/kitchen")).status_code == 200
    assert launches.renderings[0].stopped
    assert (await client.get("/v0/pages/kitchen")).status_code == 404


async def test_past_the_limit_a_new_page_is_busy(client, launches):
    await client.put("/v0/pages/a", json=SPEC)
    await client.put("/v0/pages/b", json=SPEC)
    r = await client.put("/v0/pages/c", json=SPEC)
    assert r.status_code == 503 and r.json()["error"]["code"] == "busy"


async def test_a_page_that_cannot_start_says_why(client, launches):
    client.app.state.ctx.pages.launch = Launches(fail=True)
    r = await client.put("/v0/pages/kitchen", json=SPEC)
    assert r.status_code == 500 and "chromium" in r.json()["error"]["message"]
    ended = (await client.get("/v0/pages")).json()["result"][0]
    assert ended["state"] == "ended" and "chromium" in ended["reason"]


async def test_a_page_with_no_picture_never_runs(client, launches, monkeypatch):
    monkeypatch.setattr(pages, "START_SECONDS", 0.2)

    async def blank(spec, page_id):
        return FakeRendering()

    client.app.state.ctx.pages.launch = blank
    r = await client.put("/v0/pages/kitchen", json=SPEC)
    assert r.status_code == 500 and "no picture" in r.json()["error"]["message"]


async def test_a_page_is_an_http_url(client, launches):
    r = await client.put("/v0/pages/kitchen", json={**SPEC, "url": "file:///etc/passwd"})
    assert r.status_code == 400


def test_xvfbs_display_is_read_up_to_its_newline_which_it_writes_apart():
    read, write = os.pipe()
    os.write(write, b"12")

    def newline():
        time.sleep(0.1)
        os.write(write, b"\n")
        os.close(write)

    thread = threading.Thread(target=newline)
    thread.start()
    # Closing the pipe after the number alone would end Xvfb at the newline.
    assert pages._display_told(read) == b"12\n"
    thread.join()
    os.close(read)


def test_the_encode_takes_the_picture_first_and_the_sound_as_aac_from_the_input_opened_first():
    args = encoder_args(PageSpec(url="http://ein.test/", output=SPEC["output"]), "h264_nvenc", 5)
    maps = [args[n + 1] for n, arg in enumerate(args) if arg == "-map"]
    # The picture is the first track, from the second input: the sound is opened first.
    assert maps == ["1:v", "0:a"]
    assert args[args.index("-c:a") + 1] == "aac" and args[args.index("-ar") + 1] == "48000"


def test_a_key_is_named_as_the_dom_names_it():
    assert key_event("ArrowRight") == ("ArrowRight", "ArrowRight", 39, "")
    assert key_event("Space") == (" ", "Space", 32, " ")
    assert key_event("k") == ("k", "KeyK", 75, "k")
    assert key_event("7") == ("7", "Digit7", 55, "7")
    for name in ("F13", "", "ö", "ArrowSideways"):
        with pytest.raises(EmsError):
            key_event(name)


async def test_keys_reach_a_running_page_and_an_unknown_key_is_refused(client, launches):
    await client.put("/v0/pages/kitchen", json=SPEC)
    r = await client.post("/v0/pages/kitchen/keys", json={"keys": ["ArrowRight", "Space"]})
    assert r.status_code == 200 and r.json()["result"] is None
    assert launches.renderings[0].pressed == ["ArrowRight", "Space"]
    assert (await client.post("/v0/pages/kitchen/keys", json={"keys": ["Hyper"]})).status_code == 400
    assert (await client.post("/v0/pages/kitchen/keys", json={"keys": []})).status_code == 400
    assert (await client.post("/v0/pages/elsewhere/keys", json={"keys": ["a"]})).status_code == 404
    await client.delete("/v0/pages/kitchen")
    assert (await client.post("/v0/pages/kitchen/keys", json={"keys": ["a"]})).status_code == 404


class FakeBrowser:
    """The browser's side of the DevTools pipe: it answers each call and keeps what it was sent."""

    def __init__(self):
        self.calls: list[dict] = []
        self.ours_in, self.its_out = os.pipe()
        self.its_in, self.ours_out = os.pipe()

    async def devtools(self) -> Devtools:
        loop = asyncio.get_running_loop()
        writer, _ = await loop.connect_write_pipe(asyncio.Protocol, os.fdopen(self.ours_out, "wb", buffering=0))
        self.task = asyncio.ensure_future(self._answer())
        return Devtools(await pages._pipe_reader(self.ours_in), writer)

    async def _answer(self) -> None:
        reader = await pages._pipe_reader(self.its_in)
        out = os.fdopen(self.its_out, "wb", buffering=0)
        while True:
            call = json.loads((await reader.readuntil(b"\0"))[:-1])
            self.calls.append(call)
            result = {}
            if call["method"] == "Target.getTargets":
                result = {"targetInfos": [{"type": "browser", "targetId": "b"}, {"type": "page", "targetId": "p1"}]}
            elif call["method"] == "Target.attachToTarget":
                result = {"sessionId": "s1"}
            # An event between the answers, which the client passes over.
            out.write(json.dumps({"method": "Target.targetInfoChanged", "params": {}}).encode() + b"\0")
            out.write(json.dumps({"id": call["id"], "result": result}).encode() + b"\0")


async def test_keys_go_through_devtools_to_the_page_attached_once():
    browser = FakeBrowser()
    devtools = await browser.devtools()
    await devtools.press(["ArrowLeft"])
    await devtools.press(["Space"])
    methods = [call["method"] for call in browser.calls]
    assert methods == ["Target.getTargets", "Target.attachToTarget", *["Input.dispatchKeyEvent"] * 4]
    assert browser.calls[1]["params"] == {"targetId": "p1", "flatten": True}
    down = browser.calls[4]
    assert down["sessionId"] == "s1" and down["params"] == {"type": "keyDown", "key": " ", "code": "Space", "windowsVirtualKeyCode": 32, "text": " "}
    assert browser.calls[5]["params"]["type"] == "keyUp" and "text" not in browser.calls[5]["params"]
    devtools.close()
    browser.task.cancel()


async def test_a_new_page_is_answered_once_a_keyframe_grabbed_after_its_paint_is_out(client, launches, monkeypatch):
    loaded = asyncio.Event()

    async def wait_loaded(self):
        await loaded.wait()
        return True

    monkeypatch.setattr(FakeRendering, "loaded", wait_loaded)
    real_launch = launches.__call__

    async def launch_earlier(spec, page_id):
        rendering = await real_launch(spec, page_id)
        rendering.grabbed_from = time.monotonic() - 1.0
        return rendering

    client.app.state.ctx.pages.launch = launch_earlier
    putting = asyncio.ensure_future(client.put("/v0/pages/kitchen", json=SPEC))
    await asyncio.sleep(0.2)
    assert not putting.done()
    loaded.set()
    await asyncio.sleep(0.2)
    # Loaded, but the keyframe the readers would start at is from before.
    assert not putting.done()
    page = client.app.state.ctx.pages.pages["kitchen"]
    page.outputs["ts"].keys += 1
    await asyncio.sleep(0.2)
    assert not putting.done()
    # A keyframe grabbed after the paint, two seconds into a stream whose grab began a second before it.
    mp4 = page.outputs["mp4"]
    at, width = pages._tfdts(mp4.key)[1]
    mp4.key = mp4.key[:at] + (2 * tracks(mp4.splitter.header)[1].timescale).to_bytes(width, "big") + mp4.key[at + width:]
    assert (await asyncio.wait_for(putting, 2)).status_code == 200


async def test_a_page_says_what_it_shows_and_takes_messages_when_it_can(client, launches):
    await client.put("/v0/pages/kitchen", json=SPEC)
    rendering = launches.renderings[0]
    assert (await client.get("/v0/pages/kitchen/state")).json()["result"] is None
    assert (await client.post("/v0/pages/kitchen/message", json={"message": {"seek": 80}})).json()["result"] == {"taken": False}
    rendering.state = {"title": "IMG_0001.jpg", "media": {"time": 3.5, "duration": 60, "paused": False}}
    assert (await client.get("/v0/pages/kitchen/state")).json()["result"] == rendering.state
    assert (await client.post("/v0/pages/kitchen/message", json={"message": {"seek": 80}})).json()["result"] == {"taken": True}
    assert rendering.told[-1] == ("message", {"seek": 80})
    assert (await client.post("/v0/pages/kitchen/message", json={"message": 5})).status_code == 400
    await client.delete("/v0/pages/kitchen")
    assert (await client.get("/v0/pages/kitchen/state")).status_code == 404


async def test_the_page_is_asked_through_devtools_with_the_argument_as_json():
    browser = FakeBrowser()
    devtools = await browser.devtools()
    await devtools.screen("message", {"seek": 12.5})
    evaluate = browser.calls[-1]
    assert evaluate["method"] == "Runtime.evaluate" and evaluate["sessionId"] == "s1"
    assert evaluate["params"]["returnByValue"] is True and evaluate["params"]["awaitPromise"] is True
    assert 'screen.message({"seek": 12.5})' in evaluate["params"]["expression"]
    devtools.close()
    browser.task.cancel()
