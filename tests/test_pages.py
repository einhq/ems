"""`pages`: ffmpeg's MP4 and TS cut where a reader may start, every reader served from one encode,
its HLS files served as ffmpeg writes them, and a stream that ends when its browser fails, when it
is deleted or when nobody reads it. The
rendering is stood in for; the fixtures are what ffmpeg's tee writes (two seconds at 15 fps, a
keyframe a second, a fragment every 200 ms):

    ffmpeg -f lavfi -i testsrc2=s=160x90:r=15 -t 2 -c:v libx264 -preset ultrafast -bf 0 -g 15 \
      -map 0:v -flags +global_header -f tee \
      "[f=mp4:movflags=frag_keyframe+empty_moov+default_base_moof:frag_duration=200000]page.mp4|[f=mpegts:bsfs/v=dump_extra=freq=keyframe]page.mpegts"
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile

import pytest

from ems import pages
from ems.errors import EmsError
from ems.pages import (CHROME_READ, QUICK_START_SECONDS, Mp4Splitter, PageSpec, Rendering, TsSplitter, decode_time, declared, encoder_args,
                       fragment_duration, quick_start)

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


def test_the_ts_is_cut_at_its_keyframes_and_keeps_its_tables():
    splitter = TsSplitter()
    pieces = [piece for at in range(0, len(TS), 1000) for piece in splitter.feed(TS[at:at + 1000])]
    assert b"".join(piece for _, piece in pieces) == TS
    assert splitter.header is not None and len(splitter.header) == 2 * 188
    keys = [piece for key, piece in pieces if key]
    assert len(keys) == 2
    assert all(piece[0] == 0x47 and piece[5] & 0x40 for piece in keys)


class FakeProcess:
    def __init__(self):
        self.returncode = None


class FakeRendering(Rendering):
    """Outputs fed by the test; its processes are never real."""

    def __init__(self):
        self.mp4 = asyncio.StreamReader()
        self.ts = asyncio.StreamReader()
        super().__init__({"chromium": FakeProcess(), "ffmpeg": FakeProcess()}, {"mp4": self.mp4, "ts": self.ts}, "h264_nvenc",
                         hls=tempfile.mkdtemp(prefix="ems-page-hls-"))
        self.memory = 300.0
        self.stopped = False

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


@pytest.fixture
def launches(client, monkeypatch):
    monkeypatch.setattr(pages, "WATCH_SECONDS", 0.05)
    found = Launches()
    client.app.state.ctx.pages.launch = found
    return found


SPEC = {"url": "http://ein.test:8080/__page/abc", "video": {"width": 1280, "height": 720, "fps": 30}, "idle": 5}


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


async def test_every_reader_starts_at_the_latest_keyframe_with_its_clock_at_zero(client, launches):
    await client.put("/v0/pages/kitchen", json=SPEC)
    page = client.app.state.ctx.pages.get("kitchen")
    channel, reader = page.join("mp4")
    chunks = channel.chunks(reader)
    header = await anext(chunks)
    assert header[4:8] == b"ftyp"
    first = await anext(chunks)
    assert first[4:8] == b"moof" and decode_time(first) == 0
    assert decode_time(channel.gop[0]) > 0
    second = await anext(chunks)
    assert decode_time(second) > 0
    page.leave(channel, reader)

    ts_channel, ts_reader = page.join("ts")
    start = await ts_reader.next()
    assert start == ts_channel.splitter.header
    key = await ts_reader.next()
    assert key[5] & 0x40
    assert page.status()["readers"] == {"mp4": 0, "ts": 1, "hls": 0}
    page.leave(ts_channel, ts_reader)


async def test_a_live_route_sends_the_header_then_the_stream_until_the_page_ends(client, launches):
    await client.put("/v0/pages/kitchen", json=SPEC)
    page = client.app.state.ctx.pages.get("kitchen")
    header = page.channels["ts"].splitter.header
    # The test client answers once the body is whole, so the page is deleted while it is read.
    reading = asyncio.ensure_future(client.get("/v0/pages/kitchen/live.ts"))
    for _ in range(100):
        if page.readers():
            break
        await asyncio.sleep(0.01)
    assert page.status()["readers"] == {"mp4": 0, "ts": 1, "hls": 0}
    await client.delete("/v0/pages/kitchen")
    r = await asyncio.wait_for(reading, 2)
    assert r.headers["content-type"] == "video/mp2t"
    assert r.headers["access-control-allow-origin"] == "*"
    assert r.content.startswith(header) and r.content[len(header) + 5] & 0x40
    assert page.readers() == 0


def test_a_declared_header_gains_an_mehd_once():
    splitter = Mp4Splitter()
    splitter.feed(MP4)
    header = declared(splitter.header, 86400)
    assert len(header) == len(splitter.header) + 16 and header.count(b"mehd") == 1
    assert declared(header, 86400) == header


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe is not installed")
def test_a_declared_header_says_the_stream_lasts_that_long_where_a_player_reads_it(tmp_path):
    splitter = Mp4Splitter()
    splitter.feed(MP4)
    header = declared(splitter.header, 86400)
    path = tmp_path / "declared.mp4"
    path.write_bytes(header + MP4[len(splitter.header):])
    # ffprobe reads the header's durations; the fixture's own header says nothing.
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=duration", "-of", "json", str(path)],
                           capture_output=True, text=True, check=True)
    found = json.loads(probe.stdout)
    assert float(found["format"]["duration"]) == 86400 and float(found["streams"][0]["duration"]) == 86400


async def test_a_reader_of_the_mp4_may_ask_for_its_header_to_declare_a_duration(client, launches):
    await client.put("/v0/pages/kitchen", json=SPEC)
    page = client.app.state.ctx.pages.get("kitchen")
    channel, reader = page.join("mp4")
    chunks = channel.chunks(reader, 86400)
    assert await anext(chunks) == declared(channel.splitter.header, 86400)
    page.leave(channel, reader)
    assert (await client.get("/v0/pages/kitchen/live.mp4?duration=5")).status_code == 400


def _stream() -> tuple[bytes, list[bytes]]:
    """The fixture's declared header and its fragments, their clock starting at 0, as a reader is sent them."""
    splitter = Mp4Splitter()
    fragments = [piece for _, piece in splitter.feed(MP4)]
    base = decode_time(fragments[0])
    return declared(splitter.header, 86400), [pages.rebased(fragment, base) for fragment in fragments]


def _second(header: bytes) -> int:
    at = header.index(b"mdhd") - 4
    return int(QUICK_START_SECONDS * int.from_bytes(header[at + (28 if header[at + 8] == 1 else 20):][:4], "big"))


def test_a_quick_start_holds_a_second_and_fills_out_chromes_read():
    header, fragments = _stream()
    start, shift = quick_start(header, fragments[:1])
    first = fragment_duration(fragments[0], header)
    repeated = [piece for piece in start if piece[4:8] == b"moof"]
    assert start[0] == header and start[-1][4:8] == b"free"
    assert sum(len(piece) for piece in start) % CHROME_READ == 0
    # The keyframe's fragment again and again straight after itself, a second of it in all.
    assert all(pages.at_decode_time(piece, 0) == fragments[0] for piece in repeated)
    assert [decode_time(piece) for piece in repeated] == [n * first for n in range(len(repeated))]
    assert len(repeated) * first >= _second(header) > (len(repeated) - 1) * first
    assert shift == (len(repeated) - 1) * first


def test_a_quick_start_leaves_what_already_holds_a_second_as_it_is():
    header, fragments = _stream()
    held = [f for n, f in enumerate(fragments) if sum(fragment_duration(g, header) for g in fragments[:n]) < _second(header)]
    start, shift = quick_start(header, held)
    assert shift == 0 and start[1:1 + len(held)] == held


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")
def test_a_quick_start_and_what_follows_decode_as_one_stream(tmp_path):
    header, fragments = _stream()
    start, shift = quick_start(header, fragments[:1])
    rest = [pages.rebased(fragment, -shift) for fragment in fragments[1:]]
    path = tmp_path / "quick.mp4"
    path.write_bytes(b"".join([*start, *rest]))
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    assert decoded.returncode == 0 and decoded.stderr == ""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "packet=pts", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, check=True)
    times = [int(line) for line in probe.stdout.split()]
    assert len(times) > len(fragments) and times == sorted(set(times))


async def test_a_reader_that_asks_for_a_duration_is_given_a_quick_start_and_the_rest_after_it(client, launches):
    await client.put("/v0/pages/kitchen", json=SPEC)
    page = client.app.state.ctx.pages.get("kitchen")
    channel, reader = page.join("mp4")
    base = decode_time(channel.gop[0])
    expected, shift = quick_start(declared(channel.splitter.header, 86400), [pages.rebased(f, base) for f in channel.gop])
    chunks = channel.chunks(reader, 86400)
    assert [await anext(chunks) for _ in expected] == expected
    splitter = Mp4Splitter()
    fragments = [piece for _, piece in splitter.feed(MP4)]
    after = fragments[fragments.index(channel.gop[-1]) + 1]
    launches.renderings[0].feed(MP4[len(MP4) * 3 // 4:], b"")
    following = await asyncio.wait_for(anext(chunks), 2)
    assert following == pages.rebased(after, base - shift)
    page.leave(channel, reader)


def test_the_hls_is_written_beside_the_two_streams_from_the_one_encode():
    args = encoder_args(PageSpec(url="http://ein.test/"), "h264_nvenc", 5, "/scratch/page:1/hls")
    tee = args[-1]
    assert args[-2:-1] == ["tee"] and tee.count("|") == 2
    hls = tee.split("|")[2]
    assert hls.startswith("[f=hls:bsfs/v=dump_extra=freq=keyframe:hls_time=1:")
    assert "delete_segments" in hls and hls.endswith("]/scratch/page\\:1/hls/index.m3u8")


async def test_the_hls_files_are_served_as_ffmpeg_writes_them_and_count_as_a_reader(client, launches, monkeypatch):
    monkeypatch.setattr(pages, "HLS_START_SECONDS", 2.0)
    await client.put("/v0/pages/kitchen", json=SPEC)
    page = client.app.state.ctx.pages.get("kitchen")
    folder = launches.renderings[0].hls
    playlist = b"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1.000000,\nindex7.ts\n"

    async def write_later():
        await asyncio.sleep(0.3)
        open(os.path.join(folder, "index7.ts"), "wb").write(TS[:188 * 4])
        open(os.path.join(folder, "index.m3u8"), "wb").write(playlist)

    # A playlist asked for before ffmpeg has written it is waited for.
    writing = asyncio.ensure_future(write_later())
    r = await client.get("/v0/pages/kitchen/hls/index.m3u8")
    await writing
    assert r.status_code == 200 and r.content == playlist
    assert r.headers["content-type"] == "application/vnd.apple.mpegurl"
    assert r.headers["access-control-allow-origin"] == "*"
    assert page.status()["readers"] == {"mp4": 0, "ts": 0, "hls": 1}
    segment = await client.get("/v0/pages/kitchen/hls/index7.ts")
    assert segment.headers["content-type"] == "video/mp2t" and segment.content == TS[:188 * 4]
    assert (await client.get("/v0/pages/kitchen/hls/index6.ts")).status_code == 404
    assert (await client.get("/v0/pages/kitchen/hls/..%2Fprofile")).status_code == 404
    await client.delete("/v0/pages/kitchen")
    page.hls_read = None
    assert (await client.get("/v0/pages/kitchen/hls/index.m3u8")).status_code == 404


async def test_an_hls_player_keeps_a_page_from_idling(client, launches, monkeypatch):
    await client.put("/v0/pages/kitchen", json=SPEC)
    page = client.app.state.ctx.pages.get("kitchen")
    open(os.path.join(launches.renderings[0].hls, "index.m3u8"), "wb").write(b"#EXTM3U\n")
    page.unread_since -= 10
    await client.get("/v0/pages/kitchen/hls/index.m3u8")
    await asyncio.sleep(0.2)
    assert page.ending is None


async def test_a_page_nobody_reads_for_its_idle_seconds_ends(client, launches, monkeypatch):
    await client.put("/v0/pages/kitchen", json=SPEC)
    page = client.app.state.ctx.pages.get("kitchen")
    page.unread_since -= 10
    await asyncio.wait_for(page.ended.wait(), 2)
    assert page.status()["reason"] == "idle"
    assert launches.renderings[0].stopped
    r = await client.get("/v0/pages/kitchen/live.mp4")
    assert r.status_code == 404 and "idle" in r.json()["error"]["message"]


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


async def test_deleting_a_page_stops_it_and_its_readers(client, launches):
    await client.put("/v0/pages/kitchen", json=SPEC)
    page = client.app.state.ctx.pages.get("kitchen")
    _, reader = page.join("mp4")
    assert (await client.delete("/v0/pages/kitchen")).status_code == 200
    assert reader.ended and launches.renderings[0].stopped
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
