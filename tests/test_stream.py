"""`video.stream`: the encoder chosen before the first byte, the body passed on as ffmpeg writes it,
and the encoder and its slot let go however the stream ends. ffmpeg is stood in for."""

import asyncio
import os
from collections import OrderedDict

from ems.ops import stream, transcode
from ems.tools import ffprobe as ffprobe_tool
from ems.tools import run as run_tool


def fake_probe(codec="h264", audio=True):
    streams = [{"index": 0, "codec_type": "video", "codec_name": codec, "r_frame_rate": "25/1", "width": 3840, "height": 2160}]
    if audio:
        streams.append({"index": 1, "codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 6})
    return ffprobe_tool.Probe({"format": {"duration": "60.0"}, "streams": streams}, None)


class FakeEncoding:
    def __init__(self, chunks: list[bytes], errors: str):
        self.chunks = list(chunks)
        self.errors_text = errors
        self.stopped = False
        self.process = type("P", (), {"returncode": None if chunks else 1})()

    async def read(self) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""

    def tail(self) -> str:
        return self.errors_text

    async def stop(self) -> None:
        self.stopped = True


class FakeFfmpeg:
    """Each stage whose arguments hold a marker in `fail_when` writes nothing; the others write `chunks`."""

    def __init__(self, fail_when=(), chunks=(b"ftyp", b"moof", b"mdat")):
        self.fail_when = tuple(fail_when)
        self.chunks = list(chunks)
        self.runs: list[list[str]] = []
        self.encodings: list[FakeEncoding] = []

    async def spawn(self, args):
        self.runs.append(list(args))
        failing = any(marker in " ".join(args) for marker in self.fail_when)
        encoding = FakeEncoding([] if failing else self.chunks, "Cannot load libcuda.so.1" if failing else "")
        self.encodings.append(encoding)
        return encoding


def install(monkeypatch, ffmpeg: FakeFfmpeg, probe=None, encoders="h264_nvenc libopenh264", gpu=True):
    async def probing(path, *, timeout):
        return probe or fake_probe()

    async def listing(args, *, timeout, **kw):
        listed = "".join(f" V....D {name}  an encoder\n" for name in encoders.split())
        return run_tool.Completed(list(args), 0, listed.encode(), b"")

    monkeypatch.setattr(stream, "spawn", ffmpeg.spawn)
    monkeypatch.setattr(stream, "run", listing)
    monkeypatch.setattr(stream, "require", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(stream, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(stream.ffprobe_tool, "probe", probing)
    monkeypatch.setattr(stream.ffmpeg_tool, "gpu_present", lambda: gpu)
    monkeypatch.setattr(stream, "_encoders", {})
    monkeypatch.setattr(stream, "_probes", OrderedDict())


def body(ro, **extra):
    request = {"file": {"path": str(ro / "clip.mov"), "angle": 90}}
    request.update(extra)
    return request


async def test_the_stream_is_ffmpegs_output_from_vram(client, dirs, monkeypatch):
    ro, _ = dirs
    (ro / "clip.mov").write_bytes(b"mov")
    ffmpeg = FakeFfmpeg()
    install(monkeypatch, ffmpeg)
    r = await client.post("/v0/video/stream", json=body(ro, start=95.5))
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "video/mp4" and r.headers["cache-control"] == "no-store"
    assert r.headers["ems-producer"].endswith("/h264_nvenc/cuda")
    assert r.content == b"ftypmoofmdat"
    [args] = ffmpeg.runs
    assert args[args.index("-hwaccel") + 1] == "cuda" and args[args.index("-ss") + 1] == "95.5"
    assert args.index("-ss") < args.index("-i")
    assert args[args.index("-vf") + 1] == "scale_cuda=w=min(iw\\,1920):h=min(ih\\,1080):force_original_aspect_ratio=decrease:force_divisible_by=2:format=yuv420p,transpose_cuda=dir=2"
    assert args[args.index("-c:v") + 1] == "h264_nvenc" and args[args.index("-profile:v") + 1] == "high"
    assert args[args.index("-fpsmax") + 1] == "30" and args[args.index("-ac") + 1] == "2"
    assert "frag_keyframe+empty_moov+default_base_moof" in args and args[-1] == "pipe:1"
    assert ffmpeg.encodings[0].stopped
    assert client.app.state.ctx.settings.streams == 2


async def test_a_segment_is_mpegts_of_a_set_length_on_the_sources_clock(client, dirs, monkeypatch):
    ro, _ = dirs
    (ro / "clip.mov").write_bytes(b"mov")
    ffmpeg = FakeFfmpeg()
    install(monkeypatch, ffmpeg)
    r = await client.post("/v0/video/stream", json=body(ro, start=8, duration=4, source_clock=True, format="mpegts"))
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "video/mp2t"
    [args] = ffmpeg.runs
    assert args[args.index("-ss") + 1] == "8" and args[args.index("-t") + 1] == "4" and args.index("-t") > args.index("-i")
    assert args[args.index("-output_ts_offset") + 1] == "8"
    assert args[-5:] == ["-muxdelay", "0", "-f", "mpegts", "pipe:1"]
    assert "frag_keyframe+empty_moov+default_base_moof" not in args
    install(monkeypatch, ffmpeg := FakeFfmpeg())
    await client.post("/v0/video/stream", json=body(ro, start=8))
    assert "-output_ts_offset" not in ffmpeg.runs[0] and "-t" not in ffmpeg.runs[0]
    r = await client.post("/v0/video/stream", json=body(ro, format="webm"))
    assert r.status_code == 400


async def test_a_stage_that_writes_nothing_falls_to_the_next(client, dirs, monkeypatch):
    ro, _ = dirs
    (ro / "clip.mov").write_bytes(b"mov")
    ffmpeg = FakeFfmpeg(fail_when=("h264_nvenc",))
    install(monkeypatch, ffmpeg)
    r = await client.post("/v0/video/stream", json=body(ro, audio=None))
    assert r.status_code == 200, r.text
    assert r.headers["ems-producer"].endswith("/libopenh264")
    assert [args[args.index("-c:v") + 1] for args in ffmpeg.runs] == ["h264_nvenc", "h264_nvenc", "libopenh264"]
    assert "-hwaccel" not in ffmpeg.runs[1] and ffmpeg.runs[1][ffmpeg.runs[1].index("-vf") + 1].endswith("hwupload_cuda")
    assert "-an" in ffmpeg.runs[2] and ffmpeg.runs[2][ffmpeg.runs[2].index("-vf") + 1].endswith(",format=yuv420p")
    assert all(encoding.stopped for encoding in ffmpeg.encodings)


async def test_a_codec_nvdec_cannot_read_starts_with_the_upload(client, dirs, monkeypatch):
    ro, _ = dirs
    (ro / "clip.mov").write_bytes(b"mov")
    ffmpeg = FakeFfmpeg()
    install(monkeypatch, ffmpeg, probe=fake_probe(codec="prores"))
    r = await client.post("/v0/video/stream", json=body(ro))
    assert r.status_code == 200 and r.headers["ems-producer"].endswith("/h264_nvenc/upload")


async def test_no_stage_starting_is_an_error_envelope(client, dirs, monkeypatch):
    ro, _ = dirs
    (ro / "clip.mov").write_bytes(b"mov")
    ffmpeg = FakeFfmpeg(fail_when=("-c:v",))
    install(monkeypatch, ffmpeg)
    r = await client.post("/v0/video/stream", json=body(ro))
    assert r.status_code == 500
    error = r.json()["error"]
    assert error["code"] == "tool_failed" and error["permanent"] is False and "Cannot load libcuda" in error["message"]
    assert len(ffmpeg.runs) == 3


async def test_a_file_without_a_picture_is_refused_and_a_host_without_h264_says_so(client, dirs, monkeypatch):
    ro, _ = dirs
    (ro / "clip.mov").write_bytes(b"mov")
    install(monkeypatch, FakeFfmpeg(), probe=ffprobe_tool.Probe({"format": {}, "streams": [{"index": 0, "codec_type": "audio", "codec_name": "aac"}]}, None))
    r = await client.post("/v0/video/stream", json=body(ro))
    assert r.status_code == 422 and r.json()["error"]["code"] == "refused"
    install(monkeypatch, FakeFfmpeg(), encoders="libsvtav1", gpu=False)
    r = await client.post("/v0/video/stream", json=body(ro))
    assert r.status_code == 500 and "no H.264 encoder" in r.json()["error"]["message"]


class EndlessEncoding(FakeEncoding):
    """Writes its first bytes, then nothing until it is stopped, as an encoder held by a paused reader."""

    def __init__(self):
        super().__init__([b"ftyp"], "")
        self.halt = asyncio.Event()

    async def read(self) -> bytes:
        if self.chunks:
            return self.chunks.pop(0)
        await self.halt.wait()
        return b""


async def test_streams_past_the_limit_are_busy_and_a_closed_one_frees_its_slot(client, dirs, monkeypatch):
    ro, _ = dirs
    (ro / "clip.mov").write_bytes(b"mov")
    encodings: list[EndlessEncoding] = []

    async def spawn(args):
        encodings.append(EndlessEncoding())
        return encodings[-1]

    install(monkeypatch, FakeFfmpeg())
    monkeypatch.setattr(stream, "spawn", spawn)
    held = [asyncio.create_task(client.post("/v0/video/stream", json=body(ro))) for _ in range(2)]
    while len(encodings) < 2:
        await asyncio.sleep(0.01)
    r = await client.post("/v0/video/stream", json=body(ro))
    assert r.status_code == 503 and r.json()["error"]["code"] == "busy" and r.headers["retry-after"] == "5"
    held[0].cancel()
    await asyncio.gather(held[0], return_exceptions=True)
    # The server notices the reader gone on its own task, a moment after the client lets go. Which
    # request started which encoding is the scheduler's choice: the one let go is the one stopped.
    for _ in range(200):
        if any(e.stopped for e in encodings):
            break
        await asyncio.sleep(0.01)
    gone, kept = sorted(encodings, key=lambda e: not e.stopped)
    assert gone.stopped and not kept.stopped
    kept.halt.set()
    assert (await held[1]).content == b"ftyp" and kept.stopped
    install(monkeypatch, FakeFfmpeg())
    r = await client.post("/v0/video/stream", json=body(ro))
    assert r.status_code == 200


async def test_a_path_the_caller_lacks_is_not_found_and_holds_no_slot(client, dirs, monkeypatch):
    ro, _ = dirs
    install(monkeypatch, FakeFfmpeg())
    for _ in range(3):
        r = await client.post("/v0/video/stream", json=body(ro))
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


async def test_capabilities_list_the_stream_encoders_and_the_limit(client, monkeypatch):
    install(monkeypatch, FakeFfmpeg())
    monkeypatch.setitem(transcode._encoder_cache, "auto", "av1_nvenc")
    body = (await client.get("/v0/capabilities")).json()
    assert body["encode"]["stream_encoders"] == ["h264_nvenc", "libopenh264"] and body["limits"]["streams"] == 2


async def test_a_files_probe_is_kept_across_requests_while_it_stays_the_same(tmp_path, monkeypatch):
    probed: list[str] = []

    async def probing(path, *, timeout):
        probed.append(path)
        return fake_probe() if "bad" not in path else ffprobe_tool.Probe(None, "moov atom not found")

    monkeypatch.setattr(stream.ffprobe_tool, "probe", probing)
    monkeypatch.setattr(stream, "_probes", OrderedDict())
    mount = str(tmp_path)
    for grant in ("a", "b", "c", "d"):
        (tmp_path / grant).mkdir()
    # Granted in a directory of its own at each request, the same file is probed once.
    for grant in ("a", "b"):
        (tmp_path / grant / "0-clip.mov").write_bytes(b"mov")
        os.utime(tmp_path / grant / "0-clip.mov", (1_700_000_000, 1_700_000_000))
        assert (await stream.probed(mount, str(tmp_path / grant / "0-clip.mov"), 5)).first("video")
    assert probed == [str(tmp_path / "a" / "0-clip.mov")]
    # Changed, it is probed again; another caller's file of the same name is its own.
    (tmp_path / "c" / "0-clip.mov").write_bytes(b"mov, longer")
    await stream.probed(mount, str(tmp_path / "c" / "0-clip.mov"), 5)
    await stream.probed(str(tmp_path / "c"), str(tmp_path / "a" / "0-clip.mov"), 5)
    assert len(probed) == 3
    # One that could not be read is asked again.
    (tmp_path / "d" / "0-bad.mov").write_bytes(b"bad")
    for _ in range(2):
        assert (await stream.probed(mount, str(tmp_path / "d" / "0-bad.mov"), 5)).refusal
    assert len(probed) == 5
