"""`video.transcode` and `audio.transcode`: the ladder, the windows, the refusal and the
publish, with ffmpeg and ffprobe stood in for. The encoders themselves run in the container."""

import os

from ems.ops import transcode
from ems.tools import ffprobe as ffprobe_tool
from ems.tools import run as run_tool


def fake_probe(duration="10.0", fps="25/1", codec="h264", audio=True, width=1920, height=1080):
    streams = [{"codec_type": "video", "codec_name": codec, "r_frame_rate": fps, "width": width, "height": height}]
    if audio:
        streams.append({"codec_type": "audio", "codec_name": "aac", "bit_rate": "128000", "sample_rate": "44100", "channels": 2})
    return ffprobe_tool.Probe({"format": {"duration": duration, "format_name": "mov,mp4"}, "streams": streams}, None)


class FakeFfmpeg:
    """Scripted ffmpeg: which argument makes a run fail, what it writes otherwise."""

    def __init__(self, fail_when=()):
        self.fail_when = tuple(fail_when)
        self.runs: list[list[str]] = []

    async def streaming(self, args, *, timeout, on_line, env=None):
        self.runs.append(list(args))
        out = args[-1]
        if any(marker in " ".join(args) for marker in self.fail_when):
            return run_tool.Completed(list(args), 218, b"", b"Impossible to convert between the formats\n")
        await on_line("out_time_us=5000000")
        with open(out, "wb") as f:
            f.write(b"mp4" * 100)
        return run_tool.Completed(list(args), 0, b"", b"")

    async def run(self, args, *, timeout, **kw):
        self.runs.append(list(args))
        if "-encoders" in args:
            return run_tool.Completed(list(args), 0, b" V..... av1_nvenc  NVIDIA NVENC av1 encoder\n", b"")
        if "concat" in args:
            with open(args[-1], "wb") as f:
                f.write(b"joined" * 100)
            return run_tool.Completed(list(args), 0, b"", b"")
        return run_tool.Completed(list(args), 0, b"", b"")


def install(monkeypatch, ffmpeg: FakeFfmpeg, probe=None, gpu=True):
    probes = {"count": 0}

    async def probing(path, *, timeout):
        probes["count"] += 1
        return probe or fake_probe()

    monkeypatch.setattr(transcode, "run_streaming", ffmpeg.streaming)
    monkeypatch.setattr(transcode, "run", ffmpeg.run)
    monkeypatch.setattr(transcode, "require", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(transcode.ffprobe_tool, "probe", probing)
    monkeypatch.setattr(transcode.ffmpeg_tool, "gpu_present", lambda: gpu)
    transcode._encoder_cache.clear()
    return probes


def video_body(ro, rw, **extra):
    body = {"file": {"path": str(ro / "clip.mp4"), "angle": 90}, "output": {"path": str(rw / "1920x1080.mp4"), "format": "mp4"}, "video": {"box": {"width": 1920, "height": 1080}, "quality": 32, "speed": 8}}
    body.update(extra)
    return body


async def test_stage_one_keeps_everything_in_vram(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "clip.mp4").write_bytes(b"mp4")
    ffmpeg = FakeFfmpeg()
    install(monkeypatch, ffmpeg)
    r = await client.post("/v0/video/transcode", json=video_body(ro, rw))
    body = r.json()
    assert r.status_code == 200, body
    encodes = [run for run in ffmpeg.runs if "-progress" in run]
    assert len(encodes) == 1
    args = encodes[0]
    assert args[args.index("-hwaccel") + 1] == "cuda" and "-display_rotation" in args
    assert args[args.index("-vf") + 1].endswith("transpose_cuda=dir=2")
    assert args[args.index("-cq") + 1] == "38" and args[args.index("-c:a") + 1] == "aac"
    assert args[-1].startswith(str(rw / ".ems-")) and args[-1].endswith(".mp4")
    assert (rw / "1920x1080.mp4").read_bytes() == b"mp4" * 100
    assert not [f for f in os.listdir(rw) if f.startswith(".ems-")]
    assert body["result"]["path"] == str(rw / "1920x1080.mp4") and body["result"]["bytes"] == 300
    assert body["result"]["video_codec"] == "h264" and body["result"]["width"] == 1920
    assert body["meta"]["producer"].endswith("/av1_nvenc/cuda")


async def test_a_codec_nvdec_cannot_read_skips_to_the_upload_stage(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "clip.mp4").write_bytes(b"mp4")
    ffmpeg = FakeFfmpeg()
    install(monkeypatch, ffmpeg)
    r = await client.post("/v0/video/transcode", json=video_body(ro, rw, hints={"source_codec": "dvvideo"}))
    assert r.status_code == 200, r.json()
    encodes = [run for run in ffmpeg.runs if "-progress" in run]
    assert len(encodes) == 1 and "-hwaccel" not in encodes[0]
    assert encodes[0][encodes[0].index("-vf") + 1].endswith("hwupload_cuda")
    assert r.json()["meta"]["producer"].endswith("/av1_nvenc/upload")


async def test_the_ladder_falls_to_the_cpu_and_a_mirror_never_tries_cuda(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "clip.mp4").write_bytes(b"mp4")
    ffmpeg = FakeFfmpeg(fail_when=("av1_nvenc",))
    install(monkeypatch, ffmpeg)
    r = await client.post("/v0/video/transcode", json=video_body(ro, rw, file={"path": str(ro / "clip.mp4"), "mirror": True}))
    assert r.status_code == 200, r.json()
    encodes = [run for run in ffmpeg.runs if "-progress" in run]
    assert [("-hwaccel" in e, "av1_nvenc" in e, "libsvtav1" in e) for e in encodes] == [(False, True, False), (False, False, True)]
    assert encodes[-1][encodes[-1].index("-vf") + 1] == "scale=w=min(iw\\,1920):h=min(ih\\,1080):force_original_aspect_ratio=decrease:force_divisible_by=2,hflip"
    assert r.json()["meta"]["producer"].endswith("/libsvtav1")


async def test_a_muted_clip_is_cut_before_encoding(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "clip.mp4").write_bytes(b"mp4")
    ffmpeg = FakeFfmpeg()
    install(monkeypatch, ffmpeg, gpu=False)
    r = await client.post("/v0/video/transcode", json=video_body(ro, rw, audio=None, clip={"start": 1, "duration": 20}))
    assert r.status_code == 200, r.json()
    args = [run for run in ffmpeg.runs if "-progress" in run][0]
    assert args.index("-ss") < args.index("-i") and args[args.index("-ss") + 1] == "1"
    assert args[args.index("-t") + 1] == "20" and "-an" in args and "libsvtav1" in args


async def test_over_budget_is_windowed_with_one_audio_pass_and_a_join(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "tape.dv").write_bytes(b"dv")
    ffmpeg = FakeFfmpeg()
    install(monkeypatch, ffmpeg, probe=fake_probe(duration="5040.0", codec="dvvideo"))
    r = await client.post("/v0/video/transcode", json={"file": {"path": str(ro / "tape.dv")}, "output": {"path": str(rw / "1920x1080.mp4")}, "video": {"box": {"width": 1920, "height": 1080}}})
    body = r.json()
    assert r.status_code == 200, body
    encodes = [run for run in ffmpeg.runs if "-progress" in run]
    audio = [e for e in encodes if "-vn" in e]
    windows = [e for e in encodes if "-an" in e]
    assert len(audio) == 1 and len(windows) == 6
    assert [w[w.index("-ss") + 1] for w in windows] == ["0", "840", "1680", "2520", "3360", "4200"]
    assert [w[w.index("-t") + 1] if "-t" in w else None for w in windows] == ["840"] * 5 + [None]
    assert all("av1_nvenc" in w and "hwupload_cuda" in w[w.index("-vf") + 1] for w in windows)
    join = [run for run in ffmpeg.runs if "concat" in run][0]
    assert join[join.index("-map") + 1] == "0:v:0" and "1:a:0" in join and "copy" in join
    assert (rw / "1920x1080.mp4").read_bytes() == b"joined" * 100
    assert sorted(os.listdir(rw)) == ["1920x1080.mp4"]
    assert body["meta"]["producer"].endswith("/av1_nvenc/upload")


async def test_a_source_whose_windows_cannot_be_placed_is_refused(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "tape.dv").write_bytes(b"dv")
    ffmpeg = FakeFfmpeg()
    install(monkeypatch, ffmpeg, probe=fake_probe(duration="5040.0", codec="dvvideo"))
    r = await client.post("/v0/video/transcode", json={"file": {"path": str(ro / "tape.dv")}, "output": {"path": str(rw / "x.mp4")}, "clip": {"start": 0.5}})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "refused" and r.json()["error"]["permanent"] is True
    assert "whole seconds" in r.json()["error"]["message"]
    assert os.listdir(rw) == [] and not [run for run in ffmpeg.runs if "-progress" in run]

    unmeasurable = fake_probe(duration="N/A", codec="dvvideo")
    install(monkeypatch, ffmpeg, probe=unmeasurable)
    r = await client.post("/v0/video/transcode", json={"file": {"path": str(ro / "tape.dv")}, "output": {"path": str(rw / "x.mp4")}})
    assert r.status_code == 422 and "cannot measure" in r.json()["error"]["message"]


async def test_a_failed_encode_leaves_nothing_and_is_not_permanent(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "clip.mp4").write_bytes(b"mp4")
    ffmpeg = FakeFfmpeg(fail_when=("-progress",))
    install(monkeypatch, ffmpeg, gpu=False)
    r = await client.post("/v0/video/transcode", json=video_body(ro, rw))
    assert r.status_code == 500 and r.json()["error"]["code"] == "tool_failed" and r.json()["error"]["permanent"] is False
    assert os.listdir(rw) == []


async def test_audio_transcode_keeps_the_ceilings_and_publishes(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "tape.mp3").write_bytes(b"mp3")
    ffmpeg = FakeFfmpeg()
    source = ffprobe_tool.Probe({"format": {"duration": "2756.3", "format_name": "mp3"}, "streams": [{"codec_type": "audio", "codec_name": "mp3", "bit_rate": "64000", "sample_rate": "44100", "channels": 1}]}, None)
    install(monkeypatch, ffmpeg, probe=source)
    r = await client.post("/v0/audio/transcode", json={"file": {"path": str(ro / "tape.mp3")}, "output": {"path": str(rw / "128k-44100.m4a")}, "audio": {"bitrate": "128k", "sample_rate": 44100}})
    body = r.json()
    assert r.status_code == 200, body
    args = [run for run in ffmpeg.runs if "-progress" in run][0]
    assert args[args.index("-b:a") + 1] == "64000" and args[args.index("-ac") + 1] == "1" and args[-3:-1] == ["-f", "ipod"]
    assert (rw / "128k-44100.m4a").exists() and body["result"]["bytes"] == 300 and body["result"]["remuxed"] is False
    assert body["result"]["bitrate"] == 64000 and body["result"]["channels"] == 1
    assert body["meta"]["producer"].endswith("/aac")

    silent = ffprobe_tool.Probe({"format": {"duration": "2.0", "format_name": "mp4"}, "streams": [{"codec_type": "video", "codec_name": "h264"}]}, None)
    install(monkeypatch, ffmpeg, probe=silent)
    r = await client.post("/v0/audio/transcode", json={"file": {"path": str(ro / "tape.mp3")}, "output": {"path": str(rw / "x.m4a")}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "refused"


async def test_capabilities_name_the_encoder_and_the_budget(client, monkeypatch):
    ffmpeg = FakeFfmpeg()
    install(monkeypatch, ffmpeg, gpu=False)
    r = await client.get("/v0/capabilities")
    caps = r.json()
    assert caps["encode"] == {"video_encoder": "libsvtav1", "cpu_budget": "4000/170/64"}
    assert caps["limits"]["encodes"] == 2
    assert "video.transcode" in caps["ops"] and "audio.transcode" in caps["ops"]
