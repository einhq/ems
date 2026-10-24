"""`video.storyboard`: the sheet's plan, the decode ladder, and the op with ffmpeg and magick stubbed."""

import os
import shutil

import pytest

from ems.ops import storyboard
from ems.tools import ffprobe as ffprobe_tool
from ems.tools import magick
from ems.tools import run as run_tool


def plan(duration, width, height, sar=None, **kw):
    options = {"tile": 160, "columns": 12, "max_tiles": 240, "min_interval": 1.0, **kw}
    return storyboard.storyboard_plan(duration, width, height, sar, **options)


def test_a_tile_is_the_display_shape_with_its_longer_side_the_tile():
    # PAL DV: 720x576 stored, 16:15 pixels, a 4:3 picture.
    assert (plan(180, 720, 576, 16 / 15).tile_width, plan(180, 720, 576, 16 / 15).tile_height) == (160, 120)
    assert (plan(60, 1920, 1080).tile_width, plan(60, 1920, 1080).tile_height) == (160, 90)
    # A phone held upright.
    assert (plan(60, 1080, 1920).tile_width, plan(60, 1080, 1920).tile_height) == (90, 160)


def test_tile_sides_are_even():
    shape = plan(60, 1000, 333)
    assert shape.tile_width % 2 == 0 and shape.tile_height % 2 == 0


def test_a_short_video_gets_a_frame_a_second_and_a_long_one_a_wider_interval():
    film = plan(181.2, 720, 576, 16 / 15)
    assert (film.interval, film.count, film.columns, film.rows) == (1.0, 182, 12, 16)
    tape = plan(7200, 720, 576, 16 / 15)
    assert (tape.interval, tape.count, tape.rows) == (30.0, 240, 20)


def test_a_clip_shorter_than_a_row_uses_only_the_columns_it_fills():
    clip = plan(4.5, 1920, 1080)
    assert (clip.count, clip.columns, clip.rows) == (5, 5, 1)


def test_the_card_decodes_the_whole_picture_and_the_cpu_only_its_keyframes(monkeypatch):
    monkeypatch.setattr(storyboard, "require", lambda name: f"/usr/bin/{name}")
    shape = plan(181.2, 720, 576, 16 / 15)
    attempts = storyboard.decode_attempts("/in.mp4", "/s/sheet.png", shape, gpu=True)
    assert [name for name, _ in attempts] == ["cuda", "keyframes"]
    cuda, cpu = attempts[0][1], attempts[1][1]
    assert cuda[cuda.index("-hwaccel") + 1] == "cuda" and "-skip_frame" not in cuda
    assert cuda[cuda.index("-vf") + 1] == "fps=fps=1/1:round=near,scale_cuda=160:120:interp_algo=bicubic:format=nv12,hwdownload,format=nv12,setsar=1,showinfo,tile=12x16"
    assert cpu[cpu.index("-skip_frame") + 1] == "nokey" and cpu.index("-skip_frame") < cpu.index("-i")
    assert cpu[cpu.index("-vf") + 1] == "fps=fps=1/1:round=near,scale=160:120:flags=area,setsar=1,showinfo,tile=12x16"
    assert all(run[-1] == "/s/sheet.png" and "-display_rotation" in run for _, run in attempts)
    assert [name for name, _ in storyboard.decode_attempts("/in.mp4", "/s/sheet.png", shape, gpu=False)] == ["keyframes"]


def install(monkeypatch, *, fail_cuda=False, frames=182):
    runs = []

    async def fake_probe(path, *, timeout):
        return ffprobe_tool.Probe({"streams": [{"codec_type": "video", "width": 720, "height": 576, "sample_aspect_ratio": "16:15", "duration": "181.2"}], "format": {"duration": "181.3"}}, None)

    async def fake_run(args, *, timeout, **kw):
        runs.append(list(args))
        if fail_cuda and "-hwaccel" in args:
            return run_tool.Completed(list(args), 1, b"", b"No decoder surfaces left\n")
        with open(args[-1], "wb") as f:
            f.write(b"png")
        log = "".join(f"[Parsed_showinfo_4 @ 0x1] n:{i} pts:{i} pts_time:{i}.0\n" for i in range(frames))
        return run_tool.Completed(list(args), 0, b"", log.encode())

    async def fake_render(source, angle, mirror, plans, *, timeout):
        assert angle == 0 and not mirror
        for p in plans:
            with open(p.temp, "wb") as f:
                f.write(b"avif" * 10)
        return magick.Rendered((1920, 1920), [(1920, 1920)])

    monkeypatch.setattr(storyboard.ffprobe_tool, "probe", fake_probe)
    monkeypatch.setattr(storyboard, "run", fake_run)
    monkeypatch.setattr(storyboard, "require", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(storyboard.ffmpeg_tool, "gpu_present", lambda: True)
    monkeypatch.setattr(magick, "render", fake_render)
    return runs


async def test_the_op_writes_one_sheet_and_says_how_to_read_it(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "1920x1080.mp4").write_bytes(b"mp4")
    runs = install(monkeypatch)
    r = await client.post("/v0/video/storyboard", json={"file": {"path": str(ro / "1920x1080.mp4")}, "output": {"path": str(rw / "storyboard-160.avif"), "quality": 55}})
    body = r.json()
    assert r.status_code == 200, body
    assert len(runs) == 1 and "-hwaccel" in runs[0]
    assert body["result"] == {
        "path": str(rw / "storyboard-160.avif"), "width": 1920, "height": 1920, "bytes": 40,
        "tile": {"width": 160, "height": 120}, "columns": 12, "rows": 16, "count": 182, "interval": 1.0, "duration": 181.2,
    }
    assert (rw / "storyboard-160.avif").read_bytes() == b"avif" * 10
    assert not [f for f in os.listdir(rw) if f.startswith(".ems-")]
    assert "/cuda+magick/" in body["meta"]["producer"]


async def test_without_the_card_the_keyframes_make_the_sheet(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "1920x1080.mp4").write_bytes(b"mp4")
    runs = install(monkeypatch, fail_cuda=True, frames=150)
    r = await client.post("/v0/video/storyboard", json={"file": {"path": str(ro / "1920x1080.mp4")}, "output": {"path": str(rw / "storyboard-160.avif")}})
    body = r.json()
    assert r.status_code == 200, body
    assert len(runs) == 2 and "-skip_frame" in runs[1]
    # The picture ended before the last tick: the sheet says how many tiles it holds.
    assert body["result"]["count"] == 150
    assert "/keyframes+magick/" in body["meta"]["producer"]


async def test_a_turned_file_or_an_oversized_sheet_is_refused(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "v.mp4").write_bytes(b"mp4")
    install(monkeypatch)
    r = await client.post("/v0/video/storyboard", json={"file": {"path": str(ro / "v.mp4"), "angle": 90}, "output": {"path": str(rw / "s.avif")}})
    assert r.status_code == 400 and "upright" in r.json()["error"]["message"]
    r = await client.post("/v0/video/storyboard", json={"file": {"path": str(ro / "v.mp4")}, "tile": 400, "output": {"path": str(rw / "s.avif")}})
    assert r.status_code == 400 and "4096" in r.json()["error"]["message"]


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe") and shutil.which("magick")), reason="needs ffmpeg, ffprobe and magick")
async def test_a_real_sheet_from_the_test_clip(client, roots):
    ro, rw = roots
    source = os.path.join(os.path.dirname(__file__), "data", "video1.mp4")
    shutil.copy(source, ro / "video1.mp4")
    r = await client.post("/v0/video/storyboard", json={"file": {"path": str(ro / "video1.mp4")}, "output": {"path": str(rw / "storyboard-160.avif")}})
    body = r.json()
    assert r.status_code == 200, body
    result = body["result"]
    # 5.57 s of 560x320: six ticks, 160x92 tiles in one row.
    assert (result["tile"], result["columns"], result["rows"]) == ({"width": 160, "height": 92}, 6, 1)
    assert 1 <= result["count"] <= 6
    assert (result["width"], result["height"]) == (960, 92)
    assert os.path.getsize(rw / "storyboard-160.avif") > 0
