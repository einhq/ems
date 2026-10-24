"""`video.frames` request rules; the extraction itself needs ffmpeg and is stubbed."""

import os

from ems.ops import video


async def test_frames_need_a_selection_and_a_writable_dir(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "v.mp4").write_bytes(b"mp4")
    r = await client.post("/v0/video/frames", json={"file": {"path": str(ro / "v.mp4")}, "output": {"dir": str(rw / "frames")}})
    assert r.status_code == 400 and "at" in r.json()["error"]["message"]
    r = await client.post("/v0/video/frames", json={"file": {"path": str(ro / "v.mp4")}, "at": [1.0], "output": {"dir": str(ro / "frames")}})
    assert r.status_code == 403

    async def fake(ctx, path, out_dir, *, at, count, strategy, fmt, quality, max_edge, progress, angle, mirror):
        assert at == [1.0, 2.5] and out_dir == str(rw / "frames")
        assert (angle, mirror) == (90, True)
        return [{"path": f"{out_dir}/frame-000.jpg", "t": 1.0}, {"path": f"{out_dir}/frame-001.jpg", "t": 2.5}]

    monkeypatch.setattr(video, "extract_frames", fake)
    r = await client.post("/v0/video/frames", json={"file": {"path": str(ro / "v.mp4"), "angle": 90, "mirror": True}, "at": [1.0, 2.5], "output": {"dir": str(rw / "frames")}})
    body = r.json()
    assert r.status_code == 200, body
    assert body["result"]["frames"] == [{"path": str(rw / "frames" / "frame-000.jpg"), "t": 1.0}, {"path": str(rw / "frames" / "frame-001.jpg"), "t": 2.5}]
    assert (rw / "frames").is_dir()


async def test_poster_takes_one_frame_and_fits_it_like_a_photo(client, roots, monkeypatch):
    from ems.tools import magick

    ro, rw = roots
    (ro / "v.mp4").write_bytes(b"mp4")
    seen = {}

    async def fake_extract(ctx, path, out, *, at, deinterlace):
        seen["at"], seen["deinterlace"], seen["path"] = at, deinterlace, path
        with open(out, "wb") as f:
            f.write(b"jpeg")

    async def fake_render(source, angle, mirror, plans, *, timeout):
        seen["angle"], seen["source"] = angle, os.path.basename(source)
        for plan in plans:
            with open(plan.temp, "wb") as f:
                f.write(b"a")
        return magick.Rendered((240, 320), [(240, 320)])

    monkeypatch.setattr(video, "extract_poster", fake_extract)
    monkeypatch.setattr(magick, "render", fake_render)
    r = await client.post("/v0/video/poster", json={
        "file": {"path": str(ro / "v.mp4"), "angle": 270},
        "at": 0,
        "targets": [{"output": {"path": str(rw / "512x512.avif"), "quality": 58}, "box": {"width": 512, "height": 512}}],
    })
    body = r.json()
    assert r.status_code == 200, body
    assert seen == {"at": 0.0, "deinterlace": False, "path": str(ro / "v.mp4"), "angle": 270, "source": "frame.jpg"}
    assert body["result"] == {"frame": {"width": 240, "height": 320}, "targets": [{"path": str(rw / "512x512.avif"), "width": 240, "height": 320, "bytes": 1}], "t": 0.0}
    assert (rw / "512x512.avif").read_bytes() == b"a"
    assert body["meta"]["producer"].startswith("ffmpeg/")


async def test_a_poster_past_the_end_of_the_picture_is_its_first_frame(client, roots, monkeypatch):
    from ems.tools import magick

    ro, rw = roots
    (ro / "v.mp4").write_bytes(b"mp4")
    asked = []

    async def short(ctx, path, out, *, at, deinterlace):
        asked.append(at)
        if at > 0:
            raise video.NoFrame()
        with open(out, "wb") as f:
            f.write(b"jpeg")

    async def fake_render(source, angle, mirror, plans, *, timeout):
        for plan in plans:
            with open(plan.temp, "wb") as f:
                f.write(b"a")
        return magick.Rendered((240, 320), [(240, 320)])

    monkeypatch.setattr(video, "extract_poster", short)
    monkeypatch.setattr(magick, "render", fake_render)
    r = await client.post("/v0/video/poster", json={"file": {"path": str(ro / "v.mp4")}, "at": 1, "targets": [{"output": {"path": str(rw / "x.avif")}, "box": {"width": 10}}]})
    assert r.status_code == 200, r.json()
    assert asked == [1.0, 0.0]
    assert r.json()["result"]["t"] == 0.0


async def test_a_video_with_no_frame_at_all_is_undecodable(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "v.mp4").write_bytes(b"mp4")
    asked = []

    async def nothing(ctx, path, out, *, at, deinterlace):
        asked.append(at)
        raise video.NoFrame()

    monkeypatch.setattr(video, "extract_poster", nothing)
    for at, tries in ((5, [5.0, 0.0]), (0, [0.0])):
        asked.clear()
        r = await client.post("/v0/video/poster", json={"file": {"path": str(ro / "v.mp4")}, "at": at, "targets": [{"output": {"path": str(rw / "x.avif")}, "box": {"width": 10}}]})
        assert r.status_code == 422
        assert r.json()["error"] == {"code": "undecodable", "message": "no video frame", "permanent": True}
        assert asked == tries
    assert os.listdir(rw) == []


async def test_a_seek_past_the_end_is_no_frame_whatever_ffmpeg_exits(client, roots, monkeypatch, tmp_path):
    from ems.ops.video import NoFrame, extract_poster
    from ems.tools import run as run_tool

    ro, rw = roots
    (ro / "v.mp4").write_bytes(b"mp4")

    async def failing_ffmpeg(args, *, timeout, **kw):
        return run_tool.Completed(list(args), 234, b"", b"[out#0/image2 @ 0x1] Nothing was written into output file, because at least one of its streams received no packets.\n")

    async def silent_ffmpeg(args, *, timeout, **kw):
        return run_tool.Completed(list(args), 0, b"", b"")

    monkeypatch.setattr(video, "require", lambda name: "/usr/bin/ffmpeg")
    ctx = client.app.state.ctx
    for ffmpeg in (failing_ffmpeg, silent_ffmpeg):
        monkeypatch.setattr(video, "run", ffmpeg)
        try:
            await extract_poster(ctx, str(ro / "v.mp4"), str(tmp_path / "frame.jpg"), at=5.0, deinterlace=False)
        except NoFrame:
            pass
        else:
            raise AssertionError(f"a seek past the end was not refused ({ffmpeg.__name__})")


async def test_a_failed_decode_is_the_files_fault_only_when_ffprobe_refuses_it(monkeypatch):
    from ems.tools import ffprobe

    async def refused(path, *, timeout):
        return ffprobe.Probe(None, "moov atom not found")

    async def readable(path, *, timeout):
        return ffprobe.Probe({"streams": [], "format": {}}, None)

    monkeypatch.setattr(ffprobe, "probe", refused)
    error = await ffprobe.decode_failure("/files/v.mov", "ffmpeg frame extraction failed: x", timeout=5)
    assert error.code.name == "undecodable" and error.permanent and "moov atom not found" in error.message

    monkeypatch.setattr(ffprobe, "probe", readable)
    error = await ffprobe.decode_failure("/files/v.mov", "ffmpeg frame extraction failed: x", timeout=5)
    assert error.code.name == "tool_failed" and not error.permanent


async def test_count_spreads_the_frames_over_the_whole_picture(client, roots, monkeypatch, tmp_path):
    from ems.ops.video import extract_frames
    from ems.streaming import Progress
    from ems.tools import ffprobe
    from ems.tools import run as run_tool

    ro, _ = roots
    (ro / "v.mp4").write_bytes(b"mp4")
    seeks = []

    async def probed(path, *, timeout):
        # The sound runs on past the picture; the picture's own length is what is spread over.
        return ffprobe.Probe({"streams": [{"codec_type": "video", "duration": "80.000"}], "format": {"duration": "95.5"}}, None)

    async def ffmpeg(args, *, timeout, **kw):
        start = float(args[args.index("-ss") + 1])
        seeks.append((start, args[args.index("-t") + 1] if "-t" in args else None, args[args.index("-vf") + 1].split(",")[0]))
        if start >= 70:
            return run_tool.Completed(list(args), 234, b"", b"[out#0/image2 @ 0x1] Nothing was written into output file, because at least one of its streams received no packets.\n")
        with open(args[-1], "wb") as f:
            f.write(b"jpeg")
        return run_tool.Completed(list(args), 0, b"", b"[Parsed_showinfo_1 @ 0x5f] n:   0 pts:  40960 pts_time:3.2     duration:    512\n")

    monkeypatch.setattr(video, "require", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video.ffmpeg_tool, "gpu_present", lambda: False)
    monkeypatch.setattr(ffprobe, "probe", probed)
    monkeypatch.setattr(video, "run", ffmpeg)
    ctx = client.app.state.ctx

    got = await extract_frames(ctx, str(ro / "v.mp4"), str(tmp_path), at=None, count=8, strategy="representative", fmt="jpeg", quality=93, max_edge=None, progress=Progress(None))
    assert seeks == [(i * 10.0, "10.000", "thumbnail=300") for i in range(8)]
    # Each frame's moment is where its span starts plus where in it the thumbnail filter picked;
    # the span past the end of the picture gives none.
    assert [f["t"] for f in got] == [round(i * 10.0 + 3.2, 3) for i in range(7)]
    assert [os.path.basename(f["path"]) for f in got] == [f"frame-{i:03d}.jpg" for i in range(7)]

    seeks.clear()
    await extract_frames(ctx, str(ro / "v.mp4"), str(tmp_path), at=None, count=4, strategy="even", fmt="jpeg", quality=93, max_edge=None, progress=Progress(None))
    assert [(s, t) for s, t, _ in seeks] == [(10.0, None), (30.0, None), (50.0, None), (70.0, None)]


async def test_frames_are_turned_by_the_callers_angle_alone(client, roots, monkeypatch, tmp_path):
    from ems.ops.video import SQUARE_PIXELS_FILTER, extract_frames
    from ems.streaming import Progress
    from ems.tools import ffprobe
    from ems.tools import run as run_tool

    ro, _ = roots
    (ro / "v.mp4").write_bytes(b"mp4")
    calls = []

    async def probed(path, *, timeout):
        return ffprobe.Probe({"streams": [{"codec_type": "video", "duration": "10.000"}], "format": {}}, None)

    async def ffmpeg(args, *, timeout, **kw):
        calls.append(args)
        with open(args[-1], "wb") as f:
            f.write(b"jpeg")
        return run_tool.Completed(list(args), 0, b"", b"")

    monkeypatch.setattr(video, "require", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video.ffmpeg_tool, "gpu_present", lambda: False)
    monkeypatch.setattr(ffprobe, "probe", probed)
    monkeypatch.setattr(video, "run", ffmpeg)
    ctx = client.app.state.ctx
    fit = "scale=w=min(iw\\,640):h=min(ih\\,640):force_original_aspect_ratio=decrease:force_divisible_by=2"

    await extract_frames(ctx, str(ro / "v.mp4"), str(tmp_path), at=[1.0], count=None, strategy="even", fmt="jpeg", quality=90, max_edge=640, progress=Progress(None), angle=90, mirror=True)
    await extract_frames(ctx, str(ro / "v.mp4"), str(tmp_path), at=None, count=1, strategy="even", fmt="jpeg", quality=90, max_edge=None, progress=Progress(None), angle=180)
    await extract_frames(ctx, str(ro / "v.mp4"), str(tmp_path), at=[1.0], count=None, strategy="even", fmt="jpeg", quality=90, max_edge=None, progress=Progress(None))
    chains = [args[args.index("-vf") + 1] for args in calls]
    # The fit is on the stored frame, then a counter-clockwise turn, then the flop.
    assert chains == [
        f"{SQUARE_PIXELS_FILTER},{fit},transpose=2,hflip",
        f"showinfo,{SQUARE_PIXELS_FILTER},transpose=1,transpose=1",
        SQUARE_PIXELS_FILTER,
    ]
    assert all(args[args.index("-display_rotation") + 1] == "0" for args in calls)
