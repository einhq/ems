"""`image.renditions`: the pipeline it builds, and how it publishes. The pixels need ImageMagick."""

import os
import shutil
import subprocess

import pytest

from ems.ops import image
from ems.tools import magick

HERE = os.path.dirname(os.path.abspath(__file__))


def plan(**overrides) -> magick.Plan:
    base = dict(temp="/rw/.ems-1", format="avif", quality=None, width=320, height=320, fit="cover", crop=None)
    base.update(overrides)
    return magick.Plan(**base)


def test_padding_grows_the_box_and_stays_inside_the_frame():
    rect = magick.padded_rect(0.6, 0.1, 0.2, 0.1, 0.5)
    assert rect == magick.Rect(0.5, 0.05, 0.4, 0.2)
    clamped = magick.padded_rect(0.9, 0.0, 0.2, 0.1, 0.5)
    assert clamped.x == 0.8 and clamped.width == pytest.approx(0.2) and clamped.y == 0.0
    plain = magick.padded_rect(0.1, 0.2, 0.3, 0.4, 0)
    assert (plain.x, plain.y, plain.width, plain.height) == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_fit_geometry_matches_the_ladder():
    assert magick.fit_args(320, 320, "cover") == ["-resize", "320x320^", "-gravity", "Center", "-crop", "320x320+0+0", "+repage"]
    assert magick.fit_args(800, 800, "contain") == ["-resize", "800x800>"]
    assert magick.fit_args(100, 50, "fill") == ["-resize", "100x50!"]
    assert magick.fit_args(640, None, "contain") == ["-resize", "640x>"]
    assert magick.fit_args(None, 480, "contain") == ["-resize", "x480>"]
    assert magick.fit_args(None, None, "contain") == []


def test_pipeline_decodes_once_and_clones_every_output():
    args = magick.pipeline_args("/ro/photo one.jpg", 90, True, [
        plan(temp="/rw/.a", quality=55),
        plan(temp="/rw/.b", width=800, height=800, fit="contain", quality=60),
    ])
    assert args[:6] == ["/ro/photo one.jpg", "-set", "option:filter:blur", "0.8", "-filter", "Lagrange"]
    assert args[6:9] == ["-strip", "-orient", "Undefined"]
    assert args[9:14] == ["-rotate", "-90", "+repage", "-flop", "+repage"]
    assert args.count("+clone") == 2 and args.count("+delete") == 2
    assert args[-3:] == ["-print", "frame %w %h\\n", "null:"]
    first = args.index("(")
    assert args[first:first + 12] == ["(", "+clone", "-resize", "320x320^", "-gravity", "Center", "-crop", "320x320+0+0", "+repage", "-quality", "55", "-print"]
    assert "-write" in args and args[args.index("-write") + 1] == "avif:/rw/.a"
    assert args[args.index("-write", args.index("-write") + 1) + 1] == "avif:/rw/.b"


def test_a_crop_is_cut_from_the_decoded_raster_before_it_is_fitted():
    args = magick.pipeline_args("/ro/p.jpg", 0, False, [
        plan(temp="/rw/.c", width=384, height=384, fit="contain", quality=70, crop=magick.Rect(0.5, 0.05, 0.4, 0.2)),
    ])
    assert "-rotate" not in args and "-flop" not in args
    crop = args.index("-crop")
    assert args[crop - 1] == "+gravity"
    assert args[crop + 1] == "%[fx:max(1,round(w*0.400000))]x%[fx:max(1,round(h*0.200000))]+%[fx:round(w*0.500000)]+%[fx:round(h*0.050000)]"
    assert args[crop + 2:crop + 5] == ["+repage", "-resize", "384x384>"]


def test_output_parsing_needs_every_target():
    parsed = magick.parse_output("target 1 100 50\nframe 400 200\ntarget 0 320 320\n", 2)
    assert parsed.frame == (400, 200) and parsed.sizes == [(320, 320), (100, 50)]
    with pytest.raises(Exception, match="every output"):
        magick.parse_output("frame 400 200\ntarget 0 320 320\n", 2)


def test_output_format_defaults_to_avif_and_refuses_a_contradicting_name():
    from ems.schemas import OutputSpec

    assert image.output_format(OutputSpec(path="/rw/x")) == "avif"
    assert image.output_format(OutputSpec(path="/rw/x.jpg", format="jpeg")) == "jpeg"
    assert image.output_format(OutputSpec(path="/rw/x.jpg", format="jpg")) == "jpeg"
    with pytest.raises(Exception, match="asked to hold"):
        image.output_format(OutputSpec(path="/rw/x.avif", format="webp"))
    with pytest.raises(Exception, match="unsupported image format"):
        image.output_format(OutputSpec(path="/rw/x", format="tiff"))


async def test_renditions_publish_all_or_nothing(client, roots, monkeypatch):
    ro, rw = roots
    shutil.copy(os.path.join(HERE, "data", "image1.jpg"), ro / "photo.jpg")
    body = {
        "file": {"path": str(ro / "photo.jpg"), "angle": 90, "mimetype": "image/jpeg"},
        "targets": [
            {"output": {"path": str(rw / "320x320.avif"), "quality": 55}, "box": {"width": 320, "height": 320}, "fit": "cover"},
            {"output": {"path": str(rw / "800x800.avif")}, "box": {"width": 800, "height": 800}},
        ],
    }

    r = await client.post("/v0/image/renditions", json={**body, "targets": [{"output": {"path": str(ro / "x.avif")}, "box": {"width": 10}}]})
    assert r.status_code == 403
    r = await client.post("/v0/image/renditions", json={**body, "targets": [{"output": {"path": str(rw / "x.avif"), "format": "bmp"}}]})
    assert r.status_code == 400
    r = await client.post("/v0/image/renditions", json={**body, "targets": [{"output": {"path": str(rw / "x.avif")}, "box": {}}]})
    assert r.status_code == 400 and "neither" in r.json()["error"]["message"]

    seen = {}

    async def fake_render(source, angle, mirror, plans, *, timeout):
        seen["angle"], seen["mirror"], seen["source"] = angle, mirror, source
        for plan in plans:
            with open(plan.temp, "wb") as f:
                f.write(b"avif" * 10)
        return magick.Rendered((600, 800), [(320, 320), (600, 800)])

    monkeypatch.setattr(magick, "render", fake_render)
    r = await client.post("/v0/image/renditions", json=body)
    result = r.json()
    assert r.status_code == 200, result
    assert seen == {"angle": 90, "mirror": False, "source": str(ro / "photo.jpg")}
    assert result["result"]["frame"] == {"width": 600, "height": 800}
    assert result["result"]["targets"] == [
        {"path": str(rw / "320x320.avif"), "width": 320, "height": 320, "bytes": 40},
        {"path": str(rw / "800x800.avif"), "width": 600, "height": 800, "bytes": 40},
    ]
    assert (rw / "320x320.avif").read_bytes() == b"avif" * 10
    assert not [f for f in os.listdir(rw) if f.startswith(".ems-")]
    assert result["meta"]["producer"].startswith("magick/")

    async def half_render(source, angle, mirror, plans, *, timeout):
        with open(plans[0].temp, "wb") as f:
            f.write(b"x")
        return magick.Rendered((600, 800), [(320, 320), (600, 800)])

    monkeypatch.setattr(magick, "render", half_render)
    for name in ("320x320.avif", "800x800.avif"):
        os.remove(rw / name)
    r = await client.post("/v0/image/renditions", json=body)
    assert r.status_code == 500 and r.json()["error"]["code"] == "tool_failed"
    assert sorted(os.listdir(rw)) == []


async def test_a_heif_owes_only_the_residual_turn(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "IMG_1.HEIC").write_bytes(b"heic")
    seen = {}

    async def fake_applied(path, timeout):
        return 270

    async def fake_render(source, angle, mirror, plans, *, timeout):
        seen["angle"] = angle
        for plan in plans:
            with open(plan.temp, "wb") as f:
                f.write(b"a")
        return magick.Rendered((3024, 4032), [(240, 320)])

    monkeypatch.setattr(image.decode_tool, "applied_angle", fake_applied)
    monkeypatch.setattr(magick, "render", fake_render)
    target = {"output": {"path": str(rw / "320x320.avif")}, "box": {"width": 320, "height": 320}}
    r = await client.post("/v0/image/renditions", json={"file": {"path": str(ro / "IMG_1.HEIC"), "angle": 270, "mimetype": "image/heic"}, "targets": [target]})
    assert r.status_code == 200, r.json()
    assert seen["angle"] == 0
    r = await client.post("/v0/image/renditions", json={"file": {"path": str(ro / "IMG_1.HEIC"), "angle": 180, "mimetype": "image/heic"}, "targets": [target]})
    assert r.status_code == 200 and seen["angle"] == 270


def _avif_writer_present() -> bool:
    if not shutil.which("magick"):
        return False
    done = subprocess.run(["magick", "-list", "format"], capture_output=True, text=True)
    return any(line.strip().upper().startswith("AVIF") and "rw" in line for line in done.stdout.splitlines())


needs_magick = pytest.mark.skipif(not _avif_writer_present(), reason="ImageMagick 7 with an AVIF encoder is not installed")


@needs_magick
async def test_a_rotated_crop_lands_on_the_marker(tmp_path):
    """A 400x200 raster shown at angle 270 (display 200x400) with a red block at raw (40,40)-(79,79),
    i.e. display (120,40)-(159,79): the crop of that display-frame box must be solid red."""
    source = str(tmp_path / "rotated.png")
    subprocess.run(["magick", "-size", "400x200", "gradient:blue-yellow", "-fill", "red", "-draw", "rectangle 40,40 79,79", source], check=True)
    out = str(tmp_path / "crop.png")
    plans = [magick.Plan(temp=out, format="png", quality=None, width=None, height=None, fit="contain", crop=magick.Rect(0.6, 0.1, 0.2, 0.1))]
    rendered = await magick.render(source, 270, False, plans, timeout=60)
    assert rendered.frame == (200, 400) and rendered.sizes == [(40, 40)]
    mean = subprocess.run(["magick", out, "-format", "%[fx:mean.r] %[fx:mean.g] %[fx:mean.b]", "info:"], capture_output=True, text=True, check=True).stdout.split()
    assert [round(float(v)) for v in mean] == [1, 0, 0]


@needs_magick
async def test_cloned_outputs_match_independently_decoded_ones(tmp_path):
    """No output is ever resized from another: each clone equals a single-output run of the same source."""
    source = str(tmp_path / "source.jpg")
    subprocess.run(["magick", "-size", "400x200", "gradient:red-blue", "-set", "comment", "source metadata", source], check=True)
    plans = [
        magick.Plan(temp=str(tmp_path / f"{w}.avif"), format="avif", quality=q, width=w, height=w, fit="contain", crop=None)
        for w, q in ((100, 55), (150, 60))
    ]
    rendered = await magick.render(source, 270, True, plans, timeout=60)
    assert rendered.frame == (200, 400) and rendered.sizes == [(50, 100), (75, 150)]
    for plan in plans:
        reference = str(tmp_path / f"reference-{plan.width}.avif")
        subprocess.run(["magick", source, *magick.FILTER_ARGS, *magick.STRIP_ARGS, "-rotate", "-270", "+repage", "-flop", "+repage",
                        "-resize", f"{plan.width}x{plan.height}>", "-quality", str(plan.quality), f"avif:{reference}"], check=True)
        fmt = "%w %h %[colorspace] %[orientation] %[signature]"
        ours = subprocess.run(["magick", "identify", "-format", fmt, plan.temp], capture_output=True, text=True, check=True).stdout
        theirs = subprocess.run(["magick", "identify", "-format", fmt, reference], capture_output=True, text=True, check=True).stdout
        assert ours == theirs
        # `TopLeft` is what a decoder reports for no transform at all; a leftover `irot` would read as a turn.
        assert ours.startswith(f"{plan.width // 2} {plan.height} ") and ours.split()[3] in ("TopLeft", "Undefined")


async def test_a_raw_magick_cannot_demosaic_is_rendered_from_its_embedded_preview(client, roots, monkeypatch):
    ro, rw = roots
    (ro / "shot.nef").write_bytes(b"nef")
    sources = []

    async def failing_then_fine(source, angle, mirror, plans, *, timeout):
        sources.append(os.path.basename(source))
        if source.endswith("shot.nef"):
            raise magick.EmsError(magick.TOOL_FAILED, "magick failed: Input/output error")
        for plan in plans:
            with open(plan.temp, "wb") as f:
                f.write(b"p")
        return magick.Rendered((3040, 2014), [(320, 320)])

    async def fake_preview(src, tmpdir, timeout):
        out = os.path.join(tmpdir, "preview-JpgFromRaw.jpg")
        with open(out, "wb") as f:
            f.write(b"jpeg")
        return out

    monkeypatch.setattr(magick, "render", failing_then_fine)
    monkeypatch.setattr(image.decode_tool, "_preview", fake_preview)
    target = {"output": {"path": str(rw / "320x320.avif")}, "box": {"width": 320, "height": 320}, "fit": "cover"}
    r = await client.post("/v0/image/renditions", json={"file": {"path": str(ro / "shot.nef"), "mimetype": "image/x-nikon-nef"}, "targets": [target]})
    body = r.json()
    assert r.status_code == 200, body
    assert sources == ["shot.nef", "preview-JpgFromRaw.jpg"] and body["result"]["source"] == "preview"
    assert (rw / "320x320.avif").read_bytes() == b"p"

    # A JPEG that fails is a failure: there is no preview to fall back to.
    (ro / "x.jpg").write_bytes(b"jpg")
    sources.clear()

    async def failing(source, angle, mirror, plans, *, timeout):
        raise magick.EmsError(magick.TOOL_FAILED, "magick failed")

    monkeypatch.setattr(magick, "render", failing)
    r = await client.post("/v0/image/renditions", json={"file": {"path": str(ro / "x.jpg"), "mimetype": "image/jpeg"}, "targets": [target]})
    assert r.status_code == 500 and r.json()["error"]["code"] == "tool_failed"
