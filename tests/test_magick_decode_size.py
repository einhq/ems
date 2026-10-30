import asyncio

import pytest

from ems.tools import magick
from ems.tools.run import Completed


def plan(width=None, height=None, fit="contain", crop=None):
    return magick.Plan(temp="t", format="avif", quality=None, width=width, height=height, fit=fit, crop=crop)


# The renditions EiN asks of every photo: a cover thumbnail and four contained sizes up to 2560.
PHOTO = [plan(320, 320, "cover"), plan(512, 512), plan(800, 800), plan(1600, 1600), plan(2560, 2560)]


def test_a_large_photo_decodes_at_what_its_largest_output_needs():
    assert magick.decode_size((6000, 4000), 0, PHOTO) == (2560, 1707)


def test_the_size_is_in_the_stored_frame_when_the_photo_is_turned():
    # Turned a quarter, the 2560 box binds the stored height, which is the display width.
    assert magick.decode_size((6000, 4000), 90, PHOTO) == (2560, 1707)
    assert magick.decode_size((4000, 6000), 90, PHOTO) == (1707, 2560)


def test_a_photo_needing_more_than_half_decodes_in_full():
    assert magick.decode_size((4000, 3000), 0, PHOTO) is None
    assert magick.decode_size((6000, 4000), 0, [plan()]) is None


def test_a_crop_needs_the_pixels_under_it():
    face = magick.Rect(0.4, 0.4, 0.05, 0.05)
    # 5% of 6000 is 300 px, short of the 384 asked: the full source is needed.
    assert magick.decode_size((6000, 4000), 0, [plan(384, 384, crop=face)]) is None
    wide = magick.Rect(0.0, 0.0, 0.5, 0.5)
    assert magick.decode_size((6000, 4000), 0, [plan(384, 384, crop=wide)]) == (768, 512)


def test_cover_and_fill_reach_the_looser_side():
    assert magick.decode_size((6000, 4000), 0, [plan(1000, 1000, "cover")]) == (1500, 1000)
    assert magick.decode_size((6000, 4000), 0, [plan(1000, 1000, "fill")]) == (1500, 1000)
    assert magick.decode_size((6000, 4000), 0, [plan(1000, 1000, "contain")]) == (1000, 667)


def test_the_decode_size_goes_before_the_source():
    args = magick.pipeline_args("/ro/p.jpg", 0, False, [plan(800, 800)], (1200, 800))
    assert args[:3] == ["-define", "jpeg:size=1200x800", "/ro/p.jpg"]
    assert magick.pipeline_args("/ro/p.jpg", 0, False, [plan(800, 800)])[0] == "/ro/p.jpg"


def test_render_reports_the_files_own_frame_after_a_smaller_decode(monkeypatch):
    seen = {}

    async def fake_run(args, **kwargs):
        seen["args"] = args
        return Completed(args, 0, b"target 0 2560 1707\nframe 2000 3000\n", b"")

    monkeypatch.setattr(magick, "run", fake_run)
    monkeypatch.setattr(magick, "require", lambda name: name)
    monkeypatch.setattr(magick, "jpeg_size", lambda path: (6000, 4000))
    rendered = asyncio.run(magick.render("in.jpg", 90, False, [plan(2560, 2560)], timeout=60))

    assert "jpeg:size=2560x1707" in seen["args"]
    assert rendered.frame == (4000, 6000)
    assert rendered.sizes == [(2560, 1707)]


def test_jpeg_size_reads_only_jpegs(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    Image.new("RGB", (64, 48)).save(tmp_path / "a.jpg")
    Image.new("RGB", (64, 48)).save(tmp_path / "a.png")
    assert magick.jpeg_size(str(tmp_path / "a.jpg")) == (64, 48)
    assert magick.jpeg_size(str(tmp_path / "a.png")) is None
    assert magick.jpeg_size(str(tmp_path / "missing.jpg")) is None
