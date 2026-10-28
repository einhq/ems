"""`image_io.open_rgb` with `min_side`: a JPEG comes back at a reduced scale that still covers
the asked side in both directions, and other formats come back whole.

Needs Pillow, so it skips where the model worker's packages are not installed.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ems", "modelworker"))

Image = pytest.importorskip("PIL.Image")

WIDTH, HEIGHT = 2400, 1600


def _write(tmp_path, name: str, fmt: str) -> str:
    path = str(tmp_path / name)
    Image.new("RGB", (WIDTH, HEIGHT), (120, 80, 40)).save(path, fmt)
    return path


def test_a_jpeg_decodes_at_the_smallest_scale_covering_min_side(tmp_path):
    import image_io

    path = _write(tmp_path, "photo.jpg", "JPEG")
    image = image_io.open_rgb(path, min_side=768)
    # 1/2 is the smallest of libjpeg's scales whose short side, 800, is still at least 768.
    assert image.size == (WIDTH // 2, HEIGHT // 2)
    assert image.mode == "RGB"


def test_without_min_side_a_jpeg_decodes_whole(tmp_path):
    import image_io

    path = _write(tmp_path, "photo.jpg", "JPEG")
    assert image_io.open_rgb(path).size == (WIDTH, HEIGHT)


def test_a_png_decodes_whole_whatever_min_side_asks(tmp_path):
    import image_io

    path = _write(tmp_path, "scan.png", "PNG")
    assert image_io.open_rgb(path, min_side=200).size == (WIDTH, HEIGHT)


def test_the_display_frame_turns_the_reduced_raster(tmp_path):
    import image_io

    path = _write(tmp_path, "photo.jpg", "JPEG")
    image = image_io.open_display(path, angle=90, min_side=200)
    # 1/8 covers 200 (300 x 200), and the quarter turn swaps the sides.
    assert image.size == (HEIGHT // 8, WIDTH // 8)
