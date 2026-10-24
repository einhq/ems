"""The model readers apply nothing: a HEIF `irot`, an EXIF Orientation, or both, and the raster
`image_io.open_display_bgr` hands back is the stored one every time.

This is the fact the display-frame convention (docs/api.md §3.3) rests on for the model ops:
the caller states the turn owed and EMS applies only that, so a wrong or stale tag cannot make
two ops disagree. ImageMagick, by contrast, applies the `irot` (and only the `irot`).

Needs ImageMagick with an AVIF encoder, exiftool, pillow-heif and numpy, so it runs in the
container and skips elsewhere.
"""

import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ems", "modelworker"))

WIDTH, HEIGHT = 96, 64


def _avif_encoder_present() -> bool:
    if not shutil.which("convert"):
        return False
    done = subprocess.run(["convert", "-list", "format"], capture_output=True, text=True)
    return any(line.strip().upper().startswith("AVIF") and "rw" in line for line in done.stdout.splitlines())


def _readers_present() -> bool:
    try:
        import numpy  # noqa: F401
        import pillow_heif  # noqa: F401
    except ImportError:
        return False
    return True


needs = pytest.mark.skipif(
    not (_avif_encoder_present() and shutil.which("exiftool") and _readers_present()),
    reason="needs ImageMagick with AVIF, exiftool, pillow-heif and numpy (the container has them)",
)


@needs
def test_display_readers_ignore_irot_and_exif_orientation(tmp_path):
    import image_io

    plain = tmp_path / "plain.jpg"
    subprocess.run(["convert", "-size", f"{WIDTH}x{HEIGHT}", "gradient:blue-yellow", str(plain)], check=True)

    irot_only = tmp_path / "irot-only.avif"
    subprocess.run(["convert", str(plain), "-orient", "RightTop", f"avif:{irot_only}"], check=True)

    exif_only = tmp_path / "exif-only.avif"
    subprocess.run(["convert", str(plain), f"avif:{exif_only}"], check=True)
    subprocess.run(["exiftool", "-Orientation=6", "-n", "-overwrite_original", str(exif_only)], check=True, capture_output=True)

    both = tmp_path / "both.avif"
    subprocess.run(["convert", str(plain), "-orient", "RightTop", f"avif:{both}"], check=True)
    subprocess.run(["exiftool", "-Orientation=6", "-n", "-overwrite_original", str(both)], check=True, capture_output=True)

    for path in (irot_only, exif_only, both):
        raster = image_io.open_display_bgr(str(path), 0, False)
        assert raster.shape[1] == WIDTH and raster.shape[0] == HEIGHT, f"{path.name}: {raster.shape}"

    # And the turn the caller states is the one applied.
    turned = image_io.open_display_bgr(str(exif_only), 90, False)
    assert turned.shape[1] == HEIGHT and turned.shape[0] == WIDTH
