"""Formats the model readers cannot open, decoded into something they can.

PIL with pillow-heif and rawpy reads almost everything the archive holds; this is the fallback
for the rest, and for a running EMS whose readers turn out to be missing a format. It runs only
when the reader has said it could not identify the file.

Orientation is the subtle part, and it differs per family because the decoders differ:

- **RAW does not rotate, anywhere.** The embedded preview is copied out verbatim and
  ImageMagick's RAW reader leaves the sensor raster alone, so the caller's whole `angle` is
  still owed on the decoded file.
- **HEIF rotates in ImageMagick.** libheif applies the container's `irot` on decode, so what
  comes out of `convert` is already turned by it, and the turn still owed is the caller's angle
  minus the `irot` — the *residual*. exiftool reports the `irot` as `Rotation` in quarter
  turns anticlockwise (numeric), which is why it is read with `-n` and only for a HEIF.

An EXIF Orientation moves nothing in either reader, so it never enters this arithmetic.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass

from ..errors import EmsError, UNSUPPORTED
from .run import require, run

HEIF_MIMETYPES = frozenset({
    "image/heic", "image/heif", "image/heic-sequence", "image/heif-sequence", "image/avif",
})
RAW_MIMETYPES = frozenset({
    "image/x-canon-cr2", "image/x-canon-cr3", "image/x-canon-crw", "image/x-nikon-nef", "image/x-nikon-nrw",
    "image/x-sony-arw", "image/x-sony-sr2", "image/x-adobe-dng", "image/x-olympus-orf", "image/x-fujifilm-raf",
    "image/x-panasonic-rw2", "image/x-panasonic-raw", "image/x-pentax-pef", "image/x-sigma-x3f", "image/x-minolta-mrw",
    "image/x-kodak-dcr", "image/x-kodak-kdc", "image/x-hasselblad-3fr", "image/x-phaseone-iiq", "image/x-epson-erf",
    "image/x-samsung-srw", "image/x-leica-rwl",
})

# Embedded JPEGs in the order vendors hide the full-size one. Below MIN_PREVIEW_EDGE the
# embedded image is a contact-sheet thumbnail, not a picture; a model run on it would "succeed".
PREVIEW_TAGS = ("JpgFromRaw", "PreviewImage", "OtherImage")
MIN_PREVIEW_EDGE = 512
# The models resize to ~1024 anyway, so a full-sensor demosaic is pixels thrown away.
DEMOSAIC_MAX_EDGE = 2048

UNREADABLE_MARKERS = ("cannot identify image file", "could not load image", "unsupported file format", "no decode delegate")


def is_unreadable(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in UNREADABLE_MARKERS)


def is_raw(mimetype: str | None) -> bool:
    """Whether the mimetype is one of the camera RAW families."""
    return mimetype in RAW_MIMETYPES


def is_heif(mimetype: str | None) -> bool:
    return mimetype in HEIF_MIMETYPES


@dataclass
class Decoded:
    path: str
    angle: int
    mirror: bool
    tmpdir: str | None

    def cleanup(self) -> None:
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)


def normalize_angle(angle) -> int:
    try:
        a = int(round(float(angle))) % 360
    except (TypeError, ValueError):
        return 0
    return a if a in (90, 180, 270) else 0


async def _longest_edge(path: str, timeout: float) -> int:
    done = await run([require("exiftool"), "-s3", "-ImageWidth", "-ImageHeight", path], timeout=timeout)
    if done.code != 0:
        return 0
    try:
        w, h = (int(v) for v in done.stdout.decode(errors="replace").split())
    except ValueError:
        return 0
    return max(w, h)


async def applied_angle(path: str, timeout: float) -> int:
    """The turn ImageMagick makes for a HEIF: the `irot`, in quarter turns, or none."""
    done = await run([require("exiftool"), "-s3", "-n", "-Rotation", path], timeout=timeout)
    text = done.stdout.decode(errors="replace").strip() if done.code == 0 else ""
    if not text:
        return 0
    try:
        return normalize_angle(float(text) * 90)
    except ValueError:
        return 0


async def _convert(src: str, out: str, timeout: float, extra: list[str] | None = None) -> str:
    done = await run([require("convert"), src, *(extra or []), f"jpg:{out}"], timeout=timeout)
    if done.code != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        raise EmsError(UNSUPPORTED, f"image decode failed: {done.tail() or 'convert wrote nothing'}")
    return out


async def _preview(src: str, tmpdir: str, timeout: float) -> str | None:
    exiftool = require("exiftool")
    for tag in PREVIEW_TAGS:
        out = os.path.join(tmpdir, f"preview-{tag}.jpg")
        # `-W` writes the binary tag straight to a file; a missing tag writes none and is no error.
        done = await run([exiftool, "-b", f"-{tag}", "-W", out, src], timeout=timeout)
        size = os.path.getsize(out) if done.code == 0 and os.path.exists(out) else 0
        if size > 0 and await _longest_edge(out, timeout) >= MIN_PREVIEW_EDGE:
            return out
        if os.path.exists(out):
            os.remove(out)
    return None


async def decode_for_models(path: str, mimetype: str | None, angle, mirror: bool, *, scratch: str, timeout: float) -> Decoded:
    """A file the model readers can open, and the turn still owed on it."""
    node_angle = normalize_angle(angle)
    raw, heif = is_raw(mimetype), is_heif(mimetype)
    if not raw and not heif:
        # Unknown family: let ImageMagick try, and assume it applied no turn (true for
        # everything but HEIF, whose libheif reader is the exception documented above).
        tmpdir = tempfile.mkdtemp(prefix="decode-", dir=scratch)
        try:
            decoded = await _convert(path, os.path.join(tmpdir, "decoded.jpg"), timeout, ["-resize", f"{DEMOSAIC_MAX_EDGE}x{DEMOSAIC_MAX_EDGE}>"])
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        return Decoded(decoded, node_angle, mirror, tmpdir)

    tmpdir = tempfile.mkdtemp(prefix="decode-", dir=scratch)
    try:
        if heif:
            # Full resolution: a HEIC is what a phone actually shot, so detection should see the
            # same pixels it would from the equivalent JPEG.
            decoded = await _convert(path, os.path.join(tmpdir, "decoded.jpg"), timeout)
            residual = normalize_angle(node_angle - await applied_angle(path, timeout))
            return Decoded(decoded, residual, mirror, tmpdir)
        decoded = await _preview(path, tmpdir, timeout)
        if decoded is None:
            decoded = await _convert(path, os.path.join(tmpdir, "decoded.jpg"), timeout, ["-resize", f"{DEMOSAIC_MAX_EDGE}x{DEMOSAIC_MAX_EDGE}>"])
        return Decoded(decoded, node_angle, mirror, tmpdir)
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
