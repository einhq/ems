"""Open an image the models can work with, whatever container it arrived in.

PIL and cv2 between them read a narrower set of formats than the archive holds, and every
caller here routes on the `image/` mimetype prefix — so without help a camera RAW or an iPhone
HEIC travels all the way to the model call before failing, as `cannot identify image file`.

Two packages close it:

  - **pillow-heif** registers a HEIF opener into PIL, so `.heic` behaves like any other file.
  - **rawpy** (LibRaw) reads RAW. `extract_thumb()` returns the embedded full-size JPEG — the
    same image the out-of-process path (ems/tools/decode.py) pulls with exiftool, and the one
    whose dimensions match what exiftool reports for the file, since those come from the
    preview rather than the sensor. A full `postprocess()` demosaic is the fallback for a RAW
    that carries no usable preview.

Dispatch is by *failure*, not by extension: PIL is asked first and LibRaw picks up whatever PIL
cannot identify. A format table here would be one more list of RAW formats, free to drift.

## The frame, which is not uniform

  - **RAW comes back unrotated.** rawpy hands over the embedded preview as stored — a RAF
    tagged `Rotate 270 CW` opens landscape.
  - **HEIF comes back rotated.** pillow-heif *is* libheif, and it applies the container's
    `irot`/`imir` on decode: a 4032x3024 file opens 3024x4032. It also rewrites the
    EXIF orientation to 1 and keeps no record of what it did — there is no
    `original_orientation`, and no option to turn the transform off.

So HEIF arrives already in the display frame and everything else arrives in the stored frame.
That difference is *not* resolved here, because it cannot be: pillow-heif rewrites the EXIF
orientation to 1 and keeps no record of what it applied, so the rotation it did is unrecoverable
from this side. The caller passes a **residual** angle instead — what the display frame needs minus
what the decoder already did, worked out where the file's own tag can still be read. In the
ordinary case the two agree and the residual is zero; they differ exactly when someone
has corrected a rotation by hand, and then the correction is what survives.
"""

import io

# Below this the embedded image is a contact-sheet thumbnail rather than a picture — embedding or
# detecting on one would succeed on a blur, which is worse than the failure it replaces.
MIN_PREVIEW_EDGE = 512

_heif_registered = False


def _register_heif() -> None:
    """Teach PIL about HEIF, once per process. A missing package is not fatal here.

    The caller still has the out-of-process decode to fall back on, and saying so through the
    normal `cannot identify image file` path is better than an ImportError that looks like a
    broken install.
    """
    global _heif_registered
    if _heif_registered:
        return
    _heif_registered = True
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass


def _open_raw(path: str):
    """Decode a RAW with LibRaw: embedded preview first, full demosaic if there is none."""
    import rawpy
    from PIL import Image

    with rawpy.imread(path) as raw:
        thumb = None
        try:
            thumb = raw.extract_thumb()
        except Exception:  # noqa: BLE001 - no thumbnail, or one LibRaw will not hand over
            thumb = None

        if thumb is not None:
            if thumb.format == rawpy.ThumbFormat.JPEG:
                image = Image.open(io.BytesIO(thumb.data))
            else:
                image = Image.fromarray(thumb.data)
            if max(image.size) >= MIN_PREVIEW_EDGE:
                return image.convert("RGB")

        # postprocess() must run before the `with` closes the LibRaw handle.
        return Image.fromarray(raw.postprocess()).convert("RGB")


def open_rgb(path: str, min_side: int = 0):
    """A PIL RGB image, from any format in the archive.

    Orientation is deliberately *not* applied: the caller rotates by its own angle.

    With `min_side`, a JPEG is decoded at the smallest of libjpeg's scales (1/2, 1/4, 1/8) whose
    width and height are both at least `min_side`, for a caller that resizes far below the full
    size anyway: a 24 Mpx photo decodes in a fifth of the time. Other formats decode in full.
    """
    from PIL import Image, ImageFile, UnidentifiedImageError

    # The archive holds scans whose LZW stream stops short of the declared size; take what is
    # there, as ImageMagick does.
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # PIL errors above 2x MAX_IMAGE_PIXELS (178,956,970 by default) as a decompression-bomb
    # guard. That guard is for untrusted uploads; this reads a local archive of flatbed
    # scans, where 187 Mpx is simply a high-dpi scan of a photo album page and the pipeline
    # failing on it is the only damage done. Raised rather than disabled: a corrupt header
    # asking for a terabyte should still fail fast instead of taking the container with it.
    # 500 Mpx is ~1.5 GB decoded per RGB copy, well above A3 at 1200 dpi (279 Mpx).
    Image.MAX_IMAGE_PIXELS = 500_000_000

    _register_heif()

    try:
        # .convert() forces pixel data to load, so an image outlives a caller's temp dir.
        image = Image.open(path)
        if min_side > 0:
            image.draft("RGB", (min_side, min_side))
        return image.convert("RGB")
    except UnidentifiedImageError:
        return _open_raw(path)


def open_display(path: str, angle=0, mirror=False, min_side: int = 0):
    """A PIL RGB image in the *display* frame: rotated CCW by `angle`, then flopped.

    `angle` is the **residual** the caller worked out — what the display frame needs minus whatever
    the decoder already applied — so this can rotate unconditionally without asking what format it
    just opened. For a HEIC that libheif already turned, the residual is normally zero.
    """
    from PIL import Image

    image = open_rgb(path, min_side)

    a = _normalize_angle(angle)
    # PIL rotates counter-clockwise for a positive angle, which is the display convention.
    if a:
        image = image.rotate(a, expand=True)
    if mirror:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def open_display_bgr(path: str, angle=0, mirror=False):
    """The display frame as a BGR numpy array — what InsightFace and the rest of cv2 expect."""
    import numpy as np

    return np.asarray(open_display(path, angle, mirror))[:, :, ::-1].copy()


def _normalize_angle(angle) -> int:
    """Anything that is not a supported quarter turn is treated as no rotation."""
    try:
        a = int(round(float(angle)))
    except (TypeError, ValueError):
        return 0
    a %= 360
    return a if a in (90, 180, 270) else 0
