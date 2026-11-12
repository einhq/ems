"""`image.unfade`: a faded picture's channels stretched and its neutral tones balanced, at its own
depth, and the request's refusals."""

import os

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")


def scene(depth=16):
    """A picture as it was, in OpenCV's BGR: a grey ramp across the top, colour patches below."""
    top = 65535 if depth == 16 else 255
    h, w = 120, 160
    picture = np.zeros((h, w, 3), np.float64)
    picture[:60] = np.linspace(0.02, 0.98, w)[None, :, None]
    for i, colour in enumerate([(0.2, 0.3, 0.8), (0.1, 0.7, 0.2), (0.8, 0.4, 0.1), (0.5, 0.5, 0.5)]):
        picture[60:, i * 40:(i + 1) * 40] = colour
    return np.round(picture * top).astype(np.uint16 if depth == 16 else np.uint8)


def faded(picture):
    """`picture` faded as a print's dyes fade: each channel squeezed into a band of its own and bent."""
    top = np.iinfo(picture.dtype).max
    x = picture.astype(np.float64) / top
    # Blue, green, red: the red channel's dye least faded, so the print turns red.
    bands = [(0.25, 0.75, 1.3), (0.2, 0.8, 1.15), (0.08, 0.95, 0.9)]
    out = np.stack([lo + (hi - lo) * x[:, :, c] ** g for c, (lo, hi, g) in enumerate(bands)], axis=-1)
    return np.round(out * top).astype(picture.dtype)


def save(path, picture):
    assert cv2.imwrite(str(path), picture)
    return str(path)


def unfade_request(image, out, **kw):
    return {"file": {"path": str(image)}, "output": {"path": str(out), **kw.pop("output", {})}, **kw}


def grey_spread(picture):
    """How far the grey ramp's channels lie apart, at most, as a share of the range."""
    ramp = picture[:60, 20:140].astype(np.float64) / np.iinfo(picture.dtype).max
    return float((ramp.max(axis=2) - ramp.min(axis=2)).max())


@pytest.mark.parametrize("depth", [8, 16])
async def test_a_faded_picture_gets_its_greys_and_contrast_back_at_its_own_depth(client, dirs, depth):
    ro, rw = dirs
    original = scene(depth)
    image = save(ro / "print.tif", faded(original))
    r = await client.post("/v0/image/unfade", json=unfade_request(image, rw / "unfaded.tif", output={"format": "tiff"}))
    assert r.status_code == 200, r.json()
    body = r.json()
    result = body["result"]
    assert {k: result[k] for k in ("path", "width", "height", "channels", "depth", "bytes")} == {
        "path": str(rw / "unfaded.tif"), "width": 160, "height": 120, "channels": 3, "depth": depth, "bytes": (rw / "unfaded.tif").stat().st_size}
    assert len(result["levels"]) == 3 and len(result["gammas"]) == 3
    # Red first: the red channel, the least faded, is stretched least.
    assert result["levels"][0][1] - result["levels"][0][0] > result["levels"][2][1] - result["levels"][2][0]
    assert body["meta"]["producer"] == f"unfade/1+opencv/{cv2.__version__}"
    out = cv2.imread(str(rw / "unfaded.tif"), cv2.IMREAD_UNCHANGED)
    assert out.dtype == original.dtype and out.shape == original.shape
    assert grey_spread(faded(original)) > 0.15
    assert grey_spread(out) < 0.05
    # The ramp runs from near black to near white again.
    top = np.iinfo(out.dtype).max
    assert out[:60, 2].max() < 0.1 * top and out[:60, -3].min() > 0.9 * top
    assert not [name for name in os.listdir(rw) if name.startswith(".ems-")]


async def test_without_balance_only_the_levels_move(client, dirs):
    ro, rw = dirs
    image = save(ro / "print.png", faded(scene(8)))
    r = await client.post("/v0/image/unfade", json=unfade_request(image, rw / "levels.png", balance=False))
    assert r.status_code == 200, r.json()
    assert r.json()["result"]["gammas"] == [1.0, 1.0, 1.0]


async def test_strength_0_is_the_picture_as_it_was(client, dirs):
    ro, rw = dirs
    picture = faded(scene(16))
    image = save(ro / "print.tif", picture)
    r = await client.post("/v0/image/unfade", json=unfade_request(image, rw / "same.png", strength=0))
    assert r.status_code == 200, r.json()
    assert np.array_equal(cv2.imread(str(rw / "same.png"), cv2.IMREAD_UNCHANGED), picture)


async def test_an_alpha_is_kept_and_a_grey_picture_only_stretched(client, dirs):
    ro, rw = dirs
    bgr = faded(scene(8))
    alpha = np.full(bgr.shape[:2], 7, np.uint8)
    image = save(ro / "print.png", np.dstack([bgr, alpha]))
    r = await client.post("/v0/image/unfade", json=unfade_request(image, rw / "alpha.png"))
    assert r.status_code == 200, r.json()
    assert r.json()["result"]["channels"] == 4
    assert np.array_equal(cv2.imread(str(rw / "alpha.png"), cv2.IMREAD_UNCHANGED)[:, :, 3], alpha)

    grey = np.round(np.linspace(80, 170, 160)[None, :].repeat(120, axis=0)).astype(np.uint8)
    image = save(ro / "grey.png", grey)
    r = await client.post("/v0/image/unfade", json=unfade_request(image, rw / "grey-out.png"))
    assert r.status_code == 200, r.json()
    result = r.json()["result"]
    assert result["channels"] == 1 and result["gammas"] == [1.0]
    out = cv2.imread(str(rw / "grey-out.png"), cv2.IMREAD_UNCHANGED)
    assert out.min() <= 2 and out.max() >= 253


@pytest.mark.parametrize("change, message", [
    ({"file": {"angle": 90}}, "as it is stored"),
    ({"output": {"format": "jpeg"}}, "png or tiff"),
    ({"output": {"quality": 80}}, "lossless"),
    ({"clip": 10}, "clip"),
    ({"strength": 2}, "strength"),
])
async def test_what_is_refused(client, dirs, change, message):
    ro, rw = dirs
    image = save(ro / "print.png", faded(scene(8)))
    body = unfade_request(image, rw / "out.png")
    for key, value in change.items():
        body[key] = {**body[key], **value} if isinstance(value, dict) else value
    r = await client.post("/v0/image/unfade", json=body)
    assert r.status_code == 400, r.json()
    assert message in str(r.json())
    assert not (rw / "out.png").exists()
