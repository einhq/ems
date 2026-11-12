"""`image.inpaint`: OpenCV's fill at the picture's own depth, only where the mask marks, and the
request's refusals."""

import os

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")


def gradient(height=120, width=160, depth=16, channels=3):
    """A smooth picture: each channel a ramp in its own direction, so a fill can be checked against it."""
    top = 65535 if depth == 16 else 255
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    ramps = [x / (width - 1), y / (height - 1), (x + y) / (width + height - 2), 1 - x / (width - 1)]
    planes = [np.round(ramp * top * 0.8 + top * 0.1) for ramp in ramps[:channels]]
    picture = np.dstack(planes) if channels > 1 else planes[0]
    return picture.astype(np.uint16 if depth == 16 else np.uint8)


def scratch(height=120, width=160):
    mask = np.zeros((height, width), np.uint8)
    cv2.line(mask, (10, 20), (150, 100), 255, 2)
    cv2.circle(mask, (40, 90), 4, 255, -1)
    return mask


def save(path, picture):
    assert cv2.imwrite(str(path), picture)
    return str(path)


def inpaint_request(image, mask, out, **kw):
    return {"file": {"path": str(image)}, "mask": {"path": str(mask)}, "output": {"path": str(out), **kw.pop("output", {})}, **kw}


async def test_a_16_bit_tiff_is_filled_at_16_bits_and_untouched_outside_the_mask(client, dirs):
    ro, rw = dirs
    picture = gradient()
    damaged = picture.copy()
    mask = scratch()
    damaged[mask != 0] = 0
    image, mask_path = save(ro / "master.tif", damaged), save(ro / "defects.png", mask)
    r = await client.post("/v0/image/inpaint", json=inpaint_request(image, mask_path, rw / "inpainted.tif", output={"format": "tiff"}))
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["result"] == {"path": str(rw / "inpainted.tif"), "width": 160, "height": 120, "channels": 3, "depth": 16,
                              "filled": int(np.count_nonzero(mask)), "bytes": (rw / "inpainted.tif").stat().st_size}
    assert body["meta"]["producer"] == f"opencv/{cv2.__version__}"
    out = cv2.imread(str(rw / "inpainted.tif"), cv2.IMREAD_UNCHANGED)
    assert out.dtype == np.uint16 and out.shape == damaged.shape
    assert np.array_equal(out[mask == 0], damaged[mask == 0])
    # The scratch is filled from the ramps around it, not left black.
    error = np.abs(out[mask != 0].astype(np.int64) - picture[mask != 0].astype(np.int64))
    assert error.max() < 65535 * 0.05
    assert not [name for name in os.listdir(rw) if name.startswith(".ems-")]


@pytest.mark.parametrize("method", ["telea", "ns"])
def test_an_8_bit_picture_is_filled_as_opencv_fills_its_three_channels_at_once(tmp_path, method):
    from ems.tools import inpaint

    picture = gradient(depth=8)
    mask = scratch()
    expected = cv2.inpaint(picture, mask, 3, cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS)
    assert np.array_equal(inpaint.fill(picture.copy(), mask != 0, method, 3), expected)


def test_an_alpha_channel_is_kept_as_it_is(tmp_path):
    from ems.tools import inpaint

    picture = gradient(channels=4)
    mask = scratch()
    filled = inpaint.fill(picture.copy(), mask != 0, "telea", 3)
    assert np.array_equal(filled[:, :, 3], picture[:, :, 3])
    assert not np.array_equal(filled[:, :, :3], picture[:, :, :3])


async def test_a_grey_picture_and_an_empty_mask_come_back_as_they_are(client, dirs):
    ro, rw = dirs
    picture = gradient(depth=8, channels=1)
    image = save(ro / "grey.png", picture)
    mask = save(ro / "none.png", np.zeros(picture.shape, np.uint8))
    r = await client.post("/v0/image/inpaint", json=inpaint_request(image, mask, rw / "out.png"))
    assert r.status_code == 200, r.json()
    assert r.json()["result"]["filled"] == 0 and r.json()["result"]["channels"] == 1 and r.json()["result"]["depth"] == 8
    assert np.array_equal(cv2.imread(str(rw / "out.png"), cv2.IMREAD_UNCHANGED), picture)


async def test_a_16_bit_mask_marks_every_pixel_that_is_not_0(client, dirs):
    ro, rw = dirs
    mask = np.zeros((120, 160), np.uint16)
    mask[50:52, 30:90] = 1
    image, mask_path = save(ro / "master.png", gradient()), save(ro / "mask.png", mask)
    r = await client.post("/v0/image/inpaint", json=inpaint_request(image, mask_path, rw / "out.png", method="ns", radius=5))
    assert r.status_code == 200, r.json()
    assert r.json()["result"]["filled"] == 120


@pytest.mark.parametrize("mask, message", [
    (np.zeros((120, 160, 3), np.uint8), "a mask has one"),
    (np.zeros((60, 80), np.uint8), "the mask is 80x60 and the file 160x120"),
])
async def test_a_mask_of_colour_or_of_another_size_is_refused(client, dirs, mask, message):
    ro, rw = dirs
    image, mask_path = save(ro / "master.png", gradient()), save(ro / "mask.png", mask)
    r = await client.post("/v0/image/inpaint", json=inpaint_request(image, mask_path, rw / "out.png"))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request" and message in r.json()["error"]["message"]
    assert not (rw / "out.png").exists()


async def test_a_picture_of_floats_and_one_opencv_cannot_read_are_unsupported_and_broken_bytes_undecodable(client, dirs):
    ro, rw = dirs
    mask = save(ro / "mask.png", scratch())
    floats = save(ro / "floats.tif", gradient().astype(np.float32))
    heic = ro / "photo.heic"
    heic.write_bytes(b"\0\0\0\x18ftypheic" + b"\0" * 64)
    broken = ro / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
    for image, code in ((floats, "unsupported"), (heic, "unsupported"), (broken, "undecodable")):
        r = await client.post("/v0/image/inpaint", json=inpaint_request(image, mask, rw / "out.png"))
        assert r.json()["error"]["code"] == code, (image, r.json())
        assert r.json()["error"]["permanent"] is True


@pytest.mark.parametrize("change, message", [
    ({"file": {"angle": 90}}, "angle 0, no mirror"),
    ({"mask": {"mirror": True}}, "angle 0, no mirror"),
    ({"output": {"format": "jpeg"}}, "png or tiff"),
    ({"output": {"format": "png", "quality": 90}}, "takes no quality"),
    ({"output": {"format": "png", "path": "out.tif"}}, "named as tiff but asked to hold png"),
])
async def test_what_would_turn_the_picture_or_lose_its_pixels_is_refused(client, dirs, change, message):
    ro, rw = dirs
    image, mask = save(ro / "master.png", gradient()), save(ro / "mask.png", scratch())
    body = inpaint_request(image, mask, rw / "out.png")
    for key, fields in change.items():
        if "path" in fields:
            fields = {**fields, "path": str(rw / fields["path"])}
        body[key] = {**body[key], **fields}
    r = await client.post("/v0/image/inpaint", json=body)
    assert r.status_code == 400 and message in r.json()["error"]["message"], r.json()


async def test_a_radius_outside_1_to_64_is_refused(client, dirs):
    ro, rw = dirs
    body = inpaint_request(ro / "master.png", ro / "mask.png", rw / "out.png", radius=0)
    r = await client.post("/v0/image/inpaint", json=body)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"
