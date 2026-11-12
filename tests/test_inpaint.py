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


# LaMa: the crops the tool cuts, the worker op, and the op with a scripted model worker.


def covered(marked, groups, crops):
    """How many crops take each marked pixel."""
    times = np.zeros(marked.shape, np.int32)
    for crop in crops:
        x0, y0, x1, y1 = crop.core
        times[y0:y1, x0:x1] += marked[y0:y1, x0:x1] & (groups[y0:y1, x0:x1] == crop.group)
    return times[marked]


def test_specks_far_apart_are_crops_of_their_own_with_a_margin_around_them():
    from ems.tools import inpaint

    marked = np.zeros((2000, 3000), bool)
    marked[100:104, 200:205] = True
    marked[1500:1510, 2500:2502] = True
    marked[110:112, 300:303] = True                      # within two margins of the first
    groups, crops = inpaint.plan(marked)
    assert [(c.box, c.core == c.box, c.scale) for c in crops] == [
        ((72, 0, 431, 240), True, 1.0),
        ((2372, 1372, 2630, 1638), True, 1.0),
    ]
    assert (covered(marked, groups, crops) == 1).all()


def test_a_long_thin_scratch_is_cut_into_tiles_that_take_each_pixel_once():
    from ems.tools import inpaint

    marked = np.zeros((2400, 3600), bool)
    cv2.line(marked.view(np.uint8), (50, 100), (3500, 2300), 1, 5)
    groups, crops = inpaint.plan(marked)
    assert len(crops) > 3 and all(c.scale == 1.0 for c in crops)
    assert all(c.box[2] - c.box[0] <= inpaint.LIMIT and c.box[3] - c.box[1] <= inpaint.LIMIT for c in crops)
    # Every core is shown with a margin around it, where the picture has one.
    for c in crops:
        assert c.box[0] == max(c.core[0] - inpaint.MARGIN, 0) and c.box[3] == min(c.core[3] + inpaint.MARGIN, 2400)
    assert (covered(marked, groups, crops) == 1).all()


def test_a_hole_wider_than_a_margin_is_shown_whole_and_shrunk_to_the_limit():
    from ems.tools import inpaint

    marked = np.zeros((3000, 4000), bool)
    cv2.circle(marked.view(np.uint8), (2000, 1500), 700, 1, -1)
    groups, crops = inpaint.plan(marked)
    assert len(crops) == 1
    (crop,) = crops
    assert crop.box == (1172, 672, 2829, 2329) and crop.scale == pytest.approx(1024 / 1657)
    assert inpaint.shown_size(crop) == (1024, 1024)
    assert (covered(marked, groups, crops) == 1).all()


def test_the_worker_op_paints_each_crop_in_rgb_and_writes_16_bits(monkeypatch, tmp_path):
    import sys
    import types

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))
    import inference_ops
    import model_registry as registry

    seen = []

    def fill(picture, mask):
        seen.append((picture.dtype, float(picture[0, 0, 0]), float(picture[0, 0, 2]), int(mask.sum())))
        painted = picture.copy()
        painted[mask] = (1.0, 0.0, 0.0)                  # red, as RGB
        return painted

    monkeypatch.setattr(registry, "_family_locks", {})
    monkeypatch.setattr(registry, "inpaint_device", lambda: "cpu")
    monkeypatch.setattr(registry, "lama", lambda device: types.SimpleNamespace(fill=fill))
    picture = np.zeros((40, 48, 3), np.uint8)
    picture[:, :, 2] = 255                               # red in cv2's BGR
    mask = np.zeros((40, 48), np.uint8)
    mask[10:20, 10:20] = 255
    crop = {"image": save(tmp_path / "image.png", picture), "mask": save(tmp_path / "mask.png", mask), "out": str(tmp_path / "out.png")}
    assert inference_ops.inpaint([crop]) == {"model": "big-lama"}
    assert seen == [(np.float32, 1.0, 0.0, 100)]
    out = cv2.imread(crop["out"], cv2.IMREAD_UNCHANGED)
    assert out.dtype == np.uint16 and (out[15, 15] == (0, 0, 65535)).all() and (out[0, 0] == (0, 0, 65535)).all()
    assert inference_ops.inpaint([]) == {"model": "big-lama"}


def paint_flat(colour):
    """A scripted model worker's `inpaint`: every masked pixel of a crop painted `colour` (RGB, 0..1)."""

    def handle(request):
        for crop in request["crops"]:
            picture = cv2.imread(crop["image"], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            mask = cv2.imread(crop["mask"], cv2.IMREAD_GRAYSCALE)
            assert picture.shape[:2] == mask.shape and max(mask.shape) <= 1024
            out = (picture.astype(np.float64) / np.iinfo(picture.dtype).max * 65535).astype(np.uint16)
            out[mask != 0] = [round(v * 65535) for v in reversed(colour)]
            cv2.imwrite(crop["out"], out)
        return {"model": "big-lama"}

    return handle


async def test_lama_paints_only_the_marked_pixels_of_a_large_16_bit_scan(client, dirs, fake_worker):
    ro, rw = dirs
    fake_worker.on("inpaint", paint_flat((0.5, 0.25, 0.75)))
    picture = np.full((2000, 2600, 3), 12345, np.uint16)
    mask = np.zeros((2000, 2600), np.uint8)
    cv2.line(mask, (20, 30), (2500, 1900), 255, 3)       # tiled
    cv2.circle(mask, (300, 1700), 400, 255, -1)          # shrunk
    image, mask_path = save(ro / "master.tif", picture), save(ro / "defects.png", mask)
    r = await client.post("/v0/image/inpaint", json=inpaint_request(image, mask_path, rw / "out.tif", method="lama", output={"format": "tiff"}))
    assert r.status_code == 200, r.json()
    assert r.json()["meta"]["producer"] == f"big-lama+opencv/{cv2.__version__}"
    assert r.json()["result"]["filled"] == int(np.count_nonzero(mask)) and r.json()["result"]["depth"] == 16
    out = cv2.imread(str(rw / "out.tif"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(out[mask == 0], picture[mask == 0])
    assert (out[mask != 0] == (49151, 16384, 32768)).all()   # BGR of the colour painted
    (request,) = [q for q in fake_worker.requests if q["op"] == "inpaint"]
    assert len(request["crops"]) > 2


async def test_lama_fills_a_grey_picture_and_answers_an_empty_mask_without_crops(client, dirs, fake_worker):
    ro, rw = dirs
    fake_worker.on("inpaint", paint_flat((0.2, 0.2, 0.2)))
    picture = gradient(depth=8, channels=1)
    mask = scratch()
    image, mask_path = save(ro / "grey.png", picture), save(ro / "mask.png", mask)
    empty = save(ro / "empty.png", np.zeros_like(mask))
    r = await client.post("/v0/image/inpaint", json=inpaint_request(image, mask_path, rw / "out.png", method="lama"))
    assert r.status_code == 200, r.json()
    out = cv2.imread(str(rw / "out.png"), cv2.IMREAD_UNCHANGED)
    assert out.ndim == 2 and np.array_equal(out[mask == 0], picture[mask == 0]) and (out[mask != 0] == 51).all()
    r = await client.post("/v0/image/inpaint", json=inpaint_request(image, empty, rw / "same.png", method="lama"))
    assert r.json()["result"]["filled"] == 0 and r.json()["meta"]["producer"].startswith("big-lama+")
    assert fake_worker.requests[-1]["crops"] == []
    assert np.array_equal(cv2.imread(str(rw / "same.png"), cv2.IMREAD_UNCHANGED), picture)


async def test_lama_takes_no_radius(client, dirs):
    ro, rw = dirs
    image, mask = save(ro / "master.png", gradient()), save(ro / "mask.png", scratch())
    r = await client.post("/v0/image/inpaint", json=inpaint_request(image, mask, rw / "out.png", method="lama", radius=3))
    assert r.status_code == 400 and "lama takes none" in r.json()["error"]["message"]
