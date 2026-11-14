"""`image.deblur`: the tiles, the worker op with the model stubbed, and the op with a scripted model
worker."""

import os
import sys
import types

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))


def save(path, picture):
    assert cv2.imwrite(str(path), picture)
    return str(path)


def test_the_tiles_cores_share_out_the_picture_and_are_shown_with_a_margin():
    import deblur

    covered = np.zeros((1100, 1500), np.int32)
    for (cy0, cy1, cx0, cx1), (y0, y1, x0, x1) in deblur.tiles(1100, 1500):
        covered[cy0:cy1, cx0:cx1] += 1
        assert y1 - y0 <= deblur.TILE and x1 - x0 <= deblur.TILE
        assert y0 == max(cy0 - deblur.MARGIN, 0) and x1 == min(cx1 + deblur.MARGIN, 1500)
    assert (covered == 1).all()


def checks(height, width):
    """A checkerboard of single pixels, +1 and -1: detail as fine as a picture has."""
    return np.where((np.indices((height, width)).sum(axis=0) % 2) == 0, 1.0, -1.0).astype(np.float32)[:, :, None]


@pytest.fixture
def model(monkeypatch):
    """The worker's deblurrer stubbed: what it is shown comes back with a checkerboard of a tenth
    added, and `shift` more on every pixel; the shapes it was shown are kept."""
    import model_registry as registry

    stub = types.SimpleNamespace(shown=[], shift=0.0)

    def run(rgb):
        stub.shown.append(rgb.shape)
        height, width = rgb.shape[:2]
        return np.clip(rgb + 0.1 * checks(height, width) + stub.shift, 0, 1)

    monkeypatch.setattr(registry, "_family_locks", {})
    monkeypatch.setattr(registry, "deblur_device", lambda: "cpu")
    monkeypatch.setattr(registry, "deblurrer", lambda device, blur: types.SimpleNamespace(run=run))
    return stub


def test_the_worker_op_adds_the_detail_by_strength_at_16_bits_and_keeps_the_alpha(model, tmp_path):
    import inference_ops

    picture = np.full((700, 900, 4), 20000, np.uint16)
    picture[:, :, 3] = 1234
    out = tmp_path / "out.tiff"
    answer = inference_ops.deblur(save(tmp_path / "in.png", picture), str(out), "motion", 0.5, 2048, "tiff")
    assert answer == {"model": "restormer-motion_deblurring", "width": 900, "height": 700, "channels": 4, "depth": 16}
    assert all(shape[2] == 3 and max(shape[:2]) <= 384 for shape in model.shown)
    result = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert result.dtype == np.uint16 and result.shape == picture.shape and (result[:, :, 3] == 1234).all()
    # Away from the tiles' seams and the picture's edge the checkerboard comes through at half.
    expected = 20000 + 0.05 * 65535 * checks(700, 900)[:, :, 0]
    assert np.abs(result[100:200, 100:200, 0] - expected[100:200, 100:200]).max() < 0.01 * 65535


def test_a_shift_of_the_pictures_tones_is_not_kept(model, tmp_path):
    import inference_ops

    model.shift = 0.1
    out = tmp_path / "out.png"
    inference_ops.deblur(save(tmp_path / "in.png", np.full((300, 300, 3), 100, np.uint8)), str(out), "focus", 1.0, 2048, "png")
    result = cv2.imread(str(out), cv2.IMREAD_UNCHANGED).astype(np.float32)
    assert abs(result[50:250, 50:250].mean() - 100) < 1


def test_a_picture_longer_than_the_edge_is_shown_shrunk_and_changed_whole(model, tmp_path):
    import inference_ops

    out = tmp_path / "out.png"
    inference_ops.deblur(save(tmp_path / "in.png", np.full((600, 1000, 3), 100, np.uint8)), str(out), "focus", 1.0, 500, "png")
    assert all(shape[0] <= 300 and shape[1] <= 384 for shape in model.shown)
    result = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert result.shape == (600, 1000, 3) and result[100:500, 100:900].std() > 5


def test_the_worker_op_deblurs_a_grey_picture_as_grey(model, tmp_path):
    import inference_ops

    out = tmp_path / "out.png"
    inference_ops.deblur(save(tmp_path / "in.png", np.full((100, 120), 100, np.uint8)), str(out), "focus", 1.0, 2048, "png")
    result = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert result.ndim == 2 and np.abs(result[20:80, 20:100].astype(int) - (100 + np.rint(25.5 * checks(100, 120)[20:80, 20:100, 0]))).max() <= 2


def test_a_blur_without_weights_is_the_requests_fault():
    import model_registry as registry

    with pytest.raises(ValueError):
        registry.deblur_model("wobble")


def write_grey(request):
    """A scripted model worker's `deblur`: the picture comes back mid-grey, at its own size."""
    picture = cv2.imread(request["path"], cv2.IMREAD_UNCHANGED)
    cv2.imwrite(request["out"], np.full(picture.shape, 128, picture.dtype))
    return {"model": "restormer-motion_deblurring", "width": picture.shape[1], "height": picture.shape[0], "channels": 3, "depth": 8}


async def test_the_op_writes_the_sharper_picture_where_asked(client, dirs, fake_worker):
    ro, rw = dirs
    fake_worker.on("deblur", write_grey)
    image = save(ro / "scan.png", np.zeros((60, 80, 3), np.uint8))
    r = await client.post("/v0/image/deblur", json={"file": {"path": image}, "blur": "motion", "output": {"path": str(rw / "sharp.tif"), "format": "tiff"}})
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["result"] == {"path": str(rw / "sharp.tif"), "width": 80, "height": 60, "channels": 3, "depth": 8, "bytes": (rw / "sharp.tif").stat().st_size}
    assert body["meta"]["producer"] == "restormer-motion_deblurring"
    (request,) = [q for q in fake_worker.requests if q["op"] == "deblur"]
    assert (request["blur"], request["strength"], request["edge"], request["format"]) == ("motion", 1.0, 2048, "tiff")
    assert not [name for name in os.listdir(rw) if name.startswith(".ems-")]


async def test_a_picture_the_worker_cannot_load_is_unsupported(client, dirs, fake_worker):
    ro, rw = dirs

    def refuse(request):
        raise RuntimeError("could not load image: OpenCV has no reader for photo.heic")

    fake_worker.on("deblur", refuse)
    heic = ro / "photo.heic"
    heic.write_bytes(b"\0\0\0\x18ftypheic")
    r = await client.post("/v0/image/deblur", json={"file": {"path": str(heic)}, "output": {"path": str(rw / "out.png")}})
    assert r.json()["error"]["code"] == "unsupported" and r.json()["error"]["permanent"] is True
    assert not (rw / "out.png").exists()


@pytest.mark.parametrize("change, message", [
    ({"file": {"angle": 180}}, "angle 0, no mirror"),
    ({"output": {"format": "jpeg"}}, "png or tiff"),
    ({"blur": "wobble"}, "blur"),
    ({"strength": 2}, "strength"),
    ({"edge": 100}, "edge"),
])
async def test_what_would_turn_the_picture_or_lose_its_pixels_is_refused(client, dirs, change, message):
    ro, rw = dirs
    body = {"file": {"path": save(ro / "scan.png", np.zeros((60, 80, 3), np.uint8))}, "output": {"path": str(rw / "out.png")}}
    for key, value in change.items():
        body[key] = {**body[key], **value} if isinstance(value, dict) else value
    r = await client.post("/v0/image/deblur", json=body)
    assert r.status_code == 400 and message in r.json()["error"]["message"], r.json()
