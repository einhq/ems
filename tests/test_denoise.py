"""`image.denoise`: the tiles, the worker op with the model stubbed, and the op with a scripted
model worker."""

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
    import denoise

    covered = np.zeros((1100, 1500), np.int32)
    for (cy0, cy1, cx0, cx1), (y0, y1, x0, x1) in denoise.tiles(1100, 1500):
        covered[cy0:cy1, cx0:cx1] += 1
        assert y1 - y0 <= denoise.TILE and x1 - x0 <= denoise.TILE
        assert y0 == max(cy0 - denoise.MARGIN, 0) and x1 == min(cx1 + denoise.MARGIN, 1500)
    assert (covered == 1).all()


@pytest.fixture
def flat_model(monkeypatch):
    """The worker's denoiser stubbed: every pixel it is shown answers mid-grey."""
    import model_registry as registry

    shown = []

    def run(rgb):
        shown.append(rgb.shape)
        return np.full(rgb.shape, 0.5, np.float32)

    monkeypatch.setattr(registry, "_family_locks", {})
    monkeypatch.setattr(registry, "denoise_device", lambda: "cpu")
    monkeypatch.setattr(registry, "denoiser", lambda device, noise: types.SimpleNamespace(run=run))
    return shown


def test_the_worker_op_blends_by_strength_at_16_bits_and_keeps_the_alpha(flat_model, tmp_path):
    import inference_ops

    picture = np.zeros((700, 900, 4), np.uint16)
    picture[:, :, 3] = 1234
    out = tmp_path / "out.tiff"
    answer = inference_ops.denoise(save(tmp_path / "in.png", picture), str(out), "25", 0.5, "tiff")
    assert answer == {"model": "scunet_color_25", "width": 900, "height": 700, "channels": 4, "depth": 16}
    assert all(shape[2] == 3 and max(shape[:2]) <= 512 for shape in flat_model)
    result = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert result.dtype == np.uint16 and result.shape == picture.shape
    assert (result[:, :, :3] == 16384).all() and (result[:, :, 3] == 1234).all()


def test_the_worker_op_denoises_a_grey_picture_as_grey(flat_model, tmp_path):
    import inference_ops

    out = tmp_path / "out.png"
    inference_ops.denoise(save(tmp_path / "in.png", np.full((100, 120), 255, np.uint8)), str(out), "15", 1.0, "png")
    result = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert result.ndim == 2 and (result == 128).all()


def test_a_noise_without_weights_is_the_requests_fault():
    import model_registry as registry

    with pytest.raises(ValueError):
        registry.denoise_model("50")


def write_grey(request):
    """A scripted model worker's `denoise`: the picture comes back mid-grey, at its own size."""
    picture = cv2.imread(request["path"], cv2.IMREAD_UNCHANGED)
    cv2.imwrite(request["out"], np.full(picture.shape, 128, picture.dtype))
    return {"model": "scunet_color_15", "width": picture.shape[1], "height": picture.shape[0], "channels": 3, "depth": 8}


async def test_the_op_writes_the_denoised_picture_where_asked(client, dirs, fake_worker):
    ro, rw = dirs
    fake_worker.on("denoise", write_grey)
    image = save(ro / "scan.png", np.zeros((60, 80, 3), np.uint8))
    r = await client.post("/v0/image/denoise", json={"file": {"path": image}, "output": {"path": str(rw / "quiet.tif"), "format": "tiff"}})
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["result"] == {"path": str(rw / "quiet.tif"), "width": 80, "height": 60, "channels": 3, "depth": 8, "bytes": (rw / "quiet.tif").stat().st_size}
    assert body["meta"]["producer"] == "scunet_color_15"
    (request,) = [q for q in fake_worker.requests if q["op"] == "denoise"]
    assert (request["noise"], request["strength"], request["format"]) == ("15", 0.5, "tiff")
    assert not [name for name in os.listdir(rw) if name.startswith(".ems-")]


async def test_a_picture_the_worker_cannot_load_is_unsupported(client, dirs, fake_worker):
    ro, rw = dirs

    def refuse(request):
        raise RuntimeError("could not load image: OpenCV has no reader for photo.heic")

    fake_worker.on("denoise", refuse)
    heic = ro / "photo.heic"
    heic.write_bytes(b"\0\0\0\x18ftypheic")
    r = await client.post("/v0/image/denoise", json={"file": {"path": str(heic)}, "output": {"path": str(rw / "out.png")}})
    assert r.json()["error"]["code"] == "unsupported" and r.json()["error"]["permanent"] is True
    assert not (rw / "out.png").exists()


@pytest.mark.parametrize("change, message", [
    ({"file": {"angle": 180}}, "angle 0, no mirror"),
    ({"output": {"format": "jpeg"}}, "png or tiff"),
    ({"noise": "50"}, "noise"),
    ({"strength": 2}, "strength"),
])
async def test_what_would_turn_the_picture_or_lose_its_pixels_is_refused(client, dirs, change, message):
    ro, rw = dirs
    body = {"file": {"path": save(ro / "scan.png", np.zeros((60, 80, 3), np.uint8))}, "output": {"path": str(rw / "out.png")}}
    for key, value in change.items():
        body[key] = {**body[key], **value} if isinstance(value, dict) else value
    r = await client.post("/v0/image/denoise", json=body)
    assert r.status_code == 400 and message in r.json()["error"]["message"], r.json()
