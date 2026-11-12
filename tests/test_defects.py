"""`image.defects`: the detector's tiles, the picture it is shown, the mask made of its answer, and
the op with a scripted model worker."""

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


def test_tiles_put_together_give_every_pixel_its_own_answer():
    import defects

    detector = object.__new__(defects.Detector)
    detector._run = lambda tile: tile * 1.0                  # a network that answers the picture itself
    grey = np.random.default_rng(0).random((1000, 1700), dtype=np.float32)
    assert np.array_equal(detector.probabilities(grey), grey)
    assert np.array_equal(detector.probabilities(grey, dark=True), np.maximum(grey, 1 - grey))


def test_the_detector_is_shown_the_picture_in_grey_with_its_shorter_side_the_scale():
    from ems.tools import defects

    picture = np.zeros((1200, 1600, 3), np.uint16)
    picture[:, :800] = (65535, 65535, 65535)
    shown = defects.shrink(picture, 640)
    assert shown.dtype == np.uint16 and shown.shape == (640, 853)
    assert shown[10, 10] == 65535 and shown[10, 800] == 0
    assert defects.shown_size(900, 3000, 256) == (256, 853)


def test_the_mask_is_the_answer_at_the_pictures_size_marked_from_the_threshold():
    from ems.tools import defects

    found = np.zeros((64, 80), np.uint16)
    found[:, :40] = round(0.4 * 65535)
    found[:, 40:] = round(0.39 * 65535)
    mask = defects.mask(found, 800, 640, 0.4)
    assert mask.shape == (640, 800) and mask.dtype == np.uint8
    assert (mask[:, :390] == 255).all() and (mask[:, 410:] == 0).all()


def test_the_worker_op_writes_the_probabilities_as_16_bits(monkeypatch, tmp_path):
    import inference_ops
    import model_registry as registry

    seen = []

    def probabilities(grey, dark):
        seen.append((grey.dtype, grey.shape, float(grey.max()), dark))
        return np.where(grey > 0.5, 1.0, 0.25).astype(np.float32)

    monkeypatch.setattr(registry, "_family_locks", {})
    monkeypatch.setattr(registry, "defects_device", lambda: "cpu")
    monkeypatch.setattr(registry, "defects", lambda device: types.SimpleNamespace(probabilities=probabilities))
    shown = np.zeros((40, 60), np.uint16)
    shown[:, 30:] = 65535
    out = tmp_path / "found.png"
    assert inference_ops.detect_defects(save(tmp_path / "shown.png", shown), str(out), dark=True) == {"model": "bopb-detection"}
    assert seen == [(np.float32, (40, 60), 1.0, True)]
    found = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert found.dtype == np.uint16 and found[0, 0] == 16384 and found[0, 59] == 65535


def mark_left_half(request):
    """A scripted model worker's `detect_defects`: the left half of what it is shown is damage."""
    shown = cv2.imread(request["path"], cv2.IMREAD_UNCHANGED)
    assert shown.dtype == np.uint16 and min(shown.shape) == 256
    found = np.zeros(shown.shape, np.uint16)
    found[:, : shown.shape[1] // 2] = 65535
    cv2.imwrite(request["out"], found)
    return {"model": "bopb-detection"}


async def test_a_16_bit_scan_gets_a_mask_of_its_own_size(client, dirs, fake_worker):
    ro, rw = dirs
    fake_worker.on("detect_defects", mark_left_half)
    image = save(ro / "master.tif", np.full((1000, 1400, 3), 30000, np.uint16))
    r = await client.post("/v0/image/defects", json={"file": {"path": image}, "scale": 256, "dark": True, "output": {"path": str(rw / "defects.png")}})
    assert r.status_code == 200, r.json()
    body = r.json()
    mask = cv2.imread(str(rw / "defects.png"), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (1000, 1400) and mask.dtype == np.uint8
    assert (mask[:, :690] == 255).all() and (mask[:, 710:] == 0).all()
    assert body["result"] == {"path": str(rw / "defects.png"), "width": 1400, "height": 1000, "marked": int(np.count_nonzero(mask)),
                              "fraction": pytest.approx(np.count_nonzero(mask) / mask.size)}
    assert body["meta"]["producer"] == f"bopb-detection+opencv/{cv2.__version__}"
    (request,) = [q for q in fake_worker.requests if q["op"] == "detect_defects"]
    assert request["dark"] is True
    assert not [name for name in os.listdir(rw) if name.startswith(".ems-")]


@pytest.mark.parametrize("change, message", [
    ({"file": {"angle": 90}}, "angle 0, no mirror"),
    ({"output": {"format": "tiff"}}, "written as png"),
    ({"output": {"path": "defects.jpg"}}, "written as png"),
    ({"scale": 128}, "scale"),
    ({"threshold": 1.5}, "threshold"),
])
async def test_what_would_misplace_the_mask_or_cannot_be_written_is_refused(client, dirs, change, message):
    ro, rw = dirs
    body = {"file": {"path": save(ro / "master.png", np.zeros((300, 400), np.uint8))}, "output": {"path": str(rw / "defects.png")}}
    for key, value in change.items():
        if isinstance(value, dict):
            value = {**body[key], **value}
            if "path" in value and key == "output":
                value["path"] = str(rw / value["path"].split("/")[-1])
        body[key] = value
    r = await client.post("/v0/image/defects", json=body)
    assert r.status_code == 400 and message in r.json()["error"]["message"], r.json()


async def test_a_picture_opencv_cannot_read_is_unsupported(client, dirs):
    ro, rw = dirs
    heic = ro / "photo.heic"
    heic.write_bytes(b"\0\0\0\x18ftypheic" + b"\0" * 64)
    r = await client.post("/v0/image/defects", json={"file": {"path": str(heic)}, "output": {"path": str(rw / "defects.png")}})
    assert r.json()["error"]["code"] == "unsupported"
