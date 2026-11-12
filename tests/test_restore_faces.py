"""`image.restore_faces`: aligning a face and putting it back, the worker op with the model stubbed,
and the op with a scripted model worker."""

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


def landmarks_at(x: float, y: float, scale: float):
    """The five points of a face the template's shape, `scale` of its size, its square at `(x, y)`."""
    import faces_restore

    return [[x + px * scale, y + py * scale] for px, py in faces_restore.TEMPLATE]


def test_a_face_is_aligned_to_the_template_and_goes_back_where_it_was():
    import faces_restore

    rgb = np.zeros((800, 1000, 3), np.float32)
    rgb[300:400, 500:600] = 1.0
    face, matrix = faces_restore.align(rgb, landmarks_at(400, 200, 0.5))
    assert face.shape == (512, 512, 3)
    # The square 400..656 × 200..456 of the picture, at twice its size.
    assert face[(300 - 200) * 2 + 10, (500 - 400) * 2 + 10, 0] == pytest.approx(1.0, abs=1e-3)
    box, back, weight = faces_restore.paste(face, matrix, 800, 1000, 0.5)
    y0, y1, x0, x1 = box
    assert (y0, x0) == (199, 399) and 455 <= y1 <= 458 and 655 <= x1 <= 658
    assert weight.max() == pytest.approx(0.5) and weight[0, 0] < 1e-4 and weight[-1, -1] < 1e-4
    assert np.abs(back[300 - y0 + 20, 500 - x0 + 20] - 1.0).max() < 1e-3


def test_a_face_past_the_pictures_edge_goes_back_only_where_the_picture_is():
    import faces_restore

    face, matrix = faces_restore.align(np.zeros((300, 300, 3), np.float32), landmarks_at(-100, -50, 0.6))
    box, back, weight = faces_restore.paste(face, matrix, 300, 300, 1.0)
    assert box[0] == 0 and box[2] == 0 and back.shape[:2] == weight.shape == (box[1], box[3])


@pytest.fixture
def flat_restorer(monkeypatch):
    """The worker's GFPGAN stubbed: every face it is shown comes back white."""
    import model_registry as registry

    shown = []

    def restore(face):
        shown.append(face.shape)
        return np.ones(face.shape, np.float32)

    monkeypatch.setattr(registry, "_family_locks", {})
    monkeypatch.setattr(registry, "gfpgan_device", lambda: "cpu")
    monkeypatch.setattr(registry, "gfpgan", lambda device: types.SimpleNamespace(restore=restore))
    return shown


def test_the_worker_op_changes_only_the_faces_at_16_bits_and_keeps_the_alpha(flat_restorer, tmp_path):
    import inference_ops

    picture = np.full((900, 1200, 4), 20000, np.uint16)
    out = tmp_path / "out.tiff"
    faces = [{"landmarks": landmarks_at(200, 150, 0.6)}, {"landmarks": landmarks_at(700, 400, 0.8)}]
    answer = inference_ops.restore_faces(save(tmp_path / "in.png", picture), str(out), faces, 1.0, "tiff")
    assert answer == {"model": "GFPGANv1.4", "width": 1200, "height": 900, "channels": 4, "depth": 16}
    assert flat_restorer == [(512, 512, 3), (512, 512, 3)]
    result = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert result.dtype == np.uint16 and (result[:, :, 3] == 20000).all()
    assert (result[150 + 150, 200 + 150, :3] == 65535).all() and (result[400 + 200, 700 + 200, :3] == 65535).all()
    assert (result[10, 10] == picture[10, 10]).all() and (result[880, 1180] == picture[880, 1180]).all()
    changed = (result != picture).any(axis=2)
    assert changed.mean() < 0.4


def test_the_worker_op_with_no_faces_writes_the_picture_as_it_was(flat_restorer, tmp_path):
    import inference_ops

    picture = np.random.default_rng(0).integers(0, 255, (60, 80), dtype=np.uint8)
    out = tmp_path / "out.png"
    inference_ops.restore_faces(save(tmp_path / "in.png", picture), str(out), [], 0.5, "png")
    assert flat_restorer == [] and np.array_equal(cv2.imread(str(out), cv2.IMREAD_UNCHANGED), picture)


def detected(request):
    """A scripted model worker's `detect_faces`: a large sure face, a small one and an unsure one."""
    face = lambda x, y, size, confidence: {"box": {"x": x, "y": y, "width": size, "height": size}, "confidence": confidence,
                                           "landmarks": [[x + size * 0.3, y + size * 0.4]] * 5, "embedding": []}
    return {"frame": {"width": 1000, "height": 800}, "faces": [face(100, 100, 300, 0.99), face(600, 100, 60, 0.98), face(600, 400, 200, 0.6)]}


def copied(request):
    picture = cv2.imread(request["path"], cv2.IMREAD_UNCHANGED)
    cv2.imwrite(request["out"], picture)
    return {"model": "GFPGANv1.4", "width": 1000, "height": 800, "channels": 3, "depth": 8}


async def test_only_faces_large_and_sure_enough_are_redrawn_and_the_answer_says_it_is_generated(client, dirs, fake_worker):
    ro, rw = dirs
    fake_worker.on("detect_faces", detected)
    fake_worker.on("restore_faces", copied)
    image = save(ro / "scan.png", np.zeros((800, 1000, 3), np.uint8))
    r = await client.post("/v0/image/restore_faces", json={"file": {"path": image}, "output": {"path": str(rw / "faces.png")}})
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["result"]["generative"] is True and body["result"]["left"] == 2
    assert body["result"]["faces"] == [{"box": {"x": 0.1, "y": 0.125, "width": 0.3, "height": 0.375}, "confidence": 0.99}]
    assert body["meta"]["producer"] == "retinaface-r34+GFPGANv1.4"
    (request,) = [q for q in fake_worker.requests if q["op"] == "restore_faces"]
    assert len(request["faces"]) == 1 and request["strength"] == 0.5
    assert (rw / "faces.png").exists()


async def test_boxes_and_min_size_pick_the_faces(client, dirs, fake_worker):
    ro, rw = dirs
    fake_worker.on("detect_faces", detected)
    fake_worker.on("restore_faces", copied)
    image = save(ro / "scan.png", np.zeros((800, 1000, 3), np.uint8))
    body = {"file": {"path": image}, "faces": [{"x": 0.5, "y": 0.0, "width": 0.5, "height": 0.5}], "min_size": 32, "output": {"path": str(rw / "faces.png")}}
    r = await client.post("/v0/image/restore_faces", json=body)
    assert [f["box"]["x"] for f in r.json()["result"]["faces"]] == [0.6] and r.json()["result"]["left"] == 2


@pytest.mark.parametrize("change, message", [
    ({"file": {"mirror": True}}, "angle 0, no mirror"),
    ({"output": {"format": "webp"}}, "png or tiff"),
    ({"min_size": 8}, "min_size"),
    ({"strength": -0.1}, "strength"),
])
async def test_what_would_turn_the_picture_or_lose_its_pixels_is_refused(client, dirs, change, message):
    ro, rw = dirs
    body = {"file": {"path": save(ro / "scan.png", np.zeros((80, 100, 3), np.uint8))}, "output": {"path": str(rw / "faces.png")}}
    for key, value in change.items():
        body[key] = {**body[key], **value} if isinstance(value, dict) else value
    r = await client.post("/v0/image/restore_faces", json=body)
    assert r.status_code == 400 and message in r.json()["error"]["message"], r.json()
