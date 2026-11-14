"""`image.damage`: fading measured from the change `image.unfade` would make, the detector's marks
sorted into specks and scratches, blur from the width of the sharpest edges, the scores, and the op
with a scripted model worker."""

import os

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")


def save(path, picture):
    assert cv2.imwrite(str(path), picture)
    return str(path)


def scene(depth=8):
    """A picture with its colours whole, in OpenCV's BGR: a grey ramp across the top, colour patches below."""
    top = 65535 if depth == 16 else 255
    picture = np.zeros((300, 400, 3), np.float64)
    picture[:150] = np.linspace(0.0, 1.0, 400)[None, :, None]
    for i, colour in enumerate([(0.2, 0.3, 0.8), (0.1, 0.7, 0.2), (0.8, 0.4, 0.1), (0.5, 0.5, 0.5)]):
        picture[150:, i * 100:(i + 1) * 100] = colour
    return np.round(picture * top).astype(np.uint16 if depth == 16 else np.uint8)


def faded(picture):
    """`picture` faded as a print's dyes fade: each channel squeezed into a band of its own and bent."""
    top = np.iinfo(picture.dtype).max
    x = picture.astype(np.float64) / top
    bands = [(0.25, 0.75, 1.3), (0.2, 0.8, 1.15), (0.08, 0.95, 0.9)]
    out = np.stack([lo + (hi - lo) * x[:, :, c] ** g for c, (lo, hi, g) in enumerate(bands)], axis=-1)
    return np.round(out * top).astype(picture.dtype)


@pytest.mark.parametrize("depth", [8, 16])
def test_a_faded_picture_changes_and_one_whose_colours_are_whole_does_not(depth):
    from ems.tools import damage

    whole, worn = damage.fading(scene(depth)), damage.fading(faded(scene(depth)))
    assert whole < damage.FADED[0] < damage.FADED[1] < worn
    assert damage.score(whole, damage.FADED) == 0 and damage.score(worn, damage.FADED) == 1


def test_fading_is_measured_on_the_picture_shrunk_and_leaves_it_as_it_was():
    from ems.tools import damage

    picture = faded(np.tile(scene(), (8, 8, 1)))
    before = picture.copy()
    assert damage.fading(picture) == pytest.approx(damage.fading(faded(scene())), abs=1.0)
    assert np.array_equal(picture, before)


def test_a_grey_picture_is_faded_by_its_lost_contrast_alone():
    from ems.tools import damage

    ramp = np.tile(np.linspace(0, 255, 400), (300, 1)).astype(np.uint8)
    assert damage.fading(ramp) < damage.FADED[0]
    assert damage.fading((ramp * 0.5 + 80).astype(np.uint8)) > damage.FADED[1]


def test_long_thin_parts_are_scratches_and_the_rest_specks():
    from ems.tools import damage

    marked = np.zeros((640, 960), bool)
    marked[100:400, 300:302] = True                  # a scratch, 300 px down
    for i in range(20):                              # twenty specks
        marked[500 + (i % 4) * 20: 504 + (i % 4) * 20, 100 + i * 30: 104 + i * 30] = True
    marked[600:601, 900:902] = True                  # too small to count
    marked[0:50, 500:501] = True                     # on the edge: a print's own
    assert damage.parts(marked, np.zeros(marked.shape, np.float32)) == {"specks": 20, "scratches": 1, "length": pytest.approx(300 / 640, abs=0.01)}


def test_a_short_scratch_and_a_broad_blot_are_specks():
    from ems.tools import damage

    marked = np.zeros((640, 960), bool)
    marked[100:115, 300:302] = True                  # shorter than SCRATCH_LENGTH
    marked[300:340, 500:540] = True                  # long but broad
    assert damage.parts(marked, np.zeros(marked.shape, np.float32)) == {"specks": 2, "scratches": 0, "length": 0}


def test_marks_among_the_pictures_own_detail_are_not_counted():
    from ems.tools import damage

    grey = np.zeros((640, 960), np.float32)
    grey[:, 480:] = np.random.default_rng(0).random((640, 480))  # fine texture on the right half: branches, grass
    marked = np.zeros((640, 960), bool)
    marked[100:400, 200:202] = True                  # a scratch on smooth picture
    marked[100:400, 700:702] = True                  # a light line among detail
    for x in (100, 300, 600, 800):
        marked[500:504, x:x + 4] = True
    assert damage.parts(marked, grey) == {"specks": 2, "scratches": 1, "length": pytest.approx(300 / 640, abs=0.01)}


def test_the_scores_grow_with_the_damage_and_stop_at_one():
    from ems.tools import damage

    found = np.zeros((640, 960), np.uint16)
    shown = np.full((640, 960), 30000, np.uint16)
    clean = damage.counted(found, shown, 0.4)
    assert clean == {"specks": 0, "scratches": 0, "length": 0, "marked": 0, "dusty": 0, "scratched": 0}
    for i in range(400):
        y, x = 20 + (i // 40) * 60, 20 + (i % 40) * 23
        found[y:y + 3, x:x + 3] = 65535
    found[:, 480:482] = round(0.39 * 65535)          # below the threshold
    for x in (100, 300, 500, 700):
        found[10:630, x:x + 2] = 65535
    dirty = damage.counted(found, shown, 0.4)
    assert dirty["specks"] > 300 and dirty["scratches"] == 4
    assert dirty["dusty"] == 1 and dirty["scratched"] == 1
    assert damage.score(-1, damage.DUSTY) == 0


def shapes(size=900):
    """A sharp picture in grey: discs and bars of many grey levels on a mid-grey ground, with a
    little grain, so that its edges run every way."""
    rng = np.random.default_rng(3)
    picture = np.full((size, size), 0.5, np.float32)
    for _ in range(60):
        x, y, r = rng.integers(40, size - 40, 2).tolist() + [int(rng.integers(8, 40))]
        cv2.circle(picture, (x, y), r, float(rng.uniform(0, 1)), -1)
    for _ in range(20):
        x, y = rng.integers(40, size - 40, 2).tolist()
        cv2.rectangle(picture, (x, y), (x + int(rng.integers(10, 80)), y + int(rng.integers(10, 80))), float(rng.uniform(0, 1)), -1)
    picture += rng.normal(0, 0.01, picture.shape).astype(np.float32)
    return np.clip(np.rint(picture * 255), 0, 255).astype(np.uint8)


def shaken(picture, length=17):
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1 / length
    return cv2.filter2D(picture, -1, kernel)


def test_a_sharp_picture_is_neither_blurred_nor_shaken():
    from ems.tools import damage

    edge, shake = damage.sharpness(shapes())
    assert edge < damage.BLURRED[0] and shake < damage.SHAKEN[0]


@pytest.mark.parametrize("sigma", [3, 8])
def test_a_blurred_picture_has_wide_edges_every_way(sigma):
    from ems.tools import damage

    edge, shake = damage.sharpness(cv2.GaussianBlur(shapes(), (0, 0), sigma))
    assert edge > damage.BLURRED[1] and shake < damage.SHAKEN[0]


def test_a_shaken_picture_has_its_edges_wider_across_the_movement():
    from ems.tools import damage

    _, shake = damage.sharpness(shaken(shapes()))
    assert shake > damage.SHAKEN[1]


def test_a_sharp_part_keeps_a_picture_with_shallow_focus_sharp():
    from ems.tools import damage

    picture = cv2.GaussianBlur(shapes(), (0, 0), 4)
    picture[300:600, 300:600] = shapes()[300:600, 300:600]
    edge, _ = damage.sharpness(picture)
    assert edge < damage.BLURRED[0]


def test_a_picture_without_edges_is_not_judged():
    from ems.tools import damage

    assert damage.sharpness(np.full((400, 600, 3), 128, np.uint8)) == (None, None)


def test_a_border_round_the_picture_is_not_taken_for_its_edges():
    from ems.tools import damage

    picture = cv2.GaussianBlur(shapes(), (0, 0), 3)
    picture[:30], picture[-30:], picture[:, :30], picture[:, -30:] = 0, 0, 0, 0
    edge, _ = damage.sharpness(picture)
    assert edge > damage.BLURRED[1]


def mark_a_scratch(request):
    """A scripted model worker's `detect_defects`: one long scratch down what it is shown."""
    shown = cv2.imread(request["path"], cv2.IMREAD_UNCHANGED)
    assert shown.dtype == np.uint16 and min(shown.shape) == 640
    found = np.zeros(shown.shape, np.uint16)
    found[40:600, 200:202] = 65535
    cv2.imwrite(request["out"], found)
    return {"model": "bopb-detection"}


async def test_a_faded_scratched_scan_is_scored_and_nothing_is_left_behind(client, dirs, fake_worker):
    ro, rw = dirs
    fake_worker.on("detect_defects", mark_a_scratch)
    image = save(ro / "print.tif", faded(scene(16)))
    r = await client.post("/v0/image/damage", json={"file": {"path": image}})
    assert r.status_code == 200, r.json()
    body = r.json()
    result = body["result"]
    assert result["width"] == 400 and result["height"] == 300
    assert result["faded"] == 1 and result["dusty"] == 0 and result["scratched"] == pytest.approx((560 / 640 - 0.4) / 1.2, abs=0.01)
    assert result["measures"]["scratches"] == 1 and result["measures"]["specks"] == 0
    assert result["measures"]["change"] > 22 and result["measures"]["length"] == pytest.approx(560 / 640, abs=0.01)
    assert result["blurred"] == 0 and result["shaken"] == 0 and result["measures"]["edge"] < 1.2
    assert body["meta"]["producer"] == "damage/2+bopb-detection"
    (request,) = [q for q in fake_worker.requests if q["op"] == "detect_defects"]
    assert request["dark"] is False
    assert os.listdir(rw) == []


async def test_capabilities_name_what_image_damage_answers_as(client):
    body = (await client.get("/v0/capabilities")).json()
    assert "image.damage" in body["ops"] and body["models"]["damage"] == "damage/2+bopb-detection"


async def test_a_turned_picture_and_one_opencv_cannot_read_are_refused(client, dirs):
    ro, _ = dirs
    image = save(ro / "print.png", scene())
    r = await client.post("/v0/image/damage", json={"file": {"path": image, "angle": 90}})
    assert r.status_code == 400 and "angle 0, no mirror" in r.json()["error"]["message"]
    heic = ro / "photo.heic"
    heic.write_bytes(b"\0\0\0\x18ftypheic" + b"\0" * 64)
    r = await client.post("/v0/image/damage", json={"file": {"path": str(heic)}})
    assert r.json()["error"]["code"] == "unsupported"
