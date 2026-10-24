"""`ocr.read` against the scripted worker, and the photo reader's line filter with RapidOCR stubbed."""

import os
import sys
import types

from ems.tools import decode as decode_tool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))


class Picture:
    """What the reader asks of a PIL image."""

    def __init__(self, size):
        self.size = size

    def convert(self, mode):
        return self


LINE = {"text": "Hamngatan 4", "box": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.3], [0.1, 0.3]], "score": 0.97}


async def test_a_picture_is_read_in_its_display_frame(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.jpg").write_bytes(b"jpeg")
    fake_worker.on("read_text", lambda req: {"text": "Hamngatan 4", "lines": [LINE], "width": 4000, "height": 3000, "model": "PP-OCRv6_medium"})
    r = await client.post("/v0/ocr/read", json={"file": {"path": str(ro / "a.jpg"), "angle": 90, "mirror": True, "mimetype": "image/jpeg"}})
    body = r.json()
    assert r.status_code == 200, body
    assert body["result"] == {"text": "Hamngatan 4", "lines": [LINE], "width": 4000, "height": 3000}
    assert body["meta"]["producer"] == "PP-OCRv6_medium"
    sent = fake_worker.requests[-1]
    assert sent["op"] == "read_text" and sent["paths"] == [str(ro / "a.jpg")] and sent["angle"] == 90 and sent["mirror"] is True


async def test_a_picture_without_text_answers_empty(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.jpg").write_bytes(b"jpeg")
    fake_worker.on("read_text", lambda req: {"text": "", "lines": [], "width": 640, "height": 480, "model": "PP-OCRv6_medium"})
    body = (await client.post("/v0/ocr/read", json={"file": {"path": str(ro / "a.jpg"), "mimetype": "image/jpeg"}})).json()
    assert body["result"]["text"] == "" and body["result"]["lines"] == []


async def test_a_raw_the_readers_cannot_open_is_decoded_out_of_process(client, fake_worker, roots, monkeypatch):
    ro, _ = roots
    original = ro / "a.raf"
    original.write_bytes(b"raw")
    decoded = ro / "decoded.jpg"
    decoded.write_bytes(b"jpeg")

    async def fake_decode(path, mimetype, angle, mirror, *, scratch, timeout):
        return decode_tool.Decoded(str(decoded), decode_tool.normalize_angle(angle), mirror, None)

    monkeypatch.setattr(decode_tool, "decode_for_models", fake_decode)

    def answer(req):
        if req["paths"] == [str(original)]:
            raise ValueError(f"could not load image: {original}")
        return {"text": "Hamngatan 4", "lines": [LINE], "width": 10, "height": 10, "model": "PP-OCRv6_medium"}

    fake_worker.on("read_text", answer)
    body = (await client.post("/v0/ocr/read", json={"file": {"path": str(original), "mimetype": "image/x-fujifilm-raf"}})).json()
    assert body["ok"], body
    assert [q["paths"] for q in fake_worker.requests if q["op"] == "read_text"] == [[str(original)], [str(decoded)]]


async def test_audio_has_no_text_to_read(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.mp3").write_bytes(b"mp3")
    r = await client.post("/v0/ocr/read", json={"file": {"path": str(ro / "a.mp3"), "mimetype": "audio/mpeg"}})
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported"
    assert not [q for q in fake_worker.requests if q["op"] == "read_text"]


def test_the_reader_keeps_confident_lines_of_two_letters_or_digits(monkeypatch):
    seen = {}

    class FakeOutput:
        boxes = [[[10, 20], [50, 20], [50, 30], [10, 30]], [[0, 0], [5, 0], [5, 5], [0, 5]], [[0, 50], [90, 50], [90, 60], [0, 60]]]
        txts = ("Hamngatan 4", "-B-", "BODE")
        scores = (0.97, 0.99, 0.76)

    class FakeRapidOCR:
        def __init__(self, params):
            seen.update(params)

        def __call__(self, image):
            seen["size"] = image.size
            return FakeOutput()

    fake = types.ModuleType("rapidocr")
    fake.__file__ = "/opt/venv/lib/rapidocr/__init__.py"
    fake.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake)
    import text_reader

    reader = text_reader.PhotoReader("/opt/ocr", "medium", "cpu", 1 << 30, 2000)
    monkeypatch.setattr(reader, "_crop", lambda image, box, pixels: "crop")
    out = reader.read(Picture((100, 100)))
    assert reader.model == "PP-OCRv6_medium"
    assert seen["Det.model_path"] == "/opt/ocr/PP-OCRv6_det_medium.onnx" and seen["Rec.model_path"] == "/opt/ocr/PP-OCRv6_rec_medium.onnx"
    assert seen["Cls.model_path"] == "/opt/venv/lib/rapidocr/models/ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    assert seen["EngineConfig.onnxruntime.use_cuda"] is False
    # "-B-" holds one letter, "BODE" is under the floor.
    assert out == {
        "text": "Hamngatan 4",
        "lines": [{"text": "Hamngatan 4", "box": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.3], [0.1, 0.3]], "score": 0.97}],
        "width": 100,
        "height": 100,
        "transcribe": [],
        "tentative": [],
    }


def test_a_doubtful_line_is_read_again_by_the_handwriting_recogniser(monkeypatch):
    class FakeOutput:
        boxes = [[[10, 20], [50, 20], [50, 30], [10, 30]], [[0, 50], [90, 50], [90, 60], [0, 60]], [[0, 70], [90, 70], [90, 80], [0, 80]]]
        txts = ("GÖTEBORG.", "myep", "木木木")
        scores = (0.97, 0.47, 0.79)

    class FakeRapidOCR:
        def __init__(self, params):
            pass

        def __call__(self, image):
            return FakeOutput()

    fake = types.ModuleType("rapidocr")
    fake.__file__ = "/opt/venv/lib/rapidocr/__init__.py"
    fake.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake)
    import text_reader

    reader = text_reader.PhotoReader("/opt/ocr", "medium", "cpu", 1 << 30, 1600)
    reader.hand = types.SimpleNamespace(model="multicentury-ppocrv6-medium")
    second = iter([("John", 0.96, "crop-of-john"), ("LbHb", 0.18, "crop-of-ornament")])
    asked = []
    monkeypatch.setattr(reader, "_read_hand", lambda image, box, pixels: asked.append(box) or next(second))
    monkeypatch.setattr(reader, "_crop", lambda image, box, pixels: "crop")
    out = reader.read(Picture((100, 100)))
    # The confident line is never read again; the name is kept from its second reading, the
    # ornament is dropped by it.
    assert [line["text"] for line in out["lines"]] == ["GÖTEBORG.", "John"]
    assert out["lines"][1]["score"] == 0.96
    assert asked == [FakeOutput.boxes[1], FakeOutput.boxes[2]]
    # The kept line goes on with its crop, for the transcriber to write; the dropped one does not.
    assert out["transcribe"] == [(1, "crop-of-john")]


def _hand_reader(monkeypatch, txts, scores, second, judge="recogniser"):
    """A PhotoReader over one line per entry of `txts`, its doubtful lines read again as `second`
    gives, every crop named after its line's top edge, composed for `judge`."""

    class FakeOutput:
        boxes = [[[0, y], [90, y], [90, y + 5], [0, y + 5]] for y in range(0, 5 * len(txts), 5)]

    FakeOutput.txts, FakeOutput.scores = txts, scores

    class FakeRapidOCR:
        def __init__(self, params):
            pass

        def __call__(self, image):
            return FakeOutput()

    fake = types.ModuleType("rapidocr")
    fake.__file__ = "/opt/venv/lib/rapidocr/__init__.py"
    fake.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake)
    import text_reader

    reader = text_reader.PhotoReader("/opt/ocr", "medium", "cpu", 1 << 30, 1600)
    reader.hand = types.SimpleNamespace(model="multicentury-ppocrv6-medium")
    readings = iter(second)
    monkeypatch.setattr(reader, "_read_hand", lambda image, box, pixels: (*next(readings), f"crop-{box[0][1]}"))
    monkeypatch.setattr(reader, "_crop", lambda image, box, pixels: f"crop-{box[0][1]}")
    return reader.read(Picture((100, 100)), judge)


def test_on_a_handwritten_page_a_confident_line_goes_to_the_transcriber_too(monkeypatch):
    # A letter: PP-OCRv6 is sure of its misread date line, the four lines below it are writing.
    out = _hand_reader(
        monkeypatch,
        ("arbug lou 31-12-10", "Kare", "fir pasa", "skniva", "nagra"),
        (0.86, 0.4, 0.5, 0.45, 0.3),
        [("Kära", 0.9), ("får passa", 0.85), ("skriva", 0.88), ("några", 0.82)],
    )
    assert [index for index, _ in out["transcribe"]] == [0, 1, 2, 3, 4]
    assert out["transcribe"][0] == (0, "crop-0")


def test_on_a_photo_card_the_printed_imprint_keeps_its_reading(monkeypatch):
    # A card: the photographer's imprint is print, two handwritten names beside it.
    out = _hand_reader(
        monkeypatch,
        ("KONGL. HOFFOTOGRAF.", "Elin", "H. Holm"),
        (0.95, 0.5, 0.6),
        [("Elin", 0.8), ("H. Holm", 0.9)],
    )
    assert [index for index, _ in out["transcribe"]] == [1, 2]


def test_a_page_mostly_in_print_keeps_its_confident_lines(monkeypatch):
    # Four handwritten lines among five printed ones are no letter.
    out = _hand_reader(
        monkeypatch,
        ("Ett", "Två", "Tre", "Fyra", "Fem", "sex", "sju", "åtta", "nio"),
        (0.95, 0.95, 0.95, 0.95, 0.95, 0.5, 0.5, 0.5, 0.5),
        [("sex", 0.9), ("sju", 0.9), ("åtta", 0.9), ("nio", 0.9)],
    )
    assert [index for index, _ in out["transcribe"]] == [5, 6, 7, 8]


def test_an_english_letter_is_writing_by_how_unsure_the_photo_engine_is(monkeypatch):
    # The recogniser keeps no line of an English hand; PP-OCRv6 is sure of none of them. Every line
    # it reads at 0.6 or more goes to the transcriber, the two below the floor only if it writes a word.
    out = _hand_reader(
        monkeypatch,
        ("Dear Bert", "You will have been", "anxious to hear", "that I did not", "Mauvice"),
        (0.9, 0.85, 0.7, 0.65, 0.5),
        [("Lou vill", 0.3), ("anxin", 0.4), ("Mauri", 0.2)],
        judge="certainty",
    )
    assert [index for index, _ in out["transcribe"]] == [0, 1, 2, 3]
    assert [line["text"] for line in out["lines"]] == ["Dear Bert", "You will have been", "anxious to hear", "that I did not"]
    assert out["tentative"] == [2, 3]


def test_an_english_print_page_keeps_its_confident_lines(monkeypatch):
    # Most lines read at 0.95 or more: print, so only the line the recogniser kept is transcribed.
    out = _hand_reader(
        monkeypatch,
        ("The Smithsonian Institution", "Washington, D. C.", "With great respect,", "Proof revised"),
        (0.99, 0.98, 0.97, 0.6),
        [("Proof revised", 0.85)],
        judge="certainty",
    )
    assert [index for index, _ in out["transcribe"]] == [3]
    assert out["tentative"] == []


def test_in_a_language_without_a_transcriber_a_kept_line_keeps_the_recogniser_reading(monkeypatch):
    out = _hand_reader(
        monkeypatch,
        ("Herzliche Grüße", "aus", "Mnchen", "Tante", "Grete"),
        (0.6, 0.5, 0.5, 0.5, 0.5),
        [("Herzliche Grüße", 0.9), ("aus", 0.9), ("München", 0.9), ("Tante", 0.9), ("Grete", 0.9)],
        judge=None,
    )
    assert out["transcribe"] == []
    assert out["text"] == "Herzliche Grüße\naus\nMünchen\nTante\nGrete"


def test_the_transcriber_writes_its_lines_and_a_tentative_line_needs_a_word():
    import text_reader

    reading = {
        "text": "", "width": 10, "height": 10, "tentative": [1, 2],
        "lines": [{"text": "Dear Bert", "box": [], "score": 0.9}, {"text": "anxin", "box": [], "score": 0.7}, {"text": "Mauri", "box": [], "score": 0.65}],
        "transcribe": [(0, "c0"), (1, "c1"), (2, "c2")],
    }
    out = text_reader.write(reading, ["", "anxious to hear", "."])
    # An empty transcription keeps the reading; "." is no word, so that tentative line goes.
    assert out == {"text": "Dear Bert\nanxious to hear", "width": 10, "height": 10,
                   "lines": [{"text": "Dear Bert", "box": [], "score": 0.9}, {"text": "anxious to hear", "box": [], "score": 0.7}]}


def test_kept_text_is_what_the_floor_keeps():
    import text_reader

    reading = {"width": 1, "height": 1, "detected": [
        {"text": "Hamngatan 4", "score": 0.97, "hand": None},
        {"text": "myep", "score": 0.47, "hand": ("John", 0.96)},
        {"text": "BODE", "score": 0.76, "hand": ("LbHb", 0.18)},
    ]}
    assert text_reader.kept_text(reading) == "Hamngatan 4\nJohn"


async def test_a_given_language_goes_to_the_worker_and_the_language_comes_back_in_meta(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.jpg").write_bytes(b"jpeg")
    fake_worker.on("read_text", lambda req: {"text": "Dear Bert", "lines": [LINE], "width": 10, "height": 10, "model": "m",
                                             "language": {"code": req.get("language"), "confidence": None, "given": True}})
    body = (await client.post("/v0/ocr/read", json={"file": {"path": str(ro / "a.jpg"), "mimetype": "image/jpeg"}, "language": "en"})).json()
    assert body["meta"]["language"] == {"code": "en", "confidence": None, "given": True}
    assert fake_worker.requests[-1]["language"] == "en"


async def test_a_language_is_two_letters(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.jpg").write_bytes(b"jpeg")
    body = (await client.post("/v0/ocr/read", json={"file": {"path": str(ro / "a.jpg")}, "language": "English"})).json()
    assert body["error"]["code"] == "invalid_request"
    assert not fake_worker.requests


async def test_without_a_language_the_worker_is_not_told_one(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.jpg").write_bytes(b"jpeg")
    fake_worker.on("read_text", lambda req: {"text": "", "lines": [], "width": 1, "height": 1, "model": "m", "language": None})
    body = (await client.post("/v0/ocr/read", json={"file": {"path": str(ro / "a.jpg"), "mimetype": "image/jpeg"}})).json()
    assert body["meta"]["language"] is None
    assert "language" not in fake_worker.requests[-1]
