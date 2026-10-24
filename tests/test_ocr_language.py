"""The language a reading is in, and which transcriber that sends its handwriting to."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))


def _fake_lingua(monkeypatch, answer):
    """lingua whose detector rates every text as `answer`: [(iso code, confidence), ...]."""

    class Detector:
        def compute_language_confidence_values(self, text):
            return [types.SimpleNamespace(language=types.SimpleNamespace(iso_code_639_1=types.SimpleNamespace(name=code.upper())), value=value)
                    for code, value in answer]

    class Builder:
        @staticmethod
        def from_all_languages():
            return types.SimpleNamespace(build=Detector)

    monkeypatch.setitem(sys.modules, "lingua", types.SimpleNamespace(LanguageDetectorBuilder=Builder))
    import text_language

    return text_language.LanguageIdentifier()


LETTER = "Dear Bert, you will have been anxious to hear that I did not return"


def test_a_long_enough_reading_names_its_language(monkeypatch):
    assert _fake_lingua(monkeypatch, [("en", 0.97), ("de", 0.03)]).detect(LETTER) == {"language": "en", "confidence": 0.97}


def test_a_name_on_a_card_is_no_language(monkeypatch):
    assert _fake_lingua(monkeypatch, [("sv", 0.99)]).detect("Elin Andersson 1932") is None


def test_a_reading_that_could_be_several_languages_is_none(monkeypatch):
    assert _fake_lingua(monkeypatch, [("cy", 0.45), ("en", 0.40)]).detect(LETTER) is None


class Reader:
    def __init__(self, detected):
        self.detected = detected

    def detect(self, image):
        return {"width": 10, "height": 10, "detected": self.detected}


def _line(text, score, hand=None):
    return {"text": text, "score": score, "hand": hand, "box": [], "crop": f"crop-{text}"}


@pytest.fixture
def worker(monkeypatch, tmp_path):
    """read_text with every model stood in for: the reader answers `detected`, the identifier
    `found`, and each transcriber writes its family's name in capitals."""
    import inference_ops
    import model_registry as registry

    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"jpeg")
    for name in ("htr", "htr-en"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(registry, "HTR_ROOT", str(tmp_path / "htr"))
    monkeypatch.setattr(registry, "HTR_EN_ROOT", str(tmp_path / "htr-en"))
    monkeypatch.setattr(registry, "ocr_device", lambda: "cpu")
    monkeypatch.setattr(registry, "htr_device", lambda family="htr": "cpu")
    monkeypatch.setitem(sys.modules, "image_io", types.SimpleNamespace(open_display=lambda path, angle, mirror: "image"))
    state = {"asked": []}

    def htr(device, family="htr"):
        return types.SimpleNamespace(read=lambda crops: state["asked"].append((family, crops)) or [family.upper()] * len(crops))

    monkeypatch.setattr(registry, "htr", htr)
    monkeypatch.setattr(registry, "ocr_photo", lambda device: Reader(state["detected"]))
    monkeypatch.setattr(registry, "text_language", lambda: types.SimpleNamespace(detect=lambda text: state.get("found")))
    state["read"] = lambda language=None: inference_ops.read_text([str(picture)], language=language)
    return state


LETTER_LINES = [_line(f"line {n}", 0.7, (f"hand {n}", 0.9)) for n in range(5)]


def test_a_swedish_letter_goes_to_the_swedish_transcriber(worker):
    worker["detected"], worker["found"] = LETTER_LINES, {"language": "sv", "confidence": 1.0}
    out = worker["read"]()
    assert worker["asked"] == [("htr", [f"crop-line {n}" for n in range(5)])]
    assert out["text"] == "\n".join(["HTR"] * 5)
    assert out["language"] == {"code": "sv", "confidence": 1.0, "given": False}


def test_a_picture_in_no_known_language_goes_to_the_swedish_transcriber(worker):
    worker["detected"], worker["found"] = LETTER_LINES, None
    out = worker["read"]()
    assert [family for family, _ in worker["asked"]] == ["htr"] and out["language"] is None


def test_an_english_letter_goes_to_the_english_transcriber(worker):
    worker["detected"] = [_line(f"line {n}", 0.85) for n in range(4)]
    worker["found"] = {"language": "en", "confidence": 0.99}
    out = worker["read"]()
    assert [family for family, _ in worker["asked"]] == ["htr-en"]
    assert out["text"] == "\n".join(["HTR-EN"] * 4)


def test_a_german_letter_is_not_transcribed(worker):
    worker["detected"], worker["found"] = LETTER_LINES, {"language": "de", "confidence": 0.98}
    out = worker["read"]()
    assert worker["asked"] == [] and out["text"] == "\n".join(f"hand {n}" for n in range(5))


def test_a_given_language_is_not_detected(worker):
    worker["detected"], worker["found"] = [_line(f"line {n}", 0.85) for n in range(4)], {"language": "sv", "confidence": 1.0}
    out = worker["read"]("en")
    assert [family for family, _ in worker["asked"]] == ["htr-en"]
    assert out["language"] == {"code": "en", "confidence": None, "given": True}


def test_without_the_english_transcriber_an_english_page_keeps_its_readings(worker, tmp_path):
    os.rmdir(tmp_path / "htr-en")
    worker["detected"], worker["found"] = LETTER_LINES, {"language": "en", "confidence": 0.99}
    out = worker["read"]()
    assert worker["asked"] == [] and out["text"] == "\n".join(f"hand {n}" for n in range(5))
