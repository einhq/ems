"""`inference_ops.transcribe` with the whisper model stubbed: the per-request overrides reach the
decode call and the signature, and the configured values apply when a request names none.

The decode call is exercised for real; only the model behind it is fake.

The language half: detection runs over several windows and the expected languages win when
they reach the floor, what was chosen is what the decoder is told, and a pin skips detection.
"""

import os
import sys
import types
from contextlib import nullcontext

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import inference_ops as ops  # noqa: E402
import model_registry as registry  # noqa: E402


class FakeSegment:
    def __init__(self, start, end, text, words=None):
        self.start, self.end, self.text = start, end, text
        self.no_speech_prob = 0.1
        self.words = words


class FakeModel:
    def __init__(self, calls):
        self.calls = calls

    def transcribe(self, source, **kwargs):
        self.calls.append({"source": source, **kwargs})
        info = types.SimpleNamespace(language="sv", language_probability=0.9, duration=3.0)
        return iter([FakeSegment(0.0, 1.5, "hej"), FakeSegment(1.5, 3.0, "")]), info


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    calls: list[dict] = []
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(registry, "whisper", lambda device, compute_type: FakeModel(calls))
    monkeypatch.setattr(registry, "family_lock", lambda family: nullcontext())
    monkeypatch.setattr(registry, "run_on_model_thread", lambda family, fn: fn())
    monkeypatch.setattr(registry, "WHISPER_LANGUAGE", "")
    monkeypatch.setattr(registry, "WHISPER_LANGUAGES", ())
    monkeypatch.setattr(registry, "WHISPER_LANGUAGE_FLOOR", 0.1)
    monkeypatch.setattr(registry, "WHISPER_LANGUAGE_WINDOWS", 3)
    monkeypatch.setattr(registry, "WHISPER_WORD_TIMESTAMPS", False)
    # Detection is stubbed at the vote level: it needs real audio and the real VAD, and what
    # is under test here is what the decode is told, not whisper's ear.
    monkeypatch.setattr(ops, "decode_speech_audio", lambda model, path: f"audio of {path}")
    monkeypatch.setattr(ops, "language_votes", lambda model, audio, vad, windows: [{"sv": 0.6, "no": 0.3}])
    monkeypatch.setattr(registry, "whisper_vad_parameters", lambda: {"threshold": 0.2, "speech_pad_ms": 800, "min_silence_duration_ms": 2000})
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    return str(audio), calls


def test_defaults_come_from_the_configuration(stubbed):
    path, calls = stubbed
    result = ops.transcribe(path)
    # Nothing pinned: the decoder is told what the detection chose, never left to guess.
    assert calls[0]["language"] == "sv"
    assert calls[0]["vad_parameters"]["min_silence_duration_ms"] == 2000
    assert calls[0]["word_timestamps"] is False
    assert result["speech"] is True and result["text"] == "hej" and len(result["segments"]) == 1
    assert result["language"] == "sv"
    assert result["language_probability"] == pytest.approx(0.6)
    assert result["signature"].endswith(",lang=auto:3,words=0")
    assert result["signature"] == ops.transcribe_signature()
    # The audio detection decoded is what the decoder reads, so the file is decoded once.
    assert calls[0]["source"] == f"audio of {path}"


def test_expected_language_wins_at_the_floor(stubbed, monkeypatch):
    path, calls = stubbed
    monkeypatch.setattr(registry, "WHISPER_LANGUAGES", ("sv", "en"))
    monkeypatch.setattr(ops, "language_votes", lambda *a: [{"no": 0.86, "sv": 0.10}, {"no": 0.7, "sv": 0.2}])
    result = ops.transcribe(path)
    assert calls[0]["language"] == "sv"
    assert result["language"] == "sv"
    # The probability reported is Swedish's own, on the confidence-weighted average.
    assert result["language_probability"] == pytest.approx((0.10 * 0.86 + 0.2 * 0.7) / 1.56)
    assert result["signature"].endswith(",lang=auto:3:sv,en@0.1,words=0")

    # Below the floor the top guess stands: a Polish wedding stays Polish.
    monkeypatch.setattr(ops, "language_votes", lambda *a: [{"pl": 0.98, "sv": 0.01}])
    result = ops.transcribe(path)
    assert calls[1]["language"] == "pl" and result["language"] == "pl"


def test_overrides_reach_the_decode_and_the_signature(stubbed):
    path, calls = stubbed
    result = ops.transcribe(path, language="sv", min_silence_ms=500, word_timestamps=True)
    assert calls[0]["language"] == "sv"
    assert calls[0]["vad_parameters"]["min_silence_duration_ms"] == 500
    assert calls[0]["word_timestamps"] is True
    assert ":vad=0.2/800/500," in result["signature"]
    assert result["signature"].endswith(",lang=sv,words=1")
    # The configured signature is unchanged by a request's overrides.
    assert ops.transcribe_signature().endswith(",lang=auto:3,words=0")


def test_no_speech_means_no_detection(stubbed, monkeypatch):
    path, calls = stubbed
    monkeypatch.setattr(ops, "language_votes", lambda *a: [])
    result = ops.transcribe(path)
    assert calls[0]["language"] is None
    # What the decoder itself reports, then.
    assert result["language"] == "sv" and result["language_probability"] == pytest.approx(0.9)


def test_a_pin_skips_detection(stubbed, monkeypatch):
    path, calls = stubbed
    monkeypatch.setattr(registry, "WHISPER_LANGUAGE", "sv")
    monkeypatch.setattr(ops, "language_votes", lambda *a: pytest.fail("detection ran under a pin"))
    monkeypatch.setattr(ops, "decode_speech_audio", lambda *a: pytest.fail("decoded ahead of a pinned decode"))
    ops.transcribe(path)
    assert calls[0]["language"] == "sv"
    assert calls[0]["source"] == path
    assert ops.transcribe_signature().endswith(",lang=sv,words=0")


def test_choose_language():
    """Cases from a Swedish home archive; the numbers are whisper's."""
    choose = ops.choose_language
    assert choose([], ("sv",), 0.1) is None

    # Whisper's Norwegian on a Swedish clip: Swedish claims its siblings' mass.
    choice = choose([{"no": 0.65, "nn": 0.30, "sv": 0.02, "da": 0.01}], ("sv",), 0.1)
    assert (choice["language"], choice["detected"]) == ("sv", "no")
    assert choice["probability"] == pytest.approx(0.02) and choice["detected_probability"] == pytest.approx(0.65)
    # The English camping clip with Swedish at the same 0.02 has no such mass: English stands.
    assert choose([{"en": 0.79, "nl": 0.05, "sv": 0.02, "nn": 0.02}], ("sv",), 0.1)["language"] == "en"
    # A guess whisper is unsure of is no detection; the first expected language applies.
    assert choose([{"ja": 0.33, "ko": 0.32, "en": 0.10, "sv": 0.03}], ("sv", "da"), 0.1)["language"] == "sv"
    # Foreign for real: nothing claimed, and the top guess is sure of itself.
    assert choose([{"en": 0.94, "haw": 0.01, "sv": 0.0}], ("sv",), 0.1)["language"] == "en"
    assert choose([{"it": 0.98, "en": 0.01}], ("sv",), 0.1)["language"] == "it"

    # The Danish guide: one sure window, two noise windows. Weighted by confidence, Danish
    # keeps the top; listed as expected it wins outright, and the Norwegian mass went to
    # Swedish (first sibling listed), not to Danish.
    danish = [{"da": 0.97, "nn": 0.01, "en": 0.01}, {"en": 0.75, "da": 0.13, "cy": 0.03, "nn": 0.02}, {"en": 0.60, "cy": 0.34, "nn": 0.01}]
    choice = choose(danish, ("sv", "da"), 0.1)
    assert choice["language"] == "da" and choice["detected"] == "da"
    # With Swedish alone expected, Danish is a sibling and its mass is Swedish's claim.
    assert choose(danish, ("sv",), 0.1)["language"] == "sv"
    # And with nothing expected, the weighted average alone decides.
    assert choose(danish, (), 0.1)["language"] == "da"
    # Order decides which expected language a sibling's mass goes to.
    assert choose([{"no": 0.65, "nn": 0.30, "sv": 0.02, "da": 0.01}], ("sv", "da"), 0.1)["language"] == "sv"
    assert choose([{"no": 0.65, "nn": 0.30, "sv": 0.02, "da": 0.01}], ("da", "sv"), 0.1)["language"] == "da"

    # The floor is real: below it, and sure, the top stands.
    assert choose([{"no": 0.8, "sv": 0.05}], ("sv",), 0.9)["language"] == "no"
    # An expected language whisper never scored still claims its siblings.
    assert choose([{"no": 0.8, "en": 0.2}], ("sv",), 0.1)["language"] == "sv"


def test_a_host_transcribes_in_one_place_and_says_where(stubbed, monkeypatch):
    path, _ = stubbed
    loads = []
    monkeypatch.setattr(registry, "whisper", lambda device, compute_type: (loads.append((device, compute_type)), FakeModel([]))[1])
    # No card: the CPU, in int8.
    ops.transcribe(path)
    assert loads[-1] == ("cpu", "int8")
    assert ",on=cpu:int8," in ops.transcribe_signature()
    # A card: always on it, in float16; short of room it waits rather than use the CPU.
    monkeypatch.setattr(registry, "select_device", lambda: "cuda")
    monkeypatch.setattr(registry, "_short_of_room", lambda family, max_idle=None: None)
    ops.transcribe(path)
    assert loads[-1] == ("cuda", "float16")
    assert ",on=cuda:float16," in ops.transcribe_signature()
    monkeypatch.setattr(registry, "_short_of_room", lambda family, max_idle=None: "1600MB of the card left")
    with pytest.raises(registry.CardBusy, match="whisper waits for the card"):
        ops.transcribe(path)
