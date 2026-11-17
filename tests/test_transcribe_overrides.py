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
    feature_extractor = types.SimpleNamespace(sampling_rate=16000, n_samples=480000)

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
    # Detection is stubbed at the vote level and the speech detectors at their chunks: they need
    # real audio and the real models, and what is under test here is what the decode is told,
    # not whisper's ear. Three seconds of audio, all of it speech.
    monkeypatch.setattr(ops, "decode_speech_audio", lambda model, path: [0.0] * 48000)
    monkeypatch.setattr(ops, "speech_chunks", lambda path, audio, vad: (calls.append({"vad": dict(vad)}), [{"start": 0, "end": len(audio)}])[1])
    monkeypatch.setattr(ops, "speech_samples", lambda audio, chunks: ("joined", len(audio), tuple((c["start"], c["end"]) for c in chunks)))
    monkeypatch.setattr(ops, "language_votes", lambda model, speech, windows: [{"sv": 0.6, "no": 0.3}])
    monkeypatch.setattr(registry, "whisper_vad_parameters", lambda: {"threshold": 0.2, "speech_pad_ms": 800, "min_silence_duration_ms": 2000})
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    return str(audio), calls


def decodes(calls):
    """The decode calls among `calls`, without the speech detectors' entries."""
    return [call for call in calls if "vad" not in call]


def detected(calls):
    """The settings each run gave the speech detectors."""
    return [call["vad"] for call in calls if "vad" in call]


def test_defaults_come_from_the_configuration(stubbed):
    path, calls = stubbed
    result = ops.transcribe(path)
    # Nothing pinned: the decoder is told what the detection chose, never left to guess.
    assert decodes(calls)[0]["language"] == "sv"
    assert detected(calls)[0]["min_silence_duration_ms"] == 2000
    # The detectors' chunks are what is decoded, with whisper's own VAD off, and the words are
    # always aligned; they are returned only when asked for.
    assert decodes(calls)[0]["source"] == ("joined", 48000, ((0, 48000),))
    assert decodes(calls)[0]["vad_filter"] is False and decodes(calls)[0]["word_timestamps"] is True
    assert result["speech"] is True and result["text"] == "hej" and len(result["segments"]) == 1
    assert "words" not in result["segments"][0]
    assert result["language"] == "sv"
    assert result["language_probability"] == pytest.approx(0.6)
    assert result["duration"] == pytest.approx(3.0)
    assert result["signature"].endswith(",lang=auto:3,words=0")
    assert result["signature"] == ops.transcribe_signature()


def test_expected_language_wins_at_the_floor(stubbed, monkeypatch):
    path, calls = stubbed
    monkeypatch.setattr(registry, "WHISPER_LANGUAGES", ("sv", "en"))
    monkeypatch.setattr(ops, "language_votes", lambda *a: [{"no": 0.86, "sv": 0.10}, {"no": 0.7, "sv": 0.2}])
    result = ops.transcribe(path)
    assert decodes(calls)[0]["language"] == "sv"
    assert result["language"] == "sv"
    # The probability reported is Swedish's own, on the confidence-weighted average.
    assert result["language_probability"] == pytest.approx((0.10 * 0.86 + 0.2 * 0.7) / 1.56)
    assert result["signature"].endswith(",lang=auto:3:sv,en@0.1,words=0")

    # Below the floor the top guess stands: a Polish wedding stays Polish.
    monkeypatch.setattr(ops, "language_votes", lambda *a: [{"pl": 0.98, "sv": 0.01}])
    result = ops.transcribe(path)
    assert decodes(calls)[1]["language"] == "pl" and result["language"] == "pl"


def test_overrides_reach_the_decode_and_the_signature(stubbed):
    path, calls = stubbed
    result = ops.transcribe(path, language="sv", min_silence_ms=500, word_timestamps=True)
    assert decodes(calls)[0]["language"] == "sv"
    assert detected(calls)[0]["min_silence_duration_ms"] == 500
    assert ":vad=0.2+aed0.4/800/500," in result["signature"]
    assert result["signature"].endswith(",lang=sv,words=1")
    # The configured signature is unchanged by a request's overrides.
    assert ops.transcribe_signature().endswith(",lang=auto:3,words=0")


def test_no_speech_means_no_detection(stubbed, monkeypatch):
    path, calls = stubbed
    monkeypatch.setattr(ops, "language_votes", lambda *a: [])
    result = ops.transcribe(path)
    assert decodes(calls)[0]["language"] is None
    # What the decoder itself reports, then.
    assert result["language"] == "sv" and result["language_probability"] == pytest.approx(0.9)


def test_a_pin_skips_detection(stubbed, monkeypatch):
    path, calls = stubbed
    monkeypatch.setattr(registry, "WHISPER_LANGUAGE", "sv")
    monkeypatch.setattr(ops, "language_votes", lambda *a: pytest.fail("detection ran under a pin"))
    ops.transcribe(path)
    assert decodes(calls)[0]["language"] == "sv"
    # The decoder is handed samples under a pin too, never the path.
    assert decodes(calls)[0]["source"] == ("joined", 48000, ((0, 48000),))
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
    monkeypatch.setattr(registry, "_short_of_room", lambda family, max_idle=None, keep=(): None)
    ops.transcribe(path)
    assert loads[-1] == ("cuda", "float16")
    assert ",on=cuda:float16," in ops.transcribe_signature()
    monkeypatch.setattr(registry, "_short_of_room", lambda family, max_idle=None, keep=(): "1600MB of the card left")
    with pytest.raises(registry.CardBusy, match="whisper waits for the card"):
        ops.transcribe(path)


def test_where_whisper_decodes_follows_the_card_share_and_each_place_has_its_signature(stubbed, monkeypatch):
    path, _ = stubbed
    loads = []
    monkeypatch.setattr(registry, "whisper", lambda device, compute_type: (loads.append((device, compute_type)), FakeModel([]))[1])
    monkeypatch.setattr(registry, "card_present", lambda: True)
    monkeypatch.setattr(registry, "CARD_BUDGET_MB", 0.0)
    monkeypatch.setattr(registry, "_short_of_room", lambda family, max_idle=None, keep=(): None)
    monkeypatch.setattr(registry, "cleanup_torch", lambda: None)
    places = ops.transcribe_places()
    assert [",on=cuda:float16," in places[0], ",on=cpu:int8," in places[1]] == [True, True]
    # A share that holds whisper beside a CUDA context: the card, in float16.
    monkeypatch.setattr(registry, "CARD_SHARE_MB", registry.CARD_NEED_MB["whisper"] + registry.CUDA_CONTEXT_MB)
    assert ops.transcribe(path)["signature"] == places[0] == ops.transcribe_signature()
    assert loads[-1] == ("cuda", "float16")
    # One that does not: the CPU, in int8, and the card's copy unloaded. Either signature stays current.
    monkeypatch.setattr(registry, "_cache", {"whisper:cuda:float16": object()})
    monkeypatch.setattr(registry, "CARD_SHARE_MB", 2000.0)
    assert ops.transcribe(path)["signature"] == places[1] == ops.transcribe_signature()
    assert loads[-1] == ("cpu", "int8")
    assert "whisper:cuda:float16" not in registry._cache
    assert ops.transcribe_places() == places


def test_a_host_without_a_card_has_one_place(stubbed):
    assert len(ops.transcribe_places()) == 1 and ",on=cpu:int8," in ops.transcribe_places()[0]


def test_credit_lines_are_dropped(stubbed, monkeypatch):
    """A line the decoder was sure of but whisper_credits names never reaches the transcript."""
    path, _ = stubbed

    def transcribe(self, source, **kwargs):
        info = types.SimpleNamespace(language="sv", language_probability=0.9, duration=62.6)
        return iter([FakeSegment(0.0, 62.58, "Tack för att du har tittat på den här videon!")]), info

    monkeypatch.setattr(FakeModel, "transcribe", transcribe)
    result = ops.transcribe(path)
    assert result["segments"] == [] and result["text"] == "" and result["speech"] is False


@pytest.mark.parametrize("votes, no_speech, lines, kept", [
    ({"en": 0.9}, 0.6, ["Thank you."], False),  # whisper's stock line over music
    ({"en": 0.9}, 0.3, ["Yeah, I know."], True),  # someone saying it
    ({"sv": 0.9}, 0.55, ["Tack."], True),  # an expected language
    ({"en": 0.9}, 0.55, ["Hello!", "Welcome."], True),  # more than one line
])
def test_a_lone_foreign_line_needs_a_lower_no_speech_score(stubbed, monkeypatch, votes, no_speech, lines, kept):
    """One line in a language WHISPER_LANGUAGES does not list is dropped at a lower no-speech score."""
    path, _ = stubbed
    monkeypatch.setattr(registry, "WHISPER_LANGUAGES", ("sv", "da"))
    monkeypatch.setattr(ops, "language_votes", lambda model, speech, windows: [votes])

    def transcribe(self, source, **kwargs):
        segments = [FakeSegment(float(i), float(i) + 1.0, text) for i, text in enumerate(lines)]
        for segment in segments:
            segment.no_speech_prob = no_speech
        return iter(segments), types.SimpleNamespace(language=kwargs["language"], language_probability=1.0, duration=30.0)

    monkeypatch.setattr(FakeModel, "transcribe", transcribe)
    result = ops.transcribe(path)
    assert result["speech"] is kept and len(result["segments"]) == (len(lines) if kept else 0)


def word(start, end, text):
    return types.SimpleNamespace(start=start, end=end, word=text, probability=0.9)


def test_a_segment_across_a_cut_gap_is_placed_back_by_its_words():
    """Chunks 0-2 s and 10-12 s of a file, joined: 2 s of the joined stream is 10 s of the file.
    A segment whose words fall in both becomes two entries, each in its own chunk."""
    chunks = [{"start": 0, "end": 32000}, {"start": 160000, "end": 192000}]
    segment = FakeSegment(1.0, 3.5, " Ja visst. Kom hit.", words=[
        word(1.0, 1.4, " Ja"), word(1.4, 1.9, " visst."), word(2.2, 2.8, " Kom"), word(2.8, 3.5, " hit."),
    ])
    entries = ops.restore_segments(segment, chunks, 16000)
    assert [(e["start"], e["end"], e["text"]) for e in entries] == [(1.0, 1.9, "Ja visst."), (10.2, 11.5, "Kom hit.")]
    assert [w["word"] for w in entries[1]["words"]] == [" Kom", " hit."]

    # All its words in the later chunk: one entry there, not one stretched from the earlier.
    segment = FakeSegment(1.5, 3.5, " För din egen skull.", words=[word(2.1, 2.6, " För din"), word(2.6, 3.4, " egen skull.")])
    assert [(e["start"], e["end"], e["text"]) for e in ops.restore_segments(segment, chunks, 16000)] == [(10.1, 11.4, "För din egen skull.")]

    # Without words: the chunk it starts in, ending there at the latest.
    segment = FakeSegment(1.5, 3.5, " Hej.")
    assert [(e["start"], e["end"]) for e in ops.restore_segments(segment, chunks, 16000)] == [(1.5, 2.0)]


@pytest.mark.parametrize("seconds, text, no_speech, sparse", [
    (30.0, "Thank you for watching!", 0.56, True),  # a stock line over a whole window
    (30.0, "Kan vi laga den?", 0.03, False),  # sure of it: real speech with a long pause
    (10.0, "Ja.", 0.7, False),  # short
    (30.0, " ".join(["ord"] * 15), 0.6, False),  # enough words for its length
    (20.0, "Hej", None, False),  # no score to judge by
])
def test_a_sparse_borderline_segment_is_judged_not_speech(seconds, text, no_speech, sparse):
    assert ops.is_sparse(seconds, text, no_speech) is sparse


def test_sparse_and_letterless_lines_are_dropped(stubbed, monkeypatch):
    path, _ = stubbed

    def transcribe(self, source, **kwargs):
        segments = [
            FakeSegment(0.0, 0.5, "Ja!"),
            FakeSegment(0.5, 1.0, "🎵🎵🎵"),
            FakeSegment(1.0, 1.5, "0-3"),
            FakeSegment(1.5, 31.5, "Välkommen tillbaka!"),
        ]
        segments[-1].no_speech_prob = 0.6
        return iter(segments), types.SimpleNamespace(language="sv", language_probability=0.9, duration=31.5)

    monkeypatch.setattr(FakeModel, "transcribe", transcribe)
    assert [s["text"] for s in ops.transcribe(path)["segments"]] == ["Ja!", "0-3"]


def test_no_speech_found_means_no_decode(stubbed, monkeypatch):
    path, calls = stubbed
    monkeypatch.setattr(ops, "speech_chunks", lambda path, audio, vad: [])
    monkeypatch.setattr(ops, "language_votes", lambda *a: pytest.fail("detection ran on no speech"))
    result = ops.transcribe(path)
    assert decodes(calls) == []
    assert result["speech"] is False and result["segments"] == [] and result["duration"] == pytest.approx(3.0)
    assert result["language"] is None


def test_aed_spans_end_pad_and_merge():
    """Scores per 10 ms: speech 1.0-2.0 s and 2.5-3.0 s, the gap shorter than the silence that
    ends a region, then 6.0-6.5 s. Padded by 0.5 s, kept inside the 7 s file."""
    scores = [0.0] * 700
    for a, b in ((100, 200), (250, 300), (600, 650)):
        scores[a:b] = [0.9] * (b - a)
    spans = ops.aed_spans(scores, 0.4, 0.5, 1.0, 7.0)
    assert [(round(a, 2), round(b, 2)) for a, b in spans] == [(0.5, 3.5), (5.5, 7.0)]
    # Under the threshold everywhere: nothing.
    assert ops.aed_spans([0.3] * 700, 0.4, 0.5, 1.0, 7.0) == []


def test_merge_chunks_unites_overlapping_and_touching():
    assert ops.merge_chunks([{"start": 50, "end": 80}, {"start": 0, "end": 10}, {"start": 10, "end": 20}, {"start": 60, "end": 90}]) == [
        {"start": 0, "end": 20}, {"start": 50, "end": 90},
    ]


@pytest.mark.parametrize("lines, no_speech, dropped", [
    (["Hej!"], [0.69], True),  # whisper's greeting over a near-silent clip
    (["Hej", "då!"], [0.78, 0.78], True),
    (["Oj, krossigt."], [0.16], False),  # someone saying it
    (["Hej!", "Kom hit."], [0.7, 0.3], False),  # one line the decoder heard
    (["Ett två tre fyra fem sex sju åtta nio tio elva"], [0.7], False),  # more than a few words
    (["Hej!"], [None], False),  # no score to judge by
])
def test_a_short_transcript_the_decoder_doubts_is_dropped(lines, no_speech, dropped):
    assert ops.is_short_and_unsure(lines, no_speech) is dropped


def test_a_credit_split_across_a_cut_gap_is_dropped_whole(stubbed, monkeypatch):
    """The credit filter reads the decoded line before it is split by chunk."""
    path, _ = stubbed
    monkeypatch.setattr(ops, "speech_chunks", lambda path, audio, vad: [{"start": 0, "end": 16000}, {"start": 32000, "end": 48000}])

    def transcribe(self, source, **kwargs):
        credit = FakeSegment(0.5, 1.5, " Undertexter från Amara.org", words=[word(0.5, 0.9, " Undertexter"), word(1.1, 1.5, " från Amara.org")])
        heard = FakeSegment(1.6, 1.9, " Ja, kom nu.")
        heard.no_speech_prob = 0.05
        return iter([credit, heard]), types.SimpleNamespace(language="sv", language_probability=0.9, duration=2.0)

    monkeypatch.setattr(FakeModel, "transcribe", transcribe)
    assert [s["text"] for s in ops.transcribe(path)["segments"]] == ["Ja, kom nu."]
