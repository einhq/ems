"""Window-local diarizer labels become file-wide speakers (ems/ops/_voices.py)."""

import pytest

from ems.ops._voices import cosine, merge_adjacent, stitch_windows, window_of


def voice(axis: int, noise: float = 0.0) -> list[float]:
    """A unit vector pointing mostly along `axis`, nudged by `noise` so no two are identical."""
    v = [0.0, 0.0, 0.0, 0.0]
    v[axis] = 1.0
    v[(axis + 1) % 4] = noise
    return v


def turn(start: float, end: float, local: str) -> dict:
    return {"start": start, "end": end, "localSpeaker": local}


def test_cosine():
    assert cosine([1, 0], [2, 0]) == pytest.approx(1)
    assert cosine([1, 0], [0, 3]) == pytest.approx(0)
    # 0 rather than NaN for a zero vector or a length mismatch.
    assert cosine([0, 0], [1, 1]) == 0
    assert cosine([1, 0, 0], [1, 0]) == 0


def test_window_of():
    assert window_of("w3/speaker_1") == "w3"
    assert window_of("speaker_1") == "speaker_1"


def test_joins_one_voice_heard_across_two_windows():
    turns = [turn(0, 10, "w0/speaker_0"), turn(95, 105, "w1/speaker_1")]
    assert stitch_windows(turns, [voice(0, 0.01), voice(0, 0.02)]) == ["speaker_0", "speaker_0"]


def test_keeps_different_voices_apart():
    turns = [turn(0, 10, "w0/speaker_0"), turn(95, 105, "w1/speaker_0")]
    assert stitch_windows(turns, [voice(0), voice(1)]) == ["speaker_0", "speaker_1"]


def test_never_merges_two_labels_from_the_same_window():
    # The diarizer said these are two people; identical embeddings must not overrule that.
    turns = [turn(0, 10, "w0/speaker_0"), turn(12, 20, "w0/speaker_1")]
    assert len(set(stitch_windows(turns, [voice(0), voice(0)]))) == 2


def test_does_not_chain_same_window_labels_through_a_third():
    # w0/s0 and w0/s1 are both very like w1/s0. Merging w1/s0 with either one must not then
    # pull in the other, which single-link clustering would happily do.
    turns = [turn(0, 10, "w0/speaker_0"), turn(12, 20, "w0/speaker_1"), turn(95, 105, "w1/speaker_0")]
    speakers = stitch_windows(turns, [voice(0, 0.01), voice(0, 0.02), voice(0, 0.015)])
    assert speakers[0] != speakers[1], "same-window labels stayed separate"
    assert len(set(speakers)) == 2, "the third label joined exactly one of them"


def test_names_speakers_by_first_appearance():
    turns = [turn(50, 60, "w0/speaker_1"), turn(5, 15, "w0/speaker_0")]
    assert stitch_windows(turns, [voice(1), voice(0)]) == ["speaker_1", "speaker_0"]


def test_a_turn_without_a_vector_is_unattributed():
    turns = [turn(0, 10, "w0/speaker_0"), turn(30, 30.4, "w0/speaker_9")]
    assert stitch_windows(turns, [voice(0), None]) == ["speaker_0", None]


def test_a_crumb_joins_the_anchor_it_resembles_at_the_lower_bar():
    # w2/speaker_3 has 2s of voice: not a speaker of its own, but nearer speaker_0 (cosine 0.8,
    # under the 0.85 stitch threshold) than speaker_1 (0.6), so it joins speaker_0.
    turns = [turn(0, 10, "w0/speaker_0"), turn(100, 110, "w1/speaker_0"), turn(200, 202, "w2/speaker_3")]
    tilted = [0.8, 0.6, 0.0, 0.0]
    speakers = stitch_windows(turns, [voice(0), voice(1), tilted], threshold=0.85, attach_threshold=0.5)
    assert speakers == ["speaker_0", "speaker_1", "speaker_0"]


def test_a_crumb_never_joins_an_anchor_from_its_own_window():
    turns = [turn(0, 10, "w0/speaker_0"), turn(20, 22, "w0/speaker_1")]
    assert stitch_windows(turns, [voice(0), voice(0)]) == ["speaker_0", None]


def test_a_crumb_unlike_every_anchor_is_unattributed():
    turns = [turn(0, 10, "w0/speaker_0"), turn(100, 102, "w1/speaker_0")]
    assert stitch_windows(turns, [voice(0), voice(2)]) == ["speaker_0", None]


def test_merge_leaves_unattributed_turns_apart():
    turns = [{"start": 0, "end": 5, "speaker": None}, {"start": 5, "end": 9, "speaker": None}]
    assert len(merge_adjacent(turns)) == 2


def test_merge_rejoins_a_turn_cut_at_the_boundary():
    turns = [
        {"start": 80, "end": 90, "speaker": "speaker_0", "embedding": [1]},
        {"start": 90, "end": 97, "speaker": "speaker_0", "embedding": [2]},
    ]
    # The halves' vectors go: neither was measured over the whole.
    assert merge_adjacent(turns) == [{"start": 80, "end": 97, "speaker": "speaker_0"}]


def test_merge_leaves_a_real_pause_alone():
    turns = [{"start": 0, "end": 10, "speaker": "speaker_0"}, {"start": 14, "end": 20, "speaker": "speaker_0"}]
    assert len(merge_adjacent(turns)) == 2


def test_merge_does_not_cross_a_change_of_speaker():
    turns = [{"start": 0, "end": 10, "speaker": "speaker_0"}, {"start": 10, "end": 18, "speaker": "speaker_1"}]
    assert len(merge_adjacent(turns)) == 2


def test_speakers_of_pools_each_voice_weighted_by_how_long_it_spoke():
    from ems.ops._voices import speakers_of

    turns = [
        {"start": 0.0, "end": 9.0, "speaker": "speaker_0", "embedding": [1.0, 0.0]},
        {"start": 9.0, "end": 10.0, "speaker": "speaker_1", "embedding": [0.0, 2.0]},
        {"start": 10.0, "end": 11.0, "speaker": "speaker_0", "embedding": [0.0, 1.0]},
        {"start": 11.0, "end": 11.5, "speaker": "speaker_0"},  # too short for a voiceprint
        {"start": 12.0, "end": 13.0, "speaker": None},
    ]
    speakers = speakers_of(turns)

    assert [s["id"] for s in speakers] == ["speaker_0", "speaker_1"]
    first = speakers[0]
    assert first["turns"] == [[0.0, 9.0], [10.0, 11.0], [11.0, 11.5]]
    assert first["talkSeconds"] == 10.5
    # Nine seconds of [1, 0] against one of [0, 1]: the long turn dominates.
    assert first["vector"][0] > 0.99 and abs(sum(x * x for x in first["vector"]) - 1) < 1e-9
    assert speakers[1]["vector"] == [0.0, 1.0]
    # The unattributed turn belongs to nobody.
    assert all([12.0, 13.0] not in s["turns"] for s in speakers)


def test_asd_windows_take_the_middle_of_each_speakers_longest_turns():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ems", "modelworker"))
    from inference_ops import asd_tracks, asd_windows

    speakers = [
        {"id": "speaker_0", "turns": [[0.0, 1.0], [10.0, 16.0], [20.0, 23.0], [30.0, 32.5], [40.0, 42.0]]},
        {"id": "speaker_1", "turns": [[5.0, 7.0]]},
    ]
    windows = asd_windows(speakers, window=2.0, per_speaker=3, min_turn=2.0)
    # speaker_0's three longest of those long enough (6 s, 3 s, 2.5 s), and speaker_1's one; in file order.
    assert windows == [(5.0, 7.0, "speaker_1"), (12.0, 14.0, "speaker_0"), (20.5, 22.5, "speaker_0"), (30.25, 32.25, "speaker_0")]

    # Two faces side by side, followed across frames; a third appears late.
    a = [{"frame": f, "box": (10 + f, 10, 50 + f, 50)} for f in (0, 5, 10, 15)]
    b = [{"frame": f, "box": (200, 10, 240, 50)} for f in (0, 5, 10, 15)]
    c = [{"frame": 15, "box": (400, 400, 440, 440)}]
    tracks = asd_tracks(a + b + c)
    assert sorted(len(t) for t in tracks) == [1, 4, 4]
    assert {t[0]["box"][0] for t in tracks if len(t) == 4} == {10, 200}
