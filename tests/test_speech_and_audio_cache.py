"""speech.voiceprint is one call per recording, and the audio cache extracts a file once."""

import asyncio

import pytest

from ems.tools import audio as audio_tool


@pytest.fixture
def fake_extract(monkeypatch):
    calls = []

    async def extract(path, out, *, timeout, sample_rate=16000, channels=1):
        calls.append(path)
        await asyncio.sleep(0.05)
        if path.endswith("silent.mp4"):
            return False
        with open(out, "wb") as h:
            h.write(b"RIFF" + b"\0" * 100)
        return True

    monkeypatch.setattr(audio_tool.ffmpeg_tool, "extract_audio", extract)
    return calls


async def test_voiceprint_is_one_call_per_recording(client, fake_worker, roots, fake_extract):
    """Diarize, embed every turn in one worker op, stitch the windows, re-embed what was rejoined."""
    ro, _ = roots
    (ro / "talk.mp4").write_bytes(b"mp4")
    # Two windows of 90s. One voice runs across the boundary (w0/s0 -> w1/s0), a second voice sits
    # in w0 only, and one turn is too short to embed.
    fake_worker.on("diarize", lambda req: {
        "speech": True, "duration": 120.0, "window": 90, "model": "sortformer",
        "turns": [
            {"start": 0.0, "end": 10.0, "localSpeaker": "w0/speaker_0"},
            {"start": 12.0, "end": 12.4, "localSpeaker": "w0/speaker_1"},
            {"start": 20.0, "end": 30.0, "localSpeaker": "w0/speaker_1"},
            {"start": 80.0, "end": 90.0, "localSpeaker": "w0/speaker_0"},
            {"start": 90.0, "end": 97.0, "localSpeaker": "w1/speaker_0"},
        ],
    })
    voice = {0.0: [1, 0, 0.01], 20.0: [0, 1, 0], 80.0: [1, 0, 0.02], 90.0: [1, 0, 0.03], 80.5: [1, 0, 0]}
    fake_worker.on("embed_voice_segments", lambda req: {
        "model": "titanet",
        "voiceprints": [{"embedding": voice.get(s["start"], [1, 0, 0]), "dim": 3} for s in req["spans"]],
    })

    r = await client.post("/v0/speech/voiceprint", json={"file": {"path": str(ro / "talk.mp4")}})
    body = r.json()
    assert r.status_code == 200, body
    result = body["result"]
    assert result["speech"] is True and result["audio"] is True and result["duration"] == 120.0
    assert [(t["start"], t["end"], t["speaker"]) for t in result["turns"]] == [
        (0.0, 10.0, "speaker_0"),
        (12.0, 12.4, "speaker_1"),
        (20.0, 30.0, "speaker_1"),
        (80.0, 97.0, "speaker_0"),  # rejoined across the window boundary
    ]
    assert result["turns"][1]["embedding"] is None and result["turns"][1]["dimension"] is None
    assert result["turns"][0]["dimension"] == 3
    assert body["meta"]["producer"] == "sortformer+titanet"

    ops = [(req["op"], req.get("spans")) for req in fake_worker.requests]
    assert ops[0][0] == "diarize"
    # One embed call for the four long turns, one more for the single rejoined turn.
    assert ops[1] == ("embed_voice_segments", [{"start": 0.0, "end": 10.0}, {"start": 20.0, "end": 30.0}, {"start": 80.0, "end": 90.0}, {"start": 90.0, "end": 97.0}])
    assert ops[2] == ("embed_voice_segments", [{"start": 80.0, "end": 97.0}])
    assert len(ops) == 3
    assert fake_extract == [str(ro / "talk.mp4")]

    # The op finds the turns itself; a caller cannot send spans.
    r = await client.post("/v0/speech/voiceprint", json={"file": {"path": str(ro / "talk.mp4")}, "spans": [{"start": 0, "end": 2}]})
    assert r.status_code == 400


async def test_voiceprint_of_silence(client, fake_worker, roots, fake_extract):
    ro, _ = roots
    (ro / "hum.mp4").write_bytes(b"mp4")
    (ro / "silent.mp4").write_bytes(b"mp4")
    fake_worker.on("diarize", lambda req: {"speech": False, "duration": 5.0, "window": 90, "model": "sortformer", "turns": []})
    r = await client.post("/v0/speech/voiceprint", json={"file": {"path": str(ro / "hum.mp4")}})
    assert r.json()["result"] == {"speech": False, "audio": True, "duration": 5.0, "turns": []}
    r = await client.post("/v0/speech/voiceprint", json={"file": {"path": str(ro / "silent.mp4")}})
    assert r.json()["result"] == {"speech": False, "audio": False, "duration": 0.0, "turns": []}
    assert [req["op"] for req in fake_worker.requests] == ["diarize"]


async def test_silent_file_is_cached_as_no_audio(client, fake_worker, roots, fake_extract):
    ro, _ = roots
    (ro / "silent.mp4").write_bytes(b"mp4")
    for _ in range(2):
        r = await client.post("/v0/speech/transcribe", json={"file": {"path": str(ro / "silent.mp4")}})
        assert r.json()["result"] == {"speech": False, "audio": False, "text": "", "segments": [], "duration": 0.0}
    assert fake_extract == [str(ro / "silent.mp4")]
    assert fake_worker.requests == []


async def test_concurrent_requests_share_one_extraction(tmp_path, fake_extract):
    cache = audio_tool.AudioCache(str(tmp_path / "scratch"), ttl_seconds=0.2, budget_mb=1)
    await cache.start()
    try:
        f = tmp_path / "a.mp4"
        f.write_bytes(b"x")
        wavs = await asyncio.gather(*(cache.acquire(str(f)) for _ in range(5)))
        assert len(set(wavs)) == 1 and fake_extract == [str(f)]
        for _ in range(5):
            cache.release(str(f))
        assert cache.snapshot()["entries"] == 1
        await asyncio.sleep(0.3)
        cache.sweep()
        assert cache.snapshot()["entries"] == 0
        # A changed file is a new key.
        await cache.acquire(str(f))
        f.write_bytes(b"xy")
        await cache.acquire(str(f))
        assert len(fake_extract) == 3
    finally:
        await cache.stop()


async def test_voiceprint_reply_may_exceed_the_default_line_limit(client, fake_worker, roots, fake_extract):
    """A recording's voiceprints are one reply line of megabytes, past asyncio's 64 KiB default, and it is read whole."""
    ro, _ = roots
    (ro / "tape.mp3").write_bytes(b"mp3")
    turns = [{"start": float(i * 3), "end": float(i * 3 + 2), "localSpeaker": "w0/speaker_0"} for i in range(1000)]
    fake_worker.on("diarize", lambda req: {"speech": True, "duration": 3000.0, "window": 90, "model": "sortformer", "turns": turns})
    fake_worker.on("embed_voice_segments", lambda req: {
        "model": "titanet",
        "voiceprints": [{"embedding": [0.123456789] * 192, "dim": 192} for _ in req["spans"]],
    })
    r = await client.post("/v0/speech/voiceprint", json={"file": {"path": str(ro / "tape.mp3")}})
    assert r.status_code == 200, r.text[:200]
    assert len(r.json()["result"]["turns"]) == 1000
