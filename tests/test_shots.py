"""`video.shots`: the rules from scores to cuts, the decode it asks of ffmpeg, and the op with the decode and the worker stubbed."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import shot_rules  # noqa: E402
import shots as worker_shots  # noqa: E402

from ems.ops import shots  # noqa: E402
from ems.tools import ffprobe as ffprobe_tool  # noqa: E402

FPS = 25
TIMES = [i / FPS for i in range(30 * FPS)]


def scores_with(peaks: dict[float, float]) -> list[float]:
    out = [0.01] * len(TIMES)
    for t, s in peaks.items():
        out[round(t * FPS)] = s
    return out


def test_a_cut_is_a_peak_over_the_threshold_with_none_higher_close_by():
    s = scores_with({8.0: 0.95, 8.2: 0.7, 14.0: 0.55, 20.0: 0.9})
    assert shot_rules.peaks(s, TIMES, 0.6, 0.5) == [(8.04, 0.95), (20.04, 0.9)]
    assert shot_rules.peaks(s, TIMES, 0.5, 0.5) == [(8.04, 0.95), (14.04, 0.55), (20.04, 0.9)]
    # Without suppression a neighbouring frame of the same cut counts again.
    assert (8.24, 0.7) in shot_rules.peaks(s, TIMES, 0.6, 0.0)


def test_a_cut_starts_on_the_frame_after_the_peak():
    # The network peaks on the old shot's last frame; the chapter starts on the new shot's first.
    assert shot_rules.peaks(scores_with({8.0: 0.95}), TIMES, 0.6, 0.5) == [(TIMES[round(8.0 * FPS) + 1], 0.95)]
    # A peak on the very last frame has no frame after it and stays where it is.
    assert shot_rules.peaks(scores_with({TIMES[-1]: 0.9}), TIMES, 0.6, 0.5) == [(TIMES[-1], 0.9)]


def test_leader_is_a_run_of_blank_frames_long_enough_to_be_one():
    means = [120.0] * len(TIMES)
    spreads = [40.0] * len(TIMES)
    for i in range(round(10 * FPS), round(11 * FPS)):
        means[i], spreads[i] = 5.0, 2.0
    # Two black frames are a blip, not leader.
    for i in (round(15 * FPS), round(15 * FPS) + 1):
        means[i], spreads[i] = 5.0, 2.0
    assert shot_rules.leader_gaps(means, spreads, TIMES) == [(10.0, 11.0)]
    # Half a second of night with a few lights in it is dark, but not blank.
    for i in range(round(20 * FPS), round(20.5 * FPS)):
        means[i], spreads[i] = 4.0, 11.5
    assert shot_rules.leader_gaps(means, spreads, TIMES) == [(10.0, 11.0)]


def boundaries(candidates, gaps=(), change=1.0, **kw):
    options = {"change_at": (lambda t: change(t) if callable(change) else change), "min_change": 0.1, "merge": 1.0, "min_shot": 3.0, **kw}
    return [(b["t"], b["kind"]) for b in shot_rules.boundaries(candidates, list(gaps), 30.0, **options)]


def test_a_flash_that_changes_nothing_lasting_is_no_cut():
    # The candidate at 14 s returns to the picture it left; the cuts at 8 and 20 do not.
    assert boundaries([(8.0, 0.95), (14.0, 0.8), (20.0, 0.9)], change=lambda t: 0.02 if t == 14.0 else 0.6) == [(8.0, "cut"), (20.0, "cut")]


def test_a_leader_ends_in_a_cut_and_hides_any_detection_inside_it():
    assert boundaries([(10.4, 0.9), (20.0, 0.9)], gaps=[(10.0, 11.0)]) == [(11.0, "leader"), (20.0, "cut")]
    # Leader at the very start or end of the file starts or ends nothing.
    assert boundaries([(20.0, 0.9)], gaps=[(0.0, 2.0), (29.0, 30.0)]) == [(20.0, "cut")]


def test_close_cuts_are_one_and_no_shot_is_shorter_than_the_least():
    assert boundaries([(8.0, 0.7), (8.6, 0.9)]) == [(8.6, "cut")]
    # Cuts at 8, 10 and 20: the shot 8–10 is too short, and the weaker of its two cuts goes.
    assert boundaries([(8.0, 0.95), (10.0, 0.7), (20.0, 0.9)]) == [(8.0, "cut"), (20.0, "cut")]
    # A cut a second in leaves a first shot nobody could find.
    assert boundaries([(1.0, 0.9), (20.0, 0.9)]) == [(20.0, "cut")]


def test_windows_pad_the_ends_and_keep_every_frame_once():
    places = list(worker_shots.windows(120))
    # 120 frames padded to 25 + 120 + 55: three windows 50 apart, the middle 50 of each kept.
    assert len(places) == 3
    kept = [p for window in places for p in window[25:75]]
    assert kept[:120] == list(range(120))
    assert places[0][:26] == [0] * 26 and places[-1][-1] == 119


def test_the_card_decodes_first_and_each_frame_has_its_second_in_a_file(monkeypatch, tmp_path):
    monkeypatch.setattr(shots, "require", lambda name: f"/usr/bin/{name}")
    attempts = shots.decode_attempts("/in.mp4", "/s/frames.rgb", "/s/times.txt", gpu=True)
    assert [name for name, _ in attempts] == ["cuda", "cpu"]
    cuda, cpu = attempts[0][1], attempts[1][1]
    assert cuda[cuda.index("-hwaccel") + 1] == "cuda"
    assert cuda[cuda.index("-vf") + 1].startswith("scale_cuda=48:27") and "metadata=mode=print:key=lavfi.signalstats.YAVG:file=/s/times.txt" in cuda[cuda.index("-vf") + 1]
    # 27 rows are odd: the card's frames come down as yuv444p, never nv12.
    assert "nv12" not in cuda[cuda.index("-vf") + 1] and "hwdownload,format=yuv444p" in cuda[cuda.index("-vf") + 1]
    assert cpu[cpu.index("-vf") + 1].startswith("scale=48:27,")
    assert all(run[-1] == "/s/frames.rgb" and run[run.index("-f") + 1] == "rawvideo" for _, run in attempts)
    times = tmp_path / "times.txt"
    times.write_text("frame:0    pts:0       pts_time:0\nlavfi.signalstats.YAVG=40\nframe:1    pts:1001    pts_time:0.0417\nlavfi.signalstats.YAVG=41\n")
    assert shots.frame_times(str(times)) == [0.0, 0.0417]


def install(monkeypatch, *, frames=750, gpu=True, duration="30.0"):
    async def fake_probe(path, *, timeout):
        return ffprobe_tool.Probe({"streams": [{"codec_type": "video", "duration": duration, "avg_frame_rate": "25/1"}], "format": {}}, None)

    async def fake_decode(ctx, path, scratch, *, gpu, cpu_allowed):
        refusal = None if gpu else cpu_allowed()
        if refusal is not None:
            raise shots.EmsError(shots.REFUSED, refusal, permanent=True)
        out = os.path.join(scratch, "frames.rgb")
        with open(out, "wb") as f:
            f.write(b"\0" * shots.FRAME_BYTES * frames)
        return out, [i / FPS for i in range(frames)], "cuda" if gpu else "cpu"

    monkeypatch.setattr(shots.ffprobe_tool, "probe", fake_probe)
    monkeypatch.setattr(shots, "decode", fake_decode)
    monkeypatch.setattr(shots.ffmpeg_tool, "gpu_present", lambda: gpu)


async def test_the_op_asks_the_worker_with_its_rules_and_answers_the_cuts(client, fake_worker, roots, monkeypatch):
    ro, _ = roots
    (ro / "1920x1080.mp4").write_bytes(b"mp4")
    install(monkeypatch)
    fake_worker.on("shot_boundaries", lambda req: {"cuts": [{"t": 8.04, "score": 0.97, "kind": "cut"}], "gaps": [], "frames": 750, "model": "transnetv2"})
    r = await client.post("/v0/video/shots", json={"file": {"path": str(ro / "1920x1080.mp4")}, "threshold": 0.55})
    body = r.json()
    assert r.status_code == 200, body
    sent = fake_worker.requests[-1]
    assert sent["op"] == "shot_boundaries" and len(sent["times"]) == 750 and sent["duration"] == 30.0
    assert (sent["threshold"], sent["nms"], sent["merge"], sent["min_shot"], sent["min_change"], sent["debug"]) == (0.55, 0.5, 1.0, 3.0, 0.1, False)
    # The scratch frames are gone once the worker has them.
    assert not os.path.exists(sent["path"])
    assert body["result"]["cuts"] == [{"t": 8.04, "score": 0.97, "kind": "cut"}]
    assert body["result"]["params"]["threshold"] == 0.55 and "scores" not in body["result"]
    assert body["meta"]["producer"].startswith("transnetv2+ffmpeg/") and body["meta"]["producer"].endswith("/cuda")


async def test_a_turned_file_is_refused(client, roots, monkeypatch):
    ro, _ = roots
    (ro / "v.mp4").write_bytes(b"mp4")
    install(monkeypatch)
    r = await client.post("/v0/video/shots", json={"file": {"path": str(ro / "v.mp4"), "angle": 90}})
    assert r.status_code == 400 and "upright" in r.json()["error"]["message"]


async def test_without_a_card_a_file_over_the_cpu_budget_is_refused_whole(client, fake_worker, roots, monkeypatch):
    ro, _ = roots
    (ro / "tape.mp4").write_bytes(b"mp4")
    install(monkeypatch, gpu=False, duration="3600.0")
    r = await client.post("/v0/video/shots", json={"file": {"path": str(ro / "tape.mp4")}})
    error = r.json()["error"]
    assert error["code"] == "refused" and "one decode, not windows" in error["message"]
    # A short clip fits the budget whole.
    install(monkeypatch, gpu=False)
    fake_worker.on("shot_boundaries", lambda req: {"cuts": [], "gaps": [], "frames": 750, "model": "transnetv2"})
    r = await client.post("/v0/video/shots", json={"file": {"path": str(ro / "tape.mp4")}})
    assert r.status_code == 200 and r.json()["meta"]["producer"].endswith("/cpu")
