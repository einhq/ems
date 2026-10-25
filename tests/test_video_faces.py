"""`faces.detect_video`: following faces through the sampled frames, grouping them into the
file's faces, and the op with ffprobe and the worker stubbed."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import face_tracks  # noqa: E402

from ems.ops import faces  # noqa: E402
from ems.tools import ffprobe  # noqa: E402

# Three people, as vectors far apart; a fourth close to the first, the same person in other light.
ANNA = face_tracks.unit([1.0, 0.0, 0.0, 0.0])
ANNA_LATER = face_tracks.unit([0.9, 0.0, 0.0, 0.4])
BERIT = face_tracks.unit([0.0, 1.0, 0.0, 0.0])
CARL = face_tracks.unit([0.0, 0.0, 1.0, 0.0])


def face(x: float, vector, size: float = 120.0, confidence: float = 0.9, yaw: float | None = None) -> dict:
    detection = {"box": (x, 100.0, x + size, 100.0 + size), "confidence": confidence, "unit": vector}
    if yaw is not None:
        detection["pose"] = {"pitch": 0.0, "yaw": yaw, "roll": 0.0}
    return detection


def samples(*frames, interval: float = 1.0) -> list[dict]:
    return [{"index": i, "t": round(i * interval, 3), "faces": list(found)} for i, found in enumerate(frames)]


def test_a_face_that_stays_put_is_one_track():
    tracks = face_tracks.follow(samples([face(100, ANNA)], [face(110, ANNA)], [face(120, ANNA)]), 1.0)
    assert [len(track) for track in tracks] == [3]
    assert [d["t"] for d in tracks[0]] == [0.0, 1.0, 2.0]


def test_a_face_that_moved_too_far_to_overlap_is_followed_by_its_likeness():
    tracks = face_tracks.follow(samples([face(100, ANNA)], [face(900, ANNA)]), 1.0)
    assert [len(track) for track in tracks] == [1 + 1]


def test_an_overlapping_box_of_somebody_else_does_not_continue_a_track():
    # Berit steps in front of the camera where Anna stood.
    tracks = face_tracks.follow(samples([face(100, ANNA)], [face(105, BERIT)]), 1.0)
    assert [len(track) for track in tracks] == [1, 1]


def test_two_people_side_by_side_keep_their_own_tracks():
    tracks = face_tracks.follow(samples([face(100, ANNA), face(400, BERIT)], [face(420, BERIT), face(110, ANNA)]), 1.0)
    assert sorted(len(track) for track in tracks) == [2, 2]
    for track in tracks:
        assert len({tuple(d["unit"]) for d in track}) == 1


def test_a_track_survives_a_short_gap_and_ends_after_a_long_one():
    gap = [[] for _ in range(face_tracks.TRACK_MAX_GAP)]
    assert [len(t) for t in face_tracks.follow(samples([face(100, ANNA)], *gap, [face(100, ANNA)]), 1.0)] == [2]
    longer = [[] for _ in range(face_tracks.TRACK_MAX_GAP + 1)]
    assert [len(t) for t in face_tracks.follow(samples([face(100, ANNA)], *longer, [face(100, ANNA)]), 1.0)] == [1, 1]


def test_a_gap_is_measured_in_seconds_however_often_the_video_is_sampled():
    # Two seconds unseen at a quarter of a second is eight samples, and still the same track.
    gap = [[] for _ in range(8)]
    assert [len(t) for t in face_tracks.follow(samples([face(100, ANNA)], *gap, [face(100, ANNA)], interval=0.25), 0.25)] == [2]
    longer = [[] for _ in range(9)]
    assert [len(t) for t in face_tracks.follow(samples([face(100, ANNA)], *longer, [face(100, ANNA)], interval=0.25), 0.25)] == [1, 1]
    assert face_tracks.appearances([0.0, 2.25, 4.75], 0.25, None) == [[0.0, 2.375], [4.625, 4.875]]


def test_tracks_of_one_person_at_different_times_are_one_face():
    gap = [[] for _ in range(10)]
    tracks = face_tracks.follow(samples([face(100, ANNA)], *gap, [face(500, ANNA_LATER)], [face(500, BERIT)]), 1.0)
    groups = face_tracks.group(tracks)
    assert sorted(len(g) for g in groups) == [1, 2]


def test_tracks_seen_in_the_same_frame_are_never_one_face():
    # Two near-identical faces side by side — twins, or a mirror — are two faces of the frame.
    tracks = face_tracks.follow(samples([face(100, ANNA), face(600, ANNA_LATER)]), 1.0)
    assert len(face_tracks.group(tracks)) == 2


def test_groups_join_by_the_likeness_of_their_means():
    a, b = [{"index": 0, "t": 0.0, **face(0, ANNA)}], [{"index": 5, "t": 5.0, **face(0, BERIT)}]
    # Like Anna, and half as like Berit as a join needs: joins Anna, and the two together are
    # further from Berit still.
    near = [{"index": 9, "t": 9.0, **face(0, face_tracks.unit([1.0, 0.39, 0.0, 0.0]))}]
    groups = face_tracks.group([a, b, near])
    assert sorted(len(g) for g in groups) == [1, 2]


def test_a_noisy_sighting_joins_through_the_mean_of_the_others():
    # Three blurred looks at Anna, each pulled off her by noise of its own. The third is less
    # like either of the others than a join needs, but the two together are a better Anna.
    looks = [face_tracks.unit([1.0, 0.9, 0.0, 0.0]), face_tracks.unit([1.0, 0.0, 0.9, 0.0]), face_tracks.unit([1.0, 0.0, 0.0, 1.2])]
    assert face_tracks.dot(looks[0], looks[2]) < 0.5 and face_tracks.dot(looks[1], looks[2]) < 0.5
    tracks = [[{"index": i * 5, "t": float(i * 5), **face(0, v)}] for i, v in enumerate(looks)]
    assert len(face_tracks.group(tracks, min_similarity=0.5)) == 1


def test_appearances_merge_close_sightings_and_stay_within_the_video():
    spans = face_tracks.appearances([0.0, 1.0, 2.0, 4.0, 20.0, 21.0], 1.0, 21.2)
    assert spans == [[0.0, 4.5], [19.5, 21.2]]


def test_a_face_is_answered_with_its_clearest_detection_and_its_mean_vector():
    found = samples(
        [face(100, ANNA, size=40)],
        [face(100, ANNA_LATER, size=120, yaw=60)],
        [face(100, ANNA, size=120, yaw=5)],
    )
    [only] = face_tracks.faces_of(found, 1.0, 3.0)
    assert only["detection"]["t"] == 2.0
    assert only["detections"] == 3
    assert only["appearances"] == [[0.0, 2.5]]
    assert math.isclose(sum(v * v for v in only["vector"]), 1.0)


def test_faces_come_first_seen_first():
    found = samples([face(100, BERIT)], [face(100, BERIT), face(500, CARL)], [face(500, CARL)])
    assert [f["detection"]["unit"] for f in face_tracks.faces_of(found, 1.0, None)] == [BERIT, CARL]


def test_a_face_standing_still_keeps_about_one_point_a_second_without_its_jitter():
    # Eight seconds at a quarter of a second, the detector's box off to one side now and then.
    points = [[i * 0.25, 0.5 + (0.003 if i % 4 == 2 else 0.0), 0.4, 0.05, 0.08] for i in range(33)]
    kept = face_tracks.path(points, 0.25)
    assert [p[0] for p in kept] == [float(s) for s in range(9)]
    # The median of each sighting and its neighbours: a lone stray box is gone.
    assert all(p[1] == 0.5 for p in kept)


def test_a_face_moving_steadily_keeps_a_point_a_second_and_one_that_stops_keeps_the_stop():
    steady = [[i * 0.25, 0.1 + 0.02 * i, 0.4, 0.05, 0.08] for i in range(9)]
    assert [p[0] for p in face_tracks.path(steady, 0.25)] == [0.0, 1.0, 2.0]
    # Across the frame for a second, then still: a line from the start past the stop would miss
    # the face where it stopped, so the stop is kept.
    stops = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]
    kept = face_tracks.path([[i * 0.25, x, 0.4, 0.05, 0.08] for i, x in enumerate(stops)], 0.25)
    assert [p[0] for p in kept] == [0.0, 1.0, 2.0]


def test_a_face_moving_fast_and_unevenly_keeps_every_sample():
    jumps = [0.1, 0.12, 0.3, 0.32, 0.6, 0.62, 0.9]
    kept = face_tracks.path([[i * 0.25, x, 0.4, 0.05, 0.08] for i, x in enumerate(jumps)], 0.25)
    assert len(kept) == len(jumps)


def test_a_path_keeps_both_ends_of_a_stretch_the_face_was_unseen_in():
    seen = [[t, 0.5, 0.4, 0.05, 0.08] for t in (0.0, 0.25, 0.5, 0.75, 3.0, 3.25, 3.5)]
    assert [p[0] for p in face_tracks.path(seen, 0.25)] == [0.0, 0.75, 3.0, 3.5]


def readable_video(duration: str = "63.2"):
    async def probe(path, *, timeout):
        return ffprobe.Probe({"streams": [{"codec_type": "video", "duration": duration}], "format": {"duration": "64.0"}}, None)

    return probe


async def test_detect_video_asks_the_worker_with_the_pictures_own_length(client, fake_worker, roots, monkeypatch):
    ro, _ = roots
    (ro / "v.mp4").write_bytes(b"mp4")
    monkeypatch.setattr(faces.ffprobe_tool, "probe", readable_video())
    fake_worker.on("detect_video_faces", lambda req: {
        "frame": {"width": 1920, "height": 1080}, "interval": 1.0, "samples": 63, "frames": 1580,
        "faces": [{
            "appearances": [[0.5, 9.5]], "detections": 9, "t": 4.0,
            "box": {"x": 0.4, "y": 0.2, "width": 0.1, "height": 0.18}, "confidence": 0.93,
            "landmarks": [[0.42, 0.25]], "pose": {"pitch": 1, "yaw": -8, "roll": 0}, "embedding": [0.6, 0.8],
            "track": [[1.0, 0.41, 0.2, 0.1, 0.18], [2.0, 0.4, 0.2, 0.1, 0.18]],
        }],
    })
    r = await client.post("/v0/faces/detect_video", json={"file": {"path": str(ro / "v.mp4"), "angle": 90}, "min_size": 0.02, "min_confidence": 0.7})
    body = r.json()
    assert body["ok"], body
    assert body["result"] == {
        "frame": {"width": 1920, "height": 1080}, "duration": 63.2, "interval": 1.0, "samples": 63,
        "faces": [{
            "appearances": [[0.5, 9.5]], "detections": 9, "t": 4.0,
            "box": {"x": 0.4, "y": 0.2, "width": 0.1, "height": 0.18}, "confidence": 0.93, "embedding": [0.6, 0.8],
            "track": [[1.0, 0.41, 0.2, 0.1, 0.18], [2.0, 0.4, 0.2, 0.1, 0.18]],
            "landmarks": [[0.42, 0.25]], "pose": {"pitch": 1.0, "yaw": -8.0, "roll": 0.0},
        }],
    }
    sent = fake_worker.requests[-1]
    assert sent["duration"] == 63.2 and sent["angle"] == 90 and sent["interval"] == 0.25
    assert sent["min_size"] == 0.02 and sent["min_confidence"] == 0.7


async def test_a_file_without_a_picture_shows_no_faces(client, fake_worker, roots, monkeypatch):
    ro, _ = roots
    (ro / "a.m4a").write_bytes(b"m4a")

    async def sound_only(path, *, timeout):
        return ffprobe.Probe({"streams": [{"codec_type": "audio"}], "format": {"duration": "30"}}, None)

    monkeypatch.setattr(faces.ffprobe_tool, "probe", sound_only)
    r = await client.post("/v0/faces/detect_video", json={"file": {"path": str(ro / "a.m4a")}})
    assert r.json()["result"]["faces"] == []
    assert fake_worker.requests == []


async def test_a_video_nothing_can_read_is_undecodable(client, fake_worker, roots, monkeypatch):
    ro, _ = roots
    (ro / "v.mov").write_bytes(b"mov")

    async def refused(path, *, timeout):
        return ffprobe.Probe(None, "moov atom not found")

    monkeypatch.setattr(faces.ffprobe_tool, "probe", refused)
    r = await client.post("/v0/faces/detect_video", json={"file": {"path": str(ro / "v.mov")}})
    assert r.json()["error"]["code"] == "undecodable" and r.json()["error"]["permanent"] is True

    # One ffprobe reads but the decoder cannot: the worker's verdict, the same code.
    monkeypatch.setattr(faces.ffprobe_tool, "probe", readable_video())

    class Undecodable(Exception):
        pass

    def fails(req):
        raise Undecodable("no video frame could be decoded")

    fake_worker.on("detect_video_faces", fails)
    r = await client.post("/v0/faces/detect_video", json={"file": {"path": str(ro / "v.mov")}})
    assert r.json()["error"]["code"] == "undecodable"


async def test_a_late_poster_frame_is_sought_on_the_input_first(client, roots, monkeypatch, tmp_path):
    from ems.ops import video
    from ems.tools import run as run_tool

    ro, _ = roots
    (ro / "v.mp4").write_bytes(b"mp4")
    calls = []

    async def ffmpeg(args, *, timeout, **kw):
        calls.append(list(args))
        with open(args[-1], "wb") as out:
            out.write(b"jpeg")
        return run_tool.Completed(list(args), 0, b"", b"")

    monkeypatch.setattr(video, "require", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video.ffmpeg_tool, "gpu_present", lambda: False)
    monkeypatch.setattr(video, "run", ffmpeg)
    ctx = client.app.state.ctx

    await video.extract_poster(ctx, str(ro / "v.mp4"), str(tmp_path / "late.jpg"), at=2400.0, deinterlace=False)
    args = calls[-1]
    source = args.index("-i")
    assert args[source - 2:source] == ["-ss", "2395.000"]
    assert args[args.index("-ss", source):][:2] == ["-ss", "5.000"]

    await video.extract_poster(ctx, str(ro / "v.mp4"), str(tmp_path / "early.jpg"), at=1.0, deinterlace=False)
    args = calls[-1]
    assert "-ss" not in args[:args.index("-i")]
    assert args[args.index("-ss"):][:2] == ["-ss", "1.000"]


class Frame:
    """A decoded frame, as far as `detect_video_faces` reads one: its size, and when it was shot."""

    def __init__(self, t: float):
        self.t = t
        self.shape = (360, 640, 3)


class FakeCapture:
    """`cv2.VideoCapture` over a list of frame times, 10 a second."""

    def __init__(self, times):
        self.times = times
        self.position = -1

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def get(self, prop):
        if prop == "fps":
            return 10.0
        return self.times[self.position] * 1000.0 if self.position >= 0 else 0.0

    def grab(self):
        self.position += 1
        return self.position < len(self.times)

    def retrieve(self):
        return True, Frame(self.times[self.position])

    def release(self):
        pass


class FakeFace:
    def __init__(self, x: float, vector):
        self.bbox = [x, 100.0, x + 80.0, 180.0]
        self.det_score = 0.9
        self.normed_embedding = Vectors(vector)
        self.kps = None


class Vectors(list):
    def tolist(self):
        return list(self)


def test_the_worker_samples_a_frame_an_interval_and_answers_fractions(monkeypatch):
    import types

    import inference_ops as ops
    import model_registry as registry

    seconds = [i / 10 for i in range(50)]
    cv2 = types.SimpleNamespace(CAP_PROP_FPS="fps", CAP_PROP_POS_MSEC="msec", VideoCapture=lambda path: FakeCapture(seconds))
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(ops.os.path, "exists", lambda path: True)
    monkeypatch.setattr(registry, "run_on_model_thread", lambda family, fn: fn())
    looked_at = []

    def insightface(frame):
        looked_at.append(frame.t)
        # Anna throughout; Berit only for the frame at two seconds, and too small to keep.
        found = [FakeFace(64.0, ANNA)]
        if frame.t == 2.0:
            found.append(FakeFace(400.0, BERIT))
            found[-1].bbox = [400.0, 100.0, 404.0, 104.0]
        return found

    monkeypatch.setattr(ops, "_insightface_get", insightface)
    result = ops.detect_video_faces("/files/v.mp4", interval=1.0, duration=4.9, min_size=0.05)
    assert looked_at == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert result["samples"] == 5 and result["frames"] == 50 and result["interval"] == 1.0
    assert result["frame"] == {"width": 640, "height": 360}
    [anna] = result["faces"]
    assert anna["appearances"] == [[0.0, 4.5]] and anna["detections"] == 5
    assert anna["box"] == {"x": 0.1, "y": 100 / 360, "width": 0.125, "height": 80 / 360}
    assert anna["embedding"] == ANNA
    # Every sighting's box, in time order: one a sampled frame.
    assert anna["track"] == [[t, 0.1, 0.2778, 0.125, 0.2222] for t in (0.0, 1.0, 2.0, 3.0, 4.0)]


def test_a_long_video_is_sampled_more_sparsely():
    import inference_ops as ops

    assert ops.video_face_step(1.0, 600.0) == 1.0
    assert ops.video_face_step(1.0, 2 * ops.VIDEO_FACES_MAX_SAMPLES) == 2.0
    assert ops.video_face_step(1.0, None) == 1.0
