"""`fingerprint.compute` against the scripted worker, with ffmpeg and fpcalc stubbed."""

import os

from ems.ops import fingerprint
from ems.tools import decode as decode_tool
from ems.tools.run import Completed


async def test_a_picture_gets_one_descriptor_in_its_display_frame(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.jpg").write_bytes(b"jpeg")
    fake_worker.on("copy_descriptors", lambda req: {"descriptors": [[0.6, 0.8]], "model": "sscd_disc_mixup"})
    r = await client.post("/v0/fingerprint/compute", json={"file": {"path": str(ro / "a.jpg"), "angle": 90, "mirror": True, "mimetype": "image/jpeg"}})
    body = r.json()
    assert r.status_code == 200, body
    assert body["result"] == {"fingerprints": [{"kind": "sscd-image", "dimension": 2, "vector": [0.6, 0.8], "label": "primary"}]}
    assert body["meta"]["producer"] == "sscd_disc_mixup"
    sent = fake_worker.requests[-1]
    assert sent["op"] == "copy_descriptors" and sent["paths"] == [str(ro / "a.jpg")] and sent["angle"] == 90 and sent["mirror"] is True


async def test_a_picture_the_readers_cannot_open_is_decoded_out_of_process(client, fake_worker, roots, monkeypatch):
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
        return {"descriptors": [[1.0, 0.0]], "model": "sscd_disc_mixup"}

    fake_worker.on("copy_descriptors", answer)
    r = await client.post("/v0/fingerprint/compute", json={"file": {"path": str(original), "mimetype": "image/x-fujifilm-raf"}})
    body = r.json()
    assert r.status_code == 200, body
    assert body["result"]["fingerprints"][0]["vector"] == [1.0, 0.0]
    assert [q["paths"] for q in fake_worker.requests if q["op"] == "copy_descriptors"] == [[str(original)], [str(decoded)]]


async def test_a_video_gets_a_descriptor_per_keyframe(client, fake_worker, roots, monkeypatch):
    ro, _ = roots
    (ro / "v.mp4").write_bytes(b"mp4")

    async def fake_frames(ctx, path, out_dir, *, at, count, strategy, fmt, quality, max_edge, progress):
        assert at is None and strategy == "representative"
        frames = []
        for i in range(count):
            p = os.path.join(out_dir, f"frame-{i:03d}.jpg")
            with open(p, "wb") as h:
                h.write(b"jpeg")
            frames.append({"path": p, "t": i * 10.0})
        return frames

    async def fake_producer(*names):
        return "+".join(f"{n}/1" for n in names)

    monkeypatch.setattr(fingerprint, "extract_frames", fake_frames)
    monkeypatch.setattr(fingerprint, "producer", fake_producer)
    fake_worker.on("copy_descriptors", lambda req: {"descriptors": [[float(i), 1.0] for i in range(len(req["paths"]))], "model": "sscd_disc_mixup"})
    r = await client.post("/v0/fingerprint/compute", json={"file": {"path": str(ro / "v.mp4"), "mimetype": "video/mp4", "angle": 180}})
    body = r.json()
    assert r.status_code == 200, body
    prints = body["result"]["fingerprints"]
    assert [p["label"] for p in prints] == [f"keyframe-{i}" for i in range(8)]
    assert {p["kind"] for p in prints} == {"sscd-video"} and prints[3]["vector"] == [3.0, 1.0]
    assert body["meta"]["producer"] == "sscd_disc_mixup+ffmpeg/1"
    assert fake_worker.requests[-1]["angle"] == 180


async def test_a_recording_gets_its_chromaprint_and_no_model_runs(client, fake_worker, roots, monkeypatch):
    ro, _ = roots
    (ro / "a.m4a").write_bytes(b"m4a")

    async def fake_run(args, *, timeout):
        return Completed(args, 0, b"DURATION=10\nFINGERPRINT=1,2,3\n", b"")

    async def fake_producer(*names):
        return "+".join(f"{n}/1" for n in names)

    monkeypatch.setattr(fingerprint, "require", lambda name: name)
    monkeypatch.setattr(fingerprint, "run", fake_run)
    monkeypatch.setattr(fingerprint, "producer", fake_producer)
    r = await client.post("/v0/fingerprint/compute", json={"file": {"path": str(ro / "a.m4a"), "mimetype": "audio/mp4"}})
    body = r.json()
    assert r.status_code == 200, body
    assert body["result"] == {"fingerprints": [{"kind": "chromaprint", "label": "primary", "fingerprint": [1, 2, 3]}]}
    assert body["meta"]["producer"] == "fpcalc/1"
    assert fake_worker.requests == []


async def test_a_kind_that_does_not_fit_the_file_is_refused(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.jpg").write_bytes(b"jpeg")
    r = await client.post("/v0/fingerprint/compute", json={"file": {"path": str(ro / "a.jpg"), "mimetype": "image/jpeg"}, "kinds": ["sscd-video"]})
    assert r.status_code == 400, r.json()
