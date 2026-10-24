"""`vision.embed` and `vision.embed_text` against the scripted worker, with the tools stubbed."""

import os

from ems.ops import vision
from ems.tools import decode as decode_tool


async def test_image_embed_returns_one_vector_named_by_the_worker(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.jpg").write_bytes(b"jpeg")
    fake_worker.on("embed_images", lambda req: {"embeddings": [[0.6, 0.8]], "model": "ViT-SO400M-16-SigLIP2-384__webli"})
    r = await client.post("/v0/vision/embed", json={"file": {"path": str(ro / "a.jpg"), "angle": 90, "mirror": True, "mimetype": "image/jpeg"}})
    body = r.json()
    assert r.status_code == 200, body
    assert body["result"] == {"embedding": [0.6, 0.8], "dimension": 2}
    # The producer is what the worker says ran, not a name the front keeps on its own.
    assert body["meta"]["producer"] == "ViT-SO400M-16-SigLIP2-384__webli"
    sent = fake_worker.requests[-1]
    assert sent["op"] == "embed_images" and sent["paths"] == [str(ro / "a.jpg")] and sent["angle"] == 90 and sent["mirror"] is True


async def test_image_embed_decodes_out_of_process_when_the_reader_cannot(client, fake_worker, roots, monkeypatch):
    ro, _ = roots
    original = ro / "a.raf"
    original.write_bytes(b"raw")
    decoded = ro / "decoded.jpg"
    decoded.write_bytes(b"jpeg")

    async def fake_decode(path, mimetype, angle, mirror, *, scratch, timeout):
        assert path == str(original) and mimetype == "image/x-fujifilm-raf"
        return decode_tool.Decoded(str(decoded), decode_tool.normalize_angle(angle), mirror, None)

    monkeypatch.setattr(decode_tool, "decode_for_models", fake_decode)

    def answer(req):
        if req["paths"] == [str(original)]:
            raise ValueError(f"could not load image: {original}")
        return {"embeddings": [[1.0, 0.0]], "model": "m"}

    fake_worker.on("embed_images", answer)
    r = await client.post("/v0/vision/embed", json={"file": {"path": str(original), "angle": 270, "mimetype": "image/x-fujifilm-raf"}})
    body = r.json()
    assert r.status_code == 200, body
    assert body["result"]["embedding"] == [1.0, 0.0]
    assert [q["paths"] for q in fake_worker.requests if q["op"] == "embed_images"] == [[str(original)], [str(decoded)]]


async def test_video_embed_returns_a_vector_per_keyframe(client, fake_worker, roots, monkeypatch):
    ro, _ = roots
    (ro / "v.mp4").write_bytes(b"mp4")

    async def fake_frames(ctx, path, out_dir, *, at, count, strategy, fmt, quality, max_edge, progress):
        frames = []
        for i in range(count):
            p = os.path.join(out_dir, f"frame-{i:03d}.jpg")
            with open(p, "wb") as h:
                h.write(b"jpeg")
            frames.append({"path": p, "t": i * 1.5})
        return frames

    monkeypatch.setattr(vision, "extract_frames", fake_frames)
    fake_worker.on("embed_images", lambda req: {"embeddings": [[float(i), 1.0] for i in range(len(req["paths"]))], "model": "m"})
    r = await client.post("/v0/vision/embed", json={"file": {"path": str(ro / "v.mp4"), "mimetype": "video/mp4", "angle": 270}})
    body = r.json()
    assert r.status_code == 200, body
    frames = body["result"]["frames"]
    # Eight keyframes, each with its moment in the clip.
    assert len(frames) == 8 and body["result"]["dimension"] == 2
    assert frames[2] == {"at": 3.0, "embedding": [2.0, 1.0]}
    assert fake_worker.requests[-1]["angle"] == 270


async def test_audio_is_not_a_picture(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.mp3").write_bytes(b"mp3")
    r = await client.post("/v0/vision/embed", json={"file": {"path": str(ro / "a.mp3"), "mimetype": "audio/mpeg"}})
    body = r.json()
    assert not body["ok"] and body["error"]["code"] == "unsupported"
    assert not [q for q in fake_worker.requests if q["op"] == "embed_images"]


async def test_embed_text_is_one_query_in_the_pictures_space(client, fake_worker):
    fake_worker.on("embed_image_texts", lambda req: {"embeddings": [[0.0, 1.0]], "model": "ViT-SO400M-16-SigLIP2-384__webli"})
    r = await client.post("/v0/vision/embed_text", json={"text": "segelbåt", "options": {"priority": "interactive"}})
    body = r.json()
    assert r.status_code == 200, body
    assert body["result"] == {"embedding": [0.0, 1.0], "dimension": 2}
    assert fake_worker.requests[-1] == {"op": "embed_image_texts", "texts": ["segelbåt"]}


def test_queries_are_embedded_on_the_cpu_unless_told_otherwise(monkeypatch):
    from ems.modelworker import model_registry as registry

    monkeypatch.setattr(registry, "select_device", lambda: "cuda")
    monkeypatch.setattr(registry, "IMAGE_EMBED_DEVICE", "auto")
    # The pictures go to the card; a query does not take ~2.8 GB of it from the batch families.
    assert (registry.image_embed_device(), registry.image_embed_text_device()) == ("cuda", "cpu")
    monkeypatch.setattr(registry, "IMAGE_EMBED_TEXT_DEVICE", "auto")
    assert registry.image_embed_text_device() == "cuda"
