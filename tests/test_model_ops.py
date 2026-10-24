"""The ops that forward to the model worker, against the scripted fake."""

import asyncio
import json

import pytest


async def test_text_embed_round_trip_uses_the_configured_embedder(client, fake_worker):
    fake_worker.on("embed_texts", lambda req: {"embeddings": [[0.1, 0.2, 0.3]], "model": "jina-embeddings-v5-text-small"})
    r = await client.post("/v0/text/embed", json={"text": "hello", "options": {"priority": "interactive"}})
    body = r.json()
    assert r.status_code == 200 and body["ok"]
    assert body["result"] == {"embedding": [0.1, 0.2, 0.3], "dimension": 3}
    assert body["meta"]["producer"] == "jina-embeddings-v5-text-small"
    assert fake_worker.requests[-1] == {"op": "embed_texts", "texts": ["hello"], "role": "document"}


async def test_text_embed_current_sends_the_role_and_names_the_model(client, fake_worker):
    fake_worker.on("embed_texts", lambda req: {"embeddings": [[0.6, 0.8]], "model": "jina-embeddings-v5-text-small"})
    r = await client.post("/v0/text/embed", json={"text": "segelbåt", "model": "current", "role": "query"})
    body = r.json()
    assert r.status_code == 200 and body["ok"], body
    assert body["result"] == {"embedding": [0.6, 0.8], "dimension": 2}
    # The producer is what the worker says ran, so a stored vector names its space.
    assert body["meta"]["producer"] == "jina-embeddings-v5-text-small"
    assert fake_worker.requests[-1] == {"op": "embed_texts", "texts": ["segelbåt"], "role": "query"}


async def test_text_embed_batch_answers_in_order(client, fake_worker):
    fake_worker.on("embed_texts", lambda req: {"embeddings": [[float(i), 1.0] for i, _ in enumerate(req["texts"])], "model": "m"})
    r = await client.post("/v0/text/embed", json={"texts": ["a", "b", "c"], "model": "current"})
    body = r.json()
    assert body["result"] == {"embeddings": [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], "dimension": 2}
    # Documents are the default role: the backfill is the caller that batches.
    assert fake_worker.requests[-1]["role"] == "document"


@pytest.mark.parametrize(
    "payload",
    # Any model but the configured one is refused rather than answered from another space.
    [{}, {"text": "a", "texts": ["b"]}, {"texts": []}, {"text": "a", "model": "minilm"}, {"text": "a", "model": "legacy"}],
)
async def test_text_embed_refuses_a_malformed_request(client, fake_worker, payload):
    r = await client.post("/v0/text/embed", json=payload)
    body = r.json()
    assert not body["ok"] and body["error"]["code"] == "invalid_request", body
    assert not fake_worker.requests


async def test_text_embed_over_the_token_limit_is_an_invalid_request(client, fake_worker):
    def refuse(req):
        raise ValueError("embed_texts: text 0 is 9000 tokens, over the 8192-token limit")

    fake_worker.on("embed_texts", refuse)
    r = await client.post("/v0/text/embed", json={"text": "long", "model": "current"})
    body = r.json()
    assert not body["ok"] and body["error"]["code"] == "invalid_request" and "8192" in body["error"]["message"]


async def test_a_deadline_reaches_the_worker(client, fake_worker):
    fake_worker.on("embed_texts", lambda req: {"embeddings": [[0.1]], "model": "jina"})
    r = await client.post("/v0/text/embed", json={"text": "hi", "options": {"priority": "interactive", "deadline": 1234.5}})
    assert r.json()["ok"] is True
    sent = fake_worker.requests[-1]
    # Priority is the front's admission; the worker hears only when nobody waits any more.
    assert sent["expires_at"] == 1234.5 and "interactive" not in sent

async def test_faces_detect_converts_pixels_to_fractions(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.jpg").write_bytes(b"jpeg")
    fake_worker.on("detect_faces", lambda req: {
        "frame": {"width": 200, "height": 100},
        "faces": [
            {"index": 0, "box": {"x": 20, "y": 10, "width": 40, "height": 20}, "confidence": 0.9, "embedding": [1.0], "landmarks": [[30, 15]],
             "pose": {"pitch": -4.5, "yaw": 12.0, "roll": 1.25}},
            {"index": 1, "box": {"x": 0, "y": 0, "width": 2, "height": 2}, "confidence": 0.5, "embedding": [2.0]},
        ],
    })
    r = await client.post("/v0/faces/detect", json={"file": {"path": str(ro / "a.jpg"), "angle": 90}, "min_size": 0.05})
    body = r.json()
    assert body["ok"], body
    assert body["result"]["frame"] == {"width": 200, "height": 100}
    assert body["result"]["faces"] == [{
        "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}, "confidence": 0.9, "embedding": [1.0], "landmarks": [[0.15, 0.15]],
        # Angles pass through untouched: there is nothing to normalize in degrees.
        "pose": {"pitch": -4.5, "yaw": 12.0, "roll": 1.25},
    }]
    assert fake_worker.requests[-1]["angle"] == 90 and fake_worker.requests[-1]["path"] == str(ro / "a.jpg")


async def test_faces_embed_reports_pose(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.jpg").write_bytes(b"jpeg")
    fake_worker.on("embed_face", lambda req: {"embedding": [1.0], "confidence": 0.8, "pose": {"pitch": 1, "yaw": -30, "roll": 2}})
    r = await client.post("/v0/faces/embed", json={"file": {"path": str(ro / "a.jpg")}, "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}})
    assert r.json()["result"] == {"embedding": [1.0], "confidence": 0.8, "pose": {"pitch": 1.0, "yaw": -30.0, "roll": 2.0}}

    # A worker without the 3D landmark model reports none, and the answer simply omits it.
    fake_worker.on("embed_face", lambda req: {"embedding": [1.0], "confidence": 0.8})
    r = await client.post("/v0/faces/embed", json={"file": {"path": str(ro / "a.jpg")}, "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}})
    assert r.json()["result"] == {"embedding": [1.0], "confidence": 0.8}


async def test_faces_embed_null_when_no_face(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.jpg").write_bytes(b"jpeg")
    fake_worker.on("embed_face", lambda req: None)
    r = await client.post("/v0/faces/embed", json={"file": {"path": str(ro / "a.jpg")}, "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}})
    assert r.json() == {"ok": True, "result": None, "meta": r.json()["meta"]}


async def test_worker_errors_are_classified(client, fake_worker):
    class CardBusy(RuntimeError):
        pass

    def busy(req):
        raise CardBusy("whisper waits for the card: 900MB free on the card, needs 4000MB")

    fake_worker.on("embed_texts", busy)
    r = await client.post("/v0/text/embed", json={"text": "x"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "model_unavailable" and r.json()["error"]["permanent"] is False
    assert r.headers["retry-after"] == "60"

    def expired(req):
        raise TimeoutError("embed_texts: request expired while waiting for text-embed")

    fake_worker.on("embed_texts", expired)
    r = await client.post("/v0/text/embed", json={"text": "x"})
    assert r.status_code == 504 and r.json()["error"]["code"] == "deadline_exceeded"

async def test_worker_down_is_model_unavailable(client, fake_worker):
    await fake_worker.stop()
    r = await client.post("/v0/text/embed", json={"text": "hello"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "model_unavailable"


async def test_streaming_emits_progress_then_result(client, fake_worker, roots):
    ro, _ = roots
    (ro / "a.jpg").write_bytes(b"jpeg")
    fake_worker.on("embed_images", lambda req: {"embeddings": [[0.5, 0.5]], "model": "siglip2"})
    events = []
    body = {"file": {"path": str(ro / "a.jpg"), "mimetype": "image/jpeg"}}
    async with client.stream("POST", "/v0/vision/embed", json=body, headers={"Accept": "text/event-stream"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        buffer = ""
        async for chunk in r.aiter_text():
            buffer += chunk
        for block in buffer.strip().split("\n\n"):
            lines = dict(line.split(": ", 1) for line in block.splitlines())
            events.append((lines["event"], json.loads(lines["data"])))
    assert [e[0] for e in events] == ["progress", "result"]
    assert events[0][1] == {"phase": "embed"}
    assert events[1][1]["ok"] is True and events[1][1]["result"] == {"embedding": [0.5, 0.5], "dimension": 2}

async def test_streaming_failure_is_an_error_event(client, fake_worker):
    def boom(req):
        raise RuntimeError("something odd")

    fake_worker.on("embed_texts", boom)
    async with client.stream("POST", "/v0/text/embed", json={"text": "x"}, headers={"Accept": "text/event-stream"}) as r:
        body = "".join([chunk async for chunk in r.aiter_text()])
    assert r.status_code == 200
    assert "event: error" in body and '"code":"tool_failed"' in body


async def test_queue_depth_answers_busy(client, fake_worker):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(req):
        started.set()
        await release.wait()
        return {"embeddings": [[1.0]], "model": "m"}

    fake_worker.on("embed_texts", slow)
    # limit 2 running + queue depth 1 waiting; the fourth is refused.
    tasks = [asyncio.create_task(client.post("/v0/text/embed", json={"text": str(i)})) for i in range(3)]
    await asyncio.sleep(0.2)
    refused = await client.post("/v0/text/embed", json={"text": "late"})
    assert refused.status_code == 503 and refused.json()["error"]["code"] == "busy"
    assert refused.headers["retry-after"] == "5"
    release.set()
    results = await asyncio.gather(*tasks)
    assert all(r.status_code == 200 for r in results)
