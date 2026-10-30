"""Search queries go to a model worker of their own; everything else stays with the batch worker."""

import pytest

from conftest import FakeModelWorker
from ems.app import make_card_worker, make_workers
from ems.config import Settings
from ems.worker import Worker


class Up(Worker):
    """A worker whose process the test does not start, reported as running."""

    @property
    def running(self) -> bool:
        return True


@pytest.fixture
async def query_fake(tmp_path, client):
    fake = FakeModelWorker(str(tmp_path / "query.sock"))
    await fake.start()
    client.app.state.ctx.query_worker = Up("/nonexistent/model_server.py", fake.socket_path, log=lambda m: None)
    yield fake
    await fake.stop()


def test_the_query_worker_holds_the_query_families_on_the_cpu_and_the_batch_worker_the_rest():
    env = {"EMS_MODEL_PINNED": "siglip-text,text-query,whisper", "EMS_WORKER_NICE": "7", "EMS_QUERY_WORKER_THREADS": "3"}
    batch, query = make_workers(Settings.from_env(env), env)
    assert batch.env["CFG_MODEL_PINNED"] == "whisper"
    assert batch.env["CFG_MODEL_NICE"] == "7"
    assert query is not None
    assert query.env["CFG_MODEL_PINNED"] == "text-query,siglip-text"
    # No GPU at all: a CUDA context alone would take card memory for models that never use it.
    assert query.env["CUDA_VISIBLE_DEVICES"] == ""
    assert query.env["CFG_TEXT_EMBED_QUERY_DEVICE"] == "cpu" and query.env["CFG_IMAGE_EMBED_TEXT_DEVICE"] == "cpu"
    assert query.env["CFG_MODEL_ROLE"] == "query" and query.env["CFG_MODEL_MAX_RSS"] == "6g"
    assert query.socket_path != batch.socket_path and "CFG_MODEL_NICE" not in query.env
    # A few threads, not every core: a short query waits on the slowest of them.
    assert [query.env[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CFG_MODEL_THREADS")] == ["3", "3", "3"]
    assert "OMP_NUM_THREADS" not in batch.env


def test_the_card_worker_holds_the_query_families_on_the_card():
    env = {"EMS_QUERY_WORKER_THREADS": "3", "EMS_CARD_WORKER_MAX_RSS": "5g"}
    card = make_card_worker(Settings.from_env(env), env)
    assert "CUDA_VISIBLE_DEVICES" not in card.env
    assert card.env["CFG_TEXT_EMBED_QUERY_DEVICE"] == "cuda" and card.env["CFG_IMAGE_EMBED_TEXT_DEVICE"] == "cuda"
    assert card.env["CFG_MODEL_ROLE"] == "card" and card.env["CFG_MODEL_MAX_RSS"] == "5g"
    # Which families it pins is the arbiter's choice at each start.
    assert card.env["CFG_MODEL_PINNED"] == ""
    assert card.socket_path == "/tmp/ems-card-worker.sock"


def test_without_a_query_worker_the_batch_worker_keeps_everything():
    env = {"EMS_MODEL_PINNED": "siglip-text,text-query", "EMS_QUERY_WORKER": "0"}
    batch, query = make_workers(Settings.from_env(env), env)
    assert query is None
    assert batch.env["CFG_MODEL_PINNED"] == "siglip-text,text-query" and "CFG_MODEL_NICE" not in batch.env


async def test_a_search_query_goes_to_the_query_worker_and_documents_to_the_batch_one(client, fake_worker, query_fake):
    query_fake.on("embed_texts", lambda req: {"embeddings": [[1.0]], "model": "jina-embeddings-v5-text-small"})
    query_fake.on("embed_image_texts", lambda req: {"embeddings": [[1.0]], "model": "siglip2"})
    fake_worker.on("embed_texts", lambda req: {"embeddings": [[2.0]], "model": "jina-embeddings-v5-text-small"})

    r = await client.post("/v0/text/embed", json={"text": "segelbåt", "role": "query"})
    assert r.json()["result"]["embedding"] == [1.0]
    r = await client.post("/v0/vision/embed_text", json={"text": "segelbåt"})
    assert r.status_code == 200, r.json()
    r = await client.post("/v0/text/embed", json={"texts": ["en segelbåt på havet"]})
    assert r.json()["result"]["embeddings"] == [[2.0]]

    assert [request["op"] for request in query_fake.requests] == ["embed_texts", "embed_image_texts"]
    assert [request["role"] for request in fake_worker.requests] == ["document"]


async def test_an_interactive_document_goes_to_the_query_worker_as_a_document(client, fake_worker, query_fake):
    query_fake.on("embed_texts", lambda req: {"embeddings": [[1.0]], "model": "jina-embeddings-v5-text-small"})

    r = await client.post("/v0/text/embed", json={"texts": ["Kära Anna"], "options": {"priority": "interactive"}})
    assert r.json()["result"]["embeddings"] == [[1.0]]

    assert [(request["role"], request["family"]) for request in query_fake.requests] == [("document", "query")]
    assert fake_worker.requests == []


async def test_with_the_query_worker_down_a_search_falls_back_to_the_batch_worker(client, fake_worker, tmp_path):
    client.app.state.ctx.query_worker = Worker("/nonexistent/model_server.py", str(tmp_path / "down.sock"), log=lambda m: None)
    fake_worker.on("embed_texts", lambda req: {"embeddings": [[3.0]], "model": "m"})
    r = await client.post("/v0/text/embed", json={"text": "segelbåt", "role": "query"})
    assert r.json()["result"]["embedding"] == [3.0]


async def test_health_lists_both_workers_models(client, fake_worker, query_fake):
    fake_worker.on("health", lambda req: {"loaded": ["whisper:cuda"]})
    query_fake.on("health", lambda req: {"loaded": ["text-query:cpu", "siglip-text:cpu"]})
    body = (await client.get("/v0/health")).json()
    assert body["models"] == {"whisper": {"loaded": True, "device": "cuda"}, "text-query": {"loaded": True, "device": "cpu"}, "siglip-text": {"loaded": True, "device": "cpu"}}
    assert body["worker"]["query"] == {"running": True, "restarts": 0}
