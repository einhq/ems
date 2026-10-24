"""What `health` says about each model family and each request in flight, for a status page."""

import asyncio
import threading

from ems.admission import Admission, current_op
from ems.app import NameTheOp
from ems.modelworker import model_registry as registry
from ems.ops.system import _families_from_workers


def _fresh_registry(monkeypatch):
    for name in ("_cache", "_last_used", "_loading", "_unloading", "_loaded_at", "_devices", "_sizes", "_family_locks"):
        monkeypatch.setattr(registry, name, {})
    monkeypatch.setattr(registry, "PINNED_FAMILIES", {"text-query"})
    monkeypatch.setattr(registry, "cleanup_torch", lambda: None)


def test_a_family_nothing_has_loaded_is_listed_with_no_copies(monkeypatch):
    _fresh_registry(monkeypatch)
    families = registry.family_states()
    assert families["whisper"]["copies"] == [] and families["whisper"]["model"] == registry.WHISPER_MODEL
    assert families["text-query"]["pinned"] is True and families["whisper"]["pinned"] is False
    assert "vlm" not in families


def test_a_load_in_progress_is_loading_then_loaded(monkeypatch):
    _fresh_registry(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def loader():
        entered.set()
        release.wait(2)
        return "model"

    load = threading.Thread(target=lambda: registry._get("whisper:cuda", loader))
    load.start()
    assert entered.wait(2)
    [copy] = registry.family_states()["whisper"]["copies"]
    assert copy["state"] == "loading" and copy["device"] == "cuda"

    release.set()
    load.join(2)
    [copy] = registry.family_states()["whisper"]["copies"]
    assert copy["state"] == "loaded" and copy["since"] and copy["last_used"]


def test_a_failed_load_leaves_nothing_behind(monkeypatch):
    _fresh_registry(monkeypatch)

    def loader():
        raise RuntimeError("out of memory")

    try:
        registry._get("whisper:cuda:float16", loader)
    except RuntimeError:
        pass
    assert registry.family_states()["whisper"]["copies"] == []


def test_an_eviction_is_unloading_until_the_memory_is_back(monkeypatch):
    _fresh_registry(monkeypatch)
    registry._get("whisper:cuda", lambda: "model")
    during = []
    monkeypatch.setattr(registry, "cleanup_torch", lambda: during.append(registry.family_states()["whisper"]["copies"]))

    registry.evict("whisper:cuda")
    assert [copy["state"] for copy in during[0]] == ["unloading"]
    assert registry.family_states()["whisper"]["copies"] == []


def test_a_family_in_use_is_running_with_its_queue(monkeypatch):
    _fresh_registry(monkeypatch)
    lock = registry.family_lock("whisper")
    lock.acquire()
    waiter = threading.Thread(target=lambda: (lock.acquire(), lock.release()))
    waiter.start()
    for _ in range(100):
        if lock.snapshot()[1]:
            break
        threading.Event().wait(0.01)
    state = registry.family_states()["whisper"]
    assert state["running"] is True and state["waiting"] == 1
    lock.release()
    waiter.join(2)


def test_insightface_reports_the_device_its_loader_chose(monkeypatch):
    _fresh_registry(monkeypatch)

    def loader():
        registry._devices["insightface"] = "cpu"
        return "app"

    registry._get("insightface", loader)
    [copy] = registry.family_states()["insightface"]["copies"]
    assert copy["device"] == "cpu"


def test_both_workers_copies_are_listed_under_one_family():
    batch = {"families": {
        "whisper": {"model": "m", "device": "auto", "pinned": False, "running": True, "waiting": 2, "copies": [{"key": "whisper:cuda", "state": "loaded", "device": "cuda"}]},
        "text-query": {"model": "t", "device": "auto", "pinned": False, "copies": [{"key": "text-query:cuda", "state": "loading", "device": "cuda"}]},
    }}
    query = {"families": {
        "text-query": {"model": "t", "device": "cpu", "pinned": True, "copies": [{"key": "text-query:cpu", "state": "loaded", "device": "cpu"}]},
    }}
    families = _families_from_workers(batch, query)
    assert families["whisper"]["copies"][0]["worker"] == "batch" and families["whisper"]["waiting"] == 2
    assert [(c["worker"], c["state"]) for c in families["text-query"]["copies"]] == [("batch", "loading"), ("query", "loaded")]
    assert families["text-query"]["pinned"] is True and families["text-query"]["device"] == "cpu"
    assert _families_from_workers(None, None) == {}


async def test_requests_name_the_op_and_whether_it_runs_or_waits():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=4, interactive_reserve=0)
    held, release = asyncio.Event(), asyncio.Event()

    async def one(op: str):
        current_op.set(op)
        async with admission.slot("models"):
            held.set()
            await release.wait()

    first = asyncio.create_task(one("vision.embed"))
    await held.wait()
    second = asyncio.create_task(one("speech.transcribe"))
    await asyncio.sleep(0.05)

    running, waiting = admission.requests()
    assert running["op"] == "vision.embed" and running["lane"] == "models" and "started" in running
    assert waiting["op"] == "speech.transcribe" and "started" not in waiting

    release.set()
    await asyncio.gather(first, second)
    assert admission.requests() == []


async def test_a_refused_request_is_not_listed():
    admission = Admission(limit_models=1, limit_tools=1, queue_depth=0, interactive_reserve=0)
    release = asyncio.Event()

    async def hold():
        async with admission.slot("models"):
            await release.wait()

    task = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    try:
        async with admission.slot("models"):
            pass
    except Exception:  # noqa: BLE001 - busy, which is the point
        pass
    assert [request["op"] for request in admission.requests()] == ["?"]
    release.set()
    await task


async def test_the_middleware_names_the_op_from_the_path():
    seen = []

    async def app(scope, receive, send):
        seen.append(current_op.get())

    await NameTheOp(app)({"type": "http", "path": "/v0/vision/embed"}, None, None)
    assert seen == ["vision.embed"]


async def test_health_carries_families_requests_and_build(client, fake_worker):
    fake_worker.on("health", lambda req: {"loaded": ["whisper:cuda"], "families": {
        "whisper": {"model": "m", "device": "auto", "pinned": False, "copies": [{"key": "whisper:cuda", "state": "loaded", "device": "cuda"}]},
    }})
    body = (await client.get("/v0/health")).json()
    assert body["families"]["whisper"]["copies"][0]["worker"] == "batch"
    assert body["requests"] == []
    assert body["build"]["version"]


def test_every_family_says_what_it_is_for_and_which_ops_use_it(monkeypatch):
    _fresh_registry(monkeypatch)
    for name, family in registry.family_states().items():
        assert family["purpose"] and family["ops"], name


def test_a_load_is_measured_and_the_size_outlives_the_eviction(monkeypatch):
    _fresh_registry(monkeypatch)
    readings = iter([
        {"at": 10.0, "free_vram": 9000.0, "allocated": 100.0, "rss": 2000.0},
        {"at": 25.5, "free_vram": 6000.0, "allocated": 2900.0, "rss": 2600.0},
    ])
    monkeypatch.setattr(registry, "_memory_now", lambda: next(readings))
    registry._get("whisper:cuda:float16", lambda: "model")

    [copy] = registry.family_states()["whisper"]["copies"]
    assert (copy["vram_mb"], copy["ram_mb"], copy["load_seconds"]) == (3000, 600, 15.5)

    registry.evict("whisper:cuda:float16")
    whisper = registry.family_states()["whisper"]
    assert whisper["copies"] == [] and whisper["measured"]["whisper:cuda:float16"]["vram_mb"] == 3000


def test_a_load_that_imported_torch_falls_back_to_its_allocation():
    cost = registry._memory_cost(
        {"at": 0.0, "free_vram": None, "allocated": None, "rss": None},
        {"at": 1.0, "free_vram": 5000.0, "allocated": 1200.4, "rss": None},
    )
    assert cost == {"load_seconds": 1.0, "vram_mb": 1200}


async def test_a_request_names_the_family_it_is_using(client, fake_worker):
    from ems.admission import current_request
    from ems.worker import Worker

    fake_worker.on("embed_texts", lambda req: {"embeddings": [[1.0]], "model": "m"})
    entry = {"op": "text.embed"}
    token = current_request.set(entry)
    try:
        worker: Worker = client.app.state.ctx.worker
        await worker.call("embed_texts", {"texts": ["en segelbåt"], "role": "document"})
    finally:
        current_request.reset(token)
    assert entry["family"] == "text-embed"
