"""A model worker busy loading a model does not hold up `GET /v0/health`."""

import asyncio
import time

from ems.ops import system
from ems.worker import Worker


class Up(Worker):
    """A worker whose process the test does not start, reported as running."""

    @property
    def running(self) -> bool:
        return True


async def test_a_worker_that_does_not_answer_is_reported_up_and_silent_by_what_it_last_said(client, fake_worker, monkeypatch):
    monkeypatch.setattr(system, "HEALTH_WAIT_SECONDS", 0.2)
    worker = Up("/nonexistent/model_server.py", fake_worker.socket_path, log=lambda m: None)
    client.app.state.ctx.worker = worker
    fake_worker.on("health", lambda req: {"loaded": ["htr:cpu"], "rss_mb": 900})
    await client.get("/v0/health")

    async def loading(req):
        await asyncio.sleep(1.5)
        return {"loaded": []}

    fake_worker.on("health", loading)
    started = time.monotonic()
    body = (await client.get("/v0/health")).json()
    assert time.monotonic() - started < 1
    assert body["worker"]["running"] is True and body["worker"]["answering"] is False
    assert body["worker"]["silent_for"] >= 0
    assert body["models"] == {"htr": {"loaded": True, "device": "cpu"}}
    assert body["process"]["rss_mb"] == 900


async def test_a_worker_that_is_down_is_not_running(client, fake_worker):
    await fake_worker.stop()
    body = (await client.get("/v0/health")).json()
    assert body["worker"] == {"running": False, "restarts": 0}
