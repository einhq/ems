import asyncio
import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ems.app import create_app  # noqa: E402
from ems.config import Settings  # noqa: E402
from ems.worker import Worker  # noqa: E402

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class FakeModelWorker:
    """A stand-in for model_server.py: the same line protocol on a Unix socket, scripted answers."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.handlers: dict = {}
        self.requests: list[dict] = []
        self.server: asyncio.AbstractServer | None = None

    def on(self, op: str, handler):
        self.handlers[op] = handler

    async def start(self):
        self.server = await asyncio.start_unix_server(self._serve, path=self.socket_path)

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def _serve(self, reader, writer):
        line = await reader.readline()
        request = json.loads(line)
        self.requests.append(request)
        handler = self.handlers.get(request.get("op"))
        try:
            if handler is None:
                raise ValueError(f"unknown op: {request.get('op')}")
            result = handler(request)
            if asyncio.iscoroutine(result):
                result = await result
            reply = {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            reply = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        writer.write((json.dumps(reply) + "\n").encode())
        await writer.drain()
        writer.close()


@pytest.fixture
def roots(tmp_path):
    ro = tmp_path / "ro"
    rw = tmp_path / "rw"
    ro.mkdir()
    rw.mkdir()
    return ro, rw


@pytest.fixture
async def fake_worker(tmp_path):
    worker = FakeModelWorker(str(tmp_path / "worker.sock"))
    await worker.start()
    yield worker
    await worker.stop()


@pytest.fixture
async def client(roots, fake_worker, tmp_path):
    ro, rw = roots
    settings = Settings.from_env({
        "EMS_KEY": KEY,
        "EMS_ROOTS": f"{ro}:ro,{rw}:rw",
        "EMS_SCRATCH": str(tmp_path / "scratch"),
        "EMS_WORKER_SOCKET": fake_worker.socket_path,
        "EMS_LIMIT_MODELS": "2",
        "EMS_LIMIT_TOOLS": "2",
        "EMS_QUEUE_DEPTH": "1",
    })
    worker = Worker("/nonexistent/model_server.py", fake_worker.socket_path, log=lambda m: None)
    app = create_app(settings, worker=worker, start_worker=False)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ems", headers=AUTH) as c:
            c.app = app
            yield c
