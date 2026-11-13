"""A worker started again before its process has spawned is still one worker, which spawns with the
environment it has by then; one stopped starts again (ein/core#573)."""

import asyncio
import time

from ems.worker import Worker


async def _spawned(path, count: int) -> list[str]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        lines = path.read_text().splitlines() if path.exists() else []
        if len(lines) >= count:
            return lines
        await asyncio.sleep(0.05)
    return path.read_text().splitlines() if path.exists() else []


async def test_a_second_start_before_the_spawn_starts_no_second_worker(tmp_path):
    out = tmp_path / "spawned"
    script = tmp_path / "server.py"
    script.write_text("import os, time\nopen(os.environ['OUT'], 'a').write(os.environ.get('CFG_MODEL_PINNED', '') + '\\n')\ntime.sleep(30)\n")
    worker = Worker(str(script), str(tmp_path / "worker.sock"), env={"OUT": str(out)}, log=lambda message: None)
    worker.env["CFG_MODEL_PINNED"] = ""
    await worker.start()
    worker.env["CFG_MODEL_PINNED"] = "text-query"
    await worker.start()
    try:
        assert await _spawned(out, 1) == ["text-query"]
        await asyncio.sleep(0.3)
        assert out.read_text().splitlines() == ["text-query"]
        await worker.stop()
        await worker.start()
        assert await _spawned(out, 2) == ["text-query", "text-query"]
    finally:
        await worker.stop()
