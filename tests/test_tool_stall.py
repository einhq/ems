"""`tools/run.py`: a tool is stopped for going quiet, never for taking long."""

import sys
import time

import pytest

from ems.tools.run import run, run_streaming


async def test_a_quiet_tool_is_stopped_and_says_why():
    started = time.monotonic()
    done = await run(["sleep", "30"], timeout=0.3)
    assert time.monotonic() - started < 5
    assert done.code != 0
    assert "made no progress for 0.3s" in done.stderr_text


async def test_a_busy_tool_runs_past_the_stall_limit():
    busy = "import time\nt = time.time()\nwhile time.time() - t < 1.2: pass\nprint('done')"
    done = await run([sys.executable, "-c", busy], timeout=0.3)
    assert done.code == 0 and done.stdout.strip() == b"done"


async def test_a_streaming_tool_that_writes_is_making_progress():
    # Sleeping between lines: no CPU and no bytes read, but each line is progress.
    talk = "import time\nfor i in range(6):\n    print(i, flush=True)\n    time.sleep(0.2)"
    lines = []

    async def on_line(line):
        lines.append(line)

    done = await run_streaming([sys.executable, "-c", talk], timeout=0.5, on_line=on_line)
    assert done.code == 0 and lines == [str(i) for i in range(6)]


async def test_a_deadline_is_a_wall_clock():
    import asyncio

    with pytest.raises(asyncio.TimeoutError):
        await run(["sleep", "30"], deadline=0.2)
