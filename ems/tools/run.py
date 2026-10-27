"""Spawning a tool: niced, stopped when it stalls, and killed when the caller is gone.

Every tool runs under `nice -n 10` so the HTTP process and the model worker keep their scheduler
priority while a transcode grinds. Stopping a tool kills the process group rather than the
process alone: ffmpeg and ImageMagick fork helpers, and an orphaned helper holding a GPU context
is exactly the leak a killed request must not leave behind.

How long a tool may take is not bounded. A tape's audio extraction or a CPU encode under load
runs for as long as it runs, and a clock that cut it short only made the caller retry the same
work from the start (ein/core#506). What is bounded is silence: `timeout` is how long
the tool's process group may go without using CPU or moving a byte before it counts as wedged
and is stopped. A stopped tool is a failed exit like any other — the caller reads `code` and
may fall back (the CPU after a GPU decode) or answer `tool_failed`, which is retryable.

`deadline` is a wall clock, for the few callers that want one: a person is waiting (health's
`nvidia-smi`), or the time is itself the judgement (OCR gives up on a photo that takes long).
It raises `asyncio.TimeoutError`, as `asyncio.wait_for` does.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import time
from dataclasses import dataclass

from ..errors import EmsError, TOOL_FAILED

NICE = 10
# For a tool whose CPU should only be what the model worker, itself at nice 10, leaves over.
BACKGROUND_NICE = 19

# How often a running tool's activity is read. A stall is noticed within this of its limit.
ACTIVITY_POLL_SECONDS = 10.0


@dataclass
class Completed:
    args: list[str]
    code: int
    stdout: bytes
    stderr: bytes

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode(errors="replace")

    def tail(self, lines: int = 3, limit: int = 400) -> str:
        text = self.stderr_text.strip().splitlines()
        return "\n".join(text[-lines:])[-limit:]


def which(name: str) -> str | None:
    return shutil.which(name)


async def run(args: list[str], *, timeout: float | None = None, deadline: float | None = None, stdin: bytes | None = None, cwd: str | None = None, env: dict | None = None, niceness: int = NICE) -> Completed:
    """Run to completion. Never raises for a non-zero exit — the caller reads `code`.

    `timeout` is the longest the tool may go without progress (see the module's docstring);
    a tool stopped for it comes back as a failed exit whose stderr says so. `niceness` is the
    `nice -n` the tool runs under.
    """
    nice = which("nice")
    argv = [nice, "-n", str(niceness), *args] if nice else list(args)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    watch = _Watch(proc, timeout)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(stdin), deadline)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        _kill_group(proc)
        await proc.wait()
        raise
    finally:
        watch.stop()
    return Completed(list(args), proc.returncode or 0, stdout, stderr + watch.verdict(args))


class _Watch:
    """Stops `proc`'s process group once it has gone `stall` seconds without progress.

    Progress is anything the group does: CPU time (its own and its reaped children's) or bytes
    read or written, summed over every process in the group from /proc. A decoder waiting on a
    slow disk is reading; an encoder is computing; a tool blocked on a dead mount or a wedged
    driver does neither. Without /proc (a developer's Mac) nothing is ever stopped.
    """

    def __init__(self, proc: asyncio.subprocess.Process, stall: float | None):
        self.proc = proc
        self.stall = stall
        self.stalled = False
        self.quiet_since = time.monotonic()
        self.task = asyncio.ensure_future(self._run()) if stall else None

    def touch(self) -> None:
        """Progress the group's counters cannot see, such as a line the tool wrote."""
        self.quiet_since = time.monotonic()

    async def _run(self) -> None:
        assert self.stall
        poll = min(ACTIVITY_POLL_SECONDS, max(0.05, self.stall / 4))
        last = group_activity(self.proc.pid)
        while self.proc.returncode is None:
            await asyncio.sleep(poll)
            now = group_activity(self.proc.pid)
            if now is None or now != last:
                last = now
                self.quiet_since = time.monotonic()
            elif time.monotonic() - self.quiet_since >= self.stall:
                self.stalled = True
                _kill_group(self.proc)
                return

    def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()

    def verdict(self, args: list[str]) -> bytes:
        if not self.stalled:
            return b""
        tool = os.path.basename(args[0]) if args else "tool"
        return f"\n{tool} made no progress for {self.stall:g}s and was stopped\n".encode()


def group_activity(pgid: int) -> tuple[int, int] | None:
    """CPU ticks and bytes moved by every process in group `pgid`, or None when /proc has none."""
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    cpu = moved = 0
    seen = False
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as handle:
                stat = handle.read().decode(errors="replace")
            # The command name may hold spaces and parentheses; the fields after it do not.
            fields = stat[stat.rfind(")") + 2:].split()
            if int(fields[2]) != pgid:  # pgrp
                continue
            cpu += sum(int(value) for value in fields[11:15])  # utime, stime, cutime, cstime
        except (OSError, ValueError, IndexError):
            continue
        seen = True
        try:
            with open(f"/proc/{entry}/io", "rb") as handle:
                for line in handle:
                    key, _, value = line.partition(b":")
                    if key in (b"rchar", b"wchar"):
                        moved += int(value)
        except (OSError, ValueError):
            pass
    return (cpu, moved) if seen else None


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        proc.kill()


def require(name: str) -> str:
    path = which(name)
    if path is None:
        raise EmsError(TOOL_FAILED, f"{name} is not installed in this EMS", permanent=True)
    return path


async def run_streaming(args: list[str], *, timeout: float | None, on_line, env: dict | None = None) -> Completed:
    """Like `run`, but each line the tool writes to stdout reaches `on_line` as it appears.

    For ffmpeg's `-progress pipe:1`, which writes `key=value` lines while it works; a line is
    progress too. stderr is still collected whole for the failure message.
    """
    nice = which("nice")
    argv = [nice, "-n", str(NICE), *args] if nice else list(args)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        start_new_session=True,
    )

    watch = _Watch(proc, timeout)

    async def pump_stdout() -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            watch.touch()
            try:
                await on_line(line.decode(errors="replace").rstrip("\n"))
            except Exception:  # noqa: BLE001 - a progress consumer must never kill the tool
                pass

    async def drain_stderr() -> bytes:
        assert proc.stderr is not None
        return await proc.stderr.read()

    try:
        _, stderr, _ = await asyncio.gather(pump_stdout(), drain_stderr(), proc.wait())
    except asyncio.CancelledError:
        _kill_group(proc)
        await proc.wait()
        raise
    finally:
        watch.stop()
    return Completed(list(args), proc.returncode or 0, b"", stderr + watch.verdict(args))
