"""The caller's files, when EMS does not mount them: an sshfs mount of the caller's endpoint.

A request may carry an `EMS-Files` header naming an SFTP endpoint of its caller's:
`address=host:port; user=…; secret=…; hostkey=ssh-ed25519 AAAA…`. Every path in that request is
the caller's: EMS has no files of its own. It mounts the endpoint read-write with sshfs, once per
endpoint, and the op runs on local paths for the request's, which `input`, `output` and
`output_dir` give it:

- An input (`file.path`) is read through the mount where it lies, so a tool seeks and reads only
  what it needs while it runs, and the page cache keeps what was read for the next op until the
  file's size or mtime changes.
- An output (`output.path`, `output.dir`) gets a place in the request's scratch, and what the op
  wrote is copied into the mount before the result goes out. A copy through sshfs has its bytes at
  the caller once it returns.

Without a mount — no /dev/fuse or sshfs here, an endpoint that cannot be reached, a mount that
fails during the request — the request is refused as `files_unavailable`, which the caller tries
again later. The caller grants a request exactly its own paths for as long as it runs, so nothing
is listed and nothing else is asked for. The connection is SSH with chacha20-poly1305 alone and the
host key from the header pinned.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import posixpath
import shutil
import signal
import sys
import tempfile
import time
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from .admission import current_op
from .errors import EmsError, FILES_UNAVAILABLE, NOT_FOUND, PATH_NOT_GRANTED, invalid

HEADER = "ems-files"
CIPHER = "chacha20-poly1305@openssh.com"


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    user: str
    secret: str
    hostkey: str

    @property
    def name(self) -> str:
        return f"{self.host}:{self.port}"


def parse_header(value: str) -> Endpoint:
    """`EMS-Files`'s value as an endpoint; an invalid request when anything is missing."""
    fields: dict[str, str] = {}
    for part in value.split(";"):
        key, sep, item = part.strip().partition("=")
        if sep:
            fields[key.strip()] = item.strip()
    address = fields.get("address", "")
    host, sep, port = address.rpartition(":")
    if not (sep and host and port.isdigit() and fields.get("user") and fields.get("secret") and fields.get("hostkey")):
        raise invalid(f"{HEADER}: needs address=host:port, user, secret and hostkey")
    return Endpoint(host.strip("[]"), int(port), fields["user"], fields["secret"], fields["hostkey"])


def request_paths(body: Any) -> tuple[list[str], list[str], list[str]]:
    """Every path in a request body, at any depth: inputs (`file.path`), outputs (`output.path`)
    and output directories (`output.dir`)."""
    inputs: list[str] = []
    outputs: list[str] = []
    dirs: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
            return
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            if isinstance(child, dict):
                if key == "file" and isinstance(child.get("path"), str):
                    inputs.append(child["path"])
                if key == "output" and isinstance(child.get("path"), str):
                    outputs.append(child["path"])
                if key == "output" and isinstance(child.get("dir"), str):
                    dirs.append(child["dir"])
            walk(child)

    walk(body)
    return inputs, outputs, dirs


def _log(message: str) -> None:
    """One line on the service's log, for the request the running task serves."""
    print(f"ems: {current_op.get() or 'request'}: {message}", file=sys.stderr, flush=True)


def _caller_path(path: str) -> str:
    if not path.startswith("/"):
        raise EmsError(PATH_NOT_GRANTED, f"{path}: not an absolute path")
    return posixpath.normpath(path)


async def _unmount(point: str) -> None:
    """Take a mount off `point`, one whose sshfs died too, which stays in the table until then."""
    unmount = await asyncio.create_subprocess_exec("fusermount3", "-uz", point, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await unmount.wait()


class Mounts:
    """sshfs mounts of callers' endpoints, one per endpoint, made on first use.

    A mount whose sshfs has exited, its connection gone, is made again; one of an older endpoint
    at the same address (the caller restarted, with a new key and secret) is let go.
    """

    def __init__(self, root: str | None = None):
        # Not under the scratch, which is swept: a sweep must never walk into a mount.
        self.root = root or os.path.join(tempfile.gettempdir(), "ems-mounts")
        self.possible = os.path.exists("/dev/fuse") and shutil.which("sshfs") is not None
        self._held: dict[Endpoint, tuple[str, asyncio.subprocess.Process]] = {}
        self._locks: dict[Endpoint, asyncio.Lock] = {}

    def start(self) -> None:
        if self.possible:
            os.makedirs(self.root, exist_ok=True)

    async def mount(self, endpoint: Endpoint) -> str:
        """Where `endpoint`'s `/` is mounted; `files_unavailable` when it cannot be."""
        if not self.possible:
            raise EmsError(FILES_UNAVAILABLE, f"{endpoint.name}: this host cannot mount (no /dev/fuse or sshfs)")
        held = self._held.get(endpoint)
        if held is not None and held[1].returncode is None:
            return held[0]
        async with self._locks.setdefault(endpoint, asyncio.Lock()):
            held = self._held.get(endpoint)
            if held is not None and held[1].returncode is None:
                return held[0]
            if held is not None:
                await self._let_go(endpoint)
            for other in [e for e in self._held if e.name == endpoint.name]:
                await self._let_go(other)
            point = os.path.join(self.root, hashlib.sha256(repr(endpoint).encode()).hexdigest()[:16])
            await _unmount(point)
            os.makedirs(point, exist_ok=True)
            known = os.path.join(self.root, os.path.basename(point) + ".known_hosts")
            with open(known, "w") as f:
                host = endpoint.host if endpoint.port == 22 else f"[{endpoint.host}]:{endpoint.port}"
                f.write(f"{host} {endpoint.hostkey}\n")
            process = await asyncio.create_subprocess_exec(
                "sshfs", f"{endpoint.user}@{endpoint.host}:/", point, "-f",
                # Attributes are asked for at every open, so a file that changed is never read from the
                # page cache (auto_cache keeps it while size and mtime stay the same).
                "-o", f"port={endpoint.port},password_stdin,rw,cache=no,auto_cache,attr_timeout=0,entry_timeout=0,negative_timeout=0",
                "-o", f"UserKnownHostsFile={known},StrictHostKeyChecking=yes,HostKeyAlgorithms={endpoint.hostkey.split()[0]}",
                "-o", f"Ciphers={CIPHER},PreferredAuthentications=password,PubkeyAuthentication=no",
                "-o", "ServerAliveInterval=15,ServerAliveCountMax=3",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            process.stdin.write(endpoint.secret.encode() + b"\n")
            await process.stdin.drain()
            process.stdin.close()
            # A look at a mount waits on the caller's answer, so it is made off the event loop.
            deadline = time.monotonic() + 20
            while not await asyncio.to_thread(os.path.ismount, point):
                if process.returncode is not None or time.monotonic() > deadline:
                    if process.returncode is None:
                        process.kill()
                    error = (await process.stderr.read()).decode(errors="replace").strip()
                    await process.wait()
                    await _unmount(point)
                    raise EmsError(FILES_UNAVAILABLE, f"{endpoint.name}: sshfs: {error or 'no mount'}")
                await asyncio.sleep(0.05)
            self._held[endpoint] = (point, process)
            _log(f"mounted {endpoint.name}")
            return point

    async def drop(self, endpoint: Endpoint) -> None:
        """Let go of a mount that failed a request; the next one mounts again."""
        async with self._locks.setdefault(endpoint, asyncio.Lock()):
            if endpoint in self._held:
                await self._let_go(endpoint)

    async def _let_go(self, endpoint: Endpoint) -> None:
        point, process = self._held.pop(endpoint)
        if process.returncode is None:
            process.send_signal(signal.SIGTERM)
            await process.wait()
        await _unmount(point)
        with suppress(OSError):
            os.rmdir(point)

    async def stop(self) -> None:
        for endpoint in list(self._held):
            await self._let_go(endpoint)


class Staging:
    """One request's paths: its inputs in the mount and its outputs' local places, while it runs."""

    def __init__(self, mounts: Mounts, endpoint: Endpoint, scratch: str):
        self.mounts = mounts
        self.endpoint = endpoint
        self.scratch = scratch
        self.point = ""
        self.inputs: dict[str, str] = {}
        self.outputs: dict[str, str] = {}
        self.dirs: dict[str, str] = {}

    async def prepare(self, body: Any) -> None:
        inputs, outputs, dirs = request_paths(body)
        for index, path in enumerate(dict.fromkeys(_caller_path(p) for p in outputs)):
            local_dir = os.path.join(self.scratch, "out", str(index))
            os.makedirs(local_dir)
            self.outputs[path] = os.path.join(local_dir, posixpath.basename(path))
        for index, path in enumerate(dict.fromkeys(_caller_path(p) for p in dirs)):
            local_dir = os.path.join(self.scratch, "dir", str(index))
            os.makedirs(local_dir)
            self.dirs[path] = local_dir
        paths = list(dict.fromkeys(_caller_path(p) for p in inputs))
        if not (paths or self.outputs or self.dirs):
            return
        started = time.monotonic()
        for attempt in (0, 1):
            self.point = await self.mounts.mount(self.endpoint)
            try:
                for path in paths:
                    await self._look(path)
                    self.inputs[path] = self.point + path
                break
            except EmsError as exc:
                if exc.code is not FILES_UNAVAILABLE or attempt:
                    raise
                await self.mounts.drop(self.endpoint)
        if paths:
            _log(f"{len(paths)} input(s) through the mount of {self.endpoint.name}, in {time.monotonic() - started:.2f} s")

    async def _look(self, path: str) -> None:
        """That an input is there to read, as the caller grants it."""
        try:
            await asyncio.to_thread(os.stat, self.point + path)
        except FileNotFoundError as exc:
            raise EmsError(NOT_FOUND, f"{path}: no such file") from exc
        except PermissionError as exc:
            raise EmsError(PATH_NOT_GRANTED, f"{path}: not granted to this request") from exc
        except OSError as exc:
            raise EmsError(FILES_UNAVAILABLE, f"{path}: {exc}") from exc

    def input(self, path: str) -> str:
        local = self.inputs.get(_caller_path(path))
        if local is None:
            raise EmsError(PATH_NOT_GRANTED, f"{path}: not an input of this request")
        return local

    def output(self, path: str) -> str:
        local = self.outputs.get(_caller_path(path))
        if local is None:
            raise EmsError(PATH_NOT_GRANTED, f"{path}: not an output of this request")
        return local

    def output_dir(self, path: str) -> str:
        local = self.dirs.get(_caller_path(path))
        if local is None:
            raise EmsError(PATH_NOT_GRANTED, f"{path}: not an output directory of this request")
        return local

    async def send_back(self) -> None:
        """Copy what the op made into the mount: each output it wrote, and every file of an output
        directory, which is made there first."""
        sends: list[tuple[str, str]] = [(local, remote) for remote, local in self.outputs.items() if os.path.isfile(local)]
        made = [remote_dir for remote_dir in self.dirs]
        for remote_dir, local_dir in self.dirs.items():
            for name in sorted(os.listdir(local_dir)):
                local = os.path.join(local_dir, name)
                if not name.startswith(".") and os.path.isfile(local):
                    sends.append((local, posixpath.join(remote_dir, name)))
        if not sends:
            return
        started = time.monotonic()
        try:
            for remote_dir in made:
                await asyncio.to_thread(os.makedirs, self.point + remote_dir, exist_ok=True)
            for local, remote in sends:
                await asyncio.to_thread(shutil.copyfile, local, self.point + remote)
        except PermissionError as exc:
            raise EmsError(PATH_NOT_GRANTED, f"{exc.filename}: not granted to this request") from exc
        except OSError as exc:
            raise EmsError(FILES_UNAVAILABLE, f"{self.endpoint.name}: sending an output back: {exc}") from exc
        size = sum(os.path.getsize(local) for local, _remote in sends)
        _log(f"{len(sends)} output(s) to {self.endpoint.name}: {size / 1e6:.1f} MB sent back in {time.monotonic() - started:.2f} s")

    def caller_paths(self, value: Any) -> Any:
        """`value` with every local path of this request's in its caller's spelling."""
        if isinstance(value, str):
            for remote, local in self.outputs.items():
                if value == local:
                    return remote
            for remote, local in self.dirs.items():
                if value == local:
                    return remote
                if value.startswith(local + os.sep):
                    return posixpath.join(remote, value[len(local) + 1 :])
            if self.point and value.startswith(self.point + "/"):
                return value[len(self.point) :]
            return value
        if isinstance(value, list):
            return [self.caller_paths(item) for item in value]
        if isinstance(value, dict):
            return {key: self.caller_paths(item) for key, item in value.items()}
        return value

    def close(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)


# The staging of the request the running task serves.
current: ContextVar[Staging | None] = ContextVar("ems_files_staging", default=None)


def _staging(path: str) -> Staging:
    staged = current.get()
    if not isinstance(path, str) or not path:
        raise invalid("a file path is required")
    if staged is None:
        raise invalid(f"{path}: a request that names files names their endpoint in {HEADER}")
    return staged


def input(path: str) -> str:  # noqa: A001 - the op's word for it
    """The local path an op reads an input of the running request's at."""
    return _staging(path).input(path)


def output(path: str) -> str:
    """The local path an op writes an output of the running request's to; its directory exists."""
    return _staging(path).output(path)


def output_dir(path: str) -> str:
    """The local directory an op writes an output directory's files into; it exists."""
    return _staging(path).output_dir(path)
