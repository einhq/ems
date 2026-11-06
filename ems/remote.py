"""The caller's files, when EMS does not mount them: fetched for a request, sent back before it answers.

A request may carry an `EMS-Files` header naming an SFTP endpoint of its caller's:
`address=host:port; user=…; secret=…; hostkey=ssh-ed25519 AAAA…`. Every path in that request is
then the caller's, not a local one. Before the op runs, each input (`file.path`) is copied into a
local cache and each output (`output.path`, `output.dir`) given a place in the request's scratch;
the op runs on those local paths as it would on mounted ones (`Roots` asks `current` first); and
once it succeeds, what it wrote goes back to the caller's paths before the result does. The caller
grants a request exactly its own paths for as long as it runs, so nothing is listed and nothing
else is asked for.

Inputs stay in the cache under their caller path, checked against its size and mtime, so the next
op on the same file — a video gets a dozen — does not fetch it again. A copy in use is never
evicted. The connection is SSH with chacha20-poly1305 alone and the host key from the header
pinned, one per endpoint, opened on first use and again after it drops.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import posixpath
import shutil
import time
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import asyncssh

from .errors import EmsError, FILES_UNAVAILABLE, NOT_FOUND, PATH_OUTSIDE_ROOTS, invalid

HEADER = "ems-files"
CIPHER = "chacha20-poly1305@openssh.com"

# An SFTP read or write of this size answers in one packet from EiN, and this many in flight keep
# a gigabit link full without a deep queue on the caller's side.
BLOCK_SIZE = 128 * 1024
MAX_REQUESTS = 32


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


def _caller_path(path: str) -> str:
    if not path.startswith("/"):
        raise EmsError(PATH_OUTSIDE_ROOTS, f"{path}: not an absolute path")
    return posixpath.normpath(path)


def _failure(path: str, exc: BaseException) -> EmsError:
    """What an SFTP failure on the caller's `path` is to EMS's caller."""
    if isinstance(exc, EmsError):
        return exc
    if isinstance(exc, asyncssh.SFTPNoSuchFile):
        return EmsError(NOT_FOUND, f"{path}: no such file")
    if isinstance(exc, asyncssh.SFTPPermissionDenied):
        return EmsError(PATH_OUTSIDE_ROOTS, f"{path}: not granted to this request")
    return EmsError(FILES_UNAVAILABLE, f"{path}: {type(exc).__name__}: {exc}")


class Connections:
    """One SSH connection and SFTP session per endpoint, opened on first use and again after it drops."""

    def __init__(self) -> None:
        self._open: dict[Endpoint, tuple[Any, Any]] = {}
        self._locks: dict[Endpoint, asyncio.Lock] = {}

    async def sftp(self, endpoint: Endpoint):
        held = self._open.get(endpoint)
        if held is not None and not held[0].is_closed():
            return held[1]
        async with self._locks.setdefault(endpoint, asyncio.Lock()):
            held = self._open.get(endpoint)
            if held is not None and not held[0].is_closed():
                return held[1]
            try:
                key = asyncssh.import_public_key(endpoint.hostkey)
                conn = await asyncssh.connect(
                    endpoint.host,
                    endpoint.port,
                    username=endpoint.user,
                    password=endpoint.secret,
                    known_hosts=([key], [], []),
                    server_host_key_algs=[key.get_algorithm()],
                    encryption_algs=[CIPHER],
                    compression_algs=None,
                    client_keys=None,
                    agent_path=None,
                    preferred_auth="password",
                    keepalive_interval=30,
                )
                sftp = await conn.start_sftp_client()
            except (OSError, asyncssh.Error, ValueError) as exc:
                raise EmsError(FILES_UNAVAILABLE, f"{endpoint.name}: {type(exc).__name__}: {exc}") from exc
            self._open[endpoint] = (conn, sftp)
            return sftp

    def drop(self, endpoint: Endpoint) -> None:
        held = self._open.pop(endpoint, None)
        if held is not None:
            held[0].close()

    async def close(self) -> None:
        for conn, _sftp in self._open.values():
            conn.close()
        for conn, _sftp in self._open.values():
            await conn.wait_closed()
        self._open.clear()


@dataclass
class _Copy:
    local: str
    size: int
    mtime: int
    used: float
    pins: int = 0


@dataclass
class InputCache:
    """Local copies of the callers' inputs, by caller path, size and mtime, within a byte budget."""

    root: str
    budget_bytes: int
    ttl_seconds: float
    _copies: dict[str, _Copy] = field(default_factory=dict)
    _fetching: dict[str, asyncio.Future] = field(default_factory=dict)

    def start(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(self.root, exist_ok=True)

    async def fetch(self, sftp, endpoint: Endpoint, path: str) -> _Copy:
        """The local copy of `path`, pinned until `release`; fetched unless the one held is current."""
        attrs = await sftp.stat(path)
        size, mtime = int(attrs.size or 0), int(attrs.mtime or 0)
        digest = hashlib.sha256(f"{endpoint.name}\0{path}".encode()).hexdigest()[:32]
        name = f"{digest}-{size}-{mtime}"
        while True:
            copy = self._copies.get(name)
            if copy is not None and os.path.exists(copy.local):
                copy.pins += 1
                copy.used = time.monotonic()
                return copy
            pending = self._fetching.get(name)
            if pending is None:
                break
            await asyncio.shield(pending)
        future = asyncio.get_running_loop().create_future()
        self._fetching[name] = future
        part = None
        try:
            # Tools pick a format by the name as well as by the bytes, and an original's name is
            # that of the file the caller's path leads to.
            real = await sftp.realpath(path)
            local = os.path.join(self.root, name + os.path.splitext(real)[1].lower())
            part = local + ".part"
            await sftp.get(path, part, block_size=BLOCK_SIZE, max_requests=MAX_REQUESTS)
            os.utime(part, (mtime, mtime))
            os.replace(part, local)
            copy = _Copy(local, size, mtime, time.monotonic(), pins=1)
            self._copies[name] = copy
            self._evict()
            return copy
        finally:
            if part is not None:
                with suppress(OSError):
                    os.unlink(part)
            future.set_result(None)
            del self._fetching[name]

    def release(self, copy: _Copy) -> None:
        copy.pins = max(0, copy.pins - 1)
        copy.used = time.monotonic()
        self._evict()

    def _evict(self) -> None:
        now = time.monotonic()
        total = sum(c.size for c in self._copies.values())
        for name, copy in sorted(self._copies.items(), key=lambda item: item[1].used):
            if copy.pins:
                continue
            if total <= self.budget_bytes and now - copy.used < self.ttl_seconds:
                continue
            with suppress(OSError):
                os.unlink(copy.local)
            del self._copies[name]
            total -= copy.size


class RemoteFiles:
    """What a request with an `EMS-Files` header needs: the connections and the input cache."""

    def __init__(self, scratch: str, budget_mb: int, ttl_seconds: float):
        self.connections = Connections()
        self.cache = InputCache(os.path.join(scratch, "files"), budget_mb * 1024 * 1024, ttl_seconds)

    def start(self) -> None:
        self.cache.start()

    async def stop(self) -> None:
        await self.connections.close()


class Staging:
    """One request's paths: its inputs' local copies and its outputs' local places, while it runs."""

    def __init__(self, files: RemoteFiles, endpoint: Endpoint, scratch: str):
        self.files = files
        self.endpoint = endpoint
        self.scratch = scratch
        self.inputs: dict[str, str] = {}
        self.outputs: dict[str, str] = {}
        self.dirs: dict[str, str] = {}
        self._pinned: list[_Copy] = []

    async def _sftp(self):
        return await self.files.connections.sftp(self.endpoint)

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
        for path in dict.fromkeys(_caller_path(p) for p in inputs):
            copy = await self._retrying(path, lambda sftp, path=path: self.files.cache.fetch(sftp, self.endpoint, path))
            self._pinned.append(copy)
            self.inputs[path] = copy.local

    async def _retrying(self, path: str, step):
        """Run one SFTP step, again on a fresh connection when the held one has gone stale."""
        for attempt in (0, 1):
            sftp = await self._sftp()
            try:
                return await step(sftp)
            except (asyncssh.SFTPNoSuchFile, asyncssh.SFTPPermissionDenied) as exc:
                raise _failure(path, exc) from exc
            except (OSError, asyncssh.Error) as exc:
                self.files.connections.drop(self.endpoint)
                if attempt:
                    raise _failure(path, exc) from exc

    def input(self, path: str) -> str:
        local = self.inputs.get(_caller_path(path))
        if local is None:
            raise EmsError(PATH_OUTSIDE_ROOTS, f"{path}: not an input of this request")
        return local

    def output(self, path: str) -> str:
        local = self.outputs.get(_caller_path(path))
        if local is None:
            raise EmsError(PATH_OUTSIDE_ROOTS, f"{path}: not an output of this request")
        return local

    def output_dir(self, path: str) -> str:
        local = self.dirs.get(_caller_path(path))
        if local is None:
            raise EmsError(PATH_OUTSIDE_ROOTS, f"{path}: not an output directory of this request")
        return local

    async def send_back(self) -> None:
        """Write what the op made to the caller's paths: each output it wrote, and every file in an output directory."""
        sends: list[tuple[str, str]] = [(local, remote) for remote, local in self.outputs.items() if os.path.isfile(local)]
        for remote_dir, local_dir in self.dirs.items():
            for name in sorted(os.listdir(local_dir)):
                local = os.path.join(local_dir, name)
                if not name.startswith(".") and os.path.isfile(local):
                    sends.append((local, posixpath.join(remote_dir, name)))
        for local, remote in sends:
            await self._retrying(remote, lambda sftp, local=local, remote=remote: sftp.put(local, remote, block_size=BLOCK_SIZE, max_requests=MAX_REQUESTS))

    def caller_paths(self, value: Any) -> Any:
        """`value` with every local path of this request's in its caller's spelling."""
        if isinstance(value, str):
            for remote, local in self.inputs.items():
                if value == local:
                    return remote
            for remote, local in self.outputs.items():
                if value == local:
                    return remote
            for remote, local in self.dirs.items():
                if value == local:
                    return remote
                if value.startswith(local + os.sep):
                    return posixpath.join(remote, value[len(local) + 1 :])
            return value
        if isinstance(value, list):
            return [self.caller_paths(item) for item in value]
        if isinstance(value, dict):
            return {key: self.caller_paths(item) for key, item in value.items()}
        return value

    def close(self) -> None:
        for copy in self._pinned:
            self.files.cache.release(copy)
        self._pinned.clear()
        shutil.rmtree(self.scratch, ignore_errors=True)


# The staging of the request the running task serves, if its files are its caller's.
current: ContextVar[Staging | None] = ContextVar("ems_files_staging", default=None)
