"""The caller's files, when EMS does not mount them: reached for a request, sent back before it answers.

A request may carry an `EMS-Files` header naming an SFTP endpoint of its caller's:
`address=host:port; user=…; secret=…; hostkey=ssh-ed25519 AAAA…`. Every path in that request is
then the caller's, not a local one, and the op runs on local paths for them (`Roots` asks
`current` first):

- An input (`file.path`) is read through an sshfs mount of the endpoint when this host can make
  one (`/dev/fuse`, sshfs, `EMS_FILES_MOUNT` not `0`): one read-only mount per endpoint, made on
  first use, so a tool seeks and reads only what it needs while it runs, and the kernel's page
  cache keeps what was read for the next op until the file's size or mtime changes. Without a
  mount, an input is copied whole into a local cache first, under its caller path and checked
  against its size and mtime, so the next op on the same file does not fetch it again; a copy in
  use is never evicted.
- An output (`output.path`, `output.dir`) gets a place in the request's scratch, and what the op
  wrote goes back to the caller's paths before the result does: an output file copied into the
  mount when there is one, where sshfs keeps many writes in flight, and otherwise, and for the
  files of an output directory, sent with the SFTP client.

The caller grants a request exactly its own paths for as long as it runs, so nothing is listed
and nothing else is asked for. Every connection is SSH with chacha20-poly1305 alone and the host
key from the header pinned.
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
from dataclasses import dataclass, field
from typing import Any

import asyncssh

from .admission import current_op
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


def _log(message: str) -> None:
    """One line on the service's log, for the request the running task serves."""
    print(f"ems: {current_op.get() or 'request'}: {message}", file=sys.stderr, flush=True)


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
    # Whether the pin that returned it fetched it, rather than finding it in the cache.
    fetched: bool = False


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
                copy.fetched = False
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
            copy = _Copy(local, size, mtime, time.monotonic(), pins=1, fetched=True)
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


class MountFailed(Exception):
    """sshfs could not mount an endpoint: this host cannot mount, or the endpoint refused."""


class Mounts:
    """sshfs mounts of callers' endpoints, one per endpoint, made on first use.

    A mount whose sshfs has exited, its connection gone, is made again; one of an older endpoint
    at the same address (the caller restarted, with a new key and secret) is let go. After a mount
    fails, none is tried for `RETRY_AFTER` seconds, and inputs are copied meanwhile.
    """

    RETRY_AFTER = 600

    def __init__(self, root: str, enabled: bool):
        self.root = root
        self._possible = enabled and os.path.exists("/dev/fuse") and shutil.which("sshfs") is not None
        self._failed_at = -self.RETRY_AFTER
        self._held: dict[Endpoint, tuple[str, asyncio.subprocess.Process]] = {}
        self._locks: dict[Endpoint, asyncio.Lock] = {}

    @property
    def enabled(self) -> bool:
        return self._possible and time.monotonic() - self._failed_at >= self.RETRY_AFTER

    async def mount(self, endpoint: Endpoint) -> str:
        """Where `endpoint`'s `/` is mounted."""
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
                    self._failed_at = time.monotonic()
                    raise MountFailed(f"{endpoint.name}: sshfs: {error or 'no mount'}")
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


async def _unmount(point: str) -> None:
    """Take a mount off `point`, one whose sshfs died too, which stays in the table until then."""
    unmount = await asyncio.create_subprocess_exec("fusermount3", "-uz", point, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await unmount.wait()


class RemoteFiles:
    """What a request with an `EMS-Files` header needs: the connections, the mounts and the input cache."""

    def __init__(self, scratch: str, budget_mb: int, ttl_seconds: float, mount: bool = True, mounts_root: str | None = None):
        self.connections = Connections()
        self.cache = InputCache(os.path.join(scratch, "files"), budget_mb * 1024 * 1024, ttl_seconds)
        # Not under the scratch, which is swept: a sweep must never walk into a mount.
        self.mounts = Mounts(mounts_root or os.path.join(tempfile.gettempdir(), "ems-mounts"), mount)

    def start(self) -> None:
        self.cache.start()
        if self.mounts.enabled:
            os.makedirs(self.mounts.root, exist_ok=True)

    async def stop(self) -> None:
        await self.mounts.stop()
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
        started = time.monotonic()
        paths = list(dict.fromkeys(_caller_path(p) for p in inputs))
        if paths and self.files.mounts.enabled:
            try:
                await self._mounted(paths)
                _log(f"{len(paths)} input(s) through the mount of {self.endpoint.name}, in {time.monotonic() - started:.2f} s")
                return
            except MountFailed as exc:
                _log(f"no mount ({exc}); copying the inputs instead, for the next {Mounts.RETRY_AFTER} s")
                self.inputs.clear()
        fetched = cached = 0
        for path in paths:
            copy = await self._retrying(path, lambda sftp, path=path: self.files.cache.fetch(sftp, self.endpoint, path))
            self._pinned.append(copy)
            self.inputs[path] = copy.local
            if copy.fetched:
                fetched += copy.size
            else:
                cached += copy.size
        if self.inputs:
            _log(f"{len(self.inputs)} input(s) from {self.endpoint.name}: {fetched / 1e6:.1f} MB fetched, {cached / 1e6:.1f} MB from the cache, in {time.monotonic() - started:.2f} s")

    async def _mounted(self, paths: list[str]) -> None:
        """Each input as a link in the request's scratch to its file in the endpoint's mount.

        The link bears the name of the file the caller's path leads to, since tools pick a format
        by the name as well as by the bytes; a mount that stopped answering is made again once.
        """
        for index, path in enumerate(paths):
            real = await self._retrying(path, lambda sftp, path=path: sftp.realpath(path))
            for attempt in (0, 1):
                point = await self.files.mounts.mount(self.endpoint)
                target = point + path
                try:
                    await asyncio.to_thread(os.stat, target)
                    break
                except FileNotFoundError as exc:
                    raise EmsError(NOT_FOUND, f"{path}: no such file") from exc
                except PermissionError as exc:
                    raise EmsError(PATH_OUTSIDE_ROOTS, f"{path}: not granted to this request") from exc
                except OSError as exc:
                    await self.files.mounts.drop(self.endpoint)
                    if attempt:
                        raise EmsError(FILES_UNAVAILABLE, f"{path}: {exc}") from exc
            local_dir = os.path.join(self.scratch, "in", str(index))
            os.makedirs(local_dir)
            link = os.path.join(local_dir, posixpath.basename(real) or "input")
            os.symlink(target, link)
            self.inputs[path] = link

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
        started = time.monotonic()
        files = set(self.outputs.values())
        point = await self._point() if any(local in files for local, _remote in sends) else None
        for local, remote in sends:
            if point is not None and local in files and await self._copied(local, point + remote):
                continue
            await self._retrying(remote, lambda sftp, local=local, remote=remote: sftp.put(local, remote, block_size=BLOCK_SIZE, max_requests=MAX_REQUESTS))
        if sends:
            size = sum(os.path.getsize(local) for local, _remote in sends)
            way = " through the mount" if point is not None else ""
            _log(f"{len(sends)} output(s) to {self.endpoint.name}{way}: {size / 1e6:.1f} MB sent back in {time.monotonic() - started:.2f} s")

    async def _point(self) -> str | None:
        """Where the endpoint is mounted, or None to send outputs with the SFTP client instead."""
        if not self.files.mounts.enabled:
            return None
        try:
            return await self.files.mounts.mount(self.endpoint)
        except MountFailed:
            return None

    @staticmethod
    async def _copied(local: str, target: str) -> bool:
        """Copy an output into the mount, its bytes at the caller once the copy returns; False when
        the mount refused it, for the SFTP client to send it instead."""
        try:
            await asyncio.to_thread(shutil.copyfile, local, target)
            return True
        except OSError:
            return False

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
