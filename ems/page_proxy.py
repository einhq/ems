"""Where a page's browser may connect: its page's own host and port, and the internet.

Every connection the browser makes, the page's and every redirect, fetch, frame and socket of it,
goes through a SOCKS5 proxy of its own on 127.0.0.1, which resolves the name itself and connects
only to an address on the internet (`on_internet`), never to this machine, its networks or a cloud's
metadata service. The one exception is the page's origin, its host and port as the caller wrote
them in its URL, which may be anywhere: that is the caller's own server. The address checked is the address
connected to, so a name that resolves elsewhere the second time gets no further.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import struct
from contextlib import suppress
from typing import Callable
from urllib.parse import urlsplit

# How long a connection may take to say where it goes, and to be connected there.
HANDSHAKE_SECONDS = 10.0
CONNECT_SECONDS = 10.0
COPY_BYTES = 1 << 16

SOCKS = 5
NO_AUTH = 0
CONNECT = 1
IPV4, DOMAIN, IPV6 = 1, 3, 4
# Replies: succeeded, refused by the rules, host unreachable, command not supported, address type not supported.
OK, NOT_ALLOWED, UNREACHABLE, NO_COMMAND, NO_ADDRESS_TYPE = 0, 2, 4, 7, 8


def on_internet(address: str) -> bool:
    """Whether `address` is one on the internet: not loopback, private, link-local (a cloud's
    metadata service among them), shared, reserved or multicast, an IPv4 address inside IPv6
    judged as itself."""
    found = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(found, ipaddress.IPv6Address) and found.ipv4_mapped is not None:
        found = found.ipv4_mapped
    return found.is_global and not found.is_multicast


def origin_of(url: str) -> tuple[str, int]:
    """The host and port of `url`, the port its scheme's when it names none."""
    parts = urlsplit(url)
    return (parts.hostname or "").lower(), parts.port or (443 if parts.scheme == "https" else 80)


class PageProxy:
    """The proxy for one page's browser: `origin` reached wherever it is, everything else only on
    the internet. `refused` is told each host and port refused, once."""

    def __init__(self, origin: tuple[str, int], refused: Callable[[str], None] = lambda _: None):
        self.origin = origin
        self.refused = refused
        self.told: set[str] = set()
        self.server: asyncio.Server | None = None
        self.tasks: set[asyncio.Task] = set()

    async def start(self) -> int:
        """Listen on 127.0.0.1, on a port of its own, which it answers."""
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    def close(self) -> None:
        if self.server is not None:
            self.server.close()
        for task in list(self.tasks):
            task.cancel()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.tasks.add(task)
        upstream: asyncio.StreamWriter | None = None
        try:
            asked = await asyncio.wait_for(self._asked(reader, writer), HANDSHAKE_SECONDS)
            if asked is None:
                return
            host, port = asked
            addresses = await self._addresses(host, port)
            if not addresses:
                self._refuse(f"{host}:{port}")
                writer.write(_reply(NOT_ALLOWED))
                return
            connected = await self._connect(addresses, port)
            if connected is None:
                writer.write(_reply(UNREACHABLE))
                return
            upstream_reader, upstream = connected
            writer.write(_reply(OK))
            await writer.drain()
            await asyncio.gather(_copy(reader, upstream), _copy(upstream_reader, writer))
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            self.tasks.discard(task)
            for each in (upstream, writer):
                if each is not None:
                    each.close()

    async def _asked(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> tuple[str, int] | None:
        """The host and port a connection asks for, after the greeting; None, answered, when it asks
        for something else."""
        version, count = await reader.readexactly(2)
        methods = await reader.readexactly(count)
        if version != SOCKS or NO_AUTH not in methods:
            writer.write(bytes([SOCKS, 0xFF]))
            return None
        writer.write(bytes([SOCKS, NO_AUTH]))
        version, command, _, kind = await reader.readexactly(4)
        if version != SOCKS or command != CONNECT:
            writer.write(_reply(NO_COMMAND))
            return None
        if kind == IPV4:
            host = socket.inet_ntop(socket.AF_INET, await reader.readexactly(4))
        elif kind == IPV6:
            host = socket.inet_ntop(socket.AF_INET6, await reader.readexactly(16))
        elif kind == DOMAIN:
            host = (await reader.readexactly((await reader.readexactly(1))[0])).decode("ascii", errors="replace")
        else:
            writer.write(_reply(NO_ADDRESS_TYPE))
            return None
        (port,) = struct.unpack(">H", await reader.readexactly(2))
        return host.strip("[]").lower(), port

    async def _addresses(self, host: str, port: int) -> list[str]:
        """The addresses `host` may be connected at: every one it resolves to for the page's
        origin, else those on the internet."""
        try:
            found = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except (socket.gaierror, UnicodeError):
            return []
        mine = (host, port) == self.origin
        addresses: list[str] = []
        for *_, (address, *_) in found:
            if address not in addresses and (mine or on_internet(address)):
                addresses.append(address)
        return addresses

    async def _connect(self, addresses: list[str], port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
        for address in addresses:
            with suppress(OSError, asyncio.TimeoutError):
                return await asyncio.wait_for(asyncio.open_connection(address, port), CONNECT_SECONDS)
        return None

    def _refuse(self, where: str) -> None:
        if where not in self.told:
            self.told.add(where)
            self.refused(where)


def _reply(code: int) -> bytes:
    return bytes([SOCKS, code, 0, IPV4, 0, 0, 0, 0, 0, 0])


async def _copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """`reader` into `writer` until it ends, then `writer` ended too, as far as it can be."""
    try:
        while data := await reader.read(COPY_BYTES):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except (ConnectionError, OSError):
        writer.close()
