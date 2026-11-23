"""A page's browser reaches its page's own host and port wherever it is, and otherwise only the
internet: the proxy it connects through refuses loopback, private, link-local and the like, and
says so once for each place."""

import asyncio
import socket
import struct

import pytest

from ems.page_proxy import NOT_ALLOWED, OK, PageProxy, on_internet, origin_of
from ems.pages import PageSpec, chromium_args


@pytest.mark.parametrize("address", [
    "127.0.0.1", "10.0.0.5", "172.16.0.1", "192.168.1.1", "169.254.169.254", "100.64.0.1", "0.0.0.0", "224.0.0.1",
    "::1", "fe80::1%eth0", "fd00::1", "::ffff:10.0.0.5", "::ffff:127.0.0.1",
])
def test_an_address_off_the_internet_is_not_on_it(address):
    assert not on_internet(address)


@pytest.mark.parametrize("address", ["93.184.215.14", "8.8.8.8", "2606:4700::1111", "::ffff:8.8.8.8"])
def test_an_address_on_the_internet_is(address):
    assert on_internet(address)


def test_a_pages_origin_is_its_host_and_port_the_schemes_when_it_names_none():
    assert origin_of("http://EiN:8080/__page/x") == ("ein", 8080)
    assert origin_of("http://ein.test/x") == ("ein.test", 80)
    assert origin_of("https://ein.test/x") == ("ein.test", 443)
    assert origin_of("http://[fd00::5]:8080/x") == ("fd00::5", 8080)


async def _echo() -> tuple[asyncio.Server, int]:
    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(await reader.read(100))
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _through(proxy: int, host: str, port: int) -> tuple[int, bytes]:
    """The proxy's reply to a connection to `host`:`port`, and what it echoes when it connects."""
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy)
    try:
        writer.write(bytes([5, 1, 0]))
        assert await reader.readexactly(2) == bytes([5, 0])
        name = host.encode()
        writer.write(bytes([5, 1, 0, 3, len(name)]) + name + struct.pack(">H", port))
        reply = await reader.readexactly(10)
        if reply[1] != OK:
            return reply[1], b""
        writer.write(b"hello")
        return reply[1], await reader.read(100)
    finally:
        writer.close()


async def test_the_pages_own_origin_is_reached_and_any_other_local_one_refused_once():
    origin, origin_port = await _echo()
    other, other_port = await _echo()
    refused: list[str] = []
    proxy = PageProxy(("127.0.0.1", origin_port), refused.append)
    port = await proxy.start()
    try:
        assert await _through(port, "127.0.0.1", origin_port) == (OK, b"hello")
        assert await _through(port, "127.0.0.1", other_port) == (NOT_ALLOWED, b"")
        assert await _through(port, "127.0.0.1", other_port) == (NOT_ALLOWED, b"")
        # The origin is the name the page was given, not wherever else reaches the same server.
        assert await _through(port, "localhost", origin_port) == (NOT_ALLOWED, b"")
        assert await _through(port, "169.254.169.254", 80) == (NOT_ALLOWED, b"")
        assert refused == [f"127.0.0.1:{other_port}", f"localhost:{origin_port}", "169.254.169.254:80"]
    finally:
        proxy.close()
        origin.close()
        other.close()


async def test_a_connection_that_asks_for_anything_but_connect_is_answered_no():
    proxy = PageProxy(("ein", 8080))
    port = await proxy.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(bytes([5, 1, 0]))
        assert await reader.readexactly(2) == bytes([5, 0])
        # UDP ASSOCIATE, which would carry datagrams past the proxy's rules.
        writer.write(bytes([5, 3, 0, 1]) + socket.inet_aton("8.8.8.8") + struct.pack(">H", 53))
        assert (await reader.readexactly(10))[1] == 7
        writer.close()
    finally:
        proxy.close()


def test_chromium_reaches_everything_through_the_proxy_loopback_included():
    spec = PageSpec.model_validate({"url": "http://ein:8080/x", "output": {"mp4": {"path": "/g/0.mp4"}, "ts": {"path": "/g/1.ts"}}})
    args = chromium_args(spec, "/tmp/profile", 1024, 41234, True)
    assert "--proxy-server=socks5://127.0.0.1:41234" in args
    assert "--proxy-bypass-list=<-loopback>" in args
    assert "--force-webrtc-ip-handling-policy=disable_non_proxied_udp" in args
    assert args[-1] == "http://ein:8080/x"
    assert "--no-sandbox" not in args
    assert "--no-sandbox" in chromium_args(spec, "/tmp/profile", 1024, 41234, False)
