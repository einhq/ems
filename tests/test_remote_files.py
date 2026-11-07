"""Requests whose files are their caller's (`EMS-Files`): read and written through an sshfs mount.

The caller is OpenSSH's sftp-server on loopback, behind what an ems-files endpoint puts in front
of SFTP: the challenge answered with the secret (plain), or TLS with both certificates pinned.
EMS reaches it through a real sshfs mount, each connection made by files_connect.py. The tests
that mount need sshfs, /dev/fuse, socat and sftp-server, and openssl for TLS.
"""

import base64
import hashlib
import hmac
import os
import secrets
import shutil
import socket
import subprocess
import threading
import time

import pytest

from ems.ops import video
from ems.remote import Mounts, parse_header, request_paths
from ems.errors import EmsError

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SFTP_SERVER = next((p for p in ("/usr/lib/openssh/sftp-server", "/usr/libexec/openssh/sftp-server", "/usr/libexec/sftp-server") if os.path.exists(p)), None)
MOUNTABLE = os.path.exists("/dev/fuse") and all(shutil.which(tool) for tool in ("sshfs", "fusermount3", "socat")) and SFTP_SERVER is not None
needs_mount = pytest.mark.skipif(not MOUNTABLE, reason="no sshfs, FUSE, socat or sftp-server here")


@pytest.fixture
async def mounting(client, tmp_path):
    """The app with real sshfs mounts in place of the conftest's host files."""
    mounts = Mounts(str(tmp_path / "mounts"))
    mounts.start()
    client.app.state.ctx.files = mounts
    yield mounts
    await mounts.stop()


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def serve_plain(secret: str) -> tuple[int, callable]:
    """sftp-server for each connection that answers the challenge; the port, and what stops it."""
    listener = socket.create_server(("127.0.0.1", 0))
    stopping = threading.Event()

    def one(conn: socket.socket) -> None:
        with conn:
            nonce = secrets.token_bytes(32)
            conn.sendall(b"EMS-Files " + nonce.hex().encode() + b"\n")
            answer = conn.makefile("rb").readline().strip()
            if hmac.compare_digest(answer, hmac.new(secret.encode(), nonce, hashlib.sha256).hexdigest().encode()):
                subprocess.run([SFTP_SERVER], stdin=conn.fileno(), stdout=conn.fileno(), check=False)

    def accept() -> None:
        while not stopping.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=one, args=(conn,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()

    def stop() -> None:
        stopping.set()
        listener.close()

    return listener.getsockname()[1], stop


def certificate(directory, name: str) -> tuple[str, str]:
    """A self-signed certificate for `name` and its key, PEM, as an ems-files endpoint makes them."""
    key, cert = directory / f"{name}.key", directory / f"{name}.crt"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-days", "1",
         "-keyout", str(key), "-out", str(cert), "-subj", f"/CN={name}", "-addext", f"subjectAltName=DNS:{name}"],
        check=True, capture_output=True,
    )
    return cert.read_text(), key.read_text()


def serve_tls(directory, client_cert: str) -> tuple[int, str, callable]:
    """sftp-server for each TLS connection that presents `client_cert`; the port, the server's certificate, and what stops it."""
    cert, key = certificate(directory, "ems-files")
    (directory / "server.pem").write_text(cert + key)
    (directory / "accepted.crt").write_text(client_cert)
    with socket.create_server(("127.0.0.1", 0)) as probe:
        port = probe.getsockname()[1]
    server = subprocess.Popen(
        ["socat", f"OPENSSL-LISTEN:{port},bind=127.0.0.1,reuseaddr,fork,cert={directory / 'server.pem'},cafile={directory / 'accepted.crt'},verify=1", f"EXEC:{SFTP_SERVER}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with socket.socket() as attempt:
            if attempt.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.05)
    return port, cert, lambda: (server.terminate(), server.wait(10))


@pytest.fixture(params=["plain", "tls"])
def caller(request, tmp_path):
    """The caller's files, and the `EMS-Files` header that names them, plain and over TLS."""
    files = tmp_path / "caller"
    files.mkdir()
    if request.param == "tls":
        if not shutil.which("openssl"):
            pytest.skip("no openssl here to make certificates")
        credentials = tmp_path / "credentials"
        credentials.mkdir()
        client_cert, client_key = certificate(credentials, "ems")
        port, server_cert, stop = serve_tls(credentials, client_cert)
        header = f"address=127.0.0.1:{port}; tls={b64(server_cert)}; client={b64(client_cert + client_key)}"
    else:
        secret = secrets.token_hex(32)
        port, stop = serve_plain(secret)
        header = f"address=127.0.0.1:{port}; secret={secret}"
    yield files, {"EMS-Files": header}
    stop()


def test_the_header_names_an_endpoint_and_how_it_is_reached():
    plain = parse_header("address=ein.example:2224; secret=5ec7e7")
    assert (plain.host, plain.port, plain.secret, plain.tls) == ("ein.example", 2224, "5ec7e7", "")
    tls = parse_header(f"address=[ein.example]:2224; tls={b64('server cert')}; client={b64('client cert and key')}")
    assert (tls.host, tls.port, tls.secret, tls.tls, tls.client) == ("ein.example", 2224, "", "server cert", "client cert and key")
    for wrong in ("address=ein.example:2224", "secret=5ec7e7", "address=ein.example:2224; tls=bm90", "address=ein.example:2224; tls=@@; client=@@"):
        with pytest.raises(EmsError):
            parse_header(wrong)
    assert request_paths({"file": {"path": "/a"}, "targets": [{"output": {"path": "/b"}}], "output": {"dir": "/c"}}) == (["/a"], ["/b"], ["/c"])


@needs_mount
async def test_an_input_is_read_through_the_mount_and_the_scratch_goes_with_the_request(client, mounting, caller, tmp_path):
    files, headers = caller
    (files / "note.txt").write_text("written by Anna Andersson")
    for _ in range(2):
        r = await client.post("/v0/document/extract", json={"file": {"path": str(files / "note.txt")}}, headers=headers)
        assert r.status_code == 200, r.json()
        assert r.json()["result"]["text"].strip() == "written by Anna Andersson"
    assert not [name for name in os.listdir(tmp_path / "scratch") if name.startswith("staged-")]


@needs_mount
async def test_a_changed_input_is_read_again(client, mounting, caller):
    files, headers = caller
    note = files / "note.txt"
    note.write_text("first")
    await client.post("/v0/document/extract", json={"file": {"path": str(note)}}, headers=headers)
    note.write_text("second, longer")
    r = await client.post("/v0/document/extract", json={"file": {"path": str(note)}}, headers=headers)
    assert r.json()["result"]["text"].strip() == "second, longer"


@needs_mount
async def test_frames_go_back_to_the_callers_directory_which_is_made(client, mounting, caller, monkeypatch):
    files, headers = caller
    shutil.copy(os.path.join(DATA, "video1.mp4"), files / "clip.mp4")

    async def fake(ctx, path, out_dir, *, at, count, strategy, fmt, quality, max_edge, progress, angle, mirror):
        assert path.endswith("clip.mp4") and open(path, "rb").read() == (files / "clip.mp4").read_bytes()
        frames = []
        for i, t in enumerate(at):
            with open(f"{out_dir}/frame-{i:03d}.jpg", "wb") as f:
                f.write(b"jpeg %d" % i)
            frames.append({"path": f"{out_dir}/frame-{i:03d}.jpg", "t": t})
        return frames

    monkeypatch.setattr(video, "extract_frames", fake)
    r = await client.post("/v0/video/frames", json={"file": {"path": str(files / "clip.mp4")}, "at": [1.0, 2.0], "output": {"dir": str(files / "frames")}}, headers=headers)
    assert r.status_code == 200, r.json()
    assert r.json()["result"]["frames"] == [{"path": str(files / "frames" / "frame-000.jpg"), "t": 1.0}, {"path": str(files / "frames" / "frame-001.jpg"), "t": 2.0}]
    assert (files / "frames" / "frame-001.jpg").read_bytes() == b"jpeg 1"


@needs_mount
@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg is not installed")
async def test_a_transcode_writes_the_callers_output_before_it_answers(client, mounting, caller):
    files, headers = caller
    shutil.copy(os.path.join(DATA, "video1.mp4"), files / "clip.mp4")
    (files / "sound.m4a").write_bytes(b"")
    r = await client.post("/v0/audio/transcode", json={"file": {"path": str(files / "clip.mp4")}, "output": {"path": str(files / "sound.m4a")}}, headers=headers)
    assert r.status_code == 200, r.json()
    assert (files / "sound.m4a").stat().st_size > 1000
    assert sorted(os.listdir(files)) == ["clip.mp4", "sound.m4a"]


@needs_mount
async def test_a_missing_input_is_not_found_and_a_bad_header_invalid(client, mounting, caller):
    files, headers = caller
    r = await client.post("/v0/document/extract", json={"file": {"path": str(files / "gone.txt")}}, headers=headers)
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = await client.post("/v0/document/extract", json={"file": {"path": str(files / "gone.txt")}}, headers={"EMS-Files": "address=nowhere"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


@needs_mount
async def test_a_caller_that_refuses_the_mount_is_a_failure_to_retry(client, mounting, caller, tmp_path):
    """Another secret, or another client certificate than the one the caller accepts."""
    files, headers = caller
    (files / "note.txt").write_text("x")
    endpoint = parse_header(headers["EMS-Files"])
    if endpoint.secret:
        wrong = {"EMS-Files": f"address=127.0.0.1:{endpoint.port}; secret={secrets.token_hex(32)}"}
    else:
        (tmp_path / "other").mkdir()
        other = "".join(certificate(tmp_path / "other", "ems"))
        wrong = {"EMS-Files": f"address=127.0.0.1:{endpoint.port}; tls={b64(endpoint.tls)}; client={b64(other)}"}
    r = await client.post("/v0/document/extract", json={"file": {"path": str(files / "note.txt")}}, headers=wrong)
    error = r.json()["error"]
    assert r.status_code == 502 and error["code"] == "files_unavailable" and error["permanent"] is False


async def test_a_host_that_cannot_mount_refuses_the_request(client, mounting, caller):
    files, headers = caller
    (files / "note.txt").write_text("x")
    mounting.possible = False
    r = await client.post("/v0/document/extract", json={"file": {"path": str(files / "note.txt")}}, headers=headers)
    error = r.json()["error"]
    assert r.status_code == 502 and error["code"] == "files_unavailable" and "cannot mount" in error["message"]
