"""Requests whose files are their caller's (`EMS-Files`): inputs fetched, outputs sent back.

The caller is an asyncssh SFTP server on loopback serving a directory outside every root, so
an op can only reach its files through the staging: the local mounts have none of them.
"""

import asyncio
import os
import shutil
import threading

import asyncssh
import pytest

from ems.ops import video
from ems.remote import parse_header, request_paths

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SECRET = "a-secret-made-at-start"
OPENED: list[tuple[str, int]] = []


class CountingSFTP(asyncssh.SFTPServer):
    def open(self, path, pflags, attrs):
        OPENED.append((path.decode(), pflags))
        return super().open(path, pflags, attrs)


class OneUser(asyncssh.SSHServer):
    def begin_auth(self, username):
        return True

    def password_auth_supported(self):
        return True

    def validate_password(self, username, password):
        return username == "ems" and password == SECRET


@pytest.fixture
def caller(tmp_path):
    """The caller's files, and the `EMS-Files` header that names them.

    The server runs on a loop of its own, in its own thread: EMS's look at a mount of it blocks
    until the server answers, as it would with a caller in another process.
    """
    files = tmp_path / "caller"
    files.mkdir()
    key = asyncssh.generate_private_key("ssh-ed25519")
    loop = asyncio.new_event_loop()
    started = threading.Event()
    holder = {}

    async def serve():
        holder["server"] = await asyncssh.create_server(
            OneUser, "127.0.0.1", 0, server_host_keys=[key], sftp_factory=CountingSFTP,
            encryption_algs=["chacha20-poly1305@openssh.com"],
        )
        started.set()

    thread = threading.Thread(target=lambda: (loop.run_until_complete(serve()), loop.run_forever()), daemon=True)
    thread.start()
    started.wait(10)
    port = holder["server"].sockets[0].getsockname()[1]
    hostkey = key.export_public_key().decode().strip()
    OPENED.clear()
    yield files, {"EMS-Files": f"address=127.0.0.1:{port}; user=ems; secret={SECRET}; hostkey={hostkey}"}
    loop.call_soon_threadsafe(holder["server"].close)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(10)


MOUNTABLE = os.path.exists("/dev/fuse") and shutil.which("sshfs") is not None and shutil.which("fusermount3") is not None


@pytest.fixture(params=["copy", pytest.param("mount", marks=pytest.mark.skipif(not MOUNTABLE, reason="no sshfs or FUSE here"))])
def mode(request, client):
    """Each test once with the inputs copied and once read through an sshfs mount."""
    client.app.state.ctx.files.mounts._possible = request.param == "mount"
    return request.param


def reads_of(path) -> int:
    return sum(1 for opened, pflags in OPENED if opened == str(path) and pflags & asyncssh.FXF_READ)


def test_the_header_names_an_endpoint_and_the_body_its_paths():
    endpoint = parse_header("address=ein.example:2224; user=ems; secret=s; hostkey=ssh-ed25519 AAAA")
    assert (endpoint.host, endpoint.port, endpoint.user, endpoint.hostkey) == ("ein.example", 2224, "ems", "ssh-ed25519 AAAA")
    assert request_paths({"file": {"path": "/a"}, "targets": [{"output": {"path": "/b"}}], "output": {"dir": "/c"}}) == (["/a"], ["/b"], ["/c"])


async def test_an_input_is_read_and_its_scratch_goes_with_the_request(client, caller, mode, tmp_path):
    files, headers = caller
    (files / "note.txt").write_text("written by Anna Andersson")
    for _ in range(2):
        r = await client.post("/v0/document/extract", json={"file": {"path": str(files / "note.txt")}}, headers=headers)
        assert r.status_code == 200, r.json()
        assert r.json()["result"]["text"].strip() == "written by Anna Andersson"
    # A copy is fetched once; through a mount the file is opened for each op.
    if mode == "copy":
        assert reads_of(files / "note.txt") == 1
    # The request's own scratch goes with it; only the caches stay.
    assert not [name for name in os.listdir(tmp_path / "scratch") if name.startswith("staged-")]


async def test_a_changed_input_is_read_again(client, caller, mode):
    files, headers = caller
    note = files / "note.txt"
    note.write_text("first")
    await client.post("/v0/document/extract", json={"file": {"path": str(note)}}, headers=headers)
    note.write_text("second, longer")
    r = await client.post("/v0/document/extract", json={"file": {"path": str(note)}}, headers=headers)
    assert r.json()["result"]["text"].strip() == "second, longer"
    if mode == "copy":
        assert reads_of(note) == 2


async def test_frames_go_back_to_the_callers_directory(client, caller, mode, monkeypatch):
    files, headers = caller
    shutil.copy(os.path.join(DATA, "video1.mp4"), files / "clip.mp4")
    # The caller makes a directory it grants; this one is a plain SFTP server.
    (files / "frames").mkdir()

    async def fake(ctx, path, out_dir, *, at, count, strategy, fmt, quality, max_edge, progress, angle, mirror):
        assert path.endswith(".mp4") and not path.startswith(str(files)) and open(path, "rb").read() == (files / "clip.mp4").read_bytes()
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


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg is not installed")
async def test_a_transcode_writes_the_callers_output_before_it_answers(client, caller, mode):
    files, headers = caller
    shutil.copy(os.path.join(DATA, "video1.mp4"), files / "clip.mp4")
    (files / "sound.m4a").write_bytes(b"")
    r = await client.post("/v0/audio/transcode", json={"file": {"path": str(files / "clip.mp4")}, "output": {"path": str(files / "sound.m4a")}}, headers=headers)
    assert r.status_code == 200, r.json()
    assert (files / "sound.m4a").stat().st_size > 1000
    assert sorted(os.listdir(files)) == ["clip.mp4", "sound.m4a"]


async def test_a_missing_input_is_not_found_and_a_bad_header_invalid(client, caller, mode):
    files, headers = caller
    r = await client.post("/v0/document/extract", json={"file": {"path": str(files / "gone.txt")}}, headers=headers)
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = await client.post("/v0/document/extract", json={"file": {"path": str(files / "gone.txt")}}, headers={"EMS-Files": "address=nowhere"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


async def test_an_unreachable_caller_is_a_failure_to_retry(client, caller, mode):
    files, headers = caller
    (files / "note.txt").write_text("x")
    wrong = {"EMS-Files": headers["EMS-Files"].replace(SECRET, "not-the-secret")}
    r = await client.post("/v0/document/extract", json={"file": {"path": str(files / "note.txt")}}, headers=wrong)
    error = r.json()["error"]
    assert r.status_code == 502 and error["code"] == "files_unavailable" and error["permanent"] is False
