"""sshfs's `ssh_command` for a caller's `ems-files` endpoint: one connection, then SFTP on stdin and stdout.

sshfs runs it with an ssh client's arguments, which it ignores; what it connects to comes from the
environment remote.py starts sshfs with:

- `EMS_FILES_ADDRESS`: `host:port`.
- Plain: `EMS_FILES_SECRET`. The endpoint's first line is `EMS-Files <nonce>`; the answer is
  `HMAC-SHA256(secret, nonce)` in hex on a line, and the SFTP stream follows. Nothing is encrypted.
- TLS: `EMS_FILES_CA`, the endpoint's certificate, which is pinned (and names it `ems-files`), and
  `EMS_FILES_CERT`, this side's certificate and key.

Either way it ends as socat copying the connection to stdin and stdout, so no Python stays in the
way of the bytes. It must not import from the `ems` package: sshfs runs it as a script.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import socket
import sys

# The name the endpoint's certificate carries.
SERVER_NAME = "ems-files"
# socat's buffer: an SFTP read's answer in one copy.
BUFFER = 262144
HANDSHAKE_SECONDS = 20


def main() -> None:
    host, _, port = os.environ["EMS_FILES_ADDRESS"].rpartition(":")
    host = host.strip("[]")
    ca = os.environ.get("EMS_FILES_CA")
    if ca:
        cert = os.environ["EMS_FILES_CERT"]
        os.execvp("socat", ["socat", f"-b{BUFFER}", "STDIO", f"OPENSSL:{host}:{port},cafile={ca},cert={cert},commonname={SERVER_NAME},nodelay"])
    secret = os.environ["EMS_FILES_SECRET"].encode()
    conn = socket.create_connection((host, int(port)), timeout=HANDSHAKE_SECONDS)
    # Nagle's algorithm would hold each request back until the endpoint's delayed ACK.
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    line = b""
    while not line.endswith(b"\n"):
        chunk = conn.recv(1)
        if not chunk or len(line) > 200:
            sys.exit(f"ems-files {host}:{port}: no challenge")
        line += chunk
    word, _, nonce = line.decode(errors="replace").strip().partition(" ")
    if word != "EMS-Files":
        sys.exit(f"ems-files {host}:{port}: not an ems-files endpoint")
    conn.sendall(hmac.new(secret, bytes.fromhex(nonce), hashlib.sha256).hexdigest().encode() + b"\n")
    conn.settimeout(None)
    fd = conn.detach()
    os.set_inheritable(fd, True)
    os.execvp("socat", ["socat", f"-b{BUFFER}", "STDIO", f"FD:{fd}"])


if __name__ == "__main__":
    main()
