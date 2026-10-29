"""A request whose caller gave up is dropped, not run or reported as a fault, wherever it is."""

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import model_server as ms  # noqa: E402


class Gone(Exception):
    pass


@pytest.fixture
def serve(monkeypatch):
    """Runs one request through the Handler; the caller hangs up once `gone` is set."""
    monkeypatch.setattr(ms, "log", lambda message: None)
    monkeypatch.setattr(ms.registry, "cleanup_torch", lambda: None)
    state = {"gone": False, "ran": False}

    def run(request, op):
        handler = object.__new__(ms.Handler)
        handler.rfile = io.BytesIO(json.dumps(request).encode() + b"\n")
        replies = []
        handler._reply = replies.append
        handler._hung_up = lambda: state["gone"]

        def handle(_request):
            state["ran"] = True
            return op()

        monkeypatch.setattr(ms, "handle", handle)
        handler.handle()
        return replies[0]

    return state, run


def test_a_caller_gone_before_the_start_is_not_run(serve):
    state, run = serve
    state["gone"] = True
    reply = run({"op": "embed_images"}, lambda: {})
    assert reply["error_type"] == "RequestAbandoned"
    assert "before the request started" in reply["error"]
    assert state["ran"] is False


def test_a_missing_file_after_the_caller_left_is_a_drop(serve):
    state, run = serve

    def op():
        state["gone"] = True
        raise FileNotFoundError("Image not found: /tmp/ems/embed-x/frame-000.jpg")

    reply = run({"op": "embed_images"}, op)
    assert reply["error_type"] == "RequestAbandoned"
    assert "Image not found" in reply["error"]


def test_a_missing_file_while_the_caller_waits_is_a_failure(serve):
    _, run = serve

    def op():
        raise FileNotFoundError("Image not found: /nowhere.jpg")

    reply = run({"op": "embed_images"}, op)
    assert reply["error_type"] == "FileNotFoundError"
