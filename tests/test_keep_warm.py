"""Keep-warm: a quiet model server runs one query through each pinned family, and stays out of a request's way."""

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import model_registry as registry  # noqa: E402
import model_server as ms  # noqa: E402


@pytest.fixture
def quiet(monkeypatch):
    """A server whose pins are ready and whose requests record their op instead of running a model."""
    ran: list[str] = []
    monkeypatch.setattr(ms, "KEEP_WARM_SECONDS", 0.05)
    monkeypatch.setattr(ms, "_recycling", False)
    monkeypatch.setattr(ms, "_last_work_at", time.time() - 60)
    monkeypatch.setattr(ms, "log", lambda message: None)
    monkeypatch.setattr(registry, "PINNED_FAMILIES", {"siglip-text", "text-query"})
    monkeypatch.setattr(ms, "handle", lambda request: ran.append(request["op"]))
    registry.pinned_ready.set()
    return ran


def _run(seconds: float) -> None:
    stop = threading.Event()
    thread = threading.Thread(target=ms._keep_warm, args=(stop,), daemon=True)
    thread.start()
    time.sleep(seconds)
    stop.set()
    thread.join(1)


def test_a_quiet_server_queries_each_pinned_family(quiet):
    _run(0.2)
    assert "embed_image_texts" in quiet and "embed_texts" in quiet


def test_a_recent_request_holds_it_off(quiet, monkeypatch):
    monkeypatch.setattr(ms, "KEEP_WARM_SECONDS", 0.1)
    monkeypatch.setattr(ms, "_last_work_at", time.time() + 60)
    _run(0.3)
    assert quiet == []


def test_a_family_a_request_holds_is_skipped(quiet):
    lock = registry.family_lock("siglip-text")
    lock.acquire()
    try:
        _run(0.2)
    finally:
        lock.release()
    assert "embed_image_texts" not in quiet and "embed_texts" in quiet


class _Model:
    def __init__(self):
        self.texts: list[str] = []

    def encode(self, texts, prompt_name):
        self.texts.extend(texts)


@pytest.mark.parametrize("device, lengths", [("cuda", list(range(1, ms.WARM_QUERY_WORDS + 1))), ("cpu", [1])])
def test_the_query_preload_loads_each_lengths_kernels_on_the_card(monkeypatch, device, lengths):
    model = _Model()
    monkeypatch.setattr(registry, "text_embed_query_device", lambda: device)
    monkeypatch.setattr(registry, "text_embedder", lambda family, device: model)
    ms._warm_query("text-query")()
    assert [len(text.split()) for text in model.texts] == lengths


def test_the_query_family_is_kept_warm_at_several_lengths(quiet, monkeypatch):
    texts: list[str] = []
    monkeypatch.setattr(registry, "PINNED_FAMILIES", {"text-query"})
    monkeypatch.setattr(ms, "handle", lambda request: texts.extend(request["texts"]))
    _run(0.08)
    assert [len(text.split()) for text in texts[:4]] == [1, 16, 32, 64]
