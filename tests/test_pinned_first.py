"""After a restart the pinned families load before any other: search must not wait out another family's load."""

import threading

from ems.modelworker import model_registry as registry


def test_a_family_that_is_not_pinned_loads_after_the_pins(monkeypatch):
    monkeypatch.setattr(registry, "_cache", {})
    monkeypatch.setattr(registry, "_last_used", {})
    monkeypatch.setattr(registry, "PINNED_FAMILIES", {"text-query"})
    ready = threading.Event()
    monkeypatch.setattr(registry, "pinned_ready", ready)

    order = []
    batch = threading.Thread(target=lambda: registry._get("whisper:cuda", lambda: order.append("whisper") or "whisper"))
    batch.start()
    batch.join(0.2)
    assert batch.is_alive() and order == []           # the batch family arrived first, and waits

    registry._get("text-query:cuda", lambda: order.append("text-query") or "text-query")
    ready.set()                                          # the preload is done
    batch.join(2)
    assert order == ["text-query", "whisper"]


def test_a_family_that_is_not_pinned_starts_no_work_before_the_pins(monkeypatch):
    monkeypatch.setattr(registry, "PINNED_FAMILIES", {"text-query"})
    monkeypatch.setattr(registry, "_family_executors", {})
    ready = threading.Event()
    monkeypatch.setattr(registry, "pinned_ready", ready)

    ran = []
    batch = threading.Thread(target=lambda: registry.run_on_model_thread("whisper", lambda: ran.append("whisper")))
    batch.start()
    batch.join(0.2)
    assert ran == []                                     # not even its device probe
    assert registry.run_on_model_thread("text-query", lambda: "ok") == "ok"
    ready.set()
    batch.join(2)
    assert ran == ["whisper"]
