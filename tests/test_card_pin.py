"""The card worker waits with its libraries imported and loads what `pin` names, the way it preloads at a start."""

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import model_registry as registry  # noqa: E402
import model_server as ms  # noqa: E402


def setup(monkeypatch, role="card"):
    loads: list[str] = []
    gate = threading.Event()

    def preload(stop, families):
        gate.wait(2)
        loads.extend(sorted(families))

    monkeypatch.setattr(ms, "ROLE", role)
    monkeypatch.setattr(ms, "CARD_IMPORTS", ())
    monkeypatch.setattr(ms, "_stop", threading.Event())
    monkeypatch.setattr(ms, "_preload_families", preload)
    monkeypatch.setattr(ms, "_preloads_pending", 0)
    monkeypatch.setattr(ms, "log", lambda message: None)
    monkeypatch.setattr(registry, "PINNED_FAMILIES", set())
    monkeypatch.setattr(registry, "pinned_ready", threading.Event())
    monkeypatch.setenv("CFG_MODEL_PINNED", "")
    return loads, gate


def test_pin_answers_at_once_and_loads_in_the_background(monkeypatch):
    loads, gate = setup(monkeypatch)
    assert ms.handle({"op": "pin", "families": ["text-query", "siglip-text"]}) == {"pinning": ["siglip-text", "text-query"]}
    assert loads == [] and not registry.pinned_ready.is_set()
    gate.set()
    assert registry.pinned_ready.wait(2)
    assert loads == ["siglip-text", "text-query"]
    # Pinned for good: never evicted, and preloaded again by a recycle's re-exec.
    assert registry.PINNED_FAMILIES == {"siglip-text", "text-query"}
    assert os.environ["CFG_MODEL_PINNED"] == "siglip-text,text-query"


def test_a_family_already_pinned_is_not_loaded_again(monkeypatch):
    loads, gate = setup(monkeypatch)
    gate.set()
    registry.PINNED_FAMILIES.add("text-query")
    registry.pinned_ready.set()
    assert ms.handle({"op": "pin", "families": ["text-query"]}) == {"pinning": []}
    assert loads == [] and registry.pinned_ready.is_set()


def test_the_pins_wait_for_the_startup_imports(monkeypatch):
    loads, gate = setup(monkeypatch)
    order: list[str] = []
    imported = threading.Event()
    monkeypatch.setattr(ms, "_import_card_libraries", lambda: (imported.wait(2), order.append("imports")))
    monkeypatch.setattr(ms, "_preload_families", lambda stop, families: order.extend(sorted(families)))
    ms._preload_started()
    startup = threading.Thread(target=ms._preload, args=(ms._stop, frozenset()))
    startup.start()
    ms.handle({"op": "pin", "families": ["text-query"]})
    imported.set()
    startup.join(2)
    assert registry.pinned_ready.wait(2)
    assert order == ["imports", "imports", "text-query"]


def test_only_the_card_worker_imports_ahead(monkeypatch):
    setup(monkeypatch, role="query")
    imported: list[bool] = []
    monkeypatch.setattr(ms, "_import_card_libraries", lambda: imported.append(True))
    ms._preload_started()
    ms._preload(ms._stop, frozenset())
    assert imported == [] and registry.pinned_ready.is_set()
