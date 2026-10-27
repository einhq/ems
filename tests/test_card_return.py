"""A family the CPU took for want of room on the card goes back to the card once it has room (ein/core#736)."""

import pytest

from ems.modelworker import model_registry as registry


def _card(monkeypatch, free_mb):
    """A card with `free_mb` free, EMS holding nothing else on it, and every look due."""
    monkeypatch.setattr(registry, "select_device", lambda: "cuda")
    monkeypatch.setattr(registry, "free_vram_mb", lambda: free_mb[0])
    monkeypatch.setattr(registry, "_own_card_mb", lambda: 0.0)
    monkeypatch.setattr(registry, "cleanup_torch", lambda: None)
    monkeypatch.setattr(registry, "_cache", {"insightface:cpu": object()})
    monkeypatch.setattr(registry, "_last_used", {"insightface:cpu": 0.0})
    monkeypatch.setattr(registry, "_card_return_at", {})
    monkeypatch.setattr(registry, "CARD_RETURN_SECONDS", 60.0)


def test_a_family_on_the_cpu_stays_there_while_the_card_has_no_room(monkeypatch):
    free = [600.0]
    _card(monkeypatch, free)
    assert registry.place("insightface") == "cpu"
    assert "insightface:cpu" in registry._cache


def test_a_family_on_the_cpu_goes_back_to_the_card_once_it_has_room(monkeypatch):
    free = [600.0]
    _card(monkeypatch, free)
    assert registry.place("insightface") == "cpu"
    free[0] = 9000.0
    registry._card_return_at["insightface"] -= 120     # the next look is due
    assert registry.place("insightface") == "cuda"
    assert "insightface:cpu" not in registry._cache   # its CPU copy is unloaded for the card's


def test_a_family_on_the_cpu_looks_again_only_once_a_while(monkeypatch):
    free = [600.0]
    _card(monkeypatch, free)
    assert registry.place("insightface") == "cpu"
    free[0] = 9000.0
    assert registry.place("insightface") == "cpu"     # the last look was just now
    assert "insightface:cpu" in registry._cache


def test_a_family_set_to_the_cpu_stays_there(monkeypatch):
    free = [9000.0]
    _card(monkeypatch, free)
    monkeypatch.setattr(registry, "_cache", {"ocr:cpu": object()})
    assert registry.place("ocr", "cpu") == "cpu"
    assert "ocr:cpu" in registry._cache


def test_whisper_makes_room_by_unloading_families_not_working_right_now(monkeypatch):
    # siglip-visual was used a moment ago, inside the grace, so waiting for idle families is not
    # enough; its lock is free, so whisper may unload it.
    free = [3000.0]
    monkeypatch.setattr(registry, "select_device", lambda: "cuda")
    monkeypatch.setattr(registry, "free_vram_mb", lambda: free[0])
    monkeypatch.setattr(registry, "_own_card_mb", lambda: 2600.0)
    monkeypatch.setattr(registry, "cleanup_torch", lambda: None)
    monkeypatch.setattr(registry, "_cache", {"siglip-visual:cuda": object()})
    monkeypatch.setattr(registry, "_last_used", {"siglip-visual:cuda": registry.time.time()})
    monkeypatch.setattr(registry, "_family_locks", {})

    def forget(keys):
        for key in keys:
            registry._cache.pop(key, None)
        free[0] += 2600.0

    monkeypatch.setattr(registry, "_forget", forget)
    assert registry.whisper_device() == "cuda"
    assert "siglip-visual:cuda" not in registry._cache


def test_whisper_leaves_a_family_that_is_working(monkeypatch):
    free = [3000.0]
    monkeypatch.setattr(registry, "select_device", lambda: "cuda")
    monkeypatch.setattr(registry, "free_vram_mb", lambda: free[0])
    monkeypatch.setattr(registry, "_own_card_mb", lambda: 2600.0)
    monkeypatch.setattr(registry, "cleanup_torch", lambda: None)
    monkeypatch.setattr(registry, "_cache", {"siglip-visual:cuda": object()})
    monkeypatch.setattr(registry, "_last_used", {"siglip-visual:cuda": registry.time.time()})
    lock = registry.FamilyLock()
    monkeypatch.setattr(registry, "_family_locks", {"siglip-visual": lock})
    with lock:
        with pytest.raises(registry.CardBusy):
            registry.whisper_device()
    assert "siglip-visual:cuda" in registry._cache
