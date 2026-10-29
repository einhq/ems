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


def _crawl(monkeypatch, free_mb, total_mb=9700.0):
    """A card in a crawl: the photo engine and siglip-visual resident, both used a moment ago."""
    monkeypatch.setattr(registry, "select_device", lambda: "cuda")
    monkeypatch.setattr(registry, "free_vram_mb", lambda: free_mb[0])
    monkeypatch.setattr(registry, "card_total_mb", lambda: total_mb)
    monkeypatch.setattr(registry, "_own_card_mb", lambda: 2600.0)
    monkeypatch.setattr(registry, "cleanup_torch", lambda: None)
    monkeypatch.setattr(registry, "HTR_DEVICE", "auto")
    now = registry.time.time()
    monkeypatch.setattr(registry, "_cache", {"ocr:cuda": object(), "siglip-visual:cuda": object()})
    monkeypatch.setattr(registry, "_last_used", {"ocr:cuda": now, "siglip-visual:cuda": now})
    monkeypatch.setattr(registry, "_family_locks", {})

    def forget(keys):
        for key in keys:
            registry._cache.pop(key, None)
        free_mb[0] += 2600.0 * len(keys)

    monkeypatch.setattr(registry, "_forget", forget)


def test_htr_waits_for_the_card_rather_than_go_to_the_cpu(monkeypatch):
    free = [1200.0]
    _crawl(monkeypatch, free)
    lock = registry.FamilyLock()
    registry._family_locks["siglip-visual"] = lock
    with lock:
        with pytest.raises(registry.CardBusy, match="htr waits for the card"):
            registry.htr_device()
    # The photo engine is not working right now, but the next picture needs it first.
    assert "ocr:cuda" in registry._cache and "siglip-visual:cuda" in registry._cache


def test_htr_makes_room_by_unloading_families_not_working_right_now(monkeypatch):
    free = [1200.0]
    _crawl(monkeypatch, free)
    assert registry.htr_device() == "cuda"
    assert "siglip-visual:cuda" not in registry._cache and "ocr:cuda" in registry._cache


def test_htr_on_the_card_already_stays_there(monkeypatch):
    free = [0.0]
    _crawl(monkeypatch, free)
    registry._cache["htr:cuda"] = object()
    assert registry.htr_device() == "cuda"


def test_htr_runs_on_the_cpu_of_a_card_too_small_ever_to_hold_it(monkeypatch):
    free = [1200.0]
    _crawl(monkeypatch, free, total_mb=3000.0)
    assert registry.htr_device() == "cpu"


def test_htr_set_to_the_cpu_stays_there(monkeypatch):
    free = [9000.0]
    _crawl(monkeypatch, free)
    monkeypatch.setattr(registry, "HTR_DEVICE", "cpu")
    assert registry.htr_device() == "cpu"


def _crowded_card(monkeypatch):
    """A card whose room only an idle siglip-visual, used long ago, can give; unloading it frees 2600 MB."""
    free = [600.0]
    monkeypatch.setattr(registry, "select_device", lambda: "cuda")
    monkeypatch.setattr(registry, "free_vram_mb", lambda: free[0])
    monkeypatch.setattr(registry, "_own_card_mb", lambda: 2600.0)
    monkeypatch.setattr(registry, "cleanup_torch", lambda: None)
    monkeypatch.setattr(registry, "_cache", {"siglip-visual:cuda": object()})
    monkeypatch.setattr(registry, "_last_used", {"siglip-visual:cuda": 0.0})
    monkeypatch.setattr(registry, "_family_locks", {})

    def forget(keys):
        for key in keys:
            registry._cache.pop(key, None)
        free[0] += 2600.0

    monkeypatch.setattr(registry, "_forget", forget)


def test_documents_embed_on_the_cpu_rather_than_unload_another_family(monkeypatch):
    _crowded_card(monkeypatch)
    monkeypatch.setattr(registry, "TEXT_EMBED_DEVICE", "auto")
    assert registry.text_embed_device() == "cpu"
    assert "siglip-visual:cuda" in registry._cache


def test_another_family_still_unloads_an_idle_one_for_its_room(monkeypatch):
    _crowded_card(monkeypatch)
    assert registry.place("sscd") == "cuda"
    assert "siglip-visual:cuda" not in registry._cache


def test_a_family_in_use_without_its_lock_is_not_idle(monkeypatch):
    _crowded_card(monkeypatch)
    with registry.family_lock("siglip-visual").in_use():
        assert registry.evict_idle(0) == []
        assert registry.place("sscd") == "cpu"
    assert registry.evict_idle(0) == ["siglip-visual:cuda"]
