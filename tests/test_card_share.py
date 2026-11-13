"""The card's share changes while EMS runs: what no longer fits it leaves the card (ein/core#573)."""

import os
import time

import pytest

from ems.modelworker import model_registry as registry


@pytest.fixture
def on_card(monkeypatch):
    """faces, whisper and siglip-visual on the card, least recently used first, claiming 8.4 GB
    with the CUDA context; nothing in use, and the whole card EMS's."""
    keys = {"faces:cuda": 1.0, "whisper:cuda:float16": 2.0, "siglip-visual:cuda": 3.0}
    monkeypatch.setattr(registry, "_cache", {key: object() for key in keys})
    monkeypatch.setattr(registry, "_last_used", dict(keys))
    monkeypatch.setattr(registry, "_family_locks", {})
    monkeypatch.setattr(registry, "CARD_NEED_MB", {"faces": 2000.0, "whisper": 4000.0, "siglip-visual": 2000.0})
    monkeypatch.setattr(registry, "cleanup_torch", lambda: None)
    monkeypatch.setattr(registry, "CARD_SHARE_MB", None)
    monkeypatch.setenv("CFG_CARD_SHARE_MB", "")
    return registry._cache


def test_a_lower_share_unloads_the_least_recently_used_until_the_rest_fits(on_card):
    registry.set_card_share(6500)
    assert registry.fit_card_share() == ["faces:cuda"]
    assert sorted(on_card) == ["siglip-visual:cuda", "whisper:cuda:float16"]
    # A recycle re-execs with the environment, and comes back with the share.
    assert os.environ["CFG_CARD_SHARE_MB"] == "6500"


def test_a_higher_share_unloads_nothing(on_card):
    registry.set_card_share(9000)
    assert registry.fit_card_share() == []
    registry.set_card_share(None)
    assert registry.fit_card_share() == [] and len(on_card) == 3
    assert os.environ["CFG_CARD_SHARE_MB"] == ""


def test_a_share_of_nothing_unloads_everything_and_asks_the_card_nothing(on_card, monkeypatch):
    monkeypatch.setattr(registry, "card_present", lambda: pytest.fail("a share of 0 asks the card nothing"))
    registry.set_card_share(0)
    assert sorted(registry.fit_card_share()) == ["faces:cuda", "siglip-visual:cuda", "whisper:cuda:float16"]
    assert registry.select_device() == "cpu"
    assert registry.place("faces") == "cpu"
    assert not registry.whisper_on_card()


def test_a_family_in_use_leaves_the_card_once_its_work_is_done(on_card):
    lock = registry.family_lock("whisper")
    lock.acquire()
    try:
        registry.set_card_share(2000)
        assert registry.fit_card_share() == ["faces:cuda", "siglip-visual:cuda"]
        assert "whisper:cuda:float16" in on_card
    finally:
        lock.release()
    deadline = time.monotonic() + 5
    while "whisper:cuda:float16" in on_card and time.monotonic() < deadline:
        time.sleep(0.01)
    assert on_card == {}


def test_whisper_is_on_the_card_while_the_share_holds_it_beside_a_context(monkeypatch):
    monkeypatch.setattr(registry, "card_present", lambda: True)
    monkeypatch.setattr(registry, "CARD_BUDGET_MB", 0.0)
    need = registry.CARD_NEED_MB["whisper"] + registry.CUDA_CONTEXT_MB
    for share, on_card in ((None, True), (need, True), (need - 1, False), (0, False)):
        monkeypatch.setattr(registry, "CARD_SHARE_MB", share)
        assert registry.whisper_on_card() is on_card, share
    # A card rehearsed as a smaller one counts the same way.
    monkeypatch.setattr(registry, "CARD_SHARE_MB", None)
    monkeypatch.setattr(registry, "CARD_BUDGET_MB", need - 1)
    assert not registry.whisper_on_card()
