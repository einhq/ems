"""A CUDA failure in the transcriber unloads its copy on the card and asks again later (ein/core#753)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import inference_ops as ops  # noqa: E402
import model_registry as registry  # noqa: E402


class Transcriber:
    def __init__(self, error):
        self.error = error

    def read(self, crops):
        if self.error:
            raise self.error
        return ["text"] * len(crops)


@pytest.fixture
def card(monkeypatch):
    """A transcriber on `device` that raises `error`; what was forgotten is recorded."""
    state = {"device": "cuda", "error": None, "forgotten": []}
    monkeypatch.setattr(registry, "htr_device", lambda family: state["device"])
    monkeypatch.setattr(registry, "htr", lambda device, family: Transcriber(state["error"]))
    monkeypatch.setattr(registry, "forget_copies", lambda family, device: state["forgotten"].append((family, device)))
    monkeypatch.setattr(registry, "cleanup_torch", lambda: None)
    return state


def test_lines_are_read_on_the_card(card):
    assert ops._transcribe_lines("htr", ["a", "b"]) == ["text", "text"]
    assert card["forgotten"] == []


@pytest.mark.parametrize("message", ["CUDA error: out of memory", "CUDA error: unknown error", "CUDA out of memory. Tried to allocate 20.00 MiB"])
def test_a_cuda_failure_unloads_the_copy_and_asks_again_later(card, message):
    card["error"] = RuntimeError(message)
    with pytest.raises(registry.CardBusy, match=message.split(".")[0]):
        ops._transcribe_lines("htr", ["a"])
    assert card["forgotten"] == [("htr", "cuda")]


def test_another_failure_is_the_request_s_own(card):
    card["error"] = ValueError("bad crop")
    with pytest.raises(ValueError):
        ops._transcribe_lines("htr", ["a"])
    assert card["forgotten"] == []


def test_on_the_cpu_a_failure_is_not_the_card_s(card):
    card["device"] = "cpu"
    card["error"] = RuntimeError("out of memory")
    with pytest.raises(RuntimeError):
        ops._transcribe_lines("htr", ["a"])
    assert card["forgotten"] == []
