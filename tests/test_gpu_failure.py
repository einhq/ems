"""A lost CUDA context: the worker reports it in health and replaces itself, at most once per grace period."""

import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ems", "modelworker"))

import model_server as ms  # noqa: E402

# A lost context as torch and CTranslate2 report it.
TORCH_TIMEOUT = (
    "CUDA error: the launch timed out and was terminated\n"
    "Search for `cudaErrorLaunchTimeout' in https://docs.nvidia.com/cuda/ for more information.\n"
    "The CUDA driver logged these messages, which may provide useful details:\nSticky error detected"
)
WHISPER_TIMEOUT = "CUDA failed with error the launch timed out and was terminated"


class FakeServer:
    def __init__(self):
        self.shut = threading.Event()

    def shutdown(self):
        self.shut.set()


@pytest.fixture
def worker(monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(ms, "_stop", threading.Event())
    monkeypatch.setattr(ms, "_server", server)
    monkeypatch.setattr(ms, "_gpu_failure", None)
    monkeypatch.setattr(ms, "_recycling", False)
    monkeypatch.setattr(ms, "_recycle_reason", "rss")
    monkeypatch.setattr(ms, "RECYCLED_FOR", "")
    monkeypatch.setattr(ms, "log", lambda message: None)
    return server


def test_sticky_errors_are_told_from_running_out_of_memory():
    assert ms.is_sticky_cuda_error(RuntimeError(TORCH_TIMEOUT))
    assert ms.is_sticky_cuda_error(RuntimeError(WHISPER_TIMEOUT))
    assert ms.is_sticky_cuda_error(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert not ms.is_sticky_cuda_error(RuntimeError("CUDA out of memory. Tried to allocate 286.00 MiB"))
    assert not ms.is_sticky_cuda_error(RuntimeError("CUDA failed with error out of memory"))
    assert not ms.is_sticky_cuda_error(FileNotFoundError("image not found: /x.jpg"))


def test_a_lost_context_drains_and_reexecs_once(worker):
    ms._on_sticky_cuda_error("transcribe", RuntimeError(TORCH_TIMEOUT))
    assert worker.shut.wait(1)
    assert ms._recycling and ms._stop.is_set() and ms._recycle_reason == "gpu"
    assert ms._gpu_failure["op"] == "transcribe"
    assert ms._gpu_failure["error"] == "CUDA error: the launch timed out and was terminated"

    # The requests still in flight fail the same way; they do not start more recycles.
    first = dict(ms._gpu_failure)
    ms._on_sticky_cuda_error("transcribe", RuntimeError(WHISPER_TIMEOUT))
    assert ms._gpu_failure == first


def test_health_reports_the_failure(worker):
    ms._on_sticky_cuda_error("transcribe", RuntimeError(WHISPER_TIMEOUT))
    health = ms.handle({"op": "health"})
    assert health["gpu_failure"]["error"] == WHISPER_TIMEOUT
    assert health["gpu_failure"]["op"] == "transcribe"


def test_lost_again_soon_after_a_recycle_waits_out_the_grace(worker, monkeypatch):
    monkeypatch.setattr(ms, "RECYCLED_FOR", "gpu")
    scheduled = []

    class Timer:
        def __init__(self, wait, fn):
            scheduled.append((wait, fn))

        def start(self):
            pass

    monkeypatch.setattr(ms.threading, "Timer", Timer)
    ms._on_sticky_cuda_error("transcribe", RuntimeError(TORCH_TIMEOUT))
    assert not worker.shut.is_set() and not ms._recycling
    assert ms.handle({"op": "health"})["gpu_failure"]["op"] == "transcribe"

    [(wait, fn)] = scheduled
    assert 0 < wait <= ms.GPU_RECYCLE_GRACE_SECONDS
    fn()
    assert worker.shut.wait(1) and ms._recycle_reason == "gpu"
