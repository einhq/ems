"""`python3 -m ems.healthcheck` passes on an ok health answer, sends the key, and fails otherwise."""

import io
import json
import urllib.request

from ems import healthcheck


class _Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, body=None, error=None):
    seen = {}

    def urlopen(request, timeout):
        seen["url"], seen["auth"] = request.full_url, request.get_header("Authorization")
        if error is not None:
            raise error
        return _Answer(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return seen


def test_an_ok_answer_is_healthy_and_carries_the_key(monkeypatch):
    monkeypatch.setenv("EMS_KEY", "k1")
    monkeypatch.setenv("EMS_KEYS_FILE", "/nonexistent")
    monkeypatch.setenv("EMS_PORT", "9999")
    seen = _serve(monkeypatch, {"ok": True, "worker": {"running": True}})
    assert healthcheck.main() == 0
    assert seen == {"url": "http://127.0.0.1:9999/v0/health", "auth": "Bearer k1"}


def test_no_key_sends_no_header(monkeypatch):
    monkeypatch.delenv("EMS_KEY", raising=False)
    monkeypatch.setenv("EMS_KEYS_FILE", "/nonexistent")
    seen = _serve(monkeypatch, {"ok": True, "worker": {"running": True}})
    assert healthcheck.main() == 0 and seen["auth"] is None


def test_a_refused_or_failed_answer_is_unhealthy(monkeypatch):
    monkeypatch.setenv("EMS_KEYS_FILE", "/nonexistent")
    _serve(monkeypatch, error=ConnectionRefusedError("refused"))
    assert healthcheck.main() == 1
    _serve(monkeypatch, {"ok": False})
    assert healthcheck.main() == 1


def test_a_front_without_its_batch_worker_is_unhealthy(monkeypatch):
    monkeypatch.delenv("EMS_KEY", raising=False)
    monkeypatch.setenv("EMS_KEYS_FILE", "/nonexistent")
    _serve(monkeypatch, {"ok": True, "worker": {"running": False, "restarts": 3}})
    assert healthcheck.main() == 1
