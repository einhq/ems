"""Envelopes, auth, validation and the system endpoints, through the ASGI app."""


async def test_unauthorized_without_key(client):
    r = await client.get("/v0/health", headers={"Authorization": ""})
    assert r.status_code == 401
    assert r.json() == {"ok": False, "error": {"code": "unauthorized", "message": "missing or unknown key"}}


async def test_capabilities_lists_ops_and_roots(client, tmp_path, monkeypatch):
    monkeypatch.setenv("EMS_OCR_HAND_ROOT", str(tmp_path / "absent"))
    monkeypatch.setenv("EMS_HTR_ROOT", str(tmp_path / "absent"))
    monkeypatch.setenv("EMS_HTR_EN_ROOT", str(tmp_path / "absent"))
    r = await client.get("/v0/capabilities")
    body = r.json()
    assert r.status_code == 200
    assert "faces.detect" in body["ops"] and "media.probe" in body["ops"]
    assert [root["mode"] for root in body["roots"]] == ["ro", "rw"]
    assert body["api"] == "0.1.0"
    assert "video.storyboard" in body["ops"] and "video.shots" in body["ops"]
    assert "ocr.read" in body["ops"] and body["models"]["ocr"] == "PP-OCRv6_medium"
    assert "vision.describe" not in body["ops"] and "text.generate" not in body["ops"] and "vlm" not in body["models"]


async def test_capabilities_names_the_handwriting_readers_where_the_image_has_them(client, tmp_path, monkeypatch):
    (tmp_path / "multicentury-ppocrv6-medium").mkdir()
    monkeypatch.setenv("EMS_OCR_HAND_ROOT", str(tmp_path / "multicentury-ppocrv6-medium"))
    monkeypatch.setenv("EMS_HTR_ROOT", str(tmp_path / "multicentury-htr-model"))
    monkeypatch.setenv("EMS_HTR_EN_ROOT", str(tmp_path / "trocr-large-handwritten"))
    assert (await client.get("/v0/capabilities")).json()["models"]["ocr"] == "PP-OCRv6_medium+multicentury-ppocrv6-medium"
    (tmp_path / "multicentury-htr-model").mkdir()
    body = (await client.get("/v0/capabilities")).json()
    assert body["models"]["ocr"] == "PP-OCRv6_medium+multicentury-ppocrv6-medium+multicentury-htr-model"
    (tmp_path / "trocr-large-handwritten").mkdir()
    body = (await client.get("/v0/capabilities")).json()
    assert body["models"]["ocr"] == "PP-OCRv6_medium+multicentury-ppocrv6-medium+multicentury-htr-model+trocr-large-handwritten"


async def test_validation_error_uses_the_envelope(client):
    r = await client.post("/v0/faces/detect", json={"file": {"path": "/x", "angle": 45}})
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False and body["error"]["code"] == "invalid_request" and body["error"]["permanent"] is True
    assert "angle" in body["error"]["message"]


async def test_unknown_fields_are_refused(client):
    r = await client.post("/v0/text/embed", json={"text": "hi", "bogus": 1})
    assert r.status_code == 400
    assert "bogus" in r.json()["error"]["message"]


async def test_path_outside_roots(client):
    r = await client.post("/v0/faces/detect", json={"file": {"path": "/etc/hostname"}})
    assert r.status_code == 403
    assert r.json()["error"] == {"code": "path_outside_roots", "message": "/etc/hostname: outside the configured roots", "permanent": True}


async def test_health_reports_worker_and_queues(client, fake_worker):
    fake_worker.on("health", lambda req: {"loaded": ["text-query:cuda", "insightface"], "uptime": 5, "rss_mb": 100, "max_rss_mb": 200, "recycles": 1})
    r = await client.get("/v0/health")
    body = r.json()
    assert body["ok"] and body["worker"]["running"] is True
    assert body["models"] == {"text-query": {"loaded": True, "device": "cuda"}, "insightface": {"loaded": True}}
    assert body["process"]["recycles"] == 1
    assert body["degraded"] == []
    assert body["queues"]["models"]["limit"] == 2


async def test_health_reports_a_lost_gpu_context_as_degraded(client, fake_worker):
    fake_worker.on("health", lambda req: {"loaded": ["whisper:cuda:float16"], "gpu_failure": {"since": 2, "op": "transcribe", "error": "CUDA error: the launch timed out and was terminated"}})
    r = await client.get("/v0/health")
    assert r.json()["degraded"] == [{"what": "gpu", "reason": "CUDA error: the launch timed out and was terminated", "since": 2}]


async def test_capabilities_carries_the_transcribe_signature(client, fake_worker):
    fake_worker.on("transcribe_signature", lambda req: {"signature": "whisper-large-v3:vad=0.2/400/500,nospeech=0.6,lang=auto,words=0"})
    r = await client.get("/v0/capabilities")
    assert r.json()["signatures"] == {"transcribe": "whisper-large-v3:vad=0.2/400/500,nospeech=0.6,lang=auto,words=0"}
    await fake_worker.stop()
    r = await client.get("/v0/capabilities")
    assert r.json()["signatures"] == {}


def test_parse_smi_reads_the_first_card_and_refuses_a_card_that_does_not_answer():
    from ems.ops import system

    assert system.parse_smi("37, 8123, 16311, 51\n") == {"util_pct": 37, "mem_used_mb": 8123, "mem_total_mb": 16311, "temp_c": 51}
    assert system.parse_smi("37, 8123, 16311, 51\n0, 10, 16311, 40\n")["util_pct"] == 37
    assert system.parse_smi("[N/A], 8123, 16311, 51\n") is None
    assert system.parse_smi("") is None
    assert system.parse_smi("37, 8123\n") is None
    # Power a board does not report is left out, the rest still read.
    assert system.parse_smi("37, 8123, 16311, 51, [N/A], [N/A], 610.43.02, A Card\n") == {
        "util_pct": 37, "mem_used_mb": 8123, "mem_total_mb": 16311, "temp_c": 51, "driver": "610.43.02", "name": "A Card"}
    assert system.parse_smi_apps("49, 1810\nnonsense\n") == [{"pid": 49, "used_mb": 1810}]


async def test_health_reports_the_whole_card_when_nvidia_smi_answers(client, monkeypatch):
    from ems.ops import system

    monkeypatch.setattr(system.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(system.os, "listdir", lambda path: ["nvidia0", "nvidiactl"])
    monkeypatch.setattr(system, "_smi_cache", {"at": 0.0, "stats": None})
    monkeypatch.setattr(system, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(system.nvml, "load", lambda: None)

    class Done:
        code = 0

        def __init__(self, stdout: bytes):
            self.stdout = stdout

    calls = []

    async def fake_run(args, *, deadline, **kwargs):
        calls.append(args)
        if args[1].startswith("--query-compute-apps"):
            return Done(b"49, 1810\n77, 300\n")
        return Done(b"37, 8123, 16311, 51, 21.62, 180.00, 610.43.02, NVIDIA GeForce RTX 5060 Ti\n")

    monkeypatch.setattr(system, "run", fake_run)
    client.app.state.ctx.worker.process = type("P", (), {"pid": 49, "returncode": None})()
    r = await client.get("/v0/health")
    assert r.json()["gpu"] == {
        "present": True, "util_pct": 37, "mem_used_mb": 8123, "mem_total_mb": 16311, "temp_c": 51,
        "power_w": 21.6, "power_limit_w": 180.0, "driver": "610.43.02", "name": "NVIDIA GeForce RTX 5060 Ti",
        # The batch worker by pid; a pid this EMS does not own stays unnamed.
        "processes": [{"pid": 49, "used_mb": 1810, "worker": "batch"}, {"pid": 77, "used_mb": 300}],
    }
    assert calls[0][1:] == [f"--query-gpu={system.SMI_QUERY}", "--format=csv,noheader,nounits"]
    assert calls[1][1:] == ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]

    # A second request within the cache window is answered from the first reading.
    r = await client.get("/v0/health")
    assert r.json()["gpu"]["util_pct"] == 37 and len(calls) == 2


async def test_health_reports_the_card_load_as_nvml_averages_it(client, monkeypatch):
    from ems.ops import system

    monkeypatch.setattr(system.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(system.os, "listdir", lambda path: ["nvidia0"])
    monkeypatch.setattr(system, "_smi_cache", {"at": 0.0, "stats": None})
    monkeypatch.setattr(system, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(system.nvml, "load", lambda: {"util_pct": 61.5, "dec_pct": 22.0, "enc_pct": 0.0})

    class Done:
        code = 0

        def __init__(self, stdout: bytes):
            self.stdout = stdout

    async def fake_run(args, *, deadline, **kwargs):
        return Done(b"" if args[1].startswith("--query-compute-apps") else b"3, 8123, 16311, 51\n")

    monkeypatch.setattr(system, "run", fake_run)
    gpu = (await client.get("/v0/health")).json()["gpu"]
    assert (gpu["util_pct"], gpu["dec_pct"], gpu["enc_pct"], gpu["mem_used_mb"]) == (61.5, 22.0, 0.0, 8123)


def test_nvml_mean_covers_only_the_window():
    from ems.tools import nvml

    samples = [(1_000_000, 100), (2_000_000, 40), (2_200_000, 20)]
    assert nvml.mean_since(samples, 2_000_000) == 30.0
    assert nvml.mean_since(samples, 3_000_000) is None
    assert nvml.mean_since([], 0) is None


async def test_health_without_nvidia_smi_reports_presence_alone(client, monkeypatch):
    from ems.ops import system

    monkeypatch.setattr(system, "_smi_cache", {"at": 0.0, "stats": None})
    monkeypatch.setattr(system, "which", lambda name: None)
    r = await client.get("/v0/health")
    assert set(r.json()["gpu"]) == {"present"}
