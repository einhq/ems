"""Envelopes, auth, validation and the system endpoints, through the ASGI app."""

import dataclasses


async def test_unauthorized_without_key(client):
    r = await client.get("/v0/health", headers={"Authorization": ""})
    assert r.status_code == 401
    assert r.json() == {"ok": False, "error": {"code": "unauthorized", "message": "missing or unknown key"}}


async def test_capabilities_lists_ops(client, tmp_path, monkeypatch):
    monkeypatch.setenv("EMS_OCR_HAND_ROOT", str(tmp_path / "absent"))
    monkeypatch.setenv("EMS_HTR_ROOT", str(tmp_path / "absent"))
    monkeypatch.setenv("EMS_HTR_EN_ROOT", str(tmp_path / "absent"))
    r = await client.get("/v0/capabilities")
    body = r.json()
    assert r.status_code == 200
    assert "faces.detect" in body["ops"] and "media.probe" in body["ops"]
    assert "roots" not in body
    assert body["api"] == "0.3.7"
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


async def test_a_request_that_names_files_without_their_endpoint_is_invalid(client):
    r = await client.post("/v0/faces/detect", json={"file": {"path": "/etc/hostname"}}, headers={"EMS-Files": ""})
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["code"] == "invalid_request" and "EMS-Files" in error["message"] or "ems-files" in error["message"]


async def test_health_reports_worker_and_queues(client, fake_worker):
    fake_worker.on("health", lambda req: {"loaded": ["text-query:cuda", "faces"], "uptime": 5, "rss_mb": 100, "max_rss_mb": 200, "recycles": 1})
    r = await client.get("/v0/health")
    body = r.json()
    assert body["ok"] and body["worker"]["running"] is True
    assert body["models"] == {"text-query": {"loaded": True, "device": "cuda"}, "faces": {"loaded": True}}
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
    assert r.json()["signatures"] == {"transcribe": "whisper-large-v3:vad=0.2/400/500,nospeech=0.6,lang=auto,words=0"}


async def test_capabilities_leaves_the_signature_out_until_the_worker_has_given_one(client, fake_worker):
    await fake_worker.stop()
    r = await client.get("/v0/capabilities")
    assert r.json()["signatures"] == {}


async def test_capabilities_does_not_wait_for_a_loading_worker(client, fake_worker, monkeypatch):
    import asyncio
    import time

    from ems.ops import system

    monkeypatch.setattr(system, "HEALTH_WAIT_SECONDS", 0.2)
    fake_worker.on("transcribe_signature", lambda req: {"signature": "first"})
    assert (await client.get("/v0/capabilities")).json()["signatures"] == {"transcribe": "first"}

    async def loading(req):
        await asyncio.sleep(5)
        return {"signature": "second"}

    fake_worker.on("transcribe_signature", loading)
    started = time.monotonic()
    r = await client.get("/v0/capabilities")
    assert time.monotonic() - started < 2
    assert r.json()["signatures"] == {"transcribe": "first"}


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


async def test_capabilities_say_which_instance_this_is(client):
    from ems.ops import system

    body = (await client.get("/v0/capabilities")).json()
    assert body["name"] == client.app.state.settings.name and body["name"]
    assert body["build"] == system.BUILD
    assert set(body["gpu"]) <= {"present", "name", "memory_mb"} and isinstance(body["gpu"]["present"], bool)


def test_the_gpu_summary_names_the_card_and_its_memory():
    from ems.ops import system

    assert system._gpu_summary({"present": False}) == {"present": False}
    card = {"present": True, "name": "NVIDIA GeForce RTX 5060 Ti", "mem_total_mb": 16311, "util_pct": 3}
    assert system._gpu_summary(card) == {"present": True, "name": "NVIDIA GeForce RTX 5060 Ti", "memory_mb": 16311}


def test_ems_ops_is_a_list_of_globs_and_everything_by_default():
    from ems.config import Settings

    assert Settings.from_env({}).ops == ("*",) and Settings.from_env({}).serves("speech.transcribe")
    assert Settings.from_env({"EMS_NAME": " laptop "}).name == "laptop"
    settings = Settings.from_env({"EMS_OPS": "media.*, vision.embed,"})
    assert settings.ops == ("media.*", "vision.embed")
    assert settings.serves("media.probe") and settings.serves("vision.embed")
    assert not settings.serves("vision.embed_text") and not settings.serves("speech.transcribe")


async def test_an_op_outside_ems_ops_is_not_served(client):
    client.app.state.settings = dataclasses.replace(client.app.state.settings, ops=("media.*", "text.embed"))
    client.app.state.ctx.settings = client.app.state.settings
    r = await client.post("/v0/faces/detect", json={"file": {"path": "/x"}})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "op_not_served" and r.json()["error"]["permanent"] is True
    # Before the body is looked at: a request that would be invalid is still refused as not served.
    r = await client.post("/v0/faces/detect", json={"bogus": 1})
    assert r.json()["error"]["code"] == "op_not_served"
    ops = (await client.get("/v0/capabilities")).json()["ops"]
    assert "media.probe" in ops and "text.embed" in ops and "faces.detect" not in ops
    assert (await client.get("/v0/health")).status_code == 200
    # The key is checked first.
    r = await client.post("/v0/faces/detect", json={}, headers={"Authorization": ""})
    assert r.status_code == 401


async def test_a_glob_that_matches_no_op_is_a_warning(dirs, fake_worker, tmp_path):
    from ems.app import create_app
    from ems.config import Settings
    from ems.worker import Worker

    settings = Settings.from_env({"EMS_KEY": "k", "EMS_SCRATCH": str(tmp_path / "s"), "EMS_OPS": "media.*,vision.embedd"})
    app = create_app(settings, worker=Worker("/nonexistent", fake_worker.socket_path, log=lambda m: None), start_worker=False)
    assert app.state.ctx.warnings == ["EMS_OPS names vision.embedd, which matches no op"]
