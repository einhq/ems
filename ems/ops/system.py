"""`GET /v0/health` and `GET /v0/capabilities`."""

from __future__ import annotations

import asyncio
import os
import time
import tomllib
from pathlib import Path

from fastapi import APIRouter

from .. import API_VERSION
from ..context import Context
from ..errors import EmsError
from ..tools import encode, nvml
from ..tools.run import run, which
from ..tools.versions import tool_version
from ..worker import Worker
from . import transcode

TOOLS = ("exiftool", "ffprobe", "ffmpeg", "fpcalc", "magick")


# The whole card, as `nvidia-smi` reports it: what every process on the card is doing, not what
# this one's torch has allocated. A caller's metrics sampler polls it; `nvidia-smi` costs tens
# of milliseconds and a fork, so one reading serves every health request for a few seconds
# rather than each request paying for its own. Its load figures are replaced by NVML's means over
# the last seconds where the driver keeps samples (tools/nvml.py).
SMI_QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,driver_version,name"
SMI_FIELDS = ("util_pct", "mem_used_mb", "mem_total_mb", "temp_c")
SMI_TIMEOUT = 5.0
SMI_CACHE_SECONDS = 5.0
_smi_cache: dict = {"at": 0.0, "stats": None}


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_smi(text: str) -> dict | None:
    """The first card of a `--format=csv,noheader,nounits` answer, or None for anything else.

    The four numbers a chart needs are required; power, the driver and the card's name are read
    when they are there and skipped when the card says `[N/A]` for them (power on some boards).
    """
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < len(SMI_FIELDS):
        return None
    try:
        values = [int(float(part)) for part in parts[: len(SMI_FIELDS)]]
    except ValueError:
        # "[N/A]" or "[Not Supported]" — a card, or a driver, that does not answer this query.
        return None
    stats: dict = dict(zip(SMI_FIELDS, values))
    extra = parts[len(SMI_FIELDS):]
    for key, raw in zip(("power_w", "power_limit_w"), extra[:2]):
        value = _number(raw)
        if value is not None:
            stats[key] = round(value, 1)
    if len(extra) > 2 and extra[2] and not extra[2].startswith("["):
        stats["driver"] = extra[2]
    if len(extra) > 3:
        stats["name"] = ", ".join(extra[3:])
    return stats


def parse_smi_apps(text: str) -> list[dict]:
    """`--query-compute-apps=pid,used_memory` as `[{pid, used_mb}]`, skipping lines it cannot read."""
    apps = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            apps.append({"pid": int(parts[0]), "used_mb": int(float(parts[1]))})
        except ValueError:
            continue
    return apps


async def _smi(smi: str, query: str) -> str | None:
    try:
        done = await run([smi, query, "--format=csv,noheader,nounits"], deadline=SMI_TIMEOUT)
    except (asyncio.TimeoutError, OSError):
        return None
    return done.stdout.decode(errors="replace") if done.code == 0 else None


async def _card_stats(fresh: bool = False) -> dict | None:
    """The first card as `nvidia-smi` reports it, its load as NVML averages it, or None without
    `nvidia-smi`.

    `processes` is what this container's processes hold of it — `nvidia-smi` in a container sees
    only its own pids — so the whole card less their sum is other containers', which is why the
    card and the models need not add up.
    """
    now = time.monotonic()
    if not fresh and now - _smi_cache["at"] < SMI_CACHE_SECONDS:
        return _smi_cache["stats"]
    stats = None
    smi = which("nvidia-smi")
    if smi:
        text = await _smi(smi, f"--query-gpu={SMI_QUERY}")
        stats = parse_smi(text) if text is not None else None
        if stats is not None:
            stats.update(await asyncio.to_thread(nvml.load) or {})
            apps = await _smi(smi, "--query-compute-apps=pid,used_memory")
            if apps is not None:
                stats["processes"] = parse_smi_apps(apps)
    _smi_cache.update(at=now, stats=stats)
    return stats


async def card_stats() -> dict | None:
    """The card as `nvidia-smi` sees it now, or None without one — what the CardArbiter decides from."""
    return await _card_stats(fresh=True)


async def _gpu(workers: dict[str, int | None]) -> dict:
    """The card, with each of its processes named after the worker whose pid it is."""
    present = bool([d for d in os.listdir("/dev") if d.startswith("nvidia")]) if os.path.isdir("/dev") else False
    gpu: dict = {"present": present}
    stats = await _card_stats() if present else None
    if stats:
        gpu.update({key: value for key, value in stats.items() if key != "processes"})
        if "processes" in stats:
            names = {pid: name for name, pid in workers.items() if pid is not None}
            gpu["processes"] = [{**app, **({"worker": names[app["pid"]]} if app["pid"] in names else {})} for app in stats["processes"]]
    return gpu


def _models_from_worker(health: dict | None) -> dict:
    """The worker reports `loaded` as `family[:device…]` keys; group them by family."""
    models: dict[str, dict] = {}
    for key in (health or {}).get("loaded") or []:
        family, _, rest = str(key).partition(":")
        device = rest.split(":", 1)[0] if rest else None
        entry = models.setdefault(family, {"loaded": True})
        if device:
            entry["device"] = device
    return models


def _families_from_workers(batch: dict | None, query: dict | None, card: dict | None = None) -> dict:
    """The workers' `families`, one entry per family, each copy marked with the worker holding it.

    The query worker's settings win for what it is configured with: it is started with its own
    devices (Context.for_queries) and holds the pins for its families, which the batch worker
    leaves out of its own. The card worker (ems/card.py) adds its copies and nothing else.
    """
    families: dict[str, dict] = {}
    for worker, health in (("batch", batch), ("query", query), ("card", card)):
        for name, entry in ((health or {}).get("families") or {}).items():
            copies = [{**copy, "worker": worker} for copy in entry.get("copies") or []]
            have = families.get(name)
            if have is None:
                families[name] = {**entry, "copies": copies}
                continue
            have["copies"].extend(copies)
            have["pinned"] = bool(have.get("pinned") or entry.get("pinned"))
            have["running"] = bool(have.get("running") or entry.get("running"))
            have["waiting"] = (have.get("waiting") or 0) + (entry.get("waiting") or 0)
            if worker == "query" and entry.get("pinned"):
                have["device"] = entry.get("device")
    return families


# What this EMS is, beyond the API it speaks: the version in pyproject.toml (the image copies the
# file but does not install the package), and the commit the image was built from when the build
# left one (`ems/BUILD`, written by `make build` or whatever synced the checkout, as `<commit> <commit
# date>`). Read once; neither can change under a running process.
BUILD_FILE = Path(__file__).resolve().parents[1] / "BUILD"
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _build() -> dict:
    build: dict = {}
    try:
        build["version"] = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        pass
    try:
        commit, _, date = BUILD_FILE.read_text().strip().partition(" ")
        if commit:
            build["commit"] = commit
        if date:
            build["date"] = date
    except OSError:
        pass
    return build


BUILD = _build()


def ocr_model(env) -> str:
    """The text reader's producer as the model worker names it (model_registry.ocr_model): the
    photo engine's tier, then each later reader the image has."""
    tier = env.get("EMS_OCR_PHOTO_TIER") or env.get("CFG_OCR_PHOTO_TIER") or "medium"
    roots = (env.get("EMS_OCR_HAND_ROOT") or env.get("CFG_OCR_HAND_ROOT") or "/opt/ocr/multicentury-ppocrv6-medium",
             env.get("EMS_HTR_ROOT") or env.get("CFG_HTR_ROOT") or "/opt/htr/multicentury-htr-model",
             env.get("EMS_HTR_EN_ROOT") or env.get("CFG_HTR_EN_ROOT") or "/opt/htr/trocr-large-handwritten")
    return "+".join([f"PP-OCRv6_{tier}", *(os.path.basename(os.path.normpath(root)) for root in roots if os.path.isdir(root))])


# How long health waits for each model worker's own answer. A worker loading a model can hold its
# interpreter for longer than a container health check waits (htr on fry, 2026-10-27); it is then
# reported by what it last said, as up and not answering.
HEALTH_WAIT_SECONDS = 2.0


async def _no_answer() -> None:
    return None


def _worker_state(worker: Worker, reply: dict | None) -> tuple[dict | None, dict]:
    """One model worker as health shows it: its reply, or while it is up and silent its last one, and its state."""
    state: dict = {"running": reply is not None or worker.running, "restarts": worker.restarts}
    if reply is None and worker.running:
        state["answering"] = False
        if worker.answered_at is not None:
            state["silent_for"] = round(time.time() - worker.answered_at, 1)
        reply = worker.last_health
    return reply, state


def register(router: APIRouter, ctx: Context) -> None:
    @router.get("/health")
    async def health():
        # The workers and the card are asked at once, each with its own bound, so one that is busy
        # does not hold up the answer.
        query_worker = ctx.query_worker
        card_worker = ctx.card.card if ctx.card is not None and ctx.card.card is not None and ctx.card.card.running else None
        worker, query, card, gpu = await asyncio.gather(
            ctx.worker.health(HEALTH_WAIT_SECONDS),
            query_worker.health(HEALTH_WAIT_SECONDS) if query_worker is not None and query_worker.running else _no_answer(),
            card_worker.health(HEALTH_WAIT_SECONDS) if card_worker is not None else _no_answer(),
            _gpu({
                "batch": ctx.worker.pid,
                "query": ctx.query_worker.pid if ctx.query_worker is not None else None,
                "card": ctx.card.card.pid if ctx.card is not None and ctx.card.card is not None else None,
            }),
        )
        worker, state = _worker_state(ctx.worker, worker)
        body: dict = {
            "ok": True,
            "api": API_VERSION,
            "uptime": round(time.time() - ctx.started_at, 1),
            "gpu": gpu,
            "models": _models_from_worker(worker),
            "worker": state,
            "queues": ctx.admission.snapshot(),
            "requests": ctx.admission.requests(),
            "build": BUILD,
        }
        if query_worker is not None:
            # The query families live in a worker of their own (Context.for_queries); what it has
            # loaded is listed with the rest, and its state beside the batch worker's.
            query, body["worker"]["query"] = _worker_state(query_worker, query)
            body["models"].update(_models_from_worker(query))
        if ctx.card is not None:
            body["card"] = ctx.card.snapshot()
        body["families"] = _families_from_workers(worker, query, card)
        if ctx.card is not None and ctx.card.enabled:
            # The query families' device is the query worker's setting; where each family runs in
            # each mode is the arbiter's (`card.plan`).
            stats = await _card_stats() or {}
            card_now = {**stats, "processes": body["gpu"].get("processes") or []}
            body["card"]["plan"] = ctx.card.plan(body["families"], card_now)
        if worker:
            process: dict = {}
            for src, dst in (("rss_mb", "rss_mb"), ("max_rss_mb", "max_rss_mb"), ("recycles", "recycles"), ("uptime", "uptime")):
                if src in worker:
                    process[dst] = worker[src]
            if query:
                # The query worker's own memory, under its own ceiling: its models are on the CPU,
                # so this is where they show.
                process["query"] = {k: query[k] for k in ("rss_mb", "max_rss_mb", "recycles", "uptime") if k in query}
            body["process"] = process
            if "vram_allocated_mb" in worker or "vram_reserved_mb" in worker:
                body["gpu"].update({k: worker[k] for k in ("vram_allocated_mb", "vram_reserved_mb") if k in worker})
            degraded = []
            if ctx.settings.expect_gpu and not body["gpu"].get("present"):
                degraded.append({"what": "gpu", "reason": "this host is meant to have a card, and none is visible"})
            gpu_failure = worker.get("gpu_failure")
            if gpu_failure:
                # Every model op on the card fails until the worker has replaced itself.
                degraded.append({"what": "gpu", "reason": gpu_failure.get("error", "cuda context lost"), "since": gpu_failure.get("since")})
            body["degraded"] = degraded
        body["audio_cache"] = ctx.audio.snapshot()
        if ctx.warnings:
            body["warnings"] = list(ctx.warnings)
        return body

    @router.get("/capabilities")
    async def capabilities():
        tools = {name: await tool_version(name) for name in TOOLS if which(name)}
        env = os.environ
        # What a transcript decoded right now would be stamped with: everything that decides
        # `speech.transcribe`'s output, as one string, so a caller can find transcripts made
        # with other settings. Costs no model load; absent while the worker is down.
        signatures: dict = {}
        try:
            signature = await ctx.worker.call("transcribe_signature", {})
            if isinstance(signature, dict) and signature.get("signature"):
                signatures["transcribe"] = signature["signature"]
        except EmsError:
            pass
        return {
            "signatures": signatures,
            "api": API_VERSION,
            "ops": [route.path.removeprefix("/v0/").replace("/", ".") for route in router.routes if "POST" in getattr(route, "methods", set())],
            "roots": ctx.roots.describe(),
            "tools": tools,
            "models": {
                # What text.embed writes as its producer; a change is a re-embed of every stored
                # text vector for the caller.
                "text_embed": (env.get("EMS_TEXT_EMBED_MODEL") or env.get("CFG_TEXT_EMBED_MODEL") or "jinaai/jina-embeddings-v5-text-small").split("/")[-1],
                "whisper": env.get("EMS_WHISPER_MODEL") or env.get("CFG_WHISPER_MODEL") or "large-v3",
                "faces": "insightface/buffalo_l",
                # What vision.embed writes as its producer; a change is a re-embed for the caller.
                "image_embed": env.get("EMS_IMAGE_EMBED_MODEL") or env.get("CFG_IMAGE_EMBED_MODEL") or "ViT-SO400M-16-SigLIP2-384__webli",
                # What ocr.read writes as its producer; a change is a re-read for the caller.
                "ocr": ocr_model(env),
            },
            "limits": {
                "models": ctx.settings.limit_models,
                "tools": ctx.settings.limit_tools,
                "encodes": ctx.settings.limit_encodes,
                "queue_depth": ctx.settings.queue_depth,
                "interactive_reserve": ctx.settings.interactive_reserve,
            },
            # What decides a transcode's fate on this box: the encoder that will run and the
            # CPU-decode budget a refusal is measured against. A caller records the budget's
            # signature beside a refusal, so a raised budget re-opens the gap on its own.
            "encode": {
                "video_encoder": await transcode.gpu_encoder(ctx) or encode.CPU_ENCODER,
                "cpu_budget": encode.Budget(
                    ctx.settings.cpu_encode_max_mb, ctx.settings.cpu_encode_mb_per_1k_frames, ctx.settings.cpu_encode_max_segments
                ).signature,
            },
        }
