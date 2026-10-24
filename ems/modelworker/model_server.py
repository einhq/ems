#!/usr/bin/env python3
"""
Resident model server.

Holds InsightFace, the text and picture embedders, whisper and the voice models loaded across
requests, so a request does not pay a model load.

Ops are model primitives — one model in, its raw output out. The front's ops compose them, so
nothing here needs to know what a media file is.

Protocol: newline-delimited JSON over a unix SOCK_STREAM socket, one request and one
response per connection.

  -> {"op":"transcribe","path":"/tmp/.../audio.wav"}
  -> {"op":"transcribe_signature"}
  -> {"op":"embed_texts","texts":["..."],"role":"query"|"document"}
  -> {"op":"diarize","path":"/tmp/.../audio.wav"}
  -> {"op":"embed_voice_segments","path":"/tmp/.../audio.wav","spans":[{"start":12.48,"end":17.84},...]}
  -> {"op":"active_speaker_score","video":"...","audio":"...","start":12.4,"end":17.8,"track":[...]}
  -> {"op":"active_speakers","video":"...","audio":"...","angle":90,"speakers":[{"id":"speaker_0","turns":[[3.2,9.8]]}]}
  -> {"op":"embed_images","paths":["/files/S1/01K..."],"angle":270,"mirror":false}
  -> {"op":"copy_descriptors","paths":["/files/S1/01K..."],"angle":270,"mirror":false}
  -> {"op":"read_text","paths":["/files/S1/01K..."],"angle":270,"mirror":false,"language":"en"}
  -> {"op":"shot_boundaries","path":"/scratch/…/frames.rgb","times":[0.0,0.04,…],"duration":180.5,"threshold":0.6,…}
  -> {"op":"embed_image_texts","texts":["segelbåt"]}
  -> {"op":"detect_faces","path":"/files/S1/01K...","angle":270,"mirror":false}
  -> {"op":"embed_face","path":"...","box":{"x":0.1,"y":0.2,"width":0.1,"height":0.1},"angle":270}
  -> {"op":"health"}

`angle`/`mirror` (optional, default 0/false) name the display frame an image op works in —
the caller's rotation, not the EXIF tag's. Face boxes are fractions of that frame.
  <- {"ok":true,"result":...}
  <- {"ok":false,"error":"..."}

Models load lazily on first use and are evicted after CFG_MODEL_IDLE_EVICT seconds
idle, so an audio-free crawl never pays for whisper and VRAM stays bounded. Inference serialises per family (see model_registry.family_lock).

Families in CFG_MODEL_PINNED (default: text-query) are exempt from eviction and are
preloaded at startup — they sit on the interactive search path, where a reload costs
tens of seconds and is far more noticeable than the VRAM it frees.

The server also holds itself to an RSS ceiling (CFG_MODEL_MAX_RSS, by default a
fraction of the cgroup limit). Crossing it logs the `memstats` line and the `op_counts`
breakdown, then drains and re-execs — see _rss_watchdog for why a leak must not be left
for the OOM killer to find.

A second covers a GPU context that is gone: after a sticky CUDA error every later call fails the
same way, and the server answers health while failing all of them. See _on_sticky_cuda_error.
"""

import json
import os
import select
import signal
import socketserver
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mem_diag  # noqa: E402
import model_registry as registry  # noqa: E402

SOCKET_PATH = os.environ.get('CFG_MODEL_SOCKET', '/tmp/ems-model-worker.sock')
EVICT_INTERVAL_SECONDS = 60

# How often the RSS watchdog samples. One read of /proc/self/statm, so the interval is set by
# how fast the ceiling can be crossed rather than by cost: a leak of gigabytes an hour is a few
# hundred MB per sample, well inside the headroom the default ceiling leaves.
RSS_CHECK_INTERVAL_SECONDS = float(os.environ.get('CFG_MODEL_RSS_INTERVAL', '30'))

# A cross inside this many seconds of startup does not recycle. A ceiling below what the
# pinned models legitimately need would otherwise re-exec forever, each time paying their
# reload and never getting under the line; the grace period bounds that loop to one restart
# per period and leaves the misconfiguration in the log instead of hiding it in a spin.
RSS_GRACE_SECONDS = float(os.environ.get('CFG_MODEL_RSS_GRACE', '600'))

# How long a recycle waits for in-flight requests before replacing the process anyway. A
# transcription can legitimately take minutes and is worth waiting for; a wedged one is not
# worth the OOM this exists to avoid.
DRAIN_TIMEOUT_SECONDS = float(os.environ.get('CFG_MODEL_DRAIN_TIMEOUT', '300'))

# How long a crossing waits for the server to go quiet before recycling anyway.
#
# A recycle drops every request still inside `handle`, so one taken mid-request destroys the
# work it was meant to protect. Waiting costs the backstop nothing — a leak is memory still
# held once requests finish, so it is still over the line at the next quiet moment — but the
# wait has to be bounded, because a wedged request must not hold the ceiling open until the
# cgroup OOM-kills the container instead. See _rss_watchdog.
RSS_QUIESCE_TIMEOUT_SECONDS = float(os.environ.get('CFG_MODEL_RSS_QUIESCE', '600'))

# CUDA errors that leave this process's context unusable. Unlike an out-of-memory error, every
# later call on the context fails the same way until the process is gone — while health still
# answers ok — so the server has to be replaced, not retried.
STICKY_CUDA_ERRORS = (
    "launch timed out",  # cudaErrorLaunchTimeout: a kernel killed by the driver's watchdog
    "illegal memory access",  # cudaErrorIllegalAddress
    "unspecified launch failure",  # cudaErrorLaunchFailure
    "misaligned address",
    "illegal instruction",
    "device-side assert",
    "sticky error",  # what the driver adds to any of these
)

# How soon after a recycle for a lost context another one may follow. A context lost again
# within this is likely a card that needs a reset rather than a process that needs replacing,
# and reloading the pinned families every few seconds would only hide that in a loop; so the
# next recycle waits for the rest of this, and health reports the failure meanwhile.
GPU_RECYCLE_GRACE_SECONDS = float(os.environ.get('CFG_MODEL_GPU_RECYCLE_GRACE', '600'))

# Why this process was started by a recycle ("rss" or "gpu"), carried across the re-exec.
RECYCLED_FOR = os.environ.get('CFG_MODEL_RECYCLE_REASON', '')

_started_at = time.time()
_requests = 0
_requests_lock = threading.Lock()

# Recycles this lineage of processes has already done, carried across the re-exec in the
# environment. Reading it back hours later from `podman logs` is what separates "the ceiling
# is holding a slow leak at bay" from "nothing leaks and this has never fired".
RECYCLE_COUNT = int(os.environ.get('CFG_MODEL_RECYCLE_COUNT', '0') or 0)

# In-flight request count, for the drain. `socketserver`'s own thread list cannot serve this:
# `_Threads.append` drops daemon threads on the floor, and `daemon_threads = True` is what
# lets a wedged request never hold up shutdown.
_inflight = 0
_inflight_cv = threading.Condition()

# Set while `_preload` is loading the pinned families. A model load is the largest allocation
# this process makes and it happens with no request in flight, so without this the RSS watchdog
# would count the server as quiet at exactly the moment it is doing the most work — and recycle
# into a reload of the very models that were half-loaded.
_preloading = False

# Set when a recycle is decided and cleared by an explicit shutdown, which is what makes a
# SIGTERM cancel one. Handlers keep running to completion either way; `main` acts on it after
# `serve_forever` has returned and the socket is gone.
_recycling = False

# The RSS ceiling in bytes, resolved once in `main` so `health` can report it without
# re-reading the cgroup on every poll. None when there is no ceiling.
_max_rss: int | None = None

# Set in `main`, so a request handler that finds the GPU context gone can start a recycle.
_stop: threading.Event | None = None
_server: "Server | None" = None

# Why the next re-exec happens, for RECYCLED_FOR in the process it starts.
_recycle_reason = "rss"

# The first sticky CUDA error this process met ({since, op, error}), reported by health until the
# process is replaced. None while the context is fine.
_gpu_failure: dict | None = None
_gpu_failure_lock = threading.Lock()

# Requests served per op. A total alone cannot say which op a memory series is tracking, and
# the ops differ by orders of magnitude in what they allocate — attributing growth needs the
# breakdown, taken hours after the fact from a `health` poll.
_op_counts: dict[str, int] = {}


# `query` for the worker that answers search queries alone (the front's Context.for_queries), so the
# two workers' lines can be told apart in one log.
ROLE = os.environ.get('CFG_MODEL_ROLE', '').strip()
LOG_TAG = f"[{ROLE}-server]" if ROLE else "[model-server]"


def log(message: str) -> None:
    print(f"{LOG_TAG} {message}", file=sys.stderr, flush=True)


def _lower_priority() -> None:
    """Run at CFG_MODEL_NICE, when set: the batch worker beside a query worker.

    Absolute rather than an increment, so a re-exec (the RSS recycle) lands at the same value.
    Linux keeps the nice value per thread and a new thread inherits its creator's, so this runs
    first thing in `main`, before any model, socket or watchdog thread exists — every thread the
    worker starts later, torch's and onnxruntime's pools included, then runs at it.
    """
    raw = os.environ.get('CFG_MODEL_NICE', '').strip()
    if not raw:
        return
    try:
        os.setpriority(os.PRIO_PROCESS, 0, int(raw))
        log(f"running at nice {raw}, below the query worker")
    except (ValueError, OSError) as exc:
        log(f"CFG_MODEL_NICE={raw!r} not applied: {exc}")


def _hub_cache_dir() -> str:
    """Where huggingface_hub will look for weights, resolved without importing it.

    Same precedence the library uses, so this is a report rather than a second opinion.
    """
    explicit = os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE')
    if explicit:
        return explicit
    hf_home = os.environ.get('HF_HOME')
    if not hf_home:
        xdg = os.environ.get('XDG_CACHE_HOME') or os.path.join(
            os.path.expanduser('~'), '.cache'
        )
        hf_home = os.path.join(xdg, 'huggingface')
    return os.path.join(hf_home, 'hub')


def _log_model_cache() -> None:
    """Say where the weights will be read from, and whether they are actually there.

    A miss here is not a slow start, it is ~17GB pulled from the Hub inside a request, on every
    container recreate. It is also completely silent: transformers just downloads. The cache
    follows $HOME unless HF_HOME says otherwise, and $HOME at image-build time need not be $HOME
    at run time.
    """
    cache = _hub_cache_dir()
    # faster-whisper resolves a bare model name to the Systran repo that hosts it, and
    # sentence-transformers prefixes its own org — so both are spelled out as the repo id the
    # cache is actually keyed by, not as the name the config uses.
    wanted = {
        'whisper': registry.WHISPER_MODEL if '/' in registry.WHISPER_MODEL
        else f'Systran/faster-whisper-{registry.WHISPER_MODEL}',
        'text-embed': registry.TEXT_EMBED_MODEL,
    }
    missing = [
        name for name, repo in wanted.items()
        if not os.path.isdir(os.path.join(cache, f"models--{repo.replace('/', '--')}"))
    ]
    if missing:
        log(f"WARNING hub cache {cache} has no local copy of {', '.join(sorted(missing))} — "
            f"these will be DOWNLOADED on first use, inside whatever request asks for them. "
            f"Set HF_HOME to wherever the image baked them.")
    else:
        log(f"hub cache {cache} (all baked models present)")
    # The image embedder is baked outside the hub cache (a plain directory, for onnxruntime) and
    # has no download path at all: a missing export fails every vision.embed, so say so now.
    image_embed = registry.image_embed_root()
    if not os.path.isfile(os.path.join(image_embed, "visual", "model.onnx")):
        log(f"WARNING image embedder {image_embed} is missing — vision.embed will fail. "
            f"Rebuild the image or set CFG_IMAGE_EMBED_ROOT.")
    copy_detect = os.path.join(registry.COPY_DETECT_ROOT, f"{registry.COPY_DETECT_MODEL}.torchscript.pt")
    if not os.path.isfile(copy_detect):
        log(f"WARNING copy detector {copy_detect} is missing — fingerprint.compute will fail for pictures and video. "
            f"Rebuild the image or set CFG_COPY_DETECT_ROOT.")
    if not os.path.isfile(os.path.join(registry.TRANSNET_ROOT, registry.TRANSNET_WEIGHTS)):
        log(f"WARNING shot detector {registry.TRANSNET_WEIGHTS} is missing from {registry.TRANSNET_ROOT} — video.shots will fail. "
            f"Rebuild the image or set CFG_TRANSNET_ROOT.")
    tier = registry.OCR_PHOTO_TIER
    for task in ("det", "rec"):
        if not os.path.isfile(os.path.join(registry.OCR_ROOT, f"PP-OCRv6_{task}_{tier}.onnx")):
            log(f"WARNING text reader PP-OCRv6_{task}_{tier}.onnx is missing from {registry.OCR_ROOT} — ocr.read will fail. "
                f"Rebuild the image or set CFG_OCR_ROOT.")
    for root, what in ((registry.OCR_HAND_ROOT, "handwriting recogniser"), (registry.HTR_ROOT, "handwriting transcriber"),
                       (registry.HTR_EN_ROOT, "English handwriting transcriber")):
        if not os.path.isdir(root):
            log(f"WARNING {what} {root} is missing — ocr.read reads without it, under another producer name.")


def handle(request: dict) -> dict:
    """Dispatch one request. Imports inference_ops lazily so startup stays fast."""
    op = request.get("op")

    if op == "health":
        health = {
            "loaded": registry.loaded(),
            # Every family it can load, resident or not, with what each copy is doing.
            "families": registry.family_states(),
            "uptime": round(time.time() - _started_at, 1),
            "requests": _requests,
            "op_counts": dict(_op_counts),
            "socket": SOCKET_PATH,
            "recycles": RECYCLE_COUNT,
        }
        # Where the server sits against its own ceiling. One /proc read, and it is the
        # difference between "indexing feels slow" and "it is 300MB from recycling again".
        rss = mem_diag.rss_bytes()
        if rss is not None:
            health["rss_mb"] = round(rss / 1048576)
        if _max_rss:
            health["max_rss_mb"] = round(_max_rss / 1048576)
        # torch's own accounting, which is what distinguishes "the models are big" from
        # "the caching allocator is holding blocks it will never reuse": reserved is what
        # nvidia-smi sees, allocated is what the live tensors actually need. A large gap
        # is cache, and reclaimable. Reported only if torch is already imported, so a
        # faces-only server does not load it just to answer a health check.
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    health["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 1048576)
                    health["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 1048576)
            except Exception:  # noqa: BLE001 - diagnostics must never break health
                pass
        with _gpu_failure_lock:
            if _gpu_failure is not None:
                health["gpu_failure"] = dict(_gpu_failure)
        return health

    if op == "memstats":
        # Host-RSS diagnostics, by hand over the socket. Kept out of `health` because it reads /proc/self/maps, which is O(mappings) and pointless on the
        # per-request health path. See mem_diag.py for what the numbers decide.
        return mem_diag.snapshot()

    import inference_ops as ops

    angle, mirror = request.get("angle", 0), bool(request.get("mirror", False))

    if op == "transcribe":
        path = request.get("path")
        if not path:
            raise ValueError("transcribe: missing path")
        # Optional per-request decode settings; absent means the configured value.
        min_silence = request.get("vad_min_silence_ms")
        return ops.transcribe(
            path,
            language=request.get("language"),
            min_silence_ms=int(min_silence) if min_silence is not None else None,
            word_timestamps=request.get("word_timestamps"),
        )

    if op == "transcribe_signature":
        # What a transcript decoded right now would be stamped with; no model is loaded.
        return {"signature": ops.transcribe_signature()}

    if op == "embed_texts":
        texts = request.get("texts")
        if not texts:
            raise ValueError("embed_texts: missing texts")
        return ops.embed_texts([str(text) for text in texts], str(request.get("role") or "document"), request.get("device"))

    if op == "unload":
        families = request.get("families")
        return {"unloaded": registry.unload(
            [str(family) for family in families] if families is not None else None,
            device=request.get("device"),
            wait=bool(request.get("wait", False)),
        )}

    if op == "diarize":
        path = request.get("path")
        if not path:
            raise ValueError("diarize: missing path")
        return ops.diarize(path)

    if op == "embed_voice_segments":
        path, spans = request.get("path"), request.get("spans")
        if not path or spans is None:
            raise ValueError("embed_voice_segments: missing path or spans")
        return ops.embed_voice_segments(path, spans)

    if op == "active_speakers":
        video, audio, speakers = request.get("video"), request.get("audio"), request.get("speakers")
        if not video or not audio or speakers is None:
            raise ValueError("active_speakers: missing video, audio or speakers")
        return ops.active_speakers(video, audio, speakers, request.get("angle") or 0, bool(request.get("mirror")))

    if op == "active_speaker_score":
        video, audio = request.get("video"), request.get("audio")
        start, end, track = request.get("start"), request.get("end"), request.get("track")
        if not video or not audio or start is None or end is None or track is None:
            raise ValueError("active_speaker_score: missing video, audio, start, end or track")
        return ops.active_speaker_score(video, audio, float(start), float(end), track)

    if op == "embed_images":
        paths = request.get("paths")
        if not paths:
            raise ValueError("embed_images: missing paths")
        return ops.embed_images(paths, angle, mirror)

    if op == "copy_descriptors":
        paths = request.get("paths")
        if not paths:
            raise ValueError("copy_descriptors: missing paths")
        return ops.copy_descriptors(paths, angle, mirror)

    if op == "shot_boundaries":
        path, times = request.get("path"), request.get("times")
        if not path or not times:
            raise ValueError("shot_boundaries: missing path or times")
        return ops.shot_boundaries(
            path, times, float(request.get("duration") or times[-1]),
            threshold=float(request["threshold"]), nms=float(request["nms"]), merge=float(request["merge"]),
            min_shot=float(request["min_shot"]), min_change=float(request["min_change"]), debug=bool(request.get("debug")),
        )

    if op == "read_text":
        paths = request.get("paths")
        if not paths:
            raise ValueError("read_text: missing paths")
        return ops.read_text(paths, angle, mirror, request.get("language"))

    if op == "embed_image_texts":
        texts = request.get("texts")
        if not texts:
            raise ValueError("embed_image_texts: missing texts")
        return ops.embed_image_texts(texts)

    if op == "detect_faces":
        path = request.get("path")
        if not path:
            raise ValueError("detect_faces: missing path")
        return ops.detect_faces(path, angle, mirror)

    if op == "embed_face":
        path, box = request.get("path"), request.get("box")
        if not path or box is None:
            raise ValueError("embed_face: missing path or box")
        return ops.embed_face(path, box, angle, mirror)

    raise ValueError(f"unknown op: {op}")


class Handler(socketserver.StreamRequestHandler):
    # A transcription can legitimately take minutes; don't time out on it.
    timeout = None

    def handle(self):
        global _requests, _inflight
        try:
            raw = self.rfile.readline()
        except OSError:
            return
        if not raw:
            return

        try:
            request = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._reply({"ok": False, "error": f"malformed request: {exc}"})
            return

        op = request.get("op", "?")
        with _requests_lock:
            _requests += 1
            _op_counts[op] = _op_counts.get(op, 0) + 1
        # Counted before the work and released in the same `finally` as the trim, so a
        # recycle's drain covers the reclaim too — re-exec'ing between a reply and its
        # `cleanup_torch` would be harmless, but a request still inside a model is not.
        with _inflight_cv:
            _inflight += 1

        started = time.time()
        try:
            with registry.request_scope(request.get("expires_at"), self._hung_up):
                result = handle(request)
            self._reply({"ok": True, "result": result})
            if op != "health":
                log(f"{op} ok in {round(time.time() - started, 2)}s")
        except registry.RequestAbandoned as exc:
            self._reply({"ok": False, "error": str(exc), "error_type": type(exc).__name__})
            log(f"{op} dropped after {round(time.time() - started, 2)}s: {exc}")
        except Exception as exc:  # noqa: BLE001 - every failure goes back to the client
            # `error_type` is the class name, for the HTTP front to classify the failure by
            # (CardBusy → retry later, FileNotFoundError → not found) without parsing prose.
            self._reply({"ok": False, "error": str(exc), "error_type": type(exc).__name__})
            log(f"{op} failed after {round(time.time() - started, 2)}s: {exc}")
            # The client only ever sees `str(exc)`, and one line is not enough for the
            # failures that matter here: a dtype mismatch says what happened in the model but
            # not *where*, and it may appear only after hours of uptime. The traceback names
            # the module, so print it where it costs nothing to have.
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            if is_sticky_cuda_error(exc):
                _on_sticky_cuda_error(op, exc)
        finally:
            # This is the only per-request point to return freed inference memory to the OS. The buffers are native (numpy/torch/
            # onnxruntime): gc frees almost nothing and glibc parks the bytes at each arena's
            # high-water mark, so without this the server's RSS climbs unbounded until the
            # container OOMs. Runs after _reply so the client isn't held up by the trim.
            #
            # `memstats` is exempt alongside `health`: it exists to measure the memory the trim
            # acts on, so trimming on its way out would make every reading a post-trim one and
            # hide the very growth it is there to catch.
            #
            # The query and card workers only embed search strings, a few KB each, and a trim
            # holds the interpreter for ~165 ms on fry: the next query, often the same search's
            # other half, waited for it. Theirs is the eviction loop's, once a minute.
            if op not in ("health", "memstats") and ROLE not in ("query", "card"):
                registry.cleanup_torch()
            with _inflight_cv:
                _inflight -= 1
                _inflight_cv.notify_all()

    def _hung_up(self) -> bool:
        """Whether the client has closed its end. It half-closes after its request, which is not this."""
        poller = select.poll()
        poller.register(self.connection, select.POLLHUP)
        return any(events & (select.POLLHUP | select.POLLERR) for _fd, events in poller.poll(0))

    def _reply(self, payload: dict) -> None:
        try:
            self.wfile.write((json.dumps(payload) + "\n").encode())
            self.wfile.flush()
        except OSError:
            # Client vanished — nothing to do.
            pass


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    # Clients queue on the per-family lock rather than being refused.
    request_queue_size = 64
    allow_reuse_address = True


def _preload(stop: threading.Event) -> None:
    """Load the pinned families up front, off the request path.

    Without this the pin only helps from the *second* search onwards: the first one after a
    restart still pays the full model load. Runs in a thread so a slow load never delays
    binding the socket.

    Each load takes the family lock, which is what stops it racing a request for the same
    family. `_get` loads outside the registry lock on purpose, so without this both threads
    end up inside the loader at once — and transformers is a lazy module, so a concurrent
    first import of it fails with "cannot import name … from 'transformers'" rather than
    merely doing the work twice.
    """
    global _preloading
    _preloading = True
    try:
        _preload_families(stop)
    finally:
        _preloading = False
        registry.pinned_ready.set()


def _preload_families(stop: threading.Event) -> None:
    """The loop itself, split out only so `_preload` can bracket it with `_preloading`."""
    for family in sorted(registry.PINNED_FAMILIES):
        if stop.is_set():
            return
        try:
            started = time.time()
            # Loading goes on the family's own thread, not this one. A thread that loads a
            # model acquires the same per-thread CUDA state a thread that runs one does, and
            # this thread exits as soon as the preload is done — see run_on_model_thread.
            # One query through each as well: on the card the first one sets up its kernels, which
            # took the first search after a start 2.8 s.
            if family == 'siglip-text':
                def load():
                    registry.siglip_text(registry.image_embed_text_device()).embed(["warm"])
            elif family == 'text-query':
                def load():
                    registry.text_embedder('text-query', registry.text_embed_query_device()).encode(["warm"], prompt_name="query")
            else:
                log(f"preload: no loader wired for pinned family '{family}', skipping")
                continue
            with registry.family_lock(family):
                registry.run_on_model_thread(family, load)
            log(f"preloaded {family} in {time.time() - started:.1f}s")
        except Exception as exc:  # noqa: BLE001 - preload is best-effort
            log(f"preload of {family} failed: {exc}")


def _evict_loop(stop: threading.Event) -> None:
    while not stop.wait(EVICT_INTERVAL_SECONDS):
        try:
            evicted = registry.evict_idle()
            if evicted:
                log(f"evicted idle models: {', '.join(evicted)}")

            # Reclaim regardless of whether anything was evicted. torch's caching
            # allocator keeps freed blocks reserved rather than returning them to the
            # driver, and every request brings a different input size, so the cache grows
            # into a pile of odd-sized blocks it can never reuse — while during a crawl the
            # models are in constant use and never go idle to be evicted. empty_cache is
            # cheap when there is nothing cached to free.
            registry.cleanup_torch()

            # onnxruntime's CUDA arena never gives memory back, so InsightFace walks the
            # card under sustained face work until a CUDA OOM drops the other families and
            # InsightFace itself onto the CPU. Recreating its sessions releases every arena; the reload
            # costs ~1-2s against the hundreds of images between resets.
            freed_at = registry.reset_insightface_if_low()
            if freed_at is not None:
                log(f"reset insightface sessions to release onnxruntime arenas "
                    f"(free VRAM was {freed_at:.0f}MB)")
        except Exception as exc:  # noqa: BLE001
            log(f"eviction failed: {exc}")


def _work_requests() -> int:
    """Requests that actually ran a model — `health` and `memstats` allocate nothing."""
    with _requests_lock:
        return sum(n for op, n in _op_counts.items() if op not in ("health", "memstats"))


def _rss_watchdog(stop: threading.Event, server: "Server", ceiling: int) -> None:
    """Recycle the process when its own RSS crosses `ceiling`.

    A per-request leak (glibc arena retention, onnxruntime's CUDA PerThreadContext, a torch
    allocation per request thread) looks the same from outside — indexing slows, the box swaps,
    the process is OOM-killed hours later — unless the server knows its own budget, and by the
    time the kernel notices, the whole host has spent hours in direct reclaim.

    So this is a ceiling of the server's own, well under the cgroup's: cross it and the
    process drains and rebuilds itself (reloading the pinned families) instead of being killed
    mid-request. It is a backstop, not a fix — the loud log is the point, because a recycle
    that keeps happening is the next leak, and the `memstats` line plus `op_counts` are
    exactly the evidence needed to attribute it.

    A crossing is acted on only when the server is quiet, or after
    RSS_QUIESCE_TIMEOUT_SECONDS of waiting for it to become so. Recycling on the sample that
    first goes over the line means recycling in the middle of whatever allocated the bytes,
    which for a model load is the request that most needs to survive.
    """
    warned = False
    quiet_deadline: float | None = None
    while not stop.wait(RSS_CHECK_INTERVAL_SECONDS):
        try:
            rss = mem_diag.rss_bytes()
            if rss is None or rss < ceiling:
                # Back under the line, so whatever was over it was transient and the wait
                # for a quiet moment starts again from scratch the next time.
                quiet_deadline = None
                continue

            uptime = time.time() - _started_at
            served = _work_requests()
            if uptime < RSS_GRACE_SECONDS or served == 0:
                # Over the line with nothing to blame it on: the models alone do not fit
                # under this ceiling, and re-exec'ing would only reload them into the same
                # wall. Say so once and keep serving — the cgroup is still the backstop.
                if not warned:
                    warned = True
                    log(f"WARNING rss {rss / 1048576:.0f}MB is already over the "
                        f"{ceiling / 1048576:.0f}MB ceiling after {uptime:.0f}s and "
                        f"{served} model request(s) — too early to blame a leak, not "
                        f"recycling. Raise CFG_MODEL_MAX_RSS if this is the real "
                        f"footprint.")
                continue

            # Never recycle out from under a request that could account for the growth.
            #
            # Loading a model for a request is itself a host spike, so a ceiling the resident
            # models nearly fill is crossed *by* the request — and a recycle then kills it, the
            # replacement reloads, spikes, and crosses again — and the request never finishes.
            #
            # A leak is by definition memory still held after requests finish, so deferring
            # to the next quiet moment gives up nothing: if these bytes are a leak, RSS is
            # still over the line when the last request returns. The wait is bounded so a
            # wedged request cannot hold the ceiling open until the cgroup does the killing.
            with _inflight_cv:
                busy = _inflight
            doing = f"{busy} request(s) in flight" if busy else "the pinned models still loading"
            if busy or _preloading:
                if quiet_deadline is None:
                    quiet_deadline = time.time() + RSS_QUIESCE_TIMEOUT_SECONDS
                    log(f"rss {rss / 1048576:.0f}MB over the {ceiling / 1048576:.0f}MB "
                        f"ceiling with {doing} — waiting up to "
                        f"{RSS_QUIESCE_TIMEOUT_SECONDS:.0f}s for the server to go quiet "
                        f"before recycling")
                if time.time() < quiet_deadline:
                    continue
                log(f"still {doing} {RSS_QUIESCE_TIMEOUT_SECONDS:.0f}s after the crossing "
                    f"— recycling anyway")

            _recycle(stop, server, rss, ceiling)
            return
        except Exception as exc:  # noqa: BLE001 - a broken probe must not kill the server
            log(f"rss watchdog failed: {exc}")


def _recycle(stop: threading.Event, server: "Server", rss: int, ceiling: int) -> None:
    """Log the evidence, then ask `main` to drain and re-exec."""
    global _recycling, _recycle_reason
    _recycling = True
    _recycle_reason = "rss"
    log(f"RSS CEILING CROSSED: {rss / 1048576:.0f}MB over {ceiling / 1048576:.0f}MB after "
        f"{time.time() - _started_at:.0f}s and {_work_requests()} model request(s) — "
        f"draining and re-execing (recycle #{RECYCLE_COUNT + 1})")
    # Taken before anything reclaims, and in the same shape the periodic log writes, so the
    # crossing lands in the same `grep memstats` series as the hours that led to it.
    try:
        log(f"memstats {mem_diag.format_line(mem_diag.snapshot(), registry.loaded())}")
    except Exception as exc:  # noqa: BLE001
        log(f"memstats at ceiling failed: {exc}")
    # What ran, per op. A total cannot say which op a series is tracking and the ops differ by
    # orders of magnitude in what they allocate; this is what turns "it grew" into "it grew
    # 30MB per transcription".
    with _requests_lock:
        counts = " ".join(f"{op}={n}" for op, n in sorted(_op_counts.items()))
    log(f"op_counts {counts or 'none'}")

    stop.set()
    # `shutdown()` blocks until `serve_forever` has returned, so it goes on a thread of its
    # own exactly as the signal handler does, and this one ends here.
    threading.Thread(target=server.shutdown, daemon=True).start()


def is_sticky_cuda_error(exc: BaseException) -> bool:
    """Whether `exc` says this process's CUDA context is gone (STICKY_CUDA_ERRORS)."""
    message = str(exc).lower()
    return any(marker in message for marker in STICKY_CUDA_ERRORS)


def _on_sticky_cuda_error(op: str, exc: BaseException) -> None:
    """Replace the process after its CUDA context is lost, at most once per grace period.

    A fresh process gets a fresh context, and the next request works. It goes through the same
    drain and re-exec as the RSS ceiling, so the front's restart backoff is not spent. Requests still running fail fast on the dead context,
    so the drain is short.

    The first error is recorded for health either way, so a card that will not come back shows
    up as `degraded` rather than as a server that answers ok and fails everything.
    """
    global _gpu_failure
    with _gpu_failure_lock:
        if _gpu_failure is not None:
            return
        _gpu_failure = {"since": time.time(), "op": op, "error": str(exc).strip().splitlines()[0][:200]}
    if _recycling or _stop is None or _server is None:
        return

    def recycle() -> None:
        global _recycling, _recycle_reason
        if _recycling or _stop.is_set():
            return
        log(f"GPU CONTEXT LOST: {op} failed with {_gpu_failure['error']!r} — every later CUDA "
            f"call fails the same way, so draining and re-execing for a fresh context "
            f"(recycle #{RECYCLE_COUNT + 1})")
        _recycling = True
        _recycle_reason = "gpu"
        _stop.set()
        threading.Thread(target=_server.shutdown, daemon=True).start()

    uptime = time.time() - _started_at
    if RECYCLED_FOR == "gpu" and uptime < GPU_RECYCLE_GRACE_SECONDS:
        wait = GPU_RECYCLE_GRACE_SECONDS - uptime
        log(f"GPU CONTEXT LOST AGAIN {uptime:.0f}s after a recycle for the same reason — the card "
            f"may need a reset (nvidia-smi -r) or the host a reboot. Health reports gpu_failure; "
            f"recycling again in {wait:.0f}s.")
        timer = threading.Timer(wait, recycle)
        timer.daemon = True
        timer.start()
        return
    recycle()


def _drain() -> None:
    """Wait for in-flight requests, bounded by DRAIN_TIMEOUT_SECONDS.

    By the time this runs the listening socket is closed and its path unlinked, so nothing new
    arrives: a client that connects now finds the worker unavailable, the same contract as the
    worker being stopped. This only bounds the wait on requests already inside `handle`. Going
    anyway after the timeout is no worse than the OOM kill it replaces — the client reads a
    dropped connection as unavailable too — and it keeps one wedged request from holding the
    ceiling open indefinitely.
    """
    deadline = time.time() + DRAIN_TIMEOUT_SECONDS
    with _inflight_cv:
        while _inflight and _recycling:
            remaining = deadline - time.time()
            if remaining <= 0:
                log(f"drain timed out with {_inflight} request(s) still running")
                return
            _inflight_cv.wait(min(remaining, 1.0))
    if _recycling:
        log("drained")


def _reexec() -> int:
    """Replace this process with a fresh copy of itself.

    `execv` rather than exiting: the pid survives, so the front's supervisor (Worker._supervise)
    never counts a recycle as a crash and its restart backoff is kept for a genuinely broken
    worker.

    Every inherited descriptor is marked close-on-exec first. `exec` replaces the address
    space but *not* the file table, and a surviving `/dev/nvidia*` descriptor would keep the
    old CUDA context — and its VRAM — alive in a process that can no longer reach it, which
    is the opposite of the point. Only stdin/stdout/stderr are kept.
    """
    import fcntl

    try:
        inherited = [int(name) for name in os.listdir("/proc/self/fd")]
    except OSError:
        inherited = []
    for fd in inherited:
        if fd < 3:
            continue
        try:
            flags = fcntl.fcntl(fd, fcntl.F_GETFD)
            fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
        except OSError:
            pass  # already gone, or not ours to touch

    os.environ["CFG_MODEL_RECYCLE_COUNT"] = str(RECYCLE_COUNT + 1)
    os.environ["CFG_MODEL_RECYCLE_REASON"] = _recycle_reason
    argv = [sys.executable, os.path.abspath(__file__), *sys.argv[1:]]
    log(f"re-exec {' '.join(argv)}")
    sys.stderr.flush()
    try:
        os.execv(argv[0], argv)
    except OSError as exc:
        # Nothing left to serve with — the socket is closed and unlinked. Exit non-zero so
        # the front's supervisor starts a new one.
        log(f"re-exec failed: {exc}")
        return 1
    return 0  # unreachable: execv does not return


def main() -> int:
    global _max_rss, _stop, _server

    _lower_priority()

    # A socket file left behind by a killed server would stop the bind, so clear it first.
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = Server(SOCKET_PATH, Handler)
    # A client may run as a different uid than the server; the socket carries no secrets
    # beyond what those callers already have.
    os.chmod(SOCKET_PATH, 0o666)

    stop = threading.Event()
    _stop, _server = stop, server
    evictor = threading.Thread(target=_evict_loop, args=(stop,), daemon=True)
    evictor.start()
    # Cleared before the socket binds, so no request can load another family ahead of the pins.
    registry.pinned_ready.clear()
    threading.Thread(target=_preload, args=(stop,), daemon=True).start()

    # A leak takes hours to reach the container cap, so the memory series collects unattended.
    memlog_interval = mem_diag.interval_from_env()
    if memlog_interval:
        threading.Thread(
            target=mem_diag.log_loop,
            args=(stop, log, memlog_interval, registry.loaded),
            daemon=True,
        ).start()

    _log_model_cache()

    _max_rss, ceiling_source = mem_diag.max_rss_from_env()
    if _max_rss:
        log(f"rss ceiling {_max_rss / 1048576:.0f}MB ({ceiling_source})")
        threading.Thread(
            target=_rss_watchdog, args=(stop, server, _max_rss), daemon=True
        ).start()
    else:
        log(f"rss ceiling disabled ({ceiling_source})")

    def shutdown(_signum, _frame):
        global _recycling
        log("shutting down")
        # Cancels a recycle in flight, including one already draining: an operator stopping
        # the service must not have it re-exec itself back into existence.
        _recycling = False
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    pinned = ', '.join(sorted(registry.PINNED_FAMILIES)) or 'none'
    recycled = f", recycle #{RECYCLE_COUNT}" if RECYCLE_COUNT else ""
    log(f"listening on {SOCKET_PATH} (idle evict {registry.IDLE_EVICT_SECONDS}s, "
        f"pinned: {pinned}{recycled})")
    try:
        server.serve_forever()
    finally:
        # Closing and unlinking here is what makes the drain safe: from this point a client
        # finds no socket, so the only requests left are the ones already being served.
        server.server_close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        log("stopped")

    if _recycling:
        _drain()
        # Re-checked after the drain, not only before it: a SIGTERM that lands while we are
        # waiting cancels the recycle, and an operator stopping the service must not have it
        # replace itself at the last moment.
        if _recycling:
            return _reexec()
        log("recycle cancelled by shutdown")

    return 0


if __name__ == "__main__":
    sys.exit(main())
