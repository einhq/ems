"""Configuration, from the environment only.

There is deliberately no configuration API: `GET /v0/capabilities` and `GET /v0/health` are how
a caller learns what a given EMS runs, and changing any of this is a restart. Every knob is an
`EMS_*` variable; the model worker's own `CFG_*` variables are derived from them when it is
spawned (see worker.py), so an operator sets one namespace.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .roots import Root, parse_roots


@dataclass(frozen=True)
class Settings:
    port: int
    host: str
    keys_file: str
    keys: frozenset[str]
    roots: tuple[Root, ...]
    scratch: str
    worker_socket: str
    worker_script: str
    # Admission: how many requests may be forwarded to the model worker at once (it queues per
    # model family behind that), how many tool-bound requests may run at once, and how many may
    # wait for either before a new one is answered `busy`.
    limit_models: int
    limit_tools: int
    # Transcodes hold a slot for minutes, so they have a lane of their own rather than sitting
    # in the tool lane ahead of every probe and rendition behind them.
    limit_encodes: int
    queue_depth: int
    interactive_reserve: int
    # The host is meant to have a card (EMS_EXPECT_GPU=1). Without one EMS still runs, as a host
    # with no card, and says so in health's `degraded` rather than failing to start: a card lost to
    # a broken device mapping after a reboot then shows as a degradation instead of an outage.
    expect_gpu: bool
    exiftool_workers: int
    # How long a tool may go without using CPU or moving a byte before it is stopped as wedged.
    # Not how long it may take: a tape's extraction or a CPU encode takes what it takes
    # (ems/tools/run.py).
    tool_stall_seconds: int
    audio_cache_seconds: int
    audio_cache_mb: int
    # The CPU decode's memory footprint, projected before an encode starts: RSS grows with the
    # frames decoded in one process (about 170 MB per thousand, whatever the resolution) and is
    # never given back, so a source over the budget is encoded as windows of this size — one
    # process each — and one whose windows cannot be placed is refused. Zero disables the CPU
    # chain. See ems/tools/encode.py.
    cpu_encode_max_mb: int
    cpu_encode_mb_per_1k_frames: int
    cpu_encode_max_segments: int
    # Which ffmpeg encoder makes the video renditions: empty picks av1_nvenc when the card and
    # the build offer it and libsvtav1 otherwise; a name pins it (the CPU chain still backs it).
    video_encoder: str
    # NVENC's cq is not libsvtav1's crf: +6 keeps the same `quality` meaning roughly the same
    # bytes on either encoder.
    nvenc_cq_offset: int
    nvenc_preset: str
    # Search queries (text.embed `role: query`, vision.embed_text) get a model worker of their own,
    # on the CPU, while the batch worker runs at `worker_nice`, so a query never waits behind
    # batch work for the CPU or the interpreter lock. See Context.for_queries.
    query_worker: bool = True
    query_worker_socket: str = "/tmp/ems-query-worker.sock"
    query_worker_max_rss: str = "6g"
    # Threads one query may use. A short query split across every core waits on whichever thread
    # other work has descheduled, so it gets a few rather than all.
    query_worker_threads: int = 4
    worker_nice: int = 10
    # Seconds without batch model work before the card goes to search (ems/card.py); 0 keeps
    # the query families on the CPU always.
    card_quiet_after: int = 600
    card_worker_socket: str = "/tmp/ems-card-worker.sock"
    card_worker_max_rss: str = "6g"
    # What a query family's card copy needs free, the card worker's CUDA context, and a margin
    # (ems/card.py, CardBudget).
    card_text_query_mb: int = 1800
    card_siglip_text_mb: int = 3100
    card_context_mb: int = 400
    card_margin_mb: int = 1024
    # The most of the card EMS's own processes hold, in MB; 0 means all of it. For a host whose
    # card another service shares (the assistant's llm), set to the share that service leaves EMS:
    # neither then loads into the other's part, whichever started first (ein/core#515).
    card_share_mb: int = 0

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> "Settings":
        e = dict(os.environ) if env is None else env

        def integer(name: str, default: int) -> int:
            raw = e.get(name, "")
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

        keys_file = e.get("EMS_KEYS_FILE", "/etc/ems.keys")
        here = os.path.dirname(os.path.abspath(__file__))
        return Settings(
            port=integer("EMS_PORT", 8181),
            host=e.get("EMS_HOST", "0.0.0.0"),
            keys_file=keys_file,
            keys=load_keys(keys_file, e.get("EMS_KEY")),
            roots=parse_roots(e.get("EMS_ROOTS", "")),
            scratch=e.get("EMS_SCRATCH", "/tmp/ems"),
            worker_socket=e.get("EMS_WORKER_SOCKET", "/tmp/ems-model-worker.sock"),
            worker_script=e.get("EMS_WORKER_SCRIPT", os.path.join(here, "modelworker", "model_server.py")),
            limit_models=integer("EMS_LIMIT_MODELS", 8),
            limit_tools=integer("EMS_LIMIT_TOOLS", 4),
            limit_encodes=integer("EMS_LIMIT_ENCODES", 2),
            queue_depth=integer("EMS_QUEUE_DEPTH", 64),
            interactive_reserve=integer("EMS_INTERACTIVE_RESERVE", 1),
            expect_gpu=e.get("EMS_EXPECT_GPU", "") == "1",
            exiftool_workers=integer("EMS_EXIFTOOL_WORKERS", 2),
            tool_stall_seconds=integer("EMS_TOOL_STALL", 900),
            audio_cache_seconds=integer("EMS_AUDIO_CACHE_SECONDS", 1800),
            audio_cache_mb=integer("EMS_AUDIO_CACHE_MB", 2048),
            cpu_encode_max_mb=integer("EMS_CPU_ENCODE_MAX_MB", 4000),
            cpu_encode_mb_per_1k_frames=integer("EMS_CPU_ENCODE_MB_PER_1K_FRAMES", 170),
            cpu_encode_max_segments=integer("EMS_CPU_ENCODE_MAX_SEGMENTS", 64),
            video_encoder=e.get("EMS_VIDEO_ENCODER", ""),
            nvenc_cq_offset=integer("EMS_NVENC_CQ_OFFSET", 6),
            nvenc_preset=e.get("EMS_NVENC_PRESET", "p5"),
            query_worker=e.get("EMS_QUERY_WORKER", "1").strip().lower() not in ("0", "false", "no", ""),
            query_worker_socket=e.get("EMS_QUERY_WORKER_SOCKET", "/tmp/ems-query-worker.sock"),
            query_worker_max_rss=e.get("EMS_QUERY_WORKER_MAX_RSS", "6g"),
            # Cores less two, at most four.
            query_worker_threads=integer("EMS_QUERY_WORKER_THREADS", max(1, min(4, (os.cpu_count() or 2) - 2))),
            worker_nice=integer("EMS_WORKER_NICE", 10),
            card_quiet_after=integer("EMS_CARD_QUIET_AFTER", 600),
            card_worker_socket=e.get("EMS_CARD_WORKER_SOCKET", "/tmp/ems-card-worker.sock"),
            card_worker_max_rss=e.get("EMS_CARD_WORKER_MAX_RSS", "6g"),
            card_text_query_mb=integer("EMS_CARD_TEXT_QUERY_MB", 1800),
            card_siglip_text_mb=integer("EMS_CARD_SIGLIP_TEXT_MB", 3100),
            card_context_mb=integer("EMS_CARD_CONTEXT_MB", 400),
            card_margin_mb=integer("EMS_CARD_MARGIN_MB", 1024),
            card_share_mb=integer("EMS_CARD_SHARE_MB", 0),
        )


def load_keys(path: str, inline: str | None) -> frozenset[str]:
    """One key per line; `EMS_KEY` adds one without a file.

    No file and no key means no authentication at all, which is only right on a private socket
    or a development box — the log says so at startup rather than failing, because a container
    with nothing mounted is still worth being able to probe.
    """
    keys: set[str] = set()
    if inline and inline.strip():
        keys.add(inline.strip())
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    keys.add(line)
    except FileNotFoundError:
        pass
    return frozenset(keys)
