"""The model worker: `modelworker/model_server.py`, spawned and spoken to over its Unix socket.

The worker is a separate process on purpose. It holds the models, recycles itself (a re-exec
in place) when it crosses its RSS ceiling, and can die of a CUDA fault — none of which should
take the HTTP listener or the tool ops with it. This module spawns it, restarts it when it
exits, and translates its `{ok, result|error}` line protocol into `EmsError`s.

Configuration crosses the boundary by prefix: every `EMS_<NAME>` in this process's environment
reaches the worker as `CFG_<NAME>`, the namespace the worker reads (`CFG_WHISPER_MODEL`,
`CFG_MODEL_PINNED`, …). A `CFG_*` already set passes through and wins.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from collections.abc import Callable
from typing import Any

from .admission import note_family
from .errors import (
    DEADLINE_EXCEEDED,
    INTERNAL,
    INVALID_REQUEST,
    MODEL_UNAVAILABLE,
    NOT_FOUND,
    TOOL_FAILED,
    UNSUPPORTED,
    EmsError,
)

RESTART_BACKOFF_SECONDS = (1, 2, 5, 10, 30)
CARD_BUSY_RETRY_AFTER_SECONDS = 60
STOP_GRACE_SECONDS = 20

# How a worker failure maps to a code: by the exception's type first, then by its message,
# which is all a generic exception (a RuntimeError for CUDA OOM) carries.
#
# Only a type says a failure is the request's own: a missing input is a FileNotFoundError and a
# malformed request a ValueError. A message cannot say so — "libcublas.so.12 is not found" is
# the image's fault, and matching "not found" on it made every request permanently not_found.
_TYPE_TO_CODE = {
    "CardBusy": MODEL_UNAVAILABLE,
    "OutOfMemoryError": MODEL_UNAVAILABLE,  # torch.OutOfMemoryError
    "TimeoutError": DEADLINE_EXCEEDED,
    "RequestAbandoned": DEADLINE_EXCEEDED,
    "FileNotFoundError": NOT_FOUND,
    "UnidentifiedImageError": UNSUPPORTED,
    "ImportError": INTERNAL,
    "ModuleNotFoundError": INTERNAL,
}

# The card ran out, or a library reports the aftermath of running out: CTranslate2 answers the
# request after an OOM with "invalid device ordinal" and then recovers on its own, so it waits
# like the OOM itself rather than recycling the worker (ein/core#354).
_MESSAGE_TO_CODE = (
    ("cuda oom", MODEL_UNAVAILABLE),
    ("out of memory", MODEL_UNAVAILABLE),  # torch, CTranslate2, onnxruntime
    ("failed to allocate memory", MODEL_UNAVAILABLE),  # onnxruntime's arena
    ("cudaerrormemoryallocation", MODEL_UNAVAILABLE),
    ("cublas_status_alloc_failed", MODEL_UNAVAILABLE),
    ("invalid device ordinal", MODEL_UNAVAILABLE),
    ("request expired", DEADLINE_EXCEEDED),
    ("could not load image", UNSUPPORTED),
    ("cannot identify image file", UNSUPPORTED),
)


def classify(error: str, error_type: str | None) -> EmsError:
    code = _TYPE_TO_CODE.get(error_type or "")
    if code is None:
        lowered = error.lower()
        for needle, candidate in _MESSAGE_TO_CODE:
            if needle in lowered:
                code = candidate
                break
    if code is None and error_type == "ValueError":
        code = INVALID_REQUEST
    if code is None:
        code = TOOL_FAILED
    retry_after = CARD_BUSY_RETRY_AFTER_SECONDS if code is MODEL_UNAVAILABLE else None
    return EmsError(code, error, retry_after=retry_after)


def worker_env(env: dict[str, str]) -> dict[str, str]:
    child = dict(env)
    for name, value in env.items():
        if name.startswith("EMS_"):
            child.setdefault("CFG_" + name[4:], value)
    return child


# The longest reply line the worker may send, in bytes. See Worker.call.
REPLY_LINE_LIMIT = 256 * 1024 * 1024


# The families that answer search queries: the text embedder's query side and the picture space's
# text tower. They run in a query worker of their own when there is one (Context.for_queries).
QUERY_FAMILIES = ("text-query", "siglip-text")

# Which model family each worker op runs on, so `health` can say what a request is using. Kept here
# beside the client rather than asked of the worker: it is static, and a request names it before
# the worker has picked it up. `embed_texts` depends on the role and is resolved in `call`.
WORKER_OP_FAMILIES = {
    "transcribe": "whisper",
    "diarize": "sortformer",
    "embed_voice_segments": "titanet",
    "active_speaker_score": "asd",
    "active_speakers": "asd",
    "detect_faces": "insightface",
    "embed_face": "insightface",
    "embed_images": "siglip-visual",
    "embed_image_texts": "siglip-text",
    "copy_descriptors": "sscd",
    "shot_boundaries": "transnet",
    "read_text": "ocr",
}


class Worker:
    def __init__(
        self,
        script: str,
        socket_path: str,
        *,
        env: dict[str, str] | None = None,
        overrides: dict[str, str] | None = None,
        log=print,
    ):
        self.script = script
        self.socket_path = socket_path
        self.env = worker_env(dict(os.environ) if env is None else env)
        # After the EMS_ → CFG_ mapping, so a worker can differ from the configuration it shares.
        self.env.update(overrides or {})
        self.env["CFG_MODEL_SOCKET"] = socket_path
        self.log = log
        self.process: asyncio.subprocess.Process | None = None
        self._supervisor: asyncio.Task | None = None
        self._stopping = False
        self.started_at: float | None = None
        self.restarts = 0
        # Wraps every call: `gate(op, payload)` returns the context it runs in (the CardArbiter's).
        self.gate: Callable[[str, dict[str, Any]], Any] | None = None

    @property
    def pid(self) -> int | None:
        """The worker's pid while it runs — how `health` finds it among the card's processes."""
        return getattr(self.process, "pid", None) if self.running else None

    @property
    def running(self) -> bool:
        """Whether the process is up — not whether its models have finished loading."""
        return self.process is not None and self.process.returncode is None

    # ---- lifecycle ----

    async def start(self) -> None:
        if not os.path.isfile(self.script):
            self.log(f"worker: no script at {self.script}; model ops will answer model_unavailable")
            return
        self._stopping = False
        self._supervisor = asyncio.create_task(self._supervise())

    async def _supervise(self) -> None:
        attempt = 0
        while not self._stopping:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, self.script, env=self.env, stdin=asyncio.subprocess.DEVNULL,
            )
            self.started_at = time.time()
            self.log(f"worker: started pid {self.process.pid}")
            code = await self.process.wait()
            if self._stopping:
                return
            # A re-exec on the RSS ceiling keeps the pid, so reaching here is a real exit.
            self.restarts += 1
            delay = RESTART_BACKOFF_SECONDS[min(attempt, len(RESTART_BACKOFF_SECONDS) - 1)]
            attempt += 1
            self.log(f"worker: exited with {code}; restarting in {delay}s")
            await asyncio.sleep(delay)
            if self.started_at and time.time() - self.started_at > 300:
                attempt = 0  # it ran for a while; the next failure starts the ladder over

    async def stop(self) -> None:
        self._stopping = True
        if self._supervisor:
            self._supervisor.cancel()
            await asyncio.gather(self._supervisor, return_exceptions=True)
        proc = self.process
        if proc and proc.returncode is None:
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), STOP_GRACE_SECONDS)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

    # ---- requests ----

    async def call(
        self,
        op: str,
        payload: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> Any:
        """One request, one connection. Raises EmsError on every failure."""
        if self.gate is not None:
            async with self.gate(op, payload):
                return await self._call(op, payload, deadline=deadline)
        return await self._call(op, payload, deadline=deadline)

    async def _call(
        self,
        op: str,
        payload: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> Any:
        family = WORKER_OP_FAMILIES.get(op) or (("text-query" if payload.get("role") == "query" else "text-embed") if op == "embed_texts" else None)
        if family:
            note_family(family)
        body = {"op": op, **payload}
        if deadline is not None:
            body["expires_at"] = deadline
        try:
            # One reply is one line, and a line can be megabytes: a recording's voiceprints
            # are a thousand turns of 192 floats each. asyncio's default limit is 64 KiB, and
            # crossing it does not truncate but raises "chunk is longer than limit".
            reader, writer = await asyncio.open_unix_connection(self.socket_path, limit=REPLY_LINE_LIMIT)
        except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
            raise EmsError(MODEL_UNAVAILABLE, f"model worker is not answering ({exc.__class__.__name__})", retry_after=10) from exc
        try:
            writer.write((json.dumps(body) + "\n").encode())
            await writer.drain()
            if writer.can_write_eof():
                writer.write_eof()
            reply = None
            while raw := await reader.readline():
                text = raw.decode(errors="replace").strip()
                if not text:
                    continue
                try:
                    reply = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise EmsError(INTERNAL, f"model worker returned unparseable output: {text[:200]}") from exc
                break
        finally:
            # On cancellation this is what tells the worker the caller is gone: work still
            # waiting for its model is dropped, as it is once `expires_at` passes; work already
            # running finishes and its answer is dropped.
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001 - the peer may have closed first
                pass
        if reply is None:
            raise EmsError(MODEL_UNAVAILABLE, "model worker closed the connection without answering", retry_after=10)
        if not reply.get("ok"):
            raise classify(str(reply.get("error") or "model worker returned an error"), reply.get("error_type"))
        return reply.get("result")

    async def health(self) -> dict | None:
        try:
            return await self.call("health", {})
        except EmsError:
            return None
