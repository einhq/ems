"""ffprobe as JSON, and the one thing a refusal means.

ffprobe parses the container before it looks at any stream, so it still exits 0 on a file
whose codec it cannot decode; a non-zero exit means it could not find the track table at all —
a truncated MOV whose `moov` atom was never written. That is a property of the bytes, reported
as `decodable: false` with the first useful stderr line, never as a transient error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..errors import EmsError, TOOL_FAILED, UNDECODABLE
from .run import require, run


@dataclass
class Probe:
    data: dict | None
    refusal: str | None

    @property
    def streams(self) -> list[dict]:
        return list((self.data or {}).get("streams") or [])

    @property
    def format(self) -> dict:
        return dict((self.data or {}).get("format") or {})

    def first(self, codec_type: str) -> dict | None:
        for stream in self.streams:
            if stream.get("codec_type") == codec_type:
                return stream
        return None


def _refusal(stderr: str, path: str) -> str:
    first = next((line.strip() for line in stderr.splitlines() if line.strip()), "")
    if first.startswith("["):
        first = first.split("]", 1)[-1].strip()
    prefix = f"{path}: "
    if first.startswith(prefix):
        first = first[len(prefix):]
    return first[:200]


async def probe(path: str, *, timeout: float) -> Probe:
    done = await run(
        [require("ffprobe"), "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
        timeout=timeout,
    )
    if done.code != 0:
        return Probe(None, _refusal(done.stderr_text, path) or f"ffprobe exit {done.code}")
    try:
        return Probe(json.loads(done.stdout.decode(errors="replace") or "{}"), None)
    except json.JSONDecodeError as exc:
        raise EmsError(TOOL_FAILED, "ffprobe returned unparseable JSON") from exc


async def decode_failure(path: str, message: str, *, timeout: float) -> EmsError:
    """The error for a decode that failed: whose fault it was, asked of ffprobe.

    When ffprobe cannot read the file either, the bytes are the problem and will be on every
    retry — `undecodable`, permanent. Otherwise the file is readable and something else went
    wrong (a decoder, the card, the moment) — `tool_failed`, which a caller may retry.
    """
    try:
        probed = await probe(path, timeout=timeout)
    except EmsError:
        return EmsError(TOOL_FAILED, message)
    if probed.refusal:
        return EmsError(UNDECODABLE, f"{message}: {probed.refusal}")
    return EmsError(TOOL_FAILED, message)


def fraction(value: str | None) -> float | None:
    """`30000/1001` → 29.97; `0/0` and `N/A` → None."""
    if not value or value == "N/A":
        return None
    num, _, den = value.partition("/")
    try:
        n, d = float(num), float(den) if den else 1.0
    except ValueError:
        return None
    if d == 0 or n <= 0:
        return None
    return n / d


def ratio(value: str | None) -> float | None:
    """`4:3` → 1.333…; `1:1`, `0:1` and `N/A` → None (nothing to apply)."""
    if not value or value == "N/A":
        return None
    a, _, b = value.partition(":")
    try:
        x, y = float(a), float(b)
    except ValueError:
        return None
    if x <= 0 or y <= 0 or x == y:
        return None
    return x / y


def number(value) -> float | None:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None
