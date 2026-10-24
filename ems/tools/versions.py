"""What version of each tool answers. Read once; it is what `meta.producer` names."""

from __future__ import annotations

import asyncio
import re

from .run import run, which

_cache: dict[str, str] = {}
_lock = asyncio.Lock()


async def _first_line(args: list[str]) -> str:
    try:
        done = await run(args, timeout=20)
    except Exception:  # noqa: BLE001 - a version string is never worth failing for
        return "?"
    text = (done.stdout or done.stderr).decode(errors="replace").strip().splitlines()
    return text[0].strip() if text else "?"


async def tool_version(name: str) -> str:
    """`exiftool` → `13.10`, `ffprobe` → `9.0.1`, `fpcalc` → `1.5.1`; `absent` when not installed."""
    async with _lock:
        if name in _cache:
            return _cache[name]
        if which(name) is None:
            _cache[name] = "absent"
            return "absent"
        if name == "exiftool":
            version = await _first_line(["exiftool", "-ver"])
        elif name in ("ffmpeg", "ffprobe"):
            line = await _first_line([name, "-version"])
            m = re.search(r"version\s+(\S+)", line)
            version = m.group(1) if m else line
        elif name == "fpcalc":
            line = await _first_line(["fpcalc", "-version"])
            m = re.search(r"(\d+\.\d+(?:\.\d+)?)", line)
            version = m.group(1) if m else line
        elif name in ("convert", "magick"):
            line = await _first_line([name, "-version"])
            m = re.search(r"ImageMagick\s+(\S+)", line)
            version = m.group(1) if m else line
        else:
            version = await _first_line([name, "--version"])
        _cache[name] = version
        return version


async def producer(*names: str) -> str:
    parts = []
    for name in names:
        parts.append(f"{name}/{await tool_version(name)}")
    return "+".join(parts)
