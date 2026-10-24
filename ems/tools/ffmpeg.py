"""ffmpeg for the speech ops (audio extraction), and whether a GPU is there to decode on.

A GPU is used for decoding when one is present, and the CPU is the fallback for what NVDEC
cannot read.
"""

from __future__ import annotations

import glob
import os

from .ffprobe import decode_failure
from .run import require, run


def gpu_present() -> bool:
    return bool(glob.glob("/dev/nvidia[0-9]*"))


async def extract_audio(path: str, out_wav: str, *, timeout: float, sample_rate: int = 16000, channels: int = 1) -> bool:
    """A PCM WAV of the file's sound at `out_wav`. False when the file has no audio stream.

    ffmpeg exits 0 having written nothing for a container without sound, so the size is checked
    rather than the exit code alone.
    """
    done = await run(
        [require("ffmpeg"), "-y", "-v", "error", "-vn", "-i", path, "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(channels), out_wav],
        timeout=timeout,
    )
    if done.code != 0:
        text = done.stderr_text.lower()
        if "does not contain any stream" in text or "output file is empty" in text or "no audio" in text:
            return False
        raise await decode_failure(path, f"ffmpeg audio extraction failed: {done.tail()}", timeout=timeout)
    try:
        return os.path.getsize(out_wav) > 44
    except FileNotFoundError:
        return False

