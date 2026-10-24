"""The video transcode, planned: filter chains, the encoder ladder and the CPU decode budget.

Three stages, each giving up strictly less of the GPU than the one below it:

1. **Full VRAM** — NVDEC decode, CUDA filters, NVENC encode. Needs transpose_cuda (ffmpeg 9).
2. **CPU decode, GPU encode** — the CPU chain, `hwupload_cuda`, NVENC. Keeps the encoder when
   only the decoder was the problem: NVDEC has no decoder for DV, MJPEG, Indeo or ProRes, a
   quarter of any old archive, and AV1 on the CPU is the expensive half.
3. **All CPU** — libsvtav1.

The CPU decode is what accumulates: RSS grows linearly with the frames pushed through it, about
0.17 MB per frame on either CPU stage since the decode is what they share, and is never given
back. Neither the encoder's parallelism nor malloc's arenas is the cause, so the bound is on the
job: the footprint is projected from the frame count before anything starts, and a source over the
budget is encoded as *windows* — a fresh process per window resets the accumulator — joined by
a stream copy. Only a source whose windows cannot be placed is refused, and a refusal is
permanent for that source at that budget.

`-display_rotation 0` is on every stage, and it is the flag that makes them agree: ffmpeg
autorotates from a container's display matrix by default but skips it when the decoder hands
back hardware frames, so without it stage one would not rotate while the CPU stages did. Not
`-noautorotate`, which suppresses the turn but passes the matrix on to the output for the
browser to apply. The caller's `angle` is the only thing that turns pixels — the same rule as
every other op.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Codecs NVDEC has no usable decoder for: each of these fails the full-VRAM stage outright, so
# it is skipped up front rather than learnt from a failed ffmpeg. Per codec, not per family: plain
# mpeg4 decodes and its MS variant does not. An unknown codec is not on this list and must not
# be — the ladder still finds every case this misses.
NVDEC_CANNOT_DECODE = frozenset({"dvvideo", "mjpeg", "prores", "h263", "msmpeg4v3"})

CPU_ENCODER = "libsvtav1"


@dataclass(frozen=True)
class Shape:
    """What is done to the picture, independent of where."""

    width: int | None = None
    height: int | None = None
    angle: int = 0
    mirror: bool = False
    deinterlace: bool = False


def scale_spec(width: int | None, height: int | None) -> str | None:
    """Fit inside the box, never upscale, keep the aspect ratio, stay yuv420p-encodable.

    `scale=W:H` forces exact dimensions and stretches anything that is not the box's shape;
    force_original_aspect_ratio fits instead, min() stops a small source being blown up, and
    force_divisible_by=2 keeps odd dimensions out of the encoder. Commas inside min() are
    escaped for ffmpeg's filtergraph parser.
    """
    if width and height:
        return f"w=min(iw\\,{width}):h=min(ih\\,{height}):force_original_aspect_ratio=decrease:force_divisible_by=2"
    if width:
        return f"w=min(iw\\,{width}):h=-2"
    if height:
        return f"w=-2:h=min(ih\\,{height})"
    return None


def turn_filters(angle: int, mirror: bool) -> list[str]:
    """The display frame's turn on the CPU. It is rotate-then-flop, so hflip comes after transpose."""
    parts = {90: ["transpose=2"], 270: ["transpose=1"], 180: ["transpose=1", "transpose=1"]}.get(angle, [])
    return parts + (["hflip"] if mirror else [])


def cpu_filters(shape: Shape) -> str:
    """The CPU chain."""
    parts: list[str] = []
    if shape.deinterlace:
        parts.append("yadif")
    spec = scale_spec(shape.width, shape.height)
    if spec:
        parts.append(f"scale={spec}")
    parts += turn_filters(shape.angle, shape.mirror)
    return ",".join(parts) or "null"


def cuda_filters(shape: Shape) -> str | None:
    """The same chain entirely in VRAM, or None when it cannot be expressed there.

    transpose_cuda's dir values match the CPU filter's for 0-3 and add dir=reversal for a
    half-turn. There is no hflip_cuda (only hflip_vulkan, another hwaccel), so a mirrored
    source cannot stay in VRAM end to end and gets no CUDA chain rather than an approximation;
    it still reaches NVENC through the upload chain.
    """
    if shape.mirror:
        return None
    parts: list[str] = []
    if shape.deinterlace:
        parts.append("yadif_cuda")
    spec = scale_spec(shape.width, shape.height)
    # No resize still needs one conversion so NVENC gets yuv420p rather than the decoder's format.
    parts.append(f"scale_cuda={spec}:format=yuv420p" if spec else "scale_cuda=format=yuv420p")
    parts += {90: ["transpose_cuda=dir=2"], 270: ["transpose_cuda=dir=1"], 180: ["transpose_cuda=dir=4"]}.get(shape.angle, [])
    return ",".join(parts)


def upload_filters(shape: Shape) -> str:
    """Stage two: the CPU chain, then the frames handed to VRAM in a format NVENC accepts."""
    return f"{cpu_filters(shape)},format=yuv420p,hwupload_cuda"


def cpu_encode_args(quality: int, speed: int) -> list[str]:
    return ["-c:v", CPU_ENCODER, "-preset", str(speed), "-crf", str(quality), "-pix_fmt", "yuv420p"]


def gpu_encode_args(encoder: str, quality: int, *, cq_offset: int, preset: str) -> list[str]:
    """NVENC spells quality -cq under VBR; its presets are p1..p7, so the SVT-AV1 speed is not passed.

    cq is 0-63 like SVT-AV1's crf but not calibrated to it: `cq_offset` (6 by default) keeps
    the same `quality` meaning roughly the same bytes whichever encoder runs.
    """
    return ["-c:v", encoder, "-preset", preset, "-rc", "vbr", "-cq", str(quality + cq_offset)]


@dataclass(frozen=True)
class Budget:
    max_mb: int
    mb_per_1k_frames: int
    max_segments: int

    @property
    def signature(self) -> str:
        """What a caller records with a refusal, so a raised budget re-opens the gap on its own."""
        return f"{self.max_mb}/{self.mb_per_1k_frames}/{self.max_segments}"

    @property
    def valid(self) -> bool:
        return self.max_mb >= 0 and self.mb_per_1k_frames > 0 and self.max_segments > 0


@dataclass(frozen=True)
class Plan:
    """How the CPU-decode stages run: whole, or as `windows` of `window_seconds` each."""

    frames: int
    projected_mb: int
    windows: int = 1
    window_seconds: int | None = None


@dataclass(frozen=True)
class Refusal:
    reason: str


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def project_cpu_decode(seconds: int, fps_num: int, fps_den: int, budget: Budget) -> tuple[int, int] | Refusal:
    """The frames and megabytes a whole CPU decode of `seconds` projects to, or why the budget allows none.

    Frames, not seconds: a 60 fps source costs 2.4x a 25 fps one for the same wall duration.
    Every estimate rounds up, keeping a fractional frame or megabyte on the guarded side.
    """
    if not budget.valid:
        return Refusal(f"invalid CPU encode limits ({budget.signature})")
    if budget.max_mb < 1:
        return Refusal(f"CPU decode is disabled by a budget of {budget.max_mb} MB")
    frames = ceil_div(seconds * fps_num, fps_den)
    return frames, ceil_div(frames * budget.mb_per_1k_frames, 1000)


def plan_cpu_decode(seconds: int, fps_num: int, fps_den: int, budget: Budget, *, clipped_whole_seconds: bool) -> Plan | Refusal:
    """Project a CPU decode of `seconds` at the given frame rate against the budget
    (`project_cpu_decode`), and split it into windows when it does not fit whole.

    Three steps to a window plan, and the third is not redundant: the window *length* comes
    from the budget (how many windows the footprint implies, divided into the running time);
    rounding it up to whole seconds can leave the surplus windows starting past the end and
    encoding nothing — a zero-byte segment and a join that fails — so the count is re-derived
    from the length actually chosen. The final projection checks the rounded window itself,
    because a very short, very expensive source may have no whole-second split that fits.

    `clipped_whole_seconds` is False when the clip's start or duration is not a whole number of
    seconds: windows are placed on the source clock in whole seconds, so such a clip cannot be
    split and is refused when it does not fit whole.
    """
    whole = project_cpu_decode(seconds, fps_num, fps_den, budget)
    if isinstance(whole, Refusal):
        return whole
    frames, projected = whole
    if projected <= budget.max_mb:
        return Plan(frames, projected)
    if not clipped_whole_seconds:
        return Refusal(
            f"a CPU decode of {frames} frames projects to {projected} MB, over the {budget.max_mb} MB budget, "
            "and a clip that does not start and end on whole seconds cannot be split into windows"
        )
    count = ceil_div(projected, budget.max_mb)
    window = ceil_div(seconds, count)
    if window > 0:
        count = ceil_div(seconds, window)
    if count > budget.max_segments:
        return Refusal(
            f"a CPU decode of {frames} frames would need {count} windows of {budget.max_mb} MB, over the {budget.max_segments} allowed"
        )
    if window < 1:
        return Refusal(f"a CPU decode of {frames} frames splits into windows shorter than a second")
    window_frames = ceil_div(window * fps_num, fps_den)
    window_mb = ceil_div(window_frames * budget.mb_per_1k_frames, 1000)
    if window_mb > budget.max_mb:
        return Refusal(f"the shortest whole-second window projects to {window_mb} MB, over the {budget.max_mb} MB budget")
    return Plan(frames, projected, count, window)


def seconds_to_decode(duration: float | None, *, start: float, clip_seconds: float | None) -> int | None:
    """How many whole seconds the CPU stages would decode, rounded up; None when unmeasurable.

    A clipped rendition is bounded by its own clip and must not be refused for the length of
    the source it was cut from; a whole-file one is bounded by the source, less the seek.
    """
    if clip_seconds is not None:
        return int(math.ceil(clip_seconds))
    if duration is None or duration <= 0:
        return None
    remaining = duration - start if start < duration else duration
    return int(math.ceil(remaining))
