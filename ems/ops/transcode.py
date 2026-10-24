"""`video.transcode` and `audio.transcode` — the web renditions of a clip and a recording.

The video op runs the three-stage ladder and the windowed CPU encode `ems/tools/encode.py`
plans; the audio op keeps a rendition from carrying more than its source did. Both write
through a private temporary file beside the output and rename it into place, so a caller
never sees a partial rendition and a failed or refused encode leaves nothing behind.
"""

from __future__ import annotations

import os
import re
import secrets
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import Field

from ..context import Context
from ..errors import EmsError, REFUSED, TOOL_FAILED, UNDECODABLE, invalid
from ..schemas import FileRef, OutputSpec, Strict, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import encode
from ..tools import ffmpeg as ffmpeg_tool
from ..tools import ffprobe as ffprobe_tool
from ..tools.run import require, run, run_streaming
from ..tools.versions import producer

# ---- video ----


class VideoSpec(Strict):
    codec: Literal["av1"] = "av1"
    box: "BoxSize | None" = None
    quality: int = Field(default=32, ge=0, le=63)
    speed: int = Field(default=8, ge=0, le=13)
    deinterlace: bool = False


class BoxSize(Strict):
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


class AudioSpec(Strict):
    codec: Literal["aac"] = "aac"
    bitrate: str = "128k"
    sample_rate: int | None = Field(default=None, gt=0)


class Clip(Strict):
    start: float = Field(default=0, ge=0)
    duration: float | None = Field(default=None, gt=0)


class Hints(Strict):
    source_codec: str | None = None


class VideoTranscodeRequest(WithOptions):
    file: FileRef
    output: OutputSpec
    video: VideoSpec = Field(default_factory=VideoSpec)
    audio: AudioSpec | None = Field(default_factory=AudioSpec)
    clip: Clip | None = None
    hints: Hints = Field(default_factory=Hints)


VideoSpec.model_rebuild()

_encoder_cache: dict[str, str] = {}


async def gpu_encoder(ctx: Context) -> str | None:
    """The NVENC encoder this EMS can use, or None: the setting when it names one, else av1_nvenc
    when the card and the build offer it. Probed once per process; the CPU chain backs a wrong
    answer either way, at the cost of one failed ffmpeg."""
    name = ctx.settings.video_encoder
    if name:
        return name if name.endswith("_nvenc") else None
    if "auto" not in _encoder_cache:
        found = ""
        if ffmpeg_tool.gpu_present():
            done = await run([require("ffmpeg"), "-hide_banner", "-encoders"], timeout=30)
            if done.code == 0 and re.search(r"\bav1_nvenc\b", done.stdout.decode(errors="replace")):
                found = "av1_nvenc"
        _encoder_cache["auto"] = found
    return _encoder_cache["auto"] or None


def _seek_args(clip: Clip | None) -> list[str]:
    # -ss before -i: an input-side seek jumps to the nearest preceding keyframe and decodes
    # forward (frame-accurate, -accurate_seek is the default); -t is output-side.
    return ["-ss", f"{clip.start:g}"] if clip and clip.start > 0 else []


def _clip_args(clip: Clip | None) -> list[str]:
    return ["-t", f"{clip.duration:g}"] if clip and clip.duration else []


def _audio_args(audio: AudioSpec | None) -> list[str]:
    # AAC, not Opus: Safari cannot decode Opus in MP4, and this rendition is what the UI plays.
    # A muted rendition (the hover clip) drops the stream — nothing autoplays with sound.
    return ["-an"] if audio is None else ["-c:a", "aac", "-b:a", audio.bitrate]


def _common_args(audio: AudioSpec | None) -> list[str]:
    # +faststart puts the moov atom first so playback starts before the whole file has arrived.
    return [*_audio_args(audio), "-movflags", "+faststart", "-y", "-f", "mp4"]


class Encoder:
    """One ffmpeg run with progress, killed with the request."""

    def __init__(self, ctx: Context, progress: Progress, total_seconds: float | None):
        self.ctx = ctx
        self.progress = progress
        self.total = total_seconds
        self.timeout = ctx.settings.tool_stall_seconds

    async def run(self, args: list[str], *, phase: str, base: float = 0.0, span: float = 1.0):
        async def on_line(line: str) -> None:
            if line.startswith("out_time_us=") and self.total:
                try:
                    done = int(line.split("=", 1)[1]) / 1_000_000
                except ValueError:
                    return
                await self.progress.emit(phase, fraction=base + span * min(1.0, done / self.total))

        return await run_streaming([require("ffmpeg"), "-hide_banner", "-nostats", "-progress", "pipe:1", *args], timeout=self.timeout, on_line=on_line)


async def _measure(ctx: Context, path: str) -> tuple[ffprobe_tool.Probe, float | None, tuple[int, int] | None, str | None]:
    """Duration, frame rate and picture codec of the source, from one probe."""
    probe = await ffprobe_tool.probe(path, timeout=ctx.settings.tool_stall_seconds)
    if probe.refusal:
        raise EmsError(UNDECODABLE, f"source cannot be read: {probe.refusal}")
    video = probe.first("video") or {}
    duration = ffprobe_tool.number(probe.format.get("duration"))
    fps = None
    raw = str(video.get("r_frame_rate") or "")
    num, _, den = raw.partition("/")
    try:
        n, d = int(num), int(den) if den else 1
        if n > 0 and d > 0:
            fps = (n, d)
    except ValueError:
        fps = None
    return probe, duration, fps, (str(video.get("codec_name")) if video.get("codec_name") else None)


def _describe_output(probe: ffprobe_tool.Probe, path: str, requested: str) -> dict:
    video = probe.first("video") or {}
    audio = probe.first("audio") or {}
    return {
        "path": requested,
        "width": video.get("width"),
        "height": video.get("height"),
        "duration": ffprobe_tool.number(probe.format.get("duration")),
        "bytes": os.path.getsize(path),
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
    }


async def _concat(ctx: Context, segments: list[str], audio: str | None, out: str, timeout: float) -> None:
    """Join encoded windows by stream copy; the audio, encoded once over the whole clip, is muxed in.

    The concat demuxer's list file names the segments directly. `-safe 0` because they are
    absolute. The maps are explicit: the segments were encoded `-an`, so stream 0 carries video
    alone and the audio, when there is any, is a second input.
    """
    listing = out + ".txt"
    with open(listing, "w") as f:
        for segment in segments:
            f.write("file '" + segment.replace("'", "'\\''") + "'\n")
    try:
        args = [require("ffmpeg"), "-hide_banner", "-f", "concat", "-safe", "0", "-i", listing]
        if audio:
            args += ["-i", audio, "-map", "0:v:0", "-map", "1:a:0"]
        else:
            args += ["-map", "0:v:0"]
        done = await run(args + ["-c", "copy", "-movflags", "+faststart", "-y", "-f", "mp4", out], timeout=timeout)
        if done.code != 0:
            raise EmsError(TOOL_FAILED, f"joining {len(segments)} windows failed: {done.tail()}")
    finally:
        try:
            os.remove(listing)
        except FileNotFoundError:
            pass


async def transcode_video(ctx: Context, req: VideoTranscodeRequest, progress: Progress) -> Outcome:
    src = ctx.roots.input(req.file.path)
    final = ctx.roots.output(req.output.path)
    fmt = (req.output.format or "mp4").lower()
    if fmt != "mp4":
        raise invalid(f"{req.output.path}: video renditions are mp4, not {fmt}")
    shape = encode.Shape(
        width=req.video.box.width if req.video.box else None,
        height=req.video.box.height if req.video.box else None,
        angle=req.file.angle,
        mirror=req.file.mirror,
        deinterlace=req.video.deinterlace,
    )
    settings = ctx.settings
    budget = encode.Budget(settings.cpu_encode_max_mb, settings.cpu_encode_mb_per_1k_frames, settings.cpu_encode_max_segments)
    clip = req.clip
    temp = os.path.join(os.path.dirname(final), f".ems-{secrets.token_hex(8)}.mp4")
    scratch: list[str] = []
    ran = ""

    async with ctx.admission.slot("encodes"):
        try:
            await progress.emit("probe")
            probe, duration, fps, probed_codec = await _measure(ctx, src)
            source_codec = req.hints.source_codec or probed_codec
            clip_seconds = clip.duration if clip and clip.duration else None
            total = clip_seconds if clip_seconds else (max(0.0, duration - (clip.start if clip else 0)) if duration else None)
            enc = Encoder(ctx, progress, total)
            encoder = await gpu_encoder(ctx)
            gpu_args = encode.gpu_encode_args(encoder, req.video.quality, cq_offset=settings.nvenc_cq_offset, preset=settings.nvenc_preset) if encoder else []
            cpu_args = encode.cpu_encode_args(req.video.quality, req.video.speed)
            cpu_chain = encode.cpu_filters(shape)
            cuda_chain = encode.cuda_filters(shape)
            head = ["-display_rotation", "0"]
            common = _common_args(req.audio)
            failures: list[str] = []

            # Stage one: everything in VRAM.
            if encoder and cuda_chain and source_codec not in encode.NVDEC_CANNOT_DECODE:
                await progress.emit("encode", message="gpu")
                done = await enc.run([*head, "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", *_seek_args(clip), "-i", src, *_clip_args(clip), "-vf", cuda_chain, *gpu_args, *common, temp], phase="encode")
                if done.code == 0:
                    ran = f"{encoder}/cuda"
                else:
                    failures.append(f"gpu decode: {done.tail(2, 300)}")

            if not ran:
                # Both remaining stages decode on the CPU, and the CPU decode is what accumulates —
                # so the budget is projected once here and gates both of them.
                seconds = encode.seconds_to_decode(duration, start=clip.start if clip else 0.0, clip_seconds=clip_seconds)
                if seconds is None or fps is None:
                    raise EmsError(REFUSED, f"cannot measure the source (duration={duration!r} fps={fps!r}), refusing a CPU decode", permanent=True)
                whole = clip is None or (float(clip.start).is_integer() and (clip.duration is None or float(clip.duration).is_integer()))
                plan = encode.plan_cpu_decode(seconds, fps[0], fps[1], budget, clipped_whole_seconds=whole)
                if isinstance(plan, encode.Refusal):
                    raise EmsError(REFUSED, plan.reason, permanent=True)

                upload_chain = encode.upload_filters(shape)
                if plan.windows > 1:
                    await progress.emit("encode", message=f"{plan.windows} windows of {plan.window_seconds}s on the CPU")
                    ran = await _encode_windows(enc, plan, src, temp, scratch, clip, seconds, cpu_chain, upload_chain, gpu_args, cpu_args, encoder, req.audio, probe, settings.tool_stall_seconds)
                else:
                    # Stage two: the CPU chain with the frames handed to VRAM at the end of it.
                    if encoder:
                        await progress.emit("encode", message="cpu decode, gpu encode")
                        done = await enc.run([*head, *_seek_args(clip), "-i", src, *_clip_args(clip), "-vf", upload_chain, *gpu_args, *common, temp], phase="encode")
                        if done.code == 0:
                            ran = f"{encoder}/upload"
                        else:
                            failures.append(f"{encoder}: {done.tail(2, 300)}")
                    # Stage three: the whole job on the CPU.
                    if not ran:
                        await progress.emit("encode", message="cpu")
                        done = await enc.run([*head, *_seek_args(clip), "-i", src, *_clip_args(clip), "-vf", cpu_chain, *cpu_args, *common, temp], phase="encode")
                        if done.code == 0:
                            ran = encode.CPU_ENCODER
                        else:
                            failures.append(f"{encode.CPU_ENCODER}: {done.tail(2, 300)}")
                            raise EmsError(TOOL_FAILED, "encoding failed: " + " | ".join(failures))

            if not os.path.isfile(temp) or os.path.getsize(temp) == 0:
                raise EmsError(TOOL_FAILED, "ffmpeg exited 0 but wrote no bytes")
            result_probe = await ffprobe_tool.probe(temp, timeout=settings.tool_stall_seconds)
            os.replace(temp, final)
        finally:
            for path in [temp, *scratch]:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
    result = _describe_output(result_probe, final, req.output.path)
    return Outcome(result, producer=f"{await producer('ffmpeg')}/{ran}")


async def _encode_windows(enc: Encoder, plan: encode.Plan, src: str, temp: str, scratch: list[str], clip: Clip | None, seconds: int, cpu_chain: str, upload_chain: str, gpu_args: list[str], cpu_args: list[str], encoder: str | None, audio: AudioSpec | None, probe: ffprobe_tool.Probe, timeout: float) -> str:
    """One ffmpeg process per window, joined afterwards by a stream copy.

    Inside a window the same two CPU-decode stages apply, and the one that makes the first
    window makes all of them: a window that fell from NVENC to libsvtav1 mid-file would carry
    different AV1 extradata than its neighbours, the one thing `-c copy` cannot reconcile.

    Audio is encoded once over the whole clip rather than per window: per-window AAC pads each
    one to a whole 1024-sample frame, so every boundary contributes up to 21 ms of drift.
    """
    base = temp[: -len(".mp4")]
    start = int(clip.start) if clip else 0
    head = ["-display_rotation", "0"]
    audio_path: str | None = None
    if audio is not None and probe.first("audio") is not None:
        audio_path = f"{base}-audio.mp4"
        scratch.append(audio_path)
        await enc.progress.emit("encode", message="audio")
        done = await enc.run([*_seek_args(clip), "-i", src, *_clip_args(clip), "-vn", *_audio_args(audio), "-y", "-f", "mp4", audio_path], phase="encode", span=0.0)
        if done.code != 0:
            raise EmsError(TOOL_FAILED, f"the audio pass failed: {done.tail(2, 300)}")

    assert plan.window_seconds is not None
    segments: list[str] = []
    stage = ""
    for index in range(plan.windows):
        at = start + index * plan.window_seconds
        remaining = seconds - index * plan.window_seconds
        # The last window runs to the end of what was asked for; a probed duration was rounded
        # up when the plan was made, so its sub-second tail is inside the final window's budget.
        length: list[str] = ["-t", str(plan.window_seconds)]
        if index + 1 == plan.windows:
            length = ["-t", str(remaining)] if clip and clip.duration else []
        segment = f"{base}-seg-{index}.mp4"
        scratch.append(segment)
        segments.append(segment)
        span = 1.0 / plan.windows
        ok = False
        if encoder and stage != "cpu":
            done = await enc.run([*head, "-ss", str(at), "-i", src, *length, "-vf", upload_chain, *gpu_args, "-an", "-y", "-f", "mp4", segment], phase="encode", base=index * span, span=span)
            ok = done.code == 0
            if ok:
                stage = "gpu"
        if not ok and stage != "gpu":
            done = await enc.run([*head, "-ss", str(at), "-i", src, *length, "-vf", cpu_chain, *cpu_args, "-an", "-y", "-f", "mp4", segment], phase="encode", base=index * span, span=span)
            ok = done.code == 0
            if ok:
                stage = "cpu"
        if not ok:
            raise EmsError(TOOL_FAILED, f"window {index + 1} of {plan.windows} failed: {done.tail(2, 300)}")
    await enc.progress.emit("join")
    await _concat(enc.ctx, segments, audio_path, temp, timeout)
    return f"{encoder}/upload" if stage == "gpu" else encode.CPU_ENCODER


# ---- audio ----


class AudioTranscodeSpec(Strict):
    codec: Literal["aac"] = "aac"
    bitrate: str = "128k"
    sample_rate: int = Field(default=44100, gt=0)


class AudioTranscodeRequest(WithOptions):
    file: FileRef
    output: OutputSpec
    audio: AudioTranscodeSpec = Field(default_factory=AudioTranscodeSpec)


def bits_per_second(bitrate: str) -> int:
    text = bitrate.strip().lower()
    try:
        return int(float(text[:-1]) * 1000) if text.endswith("k") else int(text)
    except ValueError as exc:
        raise invalid(f"bitrate {bitrate!r}: expected a number of bits per second or `128k`") from exc


def audio_plan(stream: dict | None, fmt: str, spec: AudioTranscodeSpec) -> tuple[list[str], dict]:
    """The encode arguments, as ceilings against the source.

    Nothing in a rendition can add information the original never held, so bitrate and sample
    rate are min(target, source), the channel count is the source's, and a source that is already
    AAC in an MP4-family container is remuxed rather than re-encoded (a generation of lossy
    loss and most of the CPU saved; an ADTS `.aac` stream is not, since it is not seekable the
    way the player needs).
    """
    stream = stream or {}
    target = bits_per_second(spec.bitrate)
    src_bps = ffprobe_tool.number(stream.get("bit_rate"))
    src_rate = ffprobe_tool.number(stream.get("sample_rate"))
    channels = stream.get("channels")
    bps = int(min(target, src_bps)) if src_bps else target
    rate = int(min(spec.sample_rate, src_rate)) if src_rate else spec.sample_rate
    remux = stream.get("codec_name") == "aac" and any(name in fmt for name in ("mp4", "m4a", "mov"))
    if remux:
        args = ["-vn", "-c:a", "copy"]
        facts = {"bitrate": int(src_bps) if src_bps else None, "sample_rate": int(src_rate) if src_rate else None, "channels": channels, "remuxed": True}
    else:
        args = ["-vn", "-c:a", "aac", "-profile:a", "aac_low", "-b:a", str(bps), "-ar", str(rate)]
        if channels == 1:
            args += ["-ac", "1"]
        facts = {"bitrate": bps, "sample_rate": rate, "channels": channels, "remuxed": False}
    # The ipod muxer, not mp4: both write an MP4 container, but ipod's ftyp brand is `M4A `,
    # which is what marks the file as audio to anything that sniffs the first 32 bytes.
    return args + ["-movflags", "+faststart", "-y", "-f", "ipod"], facts


async def transcode_audio(ctx: Context, req: AudioTranscodeRequest, progress: Progress) -> Outcome:
    src = ctx.roots.input(req.file.path)
    final = ctx.roots.output(req.output.path)
    fmt = (req.output.format or "m4a").lower()
    if fmt not in ("m4a", "mp4"):
        raise invalid(f"{req.output.path}: audio renditions are m4a, not {fmt}")
    temp = os.path.join(os.path.dirname(final), f".ems-{secrets.token_hex(8)}.m4a")
    async with ctx.admission.slot("encodes"):
        try:
            await progress.emit("probe")
            probe = await ffprobe_tool.probe(src, timeout=ctx.settings.tool_stall_seconds)
            if probe.refusal:
                raise EmsError(UNDECODABLE, f"source cannot be read: {probe.refusal}")
            stream = probe.first("audio")
            if stream is None:
                raise EmsError(REFUSED, "the file has no audio stream", permanent=True)
            args, facts = audio_plan(stream, str(probe.format.get("format_name") or ""), req.audio)
            total = ffprobe_tool.number(probe.format.get("duration"))
            await progress.emit("encode", message="remux" if facts["remuxed"] else "aac")
            done = await Encoder(ctx, progress, total).run(["-i", src, *args, temp], phase="encode")
            if done.code != 0:
                raise EmsError(TOOL_FAILED, f"encoding failed: {done.tail(2, 300)}")
            if not os.path.isfile(temp) or os.path.getsize(temp) == 0:
                raise EmsError(TOOL_FAILED, "ffmpeg exited 0 but wrote no bytes")
            out_probe = await ffprobe_tool.probe(temp, timeout=ctx.settings.tool_stall_seconds)
            size = os.path.getsize(temp)
            os.replace(temp, final)
        finally:
            try:
                os.remove(temp)
            except FileNotFoundError:
                pass
    out_stream = out_probe.first("audio") or {}
    result = {
        "path": req.output.path,
        "duration": ffprobe_tool.number(out_probe.format.get("duration")),
        "bytes": size,
        "bitrate": int(ffprobe_tool.number(out_stream.get("bit_rate")) or 0) or facts["bitrate"],
        "sample_rate": int(ffprobe_tool.number(out_stream.get("sample_rate")) or 0) or facts["sample_rate"],
        "channels": out_stream.get("channels") or facts["channels"],
        "remuxed": facts["remuxed"],
    }
    return Outcome(result, producer=f"{await producer('ffmpeg')}/{'copy' if facts['remuxed'] else 'aac'}")


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/video/transcode")
    async def video_transcode(request: Request, body: VideoTranscodeRequest):
        return await run_op(request, lambda progress: transcode_video(ctx, body, progress))

    @router.post("/audio/transcode")
    async def audio_transcode(request: Request, body: AudioTranscodeRequest):
        return await run_op(request, lambda progress: transcode_audio(ctx, body, progress))
