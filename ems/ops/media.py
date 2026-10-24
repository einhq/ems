"""`media.probe` — what is this file? One call, every tool, one normalized answer."""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
from typing import Literal

from fastapi import APIRouter, Request

from ..context import Context
from ..errors import EmsError, TOOL_FAILED
from ..schemas import FileRef, WithOptions
from ..streaming import Outcome, Progress, run_op
from ..tools import ffprobe as ffprobe_tool
from ..tools.run import which
from ..tools.versions import producer

DOCUMENT_MIMETYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.oasis.opendocument.text",
}

# In the order a caller should trust them. Values are exiftool's own strings; `zone` is filled
# from the matching offset tag when the file carries one.
CAPTURED_AT_TAGS = (
    ("DateTimeOriginal", "OffsetTimeOriginal"),
    ("CreateDate", "OffsetTimeDigitized"),
    ("CreationDate", None),
    ("MediaCreateDate", None),
    ("TrackCreateDate", None),
    ("DateTimeCreated", None),
    ("GPSDateTime", None),
    ("ModifyDate", "OffsetTime"),
)

# EXIF Orientation → (degrees clockwise to display upright, mirrored first).
ORIENTATION = {1: (0, False), 2: (0, True), 3: (180, False), 4: (180, True), 5: (90, True), 6: (90, False), 7: (270, True), 8: (270, False)}


class ProbeRequest(WithOptions):
    file: FileRef
    raw: list[Literal["exiftool", "ffprobe"]] | None = None
    hash: bool = False


def kind_of(mimetype: str | None) -> str:
    if not mimetype:
        return "other"
    if mimetype.startswith("image/"):
        return "image"
    if mimetype.startswith("video/"):
        return "video"
    if mimetype.startswith("audio/"):
        return "audio"
    if mimetype in DOCUMENT_MIMETYPES or mimetype.startswith("text/"):
        return "document"
    return "other"


def _num(value) -> float | None:
    return ffprobe_tool.number(value)


def _serial(exif: dict) -> str | None:
    for tag in ("SerialNumber", "InternalSerialNumber", "DeviceSerialNo"):
        value = exif.get(tag)
        if value is not None:
            text = str(value).replace("\x00", "").strip()
            if text:
                return text
    return None


def _captured_at(exif: dict) -> list[dict]:
    out = []
    for tag, offset_tag in CAPTURED_AT_TAGS:
        value = exif.get(tag)
        if value in (None, "", 0, "0000:00:00 00:00:00"):
            continue
        entry: dict = {"value": str(value), "source": tag}
        zone = exif.get(offset_tag) if offset_tag else None
        if zone:
            entry["zone"] = str(zone)
        out.append(entry)
    return out


def _gps(exif: dict) -> dict | None:
    lat, lon = exif.get("GPSLatitude"), exif.get("GPSLongitude")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    gps: dict = {"lat": float(lat), "lon": float(lon)}
    alt = exif.get("GPSAltitude")
    if isinstance(alt, (int, float)):
        gps["alt"] = float(alt)
    dop = exif.get("GPSDOP")
    if isinstance(dop, (int, float)):
        gps["dop"] = float(dop)
    return gps


def _picture(exif: dict, video_stream: dict | None) -> dict | None:
    picture: dict = {}
    if video_stream:
        w, h = _num(video_stream.get("width")), _num(video_stream.get("height"))
        if w and h:
            picture["width"], picture["height"] = int(w), int(h)
        codec = video_stream.get("codec_name")
        if codec and codec != "N/A":
            picture["codec"] = codec
        sar = ffprobe_tool.ratio(video_stream.get("sample_aspect_ratio"))
        picture["sar"] = round(sar, 6) if sar else 1
        fps = ffprobe_tool.fraction(video_stream.get("avg_frame_rate")) or ffprobe_tool.fraction(video_stream.get("r_frame_rate"))
        if fps:
            picture["fps"] = round(fps, 3)
        field_order = video_stream.get("field_order")
        picture["interlaced"] = bool(field_order and field_order not in ("progressive", "unknown"))
        for side in video_stream.get("side_data_list") or []:
            if "rotation" in side:
                picture["matrix_rotation"] = side["rotation"]
    if "width" not in picture:
        w, h = _num(exif.get("ImageWidth")), _num(exif.get("ImageHeight"))
        if w and h:
            picture["width"], picture["height"] = int(w), int(h)
    rotation = exif.get("Rotation")
    if isinstance(rotation, (int, float)):
        picture["rotation"] = int(rotation) % 360
    orientation = exif.get("Orientation")
    if isinstance(orientation, (int, float)) and int(orientation) in ORIENTATION:
        degrees, mirrored = ORIENTATION[int(orientation)]
        picture.setdefault("rotation", degrees)
        picture["orientation"] = int(orientation)
        if mirrored:
            picture["mirrored"] = True
    return picture or None


def _sound(audio_stream: dict | None) -> dict | None:
    if not audio_stream:
        return None
    sound: dict = {}
    codec = audio_stream.get("codec_name")
    if codec and codec != "N/A":
        sound["codec"] = codec
    channels, rate = _num(audio_stream.get("channels")), _num(audio_stream.get("sample_rate"))
    if channels:
        sound["channels"] = int(channels)
    if rate:
        sound["sample_rate"] = int(rate)
    return sound or None


def normalize(path: str, exif: dict, probe: ffprobe_tool.Probe | None, mimetype_hint: str | None) -> dict:
    mimetype = exif.get("MIMEType") or mimetype_hint or mimetypes.guess_type(path)[0]
    kind = kind_of(mimetype)
    result: dict = {"mimetype": mimetype, "kind": kind, "size": os.path.getsize(path)}

    video_stream = probe.first("video") if probe else None
    audio_stream = probe.first("audio") if probe else None
    fmt = probe.format if probe else {}

    if fmt.get("format_name"):
        result["container"] = fmt["format_name"]
    duration = _num(fmt.get("duration")) or (_num(exif.get("Duration")) if isinstance(exif.get("Duration"), (int, float)) else None)
    if duration:
        result["duration"] = round(duration, 3)
    bitrate = _num(fmt.get("bit_rate"))
    if bitrate:
        result["bitrate"] = int(bitrate)

    picture = _picture(exif, video_stream)
    if picture:
        result["picture"] = picture
    sound = _sound(audio_stream)
    if sound:
        result["sound"] = sound

    result["captured_at"] = _captured_at(exif)
    gps = _gps(exif)
    if gps:
        result["gps"] = gps

    device: dict = {}
    if exif.get("Make"):
        device["make"] = str(exif["Make"]).strip()
    if exif.get("Model"):
        device["model"] = str(exif["Model"]).strip()
    serial = _serial(exif)
    if serial:
        device["serial"] = serial
    if device:
        result["device"] = device

    if probe is not None and probe.refusal:
        result["decodable"] = False
        result["reason"] = probe.refusal
    else:
        result["decodable"] = True
    return result


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def probe_file(ctx: Context, req: ProbeRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    timeout = ctx.settings.tool_stall_seconds
    async with ctx.admission.slot("tools", interactive=req.options.interactive):
        await progress.emit("metadata")
        exif: dict = {}
        exif_error: EmsError | None = None
        if ctx.exiftool is not None:
            try:
                exif = await ctx.exiftool.extract(path)
            except EmsError as exc:
                exif_error = exc
        mimetype = exif.get("MIMEType") or req.file.mimetype or mimetypes.guess_type(path)[0]
        kind = kind_of(mimetype)

        probe: ffprobe_tool.Probe | None = None
        warnings: list[str] = []
        if kind in ("video", "audio") or (not exif and kind != "document"):
            if which("ffprobe"):
                await progress.emit("streams")
                probe = await ffprobe_tool.probe(path, timeout=timeout)
            else:
                warnings.append("ffprobe is not installed: stream facts and decodability were not measured")
        if not exif and (probe is None or probe.data is None):
            # Neither tool could read it; say so with whichever error is more specific.
            if exif_error is not None:
                raise exif_error
            raise EmsError(TOOL_FAILED, f"{req.file.path}: no tool could read this file", permanent=True)

        result = normalize(path, exif, probe, req.file.mimetype)
        if req.hash:
            await progress.emit("hash")
            result["sha256"] = await asyncio.to_thread(_sha256, path)

    wanted = set(req.raw) if req.raw is not None else {"exiftool", "ffprobe"}
    raw: dict = {}
    if "exiftool" in wanted and exif:
        raw["exiftool"] = exif
    if "ffprobe" in wanted and probe is not None and probe.data is not None:
        raw["ffprobe"] = probe.data
    result["raw"] = raw
    meta = {"warnings": warnings} if warnings else {}
    return Outcome(result, producer=await producer("exiftool", "ffprobe"), meta=meta)


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/media/probe")
    async def media_probe(request: Request, body: ProbeRequest):
        return await run_op(request, lambda progress: probe_file(ctx, body, progress))
