"""`speech.*` — transcription, speaker turns, voiceprints and active-speaker scoring.

Every op here takes any container with sound and extracts the audio itself (16 kHz mono PCM
in scratch, cached for a while — see tools/audio.py); the models never see the caller's file.
A file with no audio stream is a normal answer (`speech: false`, `turns: []`, `null`), not an
error — plenty of any archive is silent phone clips.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, Request
from pydantic import Field

from ..context import Context
from ..schemas import Box, FileRef, Strict, WithOptions
from ..streaming import Outcome, Progress, run_op
from ._models import model_slot
from ._voices import merge_adjacent, speakers_of, stitch_windows

# Shortest turn worth a voiceprint, in seconds.
#
# The voice analog of a face detector's pixel floor. Under this a turn is "mm" or "yeah" —
# too little voice to characterise, and an embedding made from it is noise that would go on
# to vote in the caller's identity step. Such turns are still reported, without a vector, so
# the timeline stays complete.
MIN_TURN_SECONDS = 1.5


class Vad(Strict):
    min_silence_ms: int | None = Field(default=None, ge=0)


class TranscribeRequest(WithOptions):
    file: FileRef
    language: str | None = Field(default=None, min_length=2, max_length=3)
    vad: Vad | None = None
    word_timestamps: bool | None = None


class DiarizeRequest(WithOptions):
    file: FileRef


class VoiceprintRequest(WithOptions):
    file: FileRef


class TrackPoint(Strict):
    t: float = Field(ge=0)
    box: Box


class ActiveSpeakerRequest(WithOptions):
    file: FileRef
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    track: list[TrackPoint]


class SpeakerTurns(Strict):
    id: str = Field(min_length=1)
    turns: list[tuple[float, float]]


class ActiveSpeakersRequest(WithOptions):
    file: FileRef
    speakers: list[SpeakerTurns]


@asynccontextmanager
async def extracted_audio(ctx: Context, path: str, progress: Progress):
    """The file's sound as a cached WAV, or None when it has none. Held for the block's duration."""
    await progress.emit("extract-audio")
    wav = await ctx.audio.acquire(path)
    try:
        yield wav
    finally:
        ctx.audio.release(path)


def _silent_transcript() -> dict:
    """A file with no sound track at all — distinct from a track nobody speaks on."""
    return {"speech": False, "audio": False, "text": "", "segments": [], "duration": 0.0}


async def transcribe(ctx: Context, req: TranscribeRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    async with model_slot(ctx, req.options):
        async with extracted_audio(ctx, path, progress) as wav:
            if wav is None:
                return Outcome(_silent_transcript(), producer="none")
            payload: dict = {"path": wav}
            if req.language:
                payload["language"] = req.language
            if req.vad and req.vad.min_silence_ms is not None:
                payload["vad_min_silence_ms"] = req.vad.min_silence_ms
            if req.word_timestamps is not None:
                payload["word_timestamps"] = req.word_timestamps
            await progress.emit("transcribe")
            raw = await ctx.worker.call("transcribe", payload, deadline=req.options.deadline)
    result = {
        "speech": bool(raw.get("speech")),
        "audio": True,
        "text": raw.get("text") or "",
        "segments": raw.get("segments") or [],
        "duration": float(raw.get("duration") or 0.0),
    }
    if raw.get("language"):
        result["language"] = raw["language"]
        result["language_probability"] = float(raw.get("language_probability") or 0.0)
    producer = raw.get("signature") or raw.get("model") or "whisper"
    return Outcome(result, producer=str(producer))


async def diarize(ctx: Context, req: DiarizeRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    async with model_slot(ctx, req.options):
        async with extracted_audio(ctx, path, progress) as wav:
            if wav is None:
                return Outcome({"speech": False, "duration": 0.0, "window": 0, "turns": []}, producer="none")
            await progress.emit("diarize")
            raw = await ctx.worker.call("diarize", {"path": wav}, deadline=req.options.deadline)
    turns = [{"start": t["start"], "end": t["end"], "speaker": t.get("localSpeaker") or t.get("speaker")} for t in raw.get("turns") or []]
    result = {"speech": bool(raw.get("speech")), "duration": float(raw.get("duration") or 0.0), "window": raw.get("window"), "turns": turns}
    return Outcome(result, producer=str(raw.get("model") or "sortformer"))


async def _embed_spans(ctx: Context, wav: str, turns: list[dict], deadline) -> tuple[list[list[float] | None], str | None]:
    """Voiceprints for the turns long enough to carry one, in order; None for the rest."""
    spans = [{"start": t["start"], "end": t["end"]} for t in turns if t["end"] - t["start"] >= MIN_TURN_SECONDS]
    if not spans:
        return [None] * len(turns), None
    raw = await ctx.worker.call("embed_voice_segments", {"path": wav, "spans": spans}, deadline=deadline)
    found = iter(raw.get("voiceprints") or [])
    vectors: list[list[float] | None] = []
    for turn in turns:
        if turn["end"] - turn["start"] < MIN_TURN_SECONDS:
            vectors.append(None)
            continue
        item = next(found, None)
        vectors.append(list(item["embedding"]) if item and item.get("embedding") else None)
    return vectors, (str(raw["model"]) if raw.get("model") else None)


async def voiceprint(ctx: Context, req: VoiceprintRequest, progress: Progress) -> Outcome:
    """Who speaks when, with a voiceprint for each turn and for each speaker — one call per recording.

    Diarizes in windows, embeds every turn in one pass over the sound, stitches the windows'
    local labels into file-wide speakers (see _voices.py) and re-embeds the turns the
    stitching joined back together. The caller never sees the window: it is sized for the
    card EMS runs on, which the caller has no way to know.
    """
    path = ctx.roots.input(req.file.path)
    deadline = req.options.deadline
    async with model_slot(ctx, req.options):
        async with extracted_audio(ctx, path, progress) as wav:
            if wav is None:
                return Outcome({"speech": False, "audio": False, "duration": 0.0, "turns": []}, producer="none")
            await progress.emit("diarize")
            raw = await ctx.worker.call("diarize", {"path": wav}, deadline=deadline)
            diarizer = str(raw.get("model") or "sortformer")
            duration = float(raw.get("duration") or 0.0)
            turns = [
                {"start": float(t["start"]), "end": float(t["end"]), "localSpeaker": str(t.get("localSpeaker") or t.get("speaker"))}
                for t in raw.get("turns") or []
            ]
            if not turns:
                return Outcome({"speech": False, "audio": True, "duration": duration, "turns": []}, producer=diarizer)

            await progress.emit("voiceprint")
            vectors, embedder = await _embed_spans(ctx, wav, turns, deadline)

            await progress.emit("stitch")
            speakers = stitch_windows(turns, vectors)
            merged = merge_adjacent([
                {"start": t["start"], "end": t["end"], "speaker": s, **({"embedding": v} if v is not None else {})}
                for t, s, v in zip(turns, speakers, vectors)
            ])

            # A merged turn spans more voice than either half did, so it deserves its own
            # measurement rather than inheriting one half's. Only the merged ones lack a vector
            # here, so this is a handful of spans, not a second pass over the file.
            rejoined = [t for t in merged if "embedding" not in t]
            if rejoined:
                again, embedder_again = await _embed_spans(ctx, wav, rejoined, deadline)
                embedder = embedder or embedder_again
                for turn, vector in zip(rejoined, again):
                    if vector is not None:
                        turn["embedding"] = vector

    result_turns = [
        {
            "start": t["start"],
            "end": t["end"],
            "speaker": t["speaker"],
            "embedding": t.get("embedding"),
            "dimension": len(t["embedding"]) if t.get("embedding") else None,
        }
        for t in merged
    ]
    producer = diarizer if embedder is None else f"{diarizer}+{embedder}"
    return Outcome(
        {"speech": True, "audio": True, "duration": duration, "turns": result_turns, "speakers": speakers_of(merged)},
        producer=producer,
    )


async def active_speaker(ctx: Context, req: ActiveSpeakerRequest, progress: Progress) -> Outcome:
    path = ctx.roots.input(req.file.path)
    async with model_slot(ctx, req.options):
        async with extracted_audio(ctx, path, progress) as wav:
            if wav is None:
                return Outcome(None, producer="none")
            await progress.emit("score")
            raw = await ctx.worker.call(
                "active_speaker_score",
                {"video": path, "audio": wav, "start": req.start, "end": req.end, "track": [p.model_dump() for p in req.track]},
                deadline=req.options.deadline,
            )
    if raw is None:
        return Outcome(None, producer="lr-asd")
    return Outcome({"score": float(raw.get("score", 0)), "frames": int(raw.get("frames", 0))}, producer="lr-asd")


async def active_speakers(ctx: Context, req: ActiveSpeakersRequest, progress: Progress) -> Outcome:
    """Per speaker of a recording, the visible face whose lips move with its voice — one call,
    one decode of the video (`inference_ops.active_speakers`)."""
    path = ctx.roots.input(req.file.path)
    async with model_slot(ctx, req.options):
        async with extracted_audio(ctx, path, progress) as wav:
            if wav is None:
                return Outcome({"speakers": [], "windows": 0, "frames": 0}, producer="none")
            await progress.emit("score")
            raw = await ctx.worker.call(
                "active_speakers",
                {
                    "video": path,
                    "audio": wav,
                    "angle": req.file.angle,
                    "mirror": req.file.mirror,
                    "speakers": [{"id": s.id, "turns": [list(t) for t in s.turns]} for s in req.speakers],
                },
                deadline=req.options.deadline,
            )
    return Outcome(
        {"speakers": raw.get("speakers") or [], "windows": int(raw.get("windows") or 0), "frames": int(raw.get("frames") or 0)},
        producer="lr-asd+insightface",
    )


def register(router: APIRouter, ctx: Context) -> None:
    @router.post("/speech/transcribe")
    async def speech_transcribe(request: Request, body: TranscribeRequest):
        return await run_op(request, lambda progress: transcribe(ctx, body, progress))

    @router.post("/speech/diarize")
    async def speech_diarize(request: Request, body: DiarizeRequest):
        return await run_op(request, lambda progress: diarize(ctx, body, progress))

    @router.post("/speech/voiceprint")
    async def speech_voiceprint(request: Request, body: VoiceprintRequest):
        return await run_op(request, lambda progress: voiceprint(ctx, body, progress))

    @router.post("/speech/active_speakers")
    async def speech_active_speakers(request: Request, body: ActiveSpeakersRequest):
        return await run_op(request, lambda progress: active_speakers(ctx, body, progress))

    @router.post("/speech/active_speaker")
    async def speech_active_speaker(request: Request, body: ActiveSpeakerRequest):
        return await run_op(request, lambda progress: active_speaker(ctx, body, progress))
