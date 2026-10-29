#!/usr/bin/env python3
"""
Model primitives, served by model_server.py.

These are model operations only — they take a file path or a string, run one model, and
return its raw output. Anything that knows what a *media file* is (mimetype routing,
keyframe extraction) belongs to the caller: the front's ops compose these.

Models come from model_registry. Heavy imports stay inside functions: importing this module
must not drag in torch.
"""

import os
import sys
import time

import model_registry as registry
from whisper_credits import is_credit_line

INSIGHTFACE = "insightface"
WHISPER = "whisper"
SORTFORMER = "sortformer"
TITANET = "titanet"
ASD = "asd"
SIGLIP_VISUAL = "siglip-visual"
SIGLIP_TEXT = "siglip-text"
SSCD = "sscd"
TRANSNET = "transnet"
OCR = "ocr"
HTR = "htr"
TEXT_EMBED = "text-embed"
TEXT_QUERY = "text-query"

def _whisper_model_name() -> str:
    return f"whisper-{registry.WHISPER_MODEL}"


def transcribe_signature(language: str | None = None, min_silence_ms: int | None = None, word_timestamps: bool | None = None) -> str:
    """Everything that decides what `transcribe` writes, as one string.

    A caller stamps a transcript with this and compares the stamp against what this returns
    *now*: a file whose stamp differs was decoded with other settings. Derived here, on the
    same side that decodes, so the two can never disagree — a literal on the caller's side
    that drifted from what is actually written would mark every file stale forever.

    So every setting that changes the output belongs in here, and nothing else does: the
    model, where and in what precision it runs on this host, the VAD, the no-speech ceiling, the
    language — pinned, or how it is detected — and whether words are timed. Beam size, the conditioning flag and the
    credit-line filter (whisper_credits) are constants in `transcribe`; if one becomes configurable it goes in too.
    """
    # Per-request overrides (the HTTP API's `language`, `vad.min_silence_ms`, `word_timestamps`)
    # take part exactly like the configured values do, so a transcript made with them is
    # stamped with what actually ran.
    vad = dict(registry.whisper_vad_parameters())
    if min_silence_ms is not None:
        vad['min_silence_duration_ms'] = int(min_silence_ms)
    lang = registry.WHISPER_LANGUAGE if language is None else language
    words = registry.WHISPER_WORD_TIMESTAMPS if word_timestamps is None else word_timestamps
    return (
        f"{_whisper_model_name()}"
        f":vad={vad['threshold']}/{vad['speech_pad_ms']}/{vad['min_silence_duration_ms']}"
        f",nospeech={registry.WHISPER_NO_SPEECH_PROB_MAX}"
        f",on={_whisper_placement()}"
        f",lang={lang or _detection_signature()}"
        f",words={1 if words else 0}"
    )


def _whisper_placement() -> str:
    """`cuda:float16` or `cpu:int8`: where this host decodes, and how (`registry.whisper_device`)."""
    device = "cuda" if registry.select_device() == "cuda" else "cpu"
    return f"{device}:{registry.WHISPER_COMPUTE[device]}"


def _detection_signature() -> str:
    """`auto:3` — windows read — or `auto:3:sv,en@0.1` with the expected languages and floor."""
    tag = f"auto:{registry.WHISPER_LANGUAGE_WINDOWS}"
    if registry.WHISPER_LANGUAGES:
        tag += f":{','.join(registry.WHISPER_LANGUAGES)}@{registry.WHISPER_LANGUAGE_FLOOR:g}"
    return tag


def decode_speech_audio(model, file_path: str):
    """The file's audio as whisper reads it: mono float samples at the model's sampling rate."""
    from faster_whisper.audio import decode_audio

    return decode_audio(file_path, sampling_rate=model.feature_extractor.sampling_rate)


def language_votes(model, audio, vad_params: dict, windows: int) -> list:
    """Whisper's language probabilities for up to `windows` 30-second windows of the speech.

    `audio` is the file decoded at the model's sampling rate (`decode_audio`). It is
    VAD-filtered the way `transcribe` will do it, so what is judged
    is the speech the decoder will hear, not the file's opening — a clip that starts on
    wind noise or a doorbell gets judged on what is said after it. The windows are spread
    across all of that speech (first, middle and last for three), so a guest speaking at
    the start is one vote among several. A trailing window shorter than five seconds says
    little and is left out unless it is all there is.

    Each entry maps every language code whisper knows to its probability for that window.
    Empty when the VAD found no speech; the decoder will find none either.
    """
    import math

    import numpy as np
    from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

    rate = model.feature_extractor.sampling_rate
    chunks = get_speech_timestamps(audio, VadOptions(**vad_params))
    if not chunks:
        return []
    speech = np.concatenate(collect_chunks(audio, chunks)[0])

    per_window = model.feature_extractor.n_samples
    count = max(1, math.ceil(len(speech) / per_window))
    if count > 1 and len(speech) - (count - 1) * per_window < 5 * rate:
        count -= 1
    if count <= windows:
        picks = range(count)
    else:
        picks = sorted({round(i * (count - 1) / (windows - 1)) for i in range(windows)})

    votes = []
    for index in picks:
        _language, _probability, all_probs = model.detect_language(audio=speech[index * per_window:(index + 1) * per_window])
        votes.append(dict(all_probs))
    return votes


# Languages whisper confuses with each other: mutually intelligible neighbours whose
# probability mass an expected language may claim. A Swedish clip can come back `no 0.65,
# nn 0.30, sv 0.02`: whisper is sure it is Scandinavian and unsure which, and decoding it as
# Swedish settles it. An English clip with Swedish at the same 0.02 has no such mass on
# Swedish's neighbours, which is what tells the two apart.
SIBLINGS = {
    "sv": ("no", "nn", "da"), "no": ("nn", "sv", "da"), "nn": ("no", "sv", "da"), "da": ("no", "nn", "sv"),
    "nl": ("af",), "af": ("nl",),
    "es": ("ca", "gl"), "ca": ("es",), "pt": ("gl",), "gl": ("pt", "es"),
    "id": ("ms", "jw"), "ms": ("id",),
    "cs": ("sk",), "sk": ("cs",),
    "ru": ("uk", "be"), "uk": ("ru",), "be": ("ru",),
    "hr": ("sr", "bs"), "sr": ("hr", "bs"), "bs": ("hr", "sr"),
    "hi": ("ur",), "ur": ("hi",),
    "zh": ("yue",), "yue": ("zh",),
}

# Below this, the top guess is not a detection. faster-whisper's own default for moving on
# to another window is the same 0.5; here it is where the archive's main language takes over.
UNSURE = 0.5


def choose_language(votes: list, expected: tuple, floor: float) -> dict | None:
    """The language to decode in, from the per-window votes; None when there were none.

    Three steps:

    1. Average the windows, each weighted by how sure whisper was of it (its top
       probability). One sure window outweighs two unsure ones (`da 0.97` against `en 0.75`
       and `en 0.60` with Welsh second — whisper's tell for noise), where a plain average
       would decode Danish as English and whisper would *translate* it.
    2. An expected language claims its own probability plus that of its SIBLINGS that are
       not expected themselves — each sibling's mass going to the first expected language
       it belongs to, so `sv,da` sends Norwegian to Swedish, not to both. The strongest
       claim wins if it reaches `floor`. A genuinely foreign clip leaves every expected
       claim near zero.
    3. If nothing was claimed and the top guess is under UNSURE (`ja 0.33, ko 0.32`),
       whisper did not detect a language, and the first expected language, the archive's
       main one, applies.

    `detected` is what plain detection would have said, kept so a log line can show when
    the rule changed the answer.
    """
    if not votes:
        return None
    total_weight = 0.0
    average: dict = {}
    for vote in votes:
        weight = max(vote.values()) if vote else 0.0
        total_weight += weight
        for code, probability in vote.items():
            average[code] = average.get(code, 0.0) + float(probability) * weight
    if total_weight > 0:
        average = {code: mass / total_weight for code, mass in average.items()}
    top = max(average, key=average.get)
    choice = top
    if expected:
        claims = {code: average.get(code, 0.0) for code in expected}
        for code, probability in average.items():
            if code in claims:
                continue
            for candidate in expected:
                if code in SIBLINGS.get(candidate, ()):
                    claims[candidate] += probability
                    break
        best = max(expected, key=claims.get)
        if claims[best] >= floor:
            choice = best
        elif average[top] < UNSURE:
            choice = expected[0]
    return {
        "language": choice,
        "probability": average.get(choice, 0.0),
        "detected": top,
        "detected_probability": average[top],
    }


def _normalize_angle(angle) -> int:
    """Quarter turns only; anything else means no rotation."""
    try:
        a = int(round(float(angle))) % 360
    except (TypeError, ValueError):
        return 0
    return a if a in (90, 180, 270) else 0


def _read_upright(image_path: str, angle, mirror):
    """Read an image into the *display* frame as BGR: rotated CCW by `angle`, then flopped.

    The file's own EXIF Orientation is not what decides that — the caller's `angle` is, because
    it may be a manual override and the tag may simply be wrong, and every other step rotates by
    it. `image_io` owns that question and reads HEIF and RAW besides. Decoded on the decode pool,
    outside the family's lock (registry.decode_all).
    """
    # Read through PIL rather than cv2.imread: cv2 has no HEIF or RAW decoder at all. The
    # rotation moves with it — `open_display` is the one place that knows which formats arrive
    # already turned.
    import image_io

    return registry.decode_all(lambda path: image_io.open_display_bgr(path, angle, mirror), [image_path])[0]


def _is_cuda_oom(exc: Exception) -> bool:
    """True if the exception is an out-of-VRAM error."""
    import torch
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    if "out of memory" in message:
        return isinstance(exc, RuntimeError) or "onnxruntime" in message
    # onnxruntime does not raise a torch error and does not say "out of memory";
    # its arena reports "Failed to allocate memory for requested buffer of size N".
    return "failed to allocate memory" in message


def transcribe(file_path: str, language: str | None = None, min_silence_ms: int | None = None, word_timestamps: bool | None = None) -> dict:
    """Transcribe an audio file with faster-whisper, on the GPU when available.

    `speech` is False when nothing was recognised — a normal outcome for the many silent
    phone clips in the archive, and the caller decides what to say about it. `model` names
    the model that actually ran, so the description's provenance can't drift from
    CFG_WHISPER_MODEL.

    `segments` carry their own start and end, in seconds. They are the whole reason this
    returns a structure rather than a string: subtitles, a transcript that follows the
    playhead, and a search hit that knows *when* something was said all need the timings,
    and once joined they cannot be recovered. `text` is still the flat join, because most
    callers only want that.

    The decode arguments are not defaults. See model_registry for why each is set:
    VAD (whisper_vad_parameters) so silence is not decoded into invented speech, tuned so
    that faint and outdoor speech still reaches the decoder; no conditioning on previous text
    so a bad segment cannot drag the rest of the file into a repetition loop; the language
    either pinned (WHISPER_LANGUAGE) or detected over several windows of the speech with the
    expected languages given the benefit of the doubt (WHISPER_LANGUAGES, `language_votes`
    and `choose_language`), so a noisy first 30 seconds cannot mislabel an entire recording;
    a ceiling on the decoder's own no-speech score (WHISPER_NO_SPEECH_PROB_MAX) that
    drops the fluent sentences it invents over a window it rated as silence; and a filter
    (whisper_credits) for the thank-you and credit lines that get under that ceiling.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"audio not found: {file_path}")

    # The effective decode settings: the request's overrides where given, the configuration
    # otherwise — the same resolution transcribe_signature performs, so stamp and decode agree.
    # Named so nothing inside `run` shadows them: its segment loop has a `words` of its own.
    vad_params = dict(registry.whisper_vad_parameters())
    if min_silence_ms is not None:
        vad_params['min_silence_duration_ms'] = int(min_silence_ms)
    lang_hint = registry.WHISPER_LANGUAGE if language is None else language
    want_words = registry.WHISPER_WORD_TIMESTAMPS if word_timestamps is None else bool(word_timestamps)

    def run():
        # Where this host transcribes, always (`whisper_device`): a card short of room makes the
        # caller wait, and a load that fails there anyway is an error it retries, never a quiet
        # move to the CPU.
        device = registry.whisper_device()
        model = registry.whisper(device, registry.WHISPER_COMPUTE[device])

        # Detection is ours, not the decoder's: whisper would read the first window of
        # the file and decide for all of it. The choice is then handed to the decoder as
        # if it were pinned, so every window is decoded in the same language.
        started = time.monotonic()
        detected = None
        # Detection needs the decoded audio, and the decoder then takes that array rather than
        # the path, so a file is decoded once. Under a pinned language the decoder reads the file.
        source = file_path
        if not lang_hint:
            source = decode_speech_audio(model, file_path)
            votes = language_votes(model, source, vad_params, registry.WHISPER_LANGUAGE_WINDOWS)
            detected = choose_language(votes, registry.WHISPER_LANGUAGES, registry.WHISPER_LANGUAGE_FLOOR)
            if detected and detected["language"] != detected["detected"]:
                print(
                    f"transcribe: {os.path.basename(file_path)}: decoding as {detected['language']} "
                    f"({detected['probability']:.2f}) over detected {detected['detected']} "
                    f"({detected['detected_probability']:.2f}) across {len(votes)} windows",
                    file=sys.stderr, flush=True,
                )
        voted = time.monotonic()
        # segments is a generator — consume it before releasing the model. The call decodes the
        # audio, runs the VAD and computes the features; the loop below runs the decoder.
        segments, info = model.transcribe(
            source,
            beam_size=5,
            language=lang_hint or (detected["language"] if detected else None),
            vad_filter=True,
            vad_parameters=vad_params,
            condition_on_previous_text=False,
            word_timestamps=want_words,
        )
        # Empty segments are dropped here rather than downstream: a cue with no text is
        # not a subtitle, and an empty chunk is not worth a vector. So are segments the
        # decoder rated as almost certainly not speech, and the credit lines it invents
        # over noise — those are its hallucinations, and they read as sentences, which is
        # exactly why they must not reach search.
        prepared = time.monotonic()
        out = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            no_speech = getattr(segment, "no_speech_prob", None)
            if no_speech is not None and no_speech >= registry.WHISPER_NO_SPEECH_PROB_MAX:
                continue
            if is_credit_line(text):
                continue
            entry = {"start": float(segment.start), "end": float(segment.end), "text": text}
            # Only present when word timestamps were asked for. A segment cannot be split
            # at a speaker change without knowing where inside it each word falls.
            words = getattr(segment, "words", None) or []
            aligned = [
                {"start": float(w.start), "end": float(w.end), "word": w.word}
                for w in words
                if w.start is not None and w.end is not None and w.word.strip()
            ]
            if aligned:
                entry["words"] = aligned
            out.append(entry)
        # Where a transcription's time goes, by phase (ein/core#741).
        print(
            f"transcribe: {os.path.basename(file_path)}: {float(info.duration or 0.0):.0f}s of audio, "
            f"language {voted - started:.1f}s, prepare {prepared - voted:.1f}s, "
            f"decode {time.monotonic() - prepared:.1f}s",
            file=sys.stderr, flush=True,
        )
        # `info` is only safe to read after the generator is drained — faster-whisper
        # fills in the detected language as it decodes the first window. When the
        # language was detected here, the reported probability is the averaged one the
        # choice was made on, not the 1.0 the decoder reports for a language it was told.
        return out, {
            "language": detected["language"] if detected else info.language,
            "language_probability": float(detected["probability"] if detected else info.language_probability or 0.0),
            "duration": float(info.duration or 0.0),
        }

    with registry.family_lock(WHISPER):
        segments, info = registry.run_on_model_thread(WHISPER, run)

    text = " ".join(segment["text"] for segment in segments)
    return {
        "text": text,
        "speech": bool(text),
        "model": _whisper_model_name(),
        "signature": transcribe_signature(lang_hint, min_silence_ms, want_words),
        "segments": segments,
        **info,
    }


def _parse_turn(line: str) -> dict | None:
    """`"12.480 17.840 speaker_0"` -> `{"start": 12.48, "end": 17.84, "speaker": "speaker_0"}`.

    Sortformer's own output format. Anything that does not parse is dropped rather than
    raised on: one malformed line must not lose a whole recording's diarization.
    """
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        start, end = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if end <= start:
        return None
    return {"start": start, "end": end, "speaker": parts[2]}


def diarize(file_path: str) -> dict:
    """Find speaker turns in an audio file. Returns turns with timings and a speaker label.

    Takes an already-demuxed 16kHz mono WAV, like `transcribe` — the caller owns ffmpeg.

    **Windowed, and the caller must know it.** Sortformer is not linear in input length, so
    this feeds it registry.DIARIZE_WINDOW_SECONDS at a time rather than the whole file (see
    the note there). Two consequences leak into the
    result and are reported rather than hidden:

      - `speaker` is LOCAL TO ITS WINDOW. The label is prefixed with the window index
        (`w0/speaker_0`) precisely so it cannot be mistaken for a file-wide identity: the
        same person in two windows gets two labels. Stitching them is the caller's job, by
        voiceprint (the front's ops/_voices.py).
      - A turn that spans a window boundary comes back as two turns meeting at that
        boundary. Once stitching has agreed they are one speaker, adjacent turns that touch
        can be merged.

    `speech` is False when no turn was found anywhere, a normal outcome for the many silent
    clips in the archive — the same contract `transcribe` uses.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"audio not found: {file_path}")

    window = registry.DIARIZE_WINDOW_SECONDS

    def run():
        import soundfile
        import tempfile

        device = registry.sortformer_device()
        model = registry.sortformer(device)
        samples, rate = soundfile.read(file_path, dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        total = len(samples) / float(rate)

        turns: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp:
            chunk_path = os.path.join(tmp, "window.wav")
            index = 0
            offset = 0.0
            while offset < total:
                span = min(window, total - offset)
                # Below this a window holds no usable speech and diarizing it only costs
                # a model call, so stop rather than emit a degenerate tail.
                if span < 2.0:
                    break
                piece = samples[int(offset * rate):int((offset + span) * rate)]
                soundfile.write(chunk_path, piece, rate)
                for line in model.diarize(audio=[chunk_path], batch_size=1)[0]:
                    turn = _parse_turn(line)
                    if turn is None:
                        continue
                    turns.append({
                        "start": round(turn["start"] + offset, 3),
                        "end": round(min(turn["end"] + offset, total), 3),
                        "localSpeaker": f"w{index}/{turn['speaker']}",
                    })
                index += 1
                offset += span

        # Sortformer groups its turns by speaker, not by time, so the caller would
        # otherwise have to know that. Hand back one timeline.
        turns.sort(key=lambda t: (t["start"], t["end"]))
        return turns, total

    with registry.family_lock(SORTFORMER):
        turns, duration = registry.run_on_model_thread(SORTFORMER, run)

    return {
        "turns": turns,
        "speech": bool(turns),
        "duration": duration,
        "window": window,
        "model": registry.SORTFORMER_MODEL.split("/")[-1],
    }


def active_speaker_score(video_path: str, audio_path: str, start: float, end: float, track: list) -> dict | None:
    """How well one visible face's lip movement matches the sound over a span.

    `track` is that face through time: `[{"t": seconds, "box": {x, y, width, height}}]` with
    the box in fractions of the frame, the same convention as every box in the API. The
    caller owns finding the face and following it; this only scores the pairing.

    Returns `{"score": 0..1, "frames": n}`, where the score is the mean probability that the
    face is speaking. None when there is too little to judge on — a track shorter than a
    handful of frames, or a span with no audio behind it.

    The two streams have to arrive at the rates the model was trained on: video at 25fps and
    MFCC at 100fps, four audio frames per video frame. Both are resampled to that here rather
    than being demanded of the caller, because the frame times come from wherever the caller
    chose to sample and will not naturally land on 25fps.
    """
    try:
        import cv2
        import numpy
        import python_speech_features
        import soundfile
    except ImportError as e:
        raise ImportError(f"Required package not installed: {e}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio not found: {audio_path}")
    if end <= start or len(track) < 4:
        return None

    # --- visual: the face, cropped the way the model was trained to see it -----------------
    capture = cv2.VideoCapture(video_path)
    try:
        crops = []
        for entry in track:
            at = float(entry.get("t", 0.0))
            box = entry.get("box") or {}
            capture.set(cv2.CAP_PROP_POS_MSEC, at * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            x0, y0 = float(box.get("x", 0)) * width, float(box.get("y", 0)) * height
            x1, y1 = x0 + float(box.get("width", 0)) * width, y0 + float(box.get("height", 0)) * height
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            crops.append(_asd_face(cv2, numpy, frame, (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2))
    finally:
        capture.release()

    if len(crops) < 4:
        return None

    # 25fps is the rate the visual frontend expects; the caller's sampling is resampled onto
    # it by nearest neighbour rather than interpolated, since these are images.
    frames = max(4, int(round((end - start) * 25)))
    index = numpy.linspace(0, len(crops) - 1, frames).round().astype(int)
    visual = numpy.stack([crops[i] for i in index]).astype(numpy.float32)

    # --- audio: MFCC at four frames per video frame ---------------------------------------
    rate = soundfile.info(audio_path).samplerate
    piece = _asd_samples(soundfile, audio_path, start, end, rate)
    if len(piece) < rate // 10:
        return None
    mfcc = python_speech_features.mfcc(piece, rate, numcep=13, winlen=0.025, winstep=0.010)

    wanted = frames * 4
    if len(mfcc) < wanted:
        mfcc = numpy.pad(mfcc, ((0, wanted - len(mfcc)), (0, 0)), mode="edge")
    audio = mfcc[:wanted].astype(numpy.float32)

    def run():
        import torch

        device = registry.lr_asd_device()
        model, head = registry.lr_asd(device)
        with torch.no_grad():
            visual_t = torch.from_numpy(visual).unsqueeze(0).to(device)
            audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
            outs_av, _outs_v = model(audio_t, visual_t)
            probabilities = torch.softmax(head(outs_av), dim=-1)[:, 1]
            return float(probabilities.mean().item())

    with registry.family_lock(ASD):
        score = registry.run_on_model_thread(ASD, run)

    return {"score": score, "frames": frames}


# --- active speakers, a recording at a time ------------------------------------------------
#
# Which visible face is talking when a speaker of a recording is heard (ein/core#507).
# LR-ASD reads lip movement against the sound at 25fps, so it needs consecutive frames: a
# contiguous window of a speaker's turn, not stills seconds apart. Each speaker is judged on its
# longest turns, and the video is decoded once, window after window in file order.

# Contiguous video scored from each chosen turn, centred in it: long enough for a few
# sentences' worth of lip movement, as LR-ASD's own evaluation reads up to 6 s at a time.
ASD_WINDOW_SECONDS = 4.0
# Turns shorter than this are not scored: too little lip movement to read.
ASD_MIN_TURN_SECONDS = 2.0
# The longest turns of each speaker that are scored.
ASD_TURNS_PER_SPEAKER = 3
# The rate LR-ASD's visual frontend was trained on.
ASD_FPS = 25
# Faces are detected on every n-th frame; the frames between take the box of their neighbours.
ASD_DETECT_EVERY = 5
# Overlap for a detection to continue a face's track (see EiN's face-track.ts for why it is low).
ASD_TRACK_MIN_IOU = 0.3
# Detections a track needs to be scored: four at every fifth frame is most of a second.
ASD_TRACK_MIN_DETECTIONS = 4
# LR-ASD's crop (its Columbia_test.py `cropScale`): the face box made square, padded by this
# much of its half-size and pushed down towards the mouth.
ASD_CROP_SCALE = 0.40
# Frames a track's box is median-smoothed over, as LR-ASD smooths it, so the crop does not jitter.
ASD_SMOOTH_FRAMES = 13


def _asd_face(cv2, numpy, frame, cx: float, cy: float, half: float):
    """One face as LR-ASD was trained to see it, from its box's centre and half its longer side.

    The reference pipeline crops a square of 2·half·(1+scale) whose top is `half` above the
    centre, so it reaches the chin and below; resizes it to 224 and keeps the middle 112. What the
    model sees is the lower face — nose, mouth, jaw — at about twice the size of a whole-head
    crop, grey. Outside the frame is the reference's grey 110.
    """
    cs = ASD_CROP_SCALE
    top, bottom = cy - half, cy + half * (1 + 2 * cs)
    left, right = cx - half * (1 + cs), cx + half * (1 + cs)
    pad = int(half * (1 + 2 * cs)) + 1
    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(110, 110, 110))
    face = padded[int(top) + pad:int(bottom) + pad, int(left) + pad:int(right) + pad]
    face = cv2.cvtColor(cv2.resize(face, (224, 224)), cv2.COLOR_BGR2GRAY)
    return face[56:168, 56:168]


def _median_smooth(numpy, values: list, k: int):
    """A running median over `k` values, the ends padded with their own edge value."""
    v = numpy.asarray(values, dtype=numpy.float64)
    if len(v) < 3:
        return v
    k = min(k, len(v) if len(v) % 2 else len(v) - 1)
    padded = numpy.pad(v, k // 2, mode="edge")
    return numpy.array([numpy.median(padded[i:i + k]) for i in range(len(v))])


def _asd_samples(soundfile, audio_path: str, start: float, end: float, rate: int):
    """Mono samples of [start, end) on the int16 scale LR-ASD's MFCCs were computed on.

    The reference reads 16-bit WAV with scipy, values ±32768; the same sound as floats in ±1 puts
    every log energy, and with it the first cepstral coefficient, about 21 lower than the model
    has ever seen.
    """
    samples, _ = soundfile.read(audio_path, start=int(start * rate), stop=int(end * rate), dtype="int16", always_2d=False)
    samples = samples.astype("float64")
    return samples.mean(axis=1) if samples.ndim > 1 else samples


def asd_windows(speakers: list, window: float = ASD_WINDOW_SECONDS, per_speaker: int = ASD_TURNS_PER_SPEAKER,
                min_turn: float = ASD_MIN_TURN_SECONDS) -> list:
    """The spans to score, in file order: `(start, end, speaker id)` for each speaker's longest
    turns that are long enough, each cut to `window` seconds around its middle."""
    out = []
    for speaker in speakers:
        turns = [(float(s), float(e)) for s, e in speaker.get("turns") or [] if float(e) - float(s) >= min_turn]
        for start, end in sorted(turns, key=lambda t: t[1] - t[0], reverse=True)[:per_speaker]:
            middle = (start + end) / 2
            half = min(window, end - start) / 2
            out.append((middle - half, middle + half, str(speaker["id"])))
    return sorted(out)


def _box_iou(a, b) -> float:
    """Overlap of two `(x0, y0, x1, y1)` boxes as intersection over union."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def asd_tracks(detections: list, min_iou: float = ASD_TRACK_MIN_IOU) -> list:
    """Per-frame detections grouped into faces through time.

    `detections` are `{"frame": index, "box": (x0, y0, x1, y1), ...}`. Greedy and forward-only,
    like EiN's face-track: each detection extends the open track whose last box it overlaps
    most, best pairing first, one detection per track per frame. Returns lists of detections.
    """
    by_frame: dict = {}
    for detection in detections:
        by_frame.setdefault(detection["frame"], []).append(detection)
    tracks: list = []
    for frame in sorted(by_frame):
        found = by_frame[frame]
        pairs = sorted(
            ((_box_iou(track[-1]["box"], d["box"]), ti, di) for ti, track in enumerate(tracks) for di, d in enumerate(found)),
            reverse=True,
        )
        used_tracks, used_found = set(), set()
        for overlap, ti, di in pairs:
            if overlap < min_iou or ti in used_tracks or di in used_found:
                continue
            tracks[ti].append(found[di])
            used_tracks.add(ti)
            used_found.add(di)
        tracks.extend([d] for di, d in enumerate(found) if di not in used_found)
    return tracks


def _track_box_at(track: list, frame: int):
    """A track's box at `frame`, interpolated between the detections either side of it."""
    before = max((d for d in track if d["frame"] <= frame), key=lambda d: d["frame"], default=track[0])
    after = min((d for d in track if d["frame"] >= frame), key=lambda d: d["frame"], default=track[-1])
    if before["frame"] == after["frame"]:
        return before["box"]
    w = (frame - before["frame"]) / (after["frame"] - before["frame"])
    return tuple(a + (b - a) * w for a, b in zip(before["box"], after["box"]))


def _display_video_frame(cv2, frame, angle, mirror):
    """A decoded frame turned into the display frame: CCW by `angle`, then flopped — the same
    turn image_io gives a still, so a box means the same thing on a frame as on a photo."""
    a = _normalize_angle(angle)
    if a == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif a == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif a == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mirror:
        frame = cv2.flip(frame, 1)
    return frame


def _read_asd_window(cv2, capture, start: float, end: float, angle, mirror) -> list:
    """The frames of [start, end) at ASD_FPS in the display frame: for each tick, the decoded
    frame it falls on. One seek, then a sequential read."""
    fps = capture.get(cv2.CAP_PROP_FPS) or float(ASD_FPS)
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    wanted = [int(round(i / ASD_FPS * fps)) for i in range(int(round((end - start) * ASD_FPS)))]
    frames, read, current = [], 0, None
    for target in wanted:
        while read <= target:
            ok, frame = capture.read()
            if not ok:
                return frames
            current, read = frame, read + 1
        frames.append(_display_video_frame(cv2, current, angle, mirror))
    return frames


def active_speakers(video_path: str, audio_path: str, speakers: list, angle=0, mirror=False) -> dict:
    """For each speaker of a recording, the visible face whose lips move with its voice.

    `speakers` are `[{"id", "turns": [[start, end], …]}]`, as `speech.voiceprint` answers them.
    Returns `{"speakers": [{"id", "score", "t", "box", "embedding", "confidence"}], "windows",
    "frames"}`: per speaker the best-scoring face over its scored windows — `score` the mean
    probability (0..1) that the face is the one speaking, `box` in fractions of the display
    frame at `t` seconds, `embedding` InsightFace's 512-dim vector of that face for the caller
    to recognise. A speaker with no face to score is left out; how high a score has to be to
    count is the caller's call.
    """
    try:
        import cv2
        import numpy
        import python_speech_features
        import soundfile
    except ImportError as e:
        raise ImportError(f"Required package not installed: {e}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio not found: {audio_path}")

    windows = asd_windows(speakers)
    rate = soundfile.info(audio_path).samplerate
    best: dict = {}
    decoded = 0

    capture = cv2.VideoCapture(video_path)
    # Raw frames: the caller's angle is what turns pixels, as everywhere else in EMS.
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    try:
        for start, end, speaker_id in windows:
            frames = _read_asd_window(cv2, capture, start, end, angle, mirror)
            decoded += len(frames)
            if len(frames) < ASD_DETECT_EVERY * ASD_TRACK_MIN_DETECTIONS:
                continue
            height, width = frames[0].shape[:2]

            def detect(frames=frames):
                found = []
                for index in range(0, len(frames), ASD_DETECT_EVERY):
                    for face in _insightface_get(frames[index]):
                        x0, y0, x1, y1 = (float(v) for v in face.bbox)
                        found.append({"frame": index, "box": (x0, y0, x1, y1), "score": float(face.det_score),
                                      "embedding": face.embedding})
                return found

            with registry.family_lock(INSIGHTFACE):
                detections = registry.run_on_model_thread(INSIGHTFACE, detect)

            samples = _asd_samples(soundfile, audio_path, start, end, rate)

            for track in asd_tracks(detections):
                if len(track) < ASD_TRACK_MIN_DETECTIONS:
                    continue
                first, last = track[0]["frame"], min(len(frames) - 1, track[-1]["frame"] + ASD_DETECT_EVERY - 1)
                boxes = [_track_box_at(track, index) for index in range(first, last + 1)]
                xs = _median_smooth(numpy, [(b[0] + b[2]) / 2 for b in boxes], ASD_SMOOTH_FRAMES)
                ys = _median_smooth(numpy, [(b[1] + b[3]) / 2 for b in boxes], ASD_SMOOTH_FRAMES)
                halves = _median_smooth(numpy, [max(b[2] - b[0], b[3] - b[1]) / 2 for b in boxes], ASD_SMOOTH_FRAMES)
                if min(halves) < 4:
                    continue
                crops = [_asd_face(cv2, numpy, frames[first + k], xs[k], ys[k], halves[k]) for k in range(len(boxes))]
                if len(crops) < ASD_FPS // 2:
                    continue
                piece = samples[int(first / ASD_FPS * rate):int((last + 1) / ASD_FPS * rate)]
                if len(piece) < rate // 10:
                    continue
                mfcc = python_speech_features.mfcc(piece, rate, numcep=13, winlen=0.025, winstep=0.010)
                wanted = len(crops) * 4
                if len(mfcc) < wanted:
                    mfcc = numpy.pad(mfcc, ((0, wanted - len(mfcc)), (0, 0)), mode="edge")
                visual = numpy.stack(crops).astype(numpy.float32)
                audio = mfcc[:wanted].astype(numpy.float32)

                def score(visual=visual, audio=audio):
                    import torch

                    device = registry.lr_asd_device()
                    model, head = registry.lr_asd(device)
                    with torch.no_grad():
                        outs_av, _ = model(torch.from_numpy(audio).unsqueeze(0).to(device), torch.from_numpy(visual).unsqueeze(0).to(device))
                        return float(torch.softmax(head(outs_av), dim=-1)[:, 1].mean().item())

                with registry.family_lock(ASD):
                    probability = registry.run_on_model_thread(ASD, score)

                if speaker_id in best and best[speaker_id]["score"] >= probability:
                    continue
                # The clearest look at the face is the one to recognise it from.
                face = max(track, key=lambda d: d["score"] * (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
                x0, y0, x1, y1 = face["box"]
                best[speaker_id] = {
                    "id": speaker_id,
                    "score": probability,
                    "t": round(start + face["frame"] / ASD_FPS, 3),
                    "box": {"x": x0 / width, "y": y0 / height, "width": (x1 - x0) / width, "height": (y1 - y0) / height},
                    "confidence": face["score"],
                    "embedding": face["embedding"].tolist(),
                }
    finally:
        capture.release()

    return {"speakers": [best[s] for s in sorted(best)], "windows": len(windows), "frames": decoded}


# --- faces through a video ------------------------------------------------------------------
#
# Who a video shows, and when (ein/core#410): a frame every `interval` seconds, InsightFace on
# each, and the detections followed and grouped into the file's faces by `face_tracks`.

# The most frames sampled from one video; a longer one is sampled more sparsely instead.
VIDEO_FACES_MAX_SAMPLES = 1800
# Frames detected per hold of the model. The next ones are decoded outside it, so another
# request for the family waits for one batch at most.
VIDEO_FACES_BATCH = 16
# Longer side a frame is cut down to before detection. The detector reads it at 640 and each
# face is read from the frame itself, so a 4K frame buys a small face little but memory.
VIDEO_FACES_MAX_EDGE = 1920


class Undecodable(Exception):
    """The input has no picture the decoder can read."""


def video_face_step(interval: float, duration: float | None) -> float:
    """Seconds between sampled frames: `interval`, or longer for a video that would otherwise
    give more than VIDEO_FACES_MAX_SAMPLES."""
    return max(interval, duration / VIDEO_FACES_MAX_SAMPLES) if duration and duration > 0 else interval


def detect_video_faces(video_path: str, angle=0, mirror=False, interval: float = 0.25, duration: float | None = None,
                       min_size: float = 0.0, min_confidence: float = 0.0) -> dict:
    """The faces a video shows: per face when it is seen, its clearest detection and one vector.

    A frame is taken every `video_face_step(interval, duration)` seconds, in the display frame
    the caller's `angle`/`mirror` name, read in one sequential pass. Detections smaller than
    `min_size` of the frame's shorter side or below `min_confidence` are left out before they are
    followed. Returns `{"frame", "interval", "samples", "frames", "faces"}`; each face is
    `{"appearances": [[start, end], …], "detections", "t", "box", "confidence", "landmarks",
    "pose"?, "embedding", "track"}`, the box and landmarks in fractions of the display frame at
    `t` seconds, `embedding` the normalized mean of the face's best detections' vectors, and
    `track` its path (`face_tracks.path`): `[t, x, y, width, height]` in time order, every
    sample while the face moves and about one a second while it stands still.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(f"Required package not installed: {e}")
    import face_tracks

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")

    step = video_face_step(float(interval), duration)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise Undecodable(f"no video stream the decoder can read: {video_path}")
    # Raw frames: the caller's angle is what turns pixels, as everywhere else in EMS.
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0

    samples: list = []
    pending: list = []
    frame_size = (0, 0)

    def detect(batch):
        out = []
        for index, t, frame in batch:
            height, width = frame.shape[:2]
            shorter = min(width, height)
            found = []
            for face in _insightface_get(frame):
                x0, y0, x1, y1 = (float(v) for v in face.bbox)
                if min(x1 - x0, y1 - y0) < min_size * shorter or float(face.det_score) < min_confidence:
                    continue
                detection = {
                    "box": (x0, y0, x1, y1),
                    "frame": (width, height),
                    "confidence": float(face.det_score),
                    "unit": face.normed_embedding.tolist(),
                    "landmarks": face.kps.tolist() if face.kps is not None else None,
                }
                pose = _pose_of(face)
                if pose is not None:
                    detection["pose"] = pose
                found.append(detection)
            out.append({"index": index, "t": t, "faces": found})
        return out

    def flush():
        if not pending:
            return
        batch = list(pending)
        pending.clear()
        with registry.family_lock(INSIGHTFACE):
            samples.extend(registry.run_on_model_thread(INSIGHTFACE, lambda: detect(batch)))

    decoded, next_t = 0, 0.0
    try:
        while capture.grab():
            decoded += 1
            t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # A container without timestamps reports 0 throughout; its frames are counted instead.
            if t <= 0 and decoded > 1 and fps > 0:
                t = (decoded - 1) / fps
            if t + 1e-3 < next_t:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                continue
            frame = _display_video_frame(cv2, frame, angle, mirror)
            height, width = frame.shape[:2]
            if max(width, height) > VIDEO_FACES_MAX_EDGE:
                scale = VIDEO_FACES_MAX_EDGE / max(width, height)
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            frame_size = (frame.shape[1], frame.shape[0])
            pending.append((len(samples) + len(pending), round(t, 3), frame))
            # On a grid of `step` from the start, not `step` after this frame: a step that is no
            # whole number of frames would otherwise drift, 0.25 s at 50 fps becoming 0.26 s.
            while next_t <= t + 1e-3:
                next_t += step
            if len(pending) >= VIDEO_FACES_BATCH:
                flush()
        flush()
    finally:
        capture.release()

    if decoded == 0:
        raise Undecodable(f"no video frame could be decoded: {video_path}")

    def fractions(detection) -> list[float]:
        x0, y0, x1, y1 = detection["box"]
        width, height = detection["frame"]
        return [round(max(0.0, x0 / width), 4), round(max(0.0, y0 / height), 4), round(min(1.0, (x1 - x0) / width), 4), round(min(1.0, (y1 - y0) / height), 4)]

    faces = []
    for face in face_tracks.faces_of(samples, step, duration):
        best = face["detection"]
        x0, y0, x1, y1 = best["box"]
        width, height = best["frame"]
        entry = {
            "appearances": face["appearances"],
            "detections": face["detections"],
            "t": best["t"],
            "box": {"x": max(0.0, x0 / width), "y": max(0.0, y0 / height), "width": min(1.0, (x1 - x0) / width), "height": min(1.0, (y1 - y0) / height)},
            "confidence": best["confidence"],
            "landmarks": [[x / width, y / height] for x, y in best["landmarks"]] if best.get("landmarks") else None,
            "embedding": face["vector"],
            "track": face_tracks.path([[d["t"], *fractions(d)] for d in face["sightings"]], step),
        }
        if best.get("pose"):
            entry["pose"] = best["pose"]
        faces.append(entry)
    return {"frame": {"width": frame_size[0], "height": frame_size[1]}, "interval": step, "samples": len(samples), "frames": decoded, "faces": faces}


def embed_voice_segments(file_path: str, spans: list) -> dict:
    """Embed many speech turns of one recording as voiceprints, in one pass over its sound.

    The voice analog of `embed_face`: given spans the diarizer already located, produce the
    vector that decides whose voice each is. `spans` is `[{"start", "end"}, ...]` in seconds;
    the answer is `{"voiceprints": [{"embedding": [...], "dim": n} | None, ...]}` in the same
    order, None where a span is too short to carry a usable voiceprint — the caller's
    duration gate should normally catch that first, but a turn clipped by a window boundary
    can be shorter than it looks.

    One call for the whole recording, because the cost that matters is not the model: decoding
    a long WAV costs many times one turn's embedding, so the samples are read once and every
    span is sliced out of them.

    The slice goes to the model in memory (`infer_segment`), not through a temp file and
    `get_embedding`, whose file round trip is many times the model's own forward. Turns go one
    at a time rather than as a padded batch, since padding changes what the model sees.

    `dim` is reported rather than assumed, so a caller sizes its storage from what the model
    actually returned.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"audio not found: {file_path}")
    if not spans:
        return {"voiceprints": [], "model": registry.TITANET_MODEL.split("/")[-1]}

    def run():
        import soundfile

        device = registry.titanet_device()
        model = registry.titanet(device)
        samples, rate = soundfile.read(file_path, dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        out: list = []
        for span in spans:
            start, end = float(span["start"]), float(span["end"])
            piece = samples[int(start * rate):int(end * rate)] if end > start else samples[:0]
            # A quarter second is below what any speaker model can characterise.
            if len(piece) < rate // 4:
                out.append(None)
                continue
            embedding, _logits = model.infer_segment(piece)
            out.append([float(value) for value in embedding.reshape(-1).tolist()])
        return out

    with registry.family_lock(TITANET):
        vectors = registry.run_on_model_thread(TITANET, run)

    return {
        "voiceprints": [None if v is None else {"embedding": v, "dim": len(v)} for v in vectors],
        "model": registry.TITANET_MODEL.split("/")[-1],
    }


def embed_images(paths: list[str], angle=0, mirror=False) -> dict:
    """SigLIP2 vectors for pictures in their display frame, one per path, in order.

    A video's keyframes arrive as several paths (the front extracts them); a photo is one. The
    model name comes back with the vectors so the caller's producer cannot drift from what ran.
    """
    import image_embed
    import image_io

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

    # Twice the model's side, so its resize still starts from more pixels than it keeps.
    min_side = 2 * max(image_embed.visual_preprocess(registry.image_embed_root())["size"])
    images = registry.decode_all(lambda path: image_io.open_display(path, angle, mirror, min_side=min_side), paths)

    def run():
        return registry.siglip_visual(registry.image_embed_device()).embed(images)

    with registry.family_lock(SIGLIP_VISUAL):
        vectors = registry.run_on_model_thread(SIGLIP_VISUAL, run)
    return {"embeddings": vectors, "model": registry.IMAGE_EMBED_MODEL}


def copy_descriptors(paths: list[str], angle=0, mirror=False) -> dict:
    """SSCD descriptors for pictures in their display frame, one per path, in order.

    A video's keyframes arrive as several paths, a photo as one. The model name comes back with
    the vectors so the caller's producer cannot drift from what ran.
    """
    import image_io
    from copy_detect import SHORT_EDGE

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

    # Twice the model's short edge, so its resize still starts from more pixels than it keeps.
    images = registry.decode_all(lambda path: image_io.open_display(path, angle, mirror, min_side=2 * SHORT_EDGE), paths)

    def run():
        return registry.sscd(registry.copy_detect_device()).describe(images)

    with registry.family_lock(SSCD):
        vectors = registry.run_on_model_thread(SSCD, run)
    return {"descriptors": vectors, "model": registry.COPY_DETECT_MODEL}


def shot_boundaries(path: str, times: list[float], duration: float, *, threshold: float, nms: float, merge: float, min_shot: float, min_change: float, debug: bool = False) -> dict:
    """A video's cuts, from its frames at 48×27 in the raw rgb24 file `path` and each frame's second.

    TransNetV2 scores every frame on its model thread; the rules that make cuts of the scores
    (shot_rules.py) and the lasting-change check read the frames again here, outside the lock.
    With `debug` the per-frame scores and every candidate come back too, to tune the rules by.
    """
    import shot_rules
    import shots

    count = min(shots.frame_count(path), len(times))
    if count == 0:
        raise ValueError("shot_boundaries: no frames")
    times = times[:count]

    def run():
        return registry.transnet(registry.transnet_device()).score_file(path, count)

    with registry.family_lock(TRANSNET):
        scores, means, spreads = registry.run_on_model_thread(TRANSNET, run)
    candidates = shot_rules.peaks(scores, times, threshold, nms)
    gaps = shot_rules.leader_gaps(means, spreads, times)
    cuts = shot_rules.boundaries(
        candidates, gaps, duration,
        change_at=lambda t: shots.lasting_change(path, times, t),
        min_change=min_change, merge=merge, min_shot=min_shot,
    )
    result = {"cuts": cuts, "gaps": [{"start": round(a, 3), "end": round(b, 3)} for a, b in gaps], "frames": count, "model": "transnetv2"}
    if debug:
        result["candidates"] = [{"t": round(t, 3), "score": round(s, 4)} for t, s in candidates]
        result["scores"] = [round(s, 4) for s in scores]
    return result


def read_text(paths: list[str], angle=0, mirror=False, language: str | None = None) -> dict:
    """The lines of text in one picture, read in its display frame, and the language they are in.

    `paths` holds one picture, as `call_on_pictures` sends it. The language is told from the lines
    the reader keeps (text_language.py) unless the caller gives it; it picks the transcriber that
    writes the handwritten lines, and how a picture is judged to be writing (text_reader.compose).
    The model name comes back with the lines so the caller's producer cannot drift from what ran.
    """
    import image_io
    import text_reader

    if len(paths) != 1:
        raise ValueError("read_text: one picture at a time")
    if not os.path.exists(paths[0]):
        raise FileNotFoundError(f"Image not found: {paths[0]}")

    clock = _Laps()
    image = registry.decode_all(lambda path: image_io.open_display(path, angle, mirror), paths)[0]
    clock.lap("decode")

    def run():
        clock.lap("ocr wait")
        engine = registry.ocr_photo(registry.ocr_device())
        clock.lap("ocr load")
        reading = engine.detect(image)
        clock.lap("ocr")
        if language:
            return reading, {"code": language, "confidence": None, "given": True}
        found = registry.text_language().detect(text_reader.kept_text(reading))
        clock.lap("language")
        return reading, found and {"code": found["language"], "confidence": found["confidence"], "given": False}

    try:
        with registry.family_lock(OCR):
            reading, found = registry.run_on_model_thread(OCR, run)
        family = registry.htr_family(found and found["code"])
        reading = text_reader.compose(reading, None if family is None else "recogniser" if family == HTR else "certainty")
        texts: list[str] = []
        if reading["transcribe"]:
            # The transcriber writes the text of the lines judged to be writing, in its own family.
            crops = [crop for _, crop in reading["transcribe"]]
            clock.note(f"{family} {len(crops)} of {len(reading['lines'])} lines")

            def transcribe():
                clock.lap(f"{family} wait")
                return _transcribe_lines(family, crops, clock)

            with registry.family_lock(family):
                texts = registry.run_on_model_thread(family, transcribe)
    finally:
        print(f"[model-server] read_text: {clock}", file=sys.stderr, flush=True)
    return {**text_reader.write(reading, texts), "language": found, "model": registry.ocr_model()}


class _Laps:
    """Where a request's time went, as named laps in the order they ran (ein/core#756)."""

    def __init__(self):
        self.at = time.monotonic()
        self.parts: list[str] = []

    def lap(self, name: str) -> None:
        now = time.monotonic()
        self.parts.append(f"{name} {now - self.at:.2f}s")
        self.at = now

    def note(self, text: str) -> None:
        self.parts.append(text)

    def __str__(self) -> str:
        return ", ".join(self.parts)


def _transcribe_lines(family: str, crops: list, clock: _Laps | None = None) -> list[str]:
    """The transcriber's text of `crops`, on its model thread inside its lock.

    The card can run out under the transcriber's beam search when the families beside it grow past
    their claims, and after such a failure the copy on the card fails every later call with "CUDA
    error: unknown error" while the rest of the process works (ein/core#753). So a CUDA failure
    unloads that copy, for the next request to load afresh, and answers `CardBusy`: the caller
    reads the picture again later. A sticky error keeps its text, for the server to recycle on.
    """
    clock = clock or _Laps()
    device = registry.htr_device(family)
    clock.lap(f"{family} card")
    try:
        transcriber = registry.htr(device, family)
        clock.lap(f"{family} load on {device}")
        texts = transcriber.read(crops)
        clock.lap(family)
        return texts
    except Exception as exc:
        if device == "cpu" or not _is_cuda_failure(exc):
            raise
        registry.forget_copies(family, device)
        registry.cleanup_torch()
        raise registry.CardBusy(f"{family} failed on the card and was unloaded: {exc}") from exc


def _is_cuda_failure(exc: Exception) -> bool:
    """Whether `exc` is the card failing a call: out of memory or any CUDA error."""
    message = str(exc)
    return "CUDA error" in message or "out of memory" in message.lower()


def embed_image_texts(texts: list[str]) -> dict:
    """Search text in the pictures' vector space (SigLIP2's text tower)."""
    def run():
        device = registry.image_embed_text_device()
        vectors = registry.siglip_text(device).embed(texts)
        return vectors

    with registry.family_lock(SIGLIP_TEXT):
        vectors = registry.run_on_model_thread(SIGLIP_TEXT, run)
    return {"embeddings": vectors, "model": registry.IMAGE_EMBED_MODEL}


def _document_device(requested: str | None) -> str:
    """Where documents embed: on the card when the caller says it is free (`card`) and it has room, else the setting.

    A copy on the CPU goes when the card takes the documents, so the family is never held twice.
    Runs on the family's model thread, inside its lock, so no request is using that copy.
    """
    if requested == "card":
        if f"{TEXT_EMBED}:cuda" in registry.loaded():
            return "cuda"
        free = registry.free_vram_mb()
        if free is not None and free >= registry.TEXT_EMBED_MIN_FREE_MB:
            registry.forget_copies(TEXT_EMBED, "cpu")
            return "cuda"
    return registry.text_embed_device()


def embed_texts(texts: list[str], role: str, device: str | None = None) -> dict:
    """TEXT_EMBED_MODEL vectors, one per text, in order: queries or documents by `role`.

    A text longer than TEXT_EMBED_MAX_TOKENS is refused, not truncated: the caller chunks, and a
    vector for the first part of a chunk would be stored as the whole of it. `device` is for
    documents: `card` puts them on the card when it has room (the front says so while no batch
    work holds it, ems/card.py).
    """
    if role not in ("query", "document"):
        raise ValueError(f"embed_texts: role must be query or document, not {role!r}")
    family = TEXT_QUERY if role == "query" else TEXT_EMBED
    requested = device

    def run():
        # Placed here, on the family's thread and inside its lock, as every family is: a family
        # moved between the card and the CPU must not be in use by another request (ein/core#736).
        device = registry.text_embed_query_device() if role == "query" else _document_device(requested)
        model = registry.text_embedder(family, device)
        limit = registry.TEXT_EMBED_MAX_TOKENS
        prefix = model.prompts.get(role, "")
        lengths = model.tokenizer([prefix + text for text in texts], add_special_tokens=True, truncation=False)["input_ids"]
        for index, ids in enumerate(lengths):
            if len(ids) > limit:
                raise ValueError(f"embed_texts: text {index} is {len(ids)} tokens, over the {limit}-token limit")
        vectors = model.encode(texts, prompt_name=role, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()

    with registry.family_lock(family):
        vectors = registry.run_on_model_thread(family, run)
    return {"embeddings": vectors, "model": registry.TEXT_EMBED_MODEL.split("/")[-1]}


def _insightface_get(img) -> list:
    """InsightFace's faces in `img`, on the device `registry.place` chooses.

    A card that runs out of memory anyway — another process grew under it — gets the same weights
    on the CPU for this image, only slower; nothing else on the card is touched for it. Load and
    run are both inside the guard: on a full card even creating the CUDA session can run out.
    """
    device = registry.insightface_device()
    try:
        return registry.insightface(force_cpu=device == "cpu").get(img)
    except Exception as exc:
        if device == "cpu" or not _is_cuda_oom(exc):
            raise
        registry.evict_insightface()
        registry.cleanup_torch()
        return registry.insightface(force_cpu=True).get(img)


def detect_faces(image_path: str, angle=0, mirror=False) -> dict:
    """Detect faces in an image using InsightFace. Returns boxes + 512-dim embeddings.

    Detection runs on the *display* frame (see `_read_upright`), so `frame` is reported
    alongside the pixel boxes: the caller normalizes against it and stores fractions.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = _read_upright(image_path, angle, mirror)
    frame_h, frame_w = img.shape[:2]

    def run():
        """Load and run InsightFace, pinned to one thread — see registry.run_on_model_thread."""
        faces = _insightface_get(img)

        results = []
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int).tolist()
            result = {
                "index": i,
                "box": {
                    "x": bbox[0],
                    "y": bbox[1],
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1],
                },
                "confidence": float(face.det_score),
                "embedding": face.embedding.tolist(),  # 512-dim vector
            }
            # Add landmarks if available (5 facial keypoints)
            if face.kps is not None:
                result["landmarks"] = face.kps.tolist()
            pose = _pose_of(face)
            if pose is not None:
                result["pose"] = pose
            results.append(result)
        return results

    with registry.family_lock(INSIGHTFACE):
        results = registry.run_on_model_thread(INSIGHTFACE, run)

    return {"frame": {"width": frame_w, "height": frame_h}, "faces": results}


def _pose_of(face) -> dict | None:
    """Head pose in degrees, from buffalo_l's 3D landmark model: `pitch` (up/down), `yaw`
    (left/right turn), `roll` (tilt). All 0 is a face looking straight into the camera.
    `FaceAnalysis` computes it for every face whether or not anyone reads it; None when the
    pack was loaded without the 3D landmark model."""
    pose = getattr(face, "pose", None)
    if pose is None or len(pose) != 3:
        return None
    pitch, yaw, roll = (float(v) for v in pose)
    return {"pitch": pitch, "yaw": yaw, "roll": roll}


def embed_face(image_path: str, box: dict, angle=0, mirror=False) -> dict | None:
    """Embed a single already-located face using InsightFace.

    `box` is fractional (0..1) and in the display frame, like every box in the API — the
    caller never has to know the image's dimensions. Crops the padded bounding box and runs
    detection on the crop so the returned 512-dim vector is properly aligned (InsightFace
    aligns from its own landmarks), which is what a face located by hand needs. Returns
    {"embedding": [...], "confidence": float, "pose"?: {…}}, or None when no face is found
    inside the box (e.g. a mistaken tag or a face too small to detect).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = _read_upright(image_path, angle, mirror)

    h, w = img.shape[:2]
    x, y = int(round(float(box["x"]) * w)), int(round(float(box["y"]) * h))
    bw, bh = int(round(float(box["width"]) * w)), int(round(float(box["height"]) * h))
    px, py = max(1, int(bw * 0.3)), max(1, int(bh * 0.3))
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img[y0:y1, x0:x1]

    def run():
        """Load and run InsightFace, pinned to one thread — see registry.run_on_model_thread."""
        faces = _insightface_get(crop)

        if not faces:
            return None
        # The tagged face is the dominant one in the crop.
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        result = {"embedding": face.embedding.tolist(), "confidence": float(face.det_score)}
        pose = _pose_of(face)
        if pose is not None:
            result["pose"] = pose
        return result

    with registry.family_lock(INSIGHTFACE):
        return registry.run_on_model_thread(INSIGHTFACE, run)
