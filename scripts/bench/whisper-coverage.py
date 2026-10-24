#!/usr/bin/env python3
"""whisper-coverage — why a transcript misses speech that is plainly audible.

When a transcript holds far fewer lines than the diarizer finds speech turns, something
between the sound and the cues is dropping speech, and there are three candidates that a
transcript alone cannot tell apart:

  - the VAD in front of whisper deciding that speech is not speech (wind, water, a child's
    voice, anything far from the microphone);
  - the language being detected wrongly, so every window is decoded as the wrong language and
    the parts it cannot force into that language come back empty or invented;
  - whisper's own 30-second windowing losing anything, which is the thing people usually
    suspect first and which this exists to rule in or out.

So run the same audio several ways and print what each returns. Differences here are the
answer; agreement means the speech never reached the decoder at all.

    make whisper-coverage \\
        WHISPER_COVERAGE_ARGS="--file /files/2R/01M1SJJZ13NMPMP7PH523JZH2R"

`--sweep` asks which VAD setting: it runs a set of candidate settings over many files at once and scores each setting
on the two numbers that matter — words recovered on clips with speech in them, and words
invented on clips without. A setting that recovers speech and also starts inventing it on
silence is not an improvement, and only the silent half of the sample can show that; so
`--silent` marks the files that the current pipeline (and a listener) found nothing in.

    make whisper-coverage WHISPER_COVERAGE_ARGS="--sweep \\
        --file /files/2R/… --file /files/4H/… --silent /files/V6/… --silent /files/7M/…"

Word counts are the metric, never the share of the timeline the cues span: a setting that
deletes speech inside a region still gets a cue stretched across the whole region. `kept`
is the count after the pipeline's own no-speech guard (model_registry.WHISPER_NO_SPEECH_PROB_MAX),
i.e. what would actually be written; `--text` prints every decode with its suspect lines
marked, which is how a number is checked against what was said.

`--languages` asks which language: it runs the pipeline's own detection
(inference_ops.language_votes, several windows across the speech) on every file, prints each
window's guesses and the average, and then what choose_language would decide for a set of
expected languages at each of several floors — so a floor can be picked by reading which
value keeps the Swedish clips Swedish without turning a foreign clip Swedish too. Where the
choice differs from whisper's own top guess, both decodes are printed for reading.

    make whisper-coverage WHISPER_COVERAGE_ARGS="--languages --expect sv \\
        --file /files/2R/… --file /files/QA/…"

`--compute` asks whether where whisper runs changes what it writes. A transcript decoded on the
card in float16 and one decoded on the CPU in int8 come from the same weights at different
precision, and EMS has done both, depending on what else was on the card. Every file is decoded
under each placement (card float16, card int8_float16, CPU int8, CPU float32), in the language the
pipeline's own vote picks on the reference, and each is scored against card float16 as a word
error rate, with its decode time as a multiple of the audio's length. A placement that agrees
with the reference to a fraction of a percent is one the same transcript can come from; one that
does not must not stand in for it (ein/core#515).

    make whisper-coverage WHISPER_COVERAGE_ARGS="--compute \\
        --file /files/2R/… --file /files/QA/…"
"""

import argparse
import os
import subprocess
import sys
import tempfile


def demux(src: str, dst: str) -> bool:
    """16kHz mono WAV of the file's audio; False when it has no audio stream at all.

    Plenty of the archive's clips carry no audio track, and ffmpeg fails to open an output
    with no streams in it. That is not an error for a coverage run — it is the one case
    where "no speech" is certain — so it is reported rather than raised.
    """
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
         "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", dst],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return True
    if "does not contain any stream" in result.stderr or "Invalid argument" in result.stderr:
        return False
    raise RuntimeError(f"ffmpeg failed on {src}: {result.stderr.strip()}")


def decode(model, wav: str, **kwargs):
    """One decode, the way the pipeline does it apart from what `kwargs` overrides."""
    segments, info = model.transcribe(wav, beam_size=5, condition_on_previous_text=False, **kwargs)
    return list(segments), info


def run(model, wav: str, label: str, **kwargs) -> None:
    """One decode, reported as coverage rather than as text."""
    segments, info = decode(model, wav, **kwargs)

    spoken = sum(s.end - s.start for s in segments)
    words = sum(len(s.text.split()) for s in segments)
    gaps = []
    previous = 0.0
    for s in segments:
        if s.start - previous > 2.0:
            gaps.append((round(previous, 1), round(s.start, 1)))
        previous = s.end

    print(f"\n--- {label}")
    print(f"    language={info.language} ({info.language_probability:.2f}) "
          f"segments={len(segments)} words={words} speech={spoken:.1f}s of {info.duration:.1f}s "
          f"({100 * spoken / max(info.duration, 0.01):.0f}% covered)")
    if gaps:
        print(f"    silent gaps over 2s: {gaps[:8]}")
    for s in segments[:6]:
        print(f"      {s.start:6.2f}-{s.end:6.2f}  {s.text.strip()[:88]}")
    if len(segments) > 6:
        print(f"      … {len(segments) - 6} more")


def diagnose(model, wav: str, language: str, vad: dict) -> None:
    """The four-way split that tells the VAD, the language and the windowing apart."""
    # What the pipeline does today.
    run(model, wav, "as the pipeline runs it (VAD on, language auto)", vad_filter=True, vad_parameters=vad)

    # Is the VAD throwing speech away?
    run(model, wav, "VAD off, language auto", vad_filter=False)

    # Is detection picking the wrong language and decoding into it?
    run(model, wav, f"VAD on, language pinned to {language}", vad_filter=True, language=language, vad_parameters=vad)

    # Both together — the ceiling this audio can give.
    run(model, wav, f"VAD off, language pinned to {language}", vad_filter=False, language=language)


# --- the sweep ---------------------------------------------------------------------------

# A segment whisper itself is unsure of. These are whisper's own signals, not ours:
# `no_speech_prob` is the decoder's belief that the window held no speech at all, and a high
# compression ratio is what a repetition loop looks like once written down. Both thresholds
# are the ones the reference implementation uses to decide to re-decode a window.
NO_SPEECH_PROB = 0.6
COMPRESSION_RATIO = 2.4


def suspect(segment, previous_text: str) -> str:
    """Why a segment is suspect, or an empty string when it is not."""
    text = segment.text.strip()
    if not text:
        return ""
    if segment.no_speech_prob is not None and segment.no_speech_prob > NO_SPEECH_PROB:
        return f"nospeech={segment.no_speech_prob:.2f}"
    if segment.compression_ratio is not None and segment.compression_ratio > COMPRESSION_RATIO:
        return f"compression={segment.compression_ratio:.1f}"
    # A line repeated verbatim is the start of a loop, whatever the decoder thought of it.
    return "repeat" if text == previous_text else ""


def measure(model, wav: str, no_speech_max: float, **kwargs) -> dict:
    """One decode, scored. `kept` is what the pipeline would write after its no-speech guard."""
    segments, info = decode(model, wav, **kwargs)
    words = 0
    suspect_words = 0
    kept = 0
    previous = ""
    lines = []
    for s in segments:
        n = len(s.text.split())
        words += n
        why = suspect(s, previous)
        if why:
            suspect_words += n
        if s.no_speech_prob is None or s.no_speech_prob < no_speech_max:
            kept += n
        previous = s.text.strip()
        # Suspect lines are marked in place, so the text can be read against the verdict.
        lines.append(f"[{why}] {previous}" if why else previous)
    return {
        "segments": len(segments),
        "words": words,
        "suspect": suspect_words,
        "kept": kept,
        "text": " ".join(lines),
        "duration": float(info.duration or 0.0),
    }


# A tight VAD setting, the row the single-knob candidates loosen one at a time.
TIGHT = {"threshold": 0.5, "speech_pad_ms": 400, "min_silence_duration_ms": 500}


def sweep_settings(pipeline: dict) -> list:
    """Candidate VAD settings, each a (label, transcribe kwargs) pair.

    The pipeline's own setting comes first and no VAD at all last — the floor and the
    ceiling every other row is read against. In between, one knob at a time before any
    combination, so a gain can be attributed.
    """
    def vad(base: dict, **overrides):
        params = dict(base)
        params.update(overrides)
        return {"vad_filter": True, "vad_parameters": params}

    short = {"threshold": "thr", "speech_pad_ms": "pad", "min_silence_duration_ms": "silence"}
    describe = ", ".join(f"{short.get(k, k)} {v}" for k, v in pipeline.items())
    return [
        (f"pipeline ({describe})", vad(pipeline)),
        ("tight (thr 0.5, pad 400, silence 500)", vad(TIGHT)),
        ("tight + threshold 0.35", vad(TIGHT, threshold=0.35)),
        ("tight + threshold 0.2", vad(TIGHT, threshold=0.2)),
        ("tight + pad 800ms", vad(TIGHT, speech_pad_ms=800)),
        ("tight + silence 2000ms", vad(TIGHT, min_silence_duration_ms=2000)),
        ("tight + threshold 0.35 + pad 800ms", vad(TIGHT, threshold=0.35, speech_pad_ms=800)),
        ("tight + threshold 0.35 + silence 2000ms", vad(TIGHT, threshold=0.35, min_silence_duration_ms=2000)),
        ("VAD off", {"vad_filter": False}),
    ]


def sweep(model, files: list, language: str, registry, only: list, show_text: bool) -> None:
    settings = sweep_settings(registry.whisper_vad_parameters())
    if only:
        settings = [(label, kwargs) for label, kwargs in settings if any(o in label for o in only)]
    no_speech_max = registry.WHISPER_NO_SPEECH_PROB_MAX
    common = {"language": language} if language else {}
    # totals[label] = [talking words, talking kept, silent words, silent kept]
    totals = {label: [0, 0, 0, 0] for label, _ in settings}
    width = max(len(label) for label, _ in settings)

    with tempfile.TemporaryDirectory() as tmp:
        for index, (path, silent) in enumerate(files):
            wav = os.path.join(tmp, f"{index}.wav")
            print(f"\n=== {'silent' if silent else 'talking'}  {path}", flush=True)
            if not demux(path, wav):
                print("    no audio stream; nothing any setting could decode", flush=True)
                continue
            print(f"    {'setting':<{width}}  segments  words  suspect  kept")
            for label, kwargs in settings:
                result = measure(model, wav, no_speech_max, **common, **kwargs)
                row = totals[label]
                offset = 2 if silent else 0
                row[offset] += result["words"]
                row[offset + 1] += result["kept"]
                print(f"    {label:<{width}}  {result['segments']:8d}  {result['words']:5d}  {result['suspect']:7d}  {result['kept']:4d}", flush=True)
                if show_text and result["text"]:
                    print(f"        {result['text'][:600]}")
            os.remove(wav)

    talking = sum(1 for _, silent in files if not silent)
    silent = len(files) - talking
    print(f"\n=== totals over {talking} talking and {silent} silent files; kept = after the no-speech guard at {no_speech_max}")
    print(f"    {'setting':<{width}}  talking words (kept)  silent words (kept)")
    for label, _ in settings:
        t, tk, s, sk = totals[label]
        print(f"    {label:<{width}}  {t:13d} ({tk:5d})  {s:12d} ({sk:5d})")


# --- the language sweep ----------------------------------------------------------------------

# Floors tried for the expected languages, so the printout answers "which value" in one run.
FLOORS = (0.02, 0.05, 0.1, 0.2, 0.3)


def top(probabilities: dict, n: int) -> str:
    """`sv 0.62  no 0.21  da 0.09` — the n most probable languages of one distribution."""
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:n]
    return "  ".join(f"{code} {probability:.2f}" for code, probability in ranked)


def languages(model, files: list, expected: tuple, registry, ops, show_text: bool) -> None:
    vad = registry.whisper_vad_parameters()
    windows = registry.WHISPER_LANGUAGE_WINDOWS
    print(f"expected languages {','.join(expected) or '(none)'}; {windows} windows per file; floors {FLOORS}")
    # verdicts[floor] = [files where an expected language won over another top guess]
    verdicts = {floor: [] for floor in FLOORS}

    with tempfile.TemporaryDirectory() as tmp:
        for index, (path, _silent) in enumerate(files):
            wav = os.path.join(tmp, f"{index}.wav")
            print(f"\n=== {path}", flush=True)
            if not demux(path, wav):
                print("    no audio stream")
                continue
            votes = ops.language_votes(model, wav, vad, windows)
            if not votes:
                print("    the VAD found no speech")
                os.remove(wav)
                continue
            for number, vote in enumerate(votes):
                print(f"    window {number + 1}: {top(vote, 4)}")
            average = {}
            for vote in votes:
                for code, probability in vote.items():
                    average[code] = average.get(code, 0.0) + probability / len(votes)
            print(f"    average:  {top(average, 5)}")
            line = []
            for floor in FLOORS:
                choice = ops.choose_language(votes, expected, floor)
                line.append(f"@{floor:g} {choice['language']}")
                if choice["language"] != choice["detected"]:
                    verdicts[floor].append(path)
            print(f"    choice:   {'  '.join(line)}")

            # Both decodes when the expected language wins at any floor: what whisper's own
            # guess writes, against what the expected language writes.
            contested = ops.choose_language(votes, expected, min(FLOORS))
            if contested["language"] != contested["detected"]:
                for label, code in ((f"decoded as detected {contested['detected']}", contested["detected"]),
                                    (f"decoded as expected {contested['language']}", contested["language"])):
                    result = measure(model, wav, registry.WHISPER_NO_SPEECH_PROB_MAX, language=code,
                                     vad_filter=True, vad_parameters=vad)
                    print(f"    {label}: {result['segments']} segments, {result['words']} words, {result['kept']} kept")
                    print(f"        {result['text'][:600 if show_text else 300]}")
            os.remove(wav)

    print("\n=== files where an expected language overrode the top guess, per floor")
    for floor in FLOORS:
        print(f"    @{floor:g}: {len(verdicts[floor])}")
        for path in verdicts[floor]:
            print(f"        {path}")


# --- compute types ------------------------------------------------------------------------

# Where a transcript can be decoded. The first is the reference: what the card loads today.
PLACEMENTS = (("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8"), ("cpu", "float32"))
# Each card setting beside the CPU setting closest to it: the two a host would have to agree, if
# whisper ran on whichever had room. int8 is int8 on both; float16 has no CPU kernel, so float32.
TWINS = ((("cuda", "int8_float16"), ("cpu", "int8")), (("cuda", "float16"), ("cpu", "float32")))


def words_of(text: str) -> list[str]:
    """Lowercase words without punctuation: what a word error rate should compare."""
    return "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text).split()


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """Substitutions, insertions and deletions from `reference` to `hypothesis`."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp in enumerate(hypothesis, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp))
        previous = current
    return previous[-1]


def compute_types(files, registry, ops, show_text: bool, placements=PLACEMENTS) -> None:
    """Each file under every placement, scored against the card's float16 — see the docstring."""
    import gc
    import time

    from faster_whisper import WhisperModel

    vad = registry.whisper_vad_parameters()
    expected = registry.WHISPER_LANGUAGES or ("sv",)
    with tempfile.TemporaryDirectory() as tmp:
        clips = []
        for index, (path, _) in enumerate(files):
            wav = os.path.join(tmp, f"{index}.wav")
            if demux(path, wav):
                clips.append((path, wav))
            else:
                print(f"{path}: no audio stream, skipped")

        texts: dict[tuple, dict[str, str]] = {}
        seconds: dict[tuple, dict[str, float]] = {}
        languages: dict[str, str | None] = {}
        durations: dict[str, float] = {}
        # One placement at a time, freed before the next: four copies of large-v3 at once would
        # not fit beside the service's own models.
        for placement in placements:
            device, compute_type = placement
            model = WhisperModel(registry.WHISPER_MODEL, device=device, compute_type=compute_type)
            texts[placement], seconds[placement] = {}, {}
            for path, wav in clips:
                if path not in languages:
                    votes = ops.language_votes(model, wav, vad, registry.WHISPER_LANGUAGE_WINDOWS)
                    chosen = ops.choose_language(votes, expected, registry.WHISPER_LANGUAGE_FLOOR)
                    languages[path] = chosen["language"] if chosen else None
                started = time.monotonic()
                segments, info = decode(model, wav, language=languages[path], vad_filter=True, vad_parameters=vad)
                seconds[placement][path] = time.monotonic() - started
                durations[path] = info.duration
                texts[placement][path] = " ".join(s.text.strip() for s in segments)
                # As it goes: a run stopped part way (out of memory, a restart) still says what it got.
                print(f"{device} {compute_type}  {path}: {info.duration:.0f}s of audio in {seconds[placement][path]:.0f}s, "
                      f"{len(words_of(texts[placement][path]))} words", flush=True)
            del model
            gc.collect()

    reference = placements[0]
    label = lambda p: f"{p[0]} {p[1]}"  # noqa: E731
    print(f"\n=== word error rate against {label(reference)}, and decode time as a multiple of the audio")
    totals = {p: [0, 0, 0.0] for p in placements}  # errors, reference words, seconds
    for path, _ in clips:
        ref_words = words_of(texts[reference][path])
        print(f"\n{path}: {durations[path]:.0f}s, language {languages[path]}, {len(ref_words)} words")
        for placement in placements:
            errors = word_errors(ref_words, words_of(texts[placement][path]))
            totals[placement][0] += errors
            totals[placement][1] += len(ref_words)
            totals[placement][2] += seconds[placement][path]
            rate = 100 * errors / max(1, len(ref_words))
            print(f"    {label(placement):22} WER {rate:5.1f}%  {seconds[placement][path] / max(durations[path], 0.01):5.2f}x")
            if show_text and placement != reference and errors:
                print(f"        {texts[placement][path][:400]}")
        if show_text:
            print(f"        reference: {texts[reference][path][:400]}")

    audio = sum(durations[path] for path, _ in clips)
    print("\n=== all files")
    for placement, (errors, words, spent) in totals.items():
        print(f"    {label(placement):22} WER {100 * errors / max(1, words):5.2f}%  {spent / max(audio, 0.01):5.2f}x  ({errors} of {words} words)")

    print("\n=== each card setting against its CPU twin: how far the same file differs by where it ran")
    for card, cpu in TWINS:
        if card not in placements or cpu not in placements:
            continue
        errors = words = 0
        for path, _ in clips:
            card_words = words_of(texts[card][path])
            errors += word_errors(card_words, words_of(texts[cpu][path]))
            words += len(card_words)
        print(f"    {label(card):22} vs {label(cpu):12} WER {100 * errors / max(1, words):5.2f}%  ({errors} of {words} words)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", action="append", default=[],
                        help="media file with speech in it (container path); repeatable")
    parser.add_argument("--silent", action="append", default=[],
                        help="media file the current pipeline found no speech in; repeatable, --sweep only")
    parser.add_argument("--language", default=None,
                        help="language to pin (diagnosis: the pinned runs, default sv; --sweep: every run, default auto-detect as the pipeline does)")
    parser.add_argument("--sweep", action="store_true",
                        help="score candidate VAD settings over every file instead of the four-way diagnosis")
    parser.add_argument("--text", action="store_true", help="--sweep: also print each decode's text")
    parser.add_argument("--setting", action="append", default=[],
                        help="--sweep: run only settings whose label contains this (repeatable); e.g. --setting pipeline --setting tight --setting off")
    parser.add_argument("--languages", action="store_true",
                        help="print the multi-window language detection and what each floor would choose, per file")
    parser.add_argument("--compute", action="store_true",
                        help="decode every file on the card and the CPU at each precision and score them against card float16")
    parser.add_argument("--placement", action="append", default=[],
                        help="--compute: only these, as device:compute_type (repeatable), the first the reference; "
                             "e.g. --placement cuda:float16 --placement cuda:int8_float16 --placement cpu:int8")
    parser.add_argument("--expect", action="append", default=[],
                        help="--languages: an expected language (repeatable); default the configured CFG_WHISPER_LANGUAGES, or sv")
    args = parser.parse_args()

    files = [(path, False) for path in args.file] + [(path, True) for path in args.silent]
    if not files:
        print("nothing to decode: pass --file (and, with --sweep, --silent)", file=sys.stderr)
        return 2
    for path, _ in files:
        if not os.path.exists(path):
            print(f"no such file: {path}", file=sys.stderr)
            return 2

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ems", "modelworker"))
    import model_registry as registry  # noqa: E402

    if args.compute:
        import inference_ops as ops  # noqa: E402
        placements = tuple(tuple(p.split(":", 1)) for p in args.placement) or PLACEMENTS
        compute_types(files, registry, ops, args.text, placements)
        return 0

    model = registry.whisper("cuda", "float16")

    if args.languages:
        import inference_ops as ops  # noqa: E402
        expected = tuple(args.expect) or registry.WHISPER_LANGUAGES or ("sv",)
        languages(model, files, expected, registry, ops, args.text)
        return 0

    if args.sweep:
        sweep(model, files, args.language or "", registry, args.setting, args.text)
        return 0

    vad = registry.whisper_vad_parameters()
    with tempfile.TemporaryDirectory() as tmp:
        for path, _ in files:
            wav = os.path.join(tmp, "audio.wav")
            print(f"file: {path}")
            if not demux(path, wav):
                print("    no audio stream; nothing to decode")
                continue
            diagnose(model, wav, args.language or "sv", vad)

    return 0


if __name__ == "__main__":
    sys.exit(main())
