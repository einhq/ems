#!/usr/bin/env python3
"""diarize-compare — speaker diarizers side by side, as `speech.voiceprint` would use them.

Each diarizer's turns go through what `speech.voiceprint` does with them — TitaNet per turn,
the windows stitched by voiceprint (ems/ops/_voices.py), touching turns merged and re-embedded,
one pooled voiceprint per speaker — so what is compared is what the archive would store
(ein/core#841). The diarizers:

- `v1`: `diar_sortformer_4spk-v1` in DIARIZE_WINDOW_SECONDS windows, as EMS runs it;
- `v2-window`: `diar_streaming_sortformer_4spk-v2` in the same windows;
- `v2`: the streaming model over the whole file in one call; its speaker cache carries the
  speakers across the file, so there is one window and nothing to stitch.

Two kinds of reference, either or both:

- `--truth DIR` with `--audio DIR` and `--voices DIR`: test-content's generated recordings, whose
  `truth/recordings/*.yaml` say who speaks when. Scored frame by frame at 10 ms, 0.25 s collar,
  speakers mapped one to one: confusion, miss and false alarm against the truth's speech. Each
  speaker's voiceprint is also matched to the nearest of the cast's own voices (`out/voices`).
- `--speakers FILE`: a JSON list of an archive's stored speakers (`file`, `person`, `source`,
  `turns`), with the recordings under `--files` as EMS sees them (`/files/<id[-2:]>/<id>`).
  A speaker someone named by hand (`source: manual`) labels the new speaker that holds most of
  its voice. Scored: how much of a named voice lands in one speaker, and whether each labelled
  speaker's voiceprint is nearest the mean of the same person's speakers in the other recordings
  — and above speaker-assign's personThreshold for them, below it for everyone else.
  The stored turns are v1's, so the first score leans towards v1; the second does not.

Every file's result is written to `--out` as it is made and read back on the next run.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Pure modules, loaded by path so the service's own imports are not needed.
voices = _load("_voices", os.path.join(ROOT, "ems", "ops", "_voices.py"))

WINDOW_SECONDS = 90.0
MIN_TURN_SECONDS = 1.5  # ems/ops/speech.py
PERSON_THRESHOLD = 0.7  # module-archive speaker-assign's personThreshold
FRAME = 0.01
COLLAR = 0.25

MODELS = {
    "v1": ("nvidia/diar_sortformer_4spk-v1", "2617bffbd820aa29d8f1fb6ab6f9ed7f0adbc996"),
    "v2": ("nvidia/diar_streaming_sortformer_4spk-v2", "84edd514b8ef68004c10086918cd62f2148cbd59"),
}
TITANET = ("nvidia/speakerverification_en_titanet_large", "0dc382f40121a5fbd34db10a2bb04d826c2be6a8")

# The model card's "very high latency" setting, in 80 ms frames: 27 s chunks, the most context
# per step. Latency means nothing for a whole file; context is what decides who is who.
STREAMING = {"chunk_len": 340, "chunk_right_context": 40, "fifo_len": 40, "spkcache_update_period": 300, "spkcache_len": 188}


def nemo_file(repo: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, f"{repo.split('/')[-1]}.nemo", revision=revision)


_models: dict = {}


def diarizer(version: str):
    if version not in _models:
        from nemo.collections.asr.models import SortformerEncLabelModel

        repo, revision = MODELS[version]
        model = SortformerEncLabelModel.restore_from(nemo_file(repo, revision), map_location="cuda", strict=False)
        model.eval()
        if version == "v2":
            for key, value in STREAMING.items():
                setattr(model.sortformer_modules, key, value)
            model.sortformer_modules._check_streaming_parameters()
        print(f"{version}: {repo}@{revision[:10]} streaming_mode={getattr(model, 'streaming_mode', None)}", flush=True)
        _models[version] = model
    return _models[version]


def titanet():
    if "titanet" not in _models:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        model = EncDecSpeakerLabelModel.restore_from(nemo_file(*TITANET), map_location="cuda")
        _models["titanet"] = model.eval()
    return _models["titanet"]


def parse_turn(line: str):
    parts = line.split()
    if len(parts) < 3:
        return None
    start, end = float(parts[0]), float(parts[1])
    return (start, end, parts[2]) if end > start else None


def run_diarizer(name: str, samples: np.ndarray, rate: int, tmp: str) -> list[dict]:
    """The diarizer's turns, `localSpeaker` prefixed with its window as inference_ops.diarize does."""
    model = diarizer("v1" if name == "v1" else "v2")
    window = None if name == "v2" else WINDOW_SECONDS
    total = len(samples) / rate
    path = os.path.join(tmp, "window.wav")
    turns, offset, index = [], 0.0, 0
    while offset < total:
        span = total - offset if window is None else min(window, total - offset)
        if span < 2.0:
            break
        soundfile.write(path, samples[int(offset * rate):int((offset + span) * rate)], rate)
        with torch.inference_mode():
            lines = model.diarize(audio=[path], batch_size=1, verbose=False)[0]
        for line in lines:
            turn = parse_turn(line)
            if turn:
                turns.append({"start": round(turn[0] + offset, 3), "end": round(min(turn[1] + offset, total), 3), "localSpeaker": f"w{index}/{turn[2]}"})
        index += 1
        offset += span
    turns.sort(key=lambda t: (t["start"], t["end"]))
    return turns


def embed(samples: np.ndarray, rate: int, spans: list[tuple[float, float]]) -> list[list[float] | None]:
    model = titanet()
    out = []
    for start, end in spans:
        piece = samples[int(start * rate):int(end * rate)]
        if end - start < MIN_TURN_SECONDS or len(piece) < rate // 4:
            out.append(None)
            continue
        with torch.inference_mode():
            vector, _ = model.infer_segment(piece)
        out.append([float(x) for x in vector.reshape(-1).tolist()])
    return out


def voiceprint(name: str, samples: np.ndarray, rate: int, tmp: str) -> dict:
    """What `speech.voiceprint` would answer with this diarizer: raw turns, speakers, timings."""
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    turns = run_diarizer(name, samples, rate, tmp)
    diarize_s = time.monotonic() - began
    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    vectors = embed(samples, rate, [(t["start"], t["end"]) for t in turns])
    labels = voices.stitch_windows(turns, vectors)
    merged = voices.merge_adjacent([
        {"start": t["start"], "end": t["end"], "speaker": s, **({"embedding": v} if v is not None else {})}
        for t, s, v in zip(turns, labels, vectors)
    ])
    rejoined = [t for t in merged if "embedding" not in t]
    for turn, vector in zip(rejoined, embed(samples, rate, [(t["start"], t["end"]) for t in rejoined])):
        if vector is not None:
            turn["embedding"] = vector
    return {
        "raw": turns,
        "speakers": voices.speakers_of(merged),
        "diarizeSeconds": round(diarize_s, 2),
        "totalSeconds": round(time.monotonic() - began, 2),
        "peakMB": round(peak_mb),
    }


def read_audio(path: str, tmp: str) -> tuple[np.ndarray, int]:
    wav = os.path.join(tmp, "audio.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-vn", "-i", path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav], check=True)
    samples, rate = soundfile.read(wav, dtype="float32", always_2d=False)
    os.remove(wav)
    return samples, rate


# ── scoring ──────────────────────────────────────────────────────────────────────────────────


def frames(intervals, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in intervals:
        mask[int(round(start / FRAME)):int(round(end / FRAME))] = True
    return mask


def overlap_seconds(a, b) -> float:
    total = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


def overlapped_share(raw: list[dict], duration: float) -> tuple[float, float]:
    """Seconds of speech, and the share of it where the diarizer hears two or more voices at once."""
    n = int(math.ceil(duration / FRAME)) + 1
    count = np.zeros(n, dtype=np.int16)
    for t in raw:
        count[int(round(t["start"] / FRAME)):int(round(t["end"] / FRAME))] += 1
    speech = int((count > 0).sum())
    return speech * FRAME, (int((count > 1).sum()) / speech) if speech else 0.0


def der(ref: dict[str, list], hyp: dict[str, list], duration: float) -> dict:
    """Frame-level miss, false alarm and confusion against `ref`'s speech, outside a collar."""
    from scipy.optimize import linear_sum_assignment

    n = int(math.ceil(duration / FRAME)) + 1
    scored = np.ones(n, dtype=bool)
    for intervals in ref.values():
        for start, end in intervals:
            for edge in (start, end):
                scored[max(0, int((edge - COLLAR) / FRAME)):int((edge + COLLAR) / FRAME)] = False
    rnames, hnames = list(ref), list(hyp)
    R = np.stack([frames(ref[k], n) & scored for k in rnames]) if rnames else np.zeros((0, n), bool)
    H = np.stack([frames(hyp[k], n) & scored for k in hnames]) if hnames else np.zeros((0, n), bool)
    nr, nh = R.sum(0), H.sum(0)
    matched = 0
    if rnames and hnames:
        cost = -(R[:, None, :] & H[None, :, :]).sum(-1)
        rows, cols = linear_sum_assignment(cost)
        matched = int(-cost[rows, cols].sum())
    total = int(nr.sum())
    miss = int(np.maximum(nr - nh, 0).sum())
    fa = int(np.maximum(nh - nr, 0).sum())
    conf = int(np.minimum(nr, nh).sum()) - matched
    return {"ref": total * FRAME, "miss": miss / total, "fa": fa / total, "conf": conf / total, "der": (miss + fa + conf) / total}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) or 1.0)


# ── the two references ───────────────────────────────────────────────────────────────────────


def truth_items(args) -> list[dict]:
    import yaml

    items = []
    for path in sorted(glob.glob(os.path.join(args.truth, "*.yaml"))):
        truth = yaml.safe_load(open(path))
        name = os.path.basename(path)[:-5]
        ref: dict[str, list] = {}
        for turn in truth["turns"]:
            ref.setdefault(turn["speaker"], []).append((float(turn["start"]), float(turn["end"])))
        items.append({"id": name, "path": os.path.join(args.audio, f"{name}.wav"), "ref": ref})
    return items


def speaker_items(args) -> list[dict]:
    stored = json.load(open(args.speakers))
    by_file: dict[str, list] = {}
    for s in stored:
        by_file.setdefault(s["file"], []).append(s)
    return [
        {"id": fid, "path": os.path.join(args.files, fid[-2:], fid), "name": rows[0].get("name"), "stored": rows}
        for fid, rows in sorted(by_file.items(), key=lambda kv: kv[1][0].get("duration") or 0)
    ]


def score_truth(items, results, names, voices_dir):
    cast = {}
    for path in sorted(glob.glob(os.path.join(voices_dir, "*.wav"))):
        with tempfile.TemporaryDirectory() as tmp:
            samples, rate = read_audio(path, tmp)
        cast[os.path.basename(path)[:-4]] = unit(embed(samples, rate, [(0.0, len(samples) / rate)])[0])
    print("\n== test content: DER (10 ms, 0.25 s collar) and nearest cast voice")
    for name in names:
        sums = {"ref": 0.0, "miss": 0.0, "fa": 0.0, "conf": 0.0}
        right = wrong = 0
        sims = []
        count_ok = 0
        for item in items:
            r = results[item["id"]][name]
            hyp = {s["id"]: [tuple(t) for t in s["turns"]] for s in r["speakers"]}
            d = der(item["ref"], hyp, r["duration"])
            for key in ("miss", "fa", "conf"):
                sums[key] += d[key] * d["ref"]
            sums["ref"] += d["ref"]
            count_ok += len(hyp) == len(item["ref"])
            for s in r["speakers"]:
                owner = max(item["ref"], key=lambda k: overlap_seconds(item["ref"][k], s["turns"]))
                vector = unit(s["vector"])
                best = max(cast, key=lambda k: float(vector @ cast[k]))
                sims.append(float(vector @ cast[owner]))
                right += best == owner
                wrong += best != owner
            print(f"  {name:10} {item['id']:24} speakers {len(hyp)}/{len(item['ref'])}  conf {d['conf']:.3f} miss {d['miss']:.3f} fa {d['fa']:.3f}")
        ref = sums["ref"]
        print(f"  {name:10} TOTAL conf {sums['conf'] / ref:.3f} miss {sums['miss'] / ref:.3f} fa {sums['fa'] / ref:.3f}"
              f"  right speaker count {count_ok}/{len(items)}  nearest cast voice {right}/{right + wrong}"
              f"  cosine to own voice median {np.median(sims):.3f} min {min(sims):.3f}")


def score_speakers(items, results, names):
    print("\n== archive sample: named voices")
    for name in names:
        coverage = []
        labelled = []  # (file, person, vector)
        speakers = 0
        overlapped = []
        for item in items:
            r = results[item["id"]][name]
            speakers += len(r["speakers"])
            overlapped.append(r["overlapped"])
            named = [s for s in item["stored"] if s["source"] == "manual" and s["person"]]
            for s in named:
                held = [overlap_seconds(s["turns"], ours["turns"]) for ours in r["speakers"]]
                talk = sum(e - b for b, e in s["turns"])
                coverage.append((max(held) if held else 0.0) / talk if talk else 0.0)
            for ours in r["speakers"]:
                held = [(overlap_seconds(s["turns"], ours["turns"]), s) for s in item["stored"]]
                if not held:
                    continue
                seconds, owner = max(held, key=lambda h: h[0])
                if owner["source"] == "manual" and owner["person"] and seconds >= 0.5 * ours["talkSeconds"]:
                    labelled.append((item["id"], owner["person"], unit(ours["vector"]), ours["talkSeconds"]))
        right = total = above = false_accept = 0
        for fid, person, vector, _ in labelled:
            means = {}
            for other_fid, other_person, other_vector, seconds in labelled:
                if other_fid != fid:
                    means.setdefault(other_person, []).append(other_vector * seconds)
            if person not in means:
                continue
            means = {p: unit(np.sum(v, axis=0)) for p, v in means.items()}
            sims = {p: float(vector @ m) for p, m in means.items()}
            total += 1
            right += max(sims, key=sims.get) == person
            above += sims[person] >= PERSON_THRESHOLD
            false_accept += any(s >= PERSON_THRESHOLD for p, s in sims.items() if p != person)
        print(f"  {name:10} speakers {speakers}  overlapped speech {np.mean(overlapped):.3f}"
              f"  named voice in one speaker {np.mean(coverage):.3f} (median {np.median(coverage):.3f})"
              f"  labelled {len(labelled)}  nearest person right {right}/{total}"
              f"  ≥{PERSON_THRESHOLD} to own {above}/{total}  ≥{PERSON_THRESHOLD} to another {false_accept}/{total}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diarizers", default="v1,v2-window,v2")
    parser.add_argument("--truth")
    parser.add_argument("--audio")
    parser.add_argument("--voices")
    parser.add_argument("--speakers")
    parser.add_argument("--files", default="/files")
    parser.add_argument("--limit", type=int, default=0, help="first N archive recordings, shortest first")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    names = args.diarizers.split(",")
    os.makedirs(args.out, exist_ok=True)

    truth = truth_items(args) if args.truth else []
    stored = speaker_items(args) if args.speakers else []
    if args.limit:
        stored = stored[:args.limit]
    results: dict = {}
    for item in truth + stored:
        cache = os.path.join(args.out, f"{item['id']}.json")
        results[item["id"]] = json.load(open(cache)) if os.path.exists(cache) else {}
        missing = [n for n in names if n not in results[item["id"]]]
        if not missing:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            samples, rate = read_audio(item["path"], tmp)
            duration = len(samples) / rate
            for name in missing:
                r = voiceprint(name, samples, rate, tmp)
                r["duration"] = duration
                r["speech"], r["overlapped"] = overlapped_share(r["raw"], duration)
                results[item["id"]][name] = r
                print(f"{item.get('name') or item['id']:40} {duration:7.0f}s {name:10} speakers {len(r['speakers']):2}"
                      f"  turns {len(r['raw']):4}  overlapped {r['overlapped']:.3f}  diarize {r['diarizeSeconds']:6.1f}s"
                      f"  all {r['totalSeconds']:6.1f}s  peak {r['peakMB']:5} MB", flush=True)
        with open(cache + ".new", "w") as handle:
            json.dump(results[item["id"]], handle)
        os.replace(cache + ".new", cache)

    if truth:
        score_truth(truth, results, names, args.voices)
    if stored:
        score_speakers(stored, results, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
