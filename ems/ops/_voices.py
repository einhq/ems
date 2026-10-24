"""Turning windowed diarization back into one speaker per recording.

The diarizer cannot take a whole file — Sortformer is quadratic in input length, so the
worker feeds it fixed windows (model_registry.DIARIZE_WINDOW_SECONDS). The cost is that its speaker indices restart with every window: the same
person is `w0/speaker_0` in the first window and maybe `w1/speaker_2` in the second, and
nothing in the output says so. This joins them back up using the voiceprints.

The window is a fact about the hardware, so undoing it is EMS's job: `speech.voiceprint`
returns one speaker per voice across the whole file.

**Why this is not plain clustering.** Two *different* labels inside one window are the
diarizer's own statement that those are different people, so they must never end up in the
same speaker however similar their embeddings look. A single-link pass would silently fuse
them; here the constraint is known, so it is honoured. Respecting it also makes the result
better than clustering turns directly: a window-local speaker pools several turns into one
centroid, so the comparison runs on more voice than any single utterance carries.
"""

from __future__ import annotations

import math

# Cosine similarity above which two window-local speakers are taken to be one person.
#
# Erring high is the safer direction — leaving one person split across two labels within a
# file is recoverable by the cross-file identity step that follows, whereas fusing two people
# inside a file hides one of them from it entirely. Not yet checked against labelled
# false-accept/false-reject examples from the archive.
STITCH_THRESHOLD = 0.6

# Embedded voice, in seconds, a window-local label needs to found a speaker of its own.
#
# A long recording has a few labels carrying nearly all the voice and many with a turn or two:
# a centroid made of one short utterance rarely resembles anything, and one made of nothing
# cannot. So a label under this much voice is a crumb: it may join a speaker but never be one.
ANCHOR_MIN_SECONDS = 10.0

# Similarity at which a crumb joins an anchor. Lower than STITCH_THRESHOLD on purpose: the
# anchor's centroid is minutes of voice, so the comparison is one noisy vector against a firm
# one rather than two noisy ones against each other, and the cost of a wrong join is one brief
# turn under the wrong label — assignment still votes on that turn's own vector.
ATTACH_THRESHOLD = 0.5

# Two halves of a turn cut at a window boundary meet within rounding, not at the same float.
MERGE_GAP_SECONDS = 0.25


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, or 0 when either has no magnitude."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    aa = sum(x * x for x in a)
    bb = sum(y * y for y in b)
    if aa == 0 or bb == 0:
        return 0.0
    return dot / math.sqrt(aa * bb)


def window_of(local_speaker: str) -> str:
    """The window part of a `w<n>/speaker_<n>` label, or the whole label if it carries none."""
    slash = local_speaker.find("/")
    return local_speaker if slash == -1 else local_speaker[:slash]


def _centroid(vectors: list[list[float]]) -> list[float]:
    out = [0.0] * len(vectors[0])
    for vector in vectors:
        for i, value in enumerate(vector):
            out[i] += value
    return [value / len(vectors) for value in out]


def stitch_windows(
    turns: list[dict],
    vectors: list[list[float] | None],
    threshold: float = STITCH_THRESHOLD,
    anchor_seconds: float = ANCHOR_MIN_SECONDS,
    attach_threshold: float = ATTACH_THRESHOLD,
) -> list[str | None]:
    """Resolve window-local speakers to file-level ones.

    `turns` are the diarizer's `{"start", "end", "localSpeaker"}` and `vectors[i]` is the
    voiceprint of `turns[i]`, or None where it was too short to embed. Returns one file-wide
    label (`speaker_0`, `speaker_1`, … ordered by first appearance) per turn, in order, or
    None for a turn nothing could attribute.

    Labels with at least `anchor_seconds` of embedded voice are the anchors, and only they
    found speakers: greedy agglomeration over their centroids, most similar pair first,
    refusing any merge that would put two labels from the same window together. Greedy rather
    than optimal because the input is tiny — a handful of labels per window — and because
    taking the strongest evidence first is what makes the refusals land on the weak pairs.

    Every other label is a crumb. One with a centroid joins the most similar anchor at
    `attach_threshold`, again never one holding its own window; one with no vector, or no
    anchor like it, leaves its turns unattributed. The turns are still reported — a stretch
    where somebody spoke whom nothing recognised is information — but they are not a speaker.
    """
    by_label: dict[str, dict] = {}
    for turn, vector in zip(turns, vectors):
        if vector is None:
            continue
        entry = by_label.setdefault(turn["localSpeaker"], {"vectors": [], "first": turn["start"], "seconds": 0.0})
        entry["vectors"].append(vector)
        entry["first"] = min(entry["first"], turn["start"])
        entry["seconds"] += turn["end"] - turn["start"]

    ordered = sorted(by_label.items(), key=lambda item: (item[1]["first"], item[0]))
    clusters = [
        {
            "labels": [label],
            "windows": {window_of(label)},
            "centroid": _centroid(entry["vectors"]),
            "count": len(entry["vectors"]),
            "first": entry["first"],
        }
        for label, entry in ordered
        if entry["seconds"] >= anchor_seconds
    ]

    while True:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # The diarizer already said these are different people; no similarity overrides it.
                if clusters[i]["windows"] & clusters[j]["windows"]:
                    continue
                sim = cosine(clusters[i]["centroid"], clusters[j]["centroid"])
                if sim >= threshold and (best is None or sim > best[2]):
                    best = (i, j, sim)
        if best is None:
            break
        i, j, _ = best
        a, b = clusters[i], clusters[j]
        total = a["count"] + b["count"]
        clusters[i] = {
            "labels": a["labels"] + b["labels"],
            "windows": a["windows"] | b["windows"],
            # Weighted by how many turns each side pooled, so a long speaker is not pulled off
            # course by a single short utterance merging into it.
            "centroid": [(x * a["count"] + y * b["count"]) / total for x, y in zip(a["centroid"], b["centroid"])],
            "count": total,
            "first": min(a["first"], b["first"]),
        }
        del clusters[j]

    # A crumb joins the anchor it most resembles. Anchors are settled by now, so the join
    # neither moves a centroid nor depends on the order the crumbs are tried in.
    for label, entry in ordered:
        if entry["seconds"] >= anchor_seconds:
            continue
        centroid = _centroid(entry["vectors"])
        best = None
        for cluster in clusters:
            if window_of(label) in cluster["windows"]:
                continue
            sim = cosine(centroid, cluster["centroid"])
            if sim >= attach_threshold and (best is None or sim > best[1]):
                best = (cluster, sim)
        if best is not None:
            best[0]["labels"].append(label)

    # Name speakers by when they are first heard, so the labels read in file order.
    clusters.sort(key=lambda c: c["first"])
    speaker_of: dict[str, str] = {}
    for index, cluster in enumerate(clusters):
        for label in cluster["labels"]:
            speaker_of[label] = f"speaker_{index}"

    return [speaker_of.get(turn["localSpeaker"]) for turn in turns]


def merge_adjacent(turns: list[dict], gap: float = MERGE_GAP_SECONDS) -> list[dict]:
    """Merge turns that touch and belong to the same speaker.

    Windowing cuts a turn that spans a boundary into two meeting at it. Once stitching has
    agreed both sides are one voice, they are one turn again — and a turn that is whole is a
    longer, better sample for the voiceprint that decides who it is. `turns` carry `start`,
    `end` and `speaker`; a merged turn keeps the first half's other fields and loses its
    `embedding`, since neither half's vector was measured over the whole. Two unattributed
    turns are never merged: nothing says they are the same voice.
    """
    out: list[dict] = []
    for turn in sorted(turns, key=lambda t: (t["start"], t["end"])):
        last = out[-1] if out else None
        if last is not None and turn["speaker"] is not None and last["speaker"] == turn["speaker"] and turn["start"] - last["end"] <= gap:
            last["end"] = max(last["end"], turn["end"])
            last.pop("embedding", None)
            continue
        out.append(dict(turn))
    return out


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else list(vector)


def speakers_of(turns: list[dict]) -> list[dict]:
    """The file-wide speakers of a recording, one entry each, in the order they are first heard.

    `turns` are `{"start", "end", "speaker", "embedding"?}` as `speech.voiceprint` answers them.
    Each speaker gets its turns (`[[start, end], …]` in time order), its talk time, one voiceprint
    for all of its voice — the mean of its turns' unit vectors weighted by each turn's length,
    then made unit again, so a long turn counts for more than an "mm". A speaker has at least one
    voiced turn by construction (stitching
    founds speakers on voiced turns only); a turn nothing attributed belongs to no speaker.
    """
    by_speaker: dict[str, list[dict]] = {}
    for turn in sorted(turns, key=lambda t: (t["start"], t["end"])):
        if turn.get("speaker") is not None:
            by_speaker.setdefault(turn["speaker"], []).append(turn)

    out = []
    for speaker, own in by_speaker.items():
        voiced = [t for t in own if t.get("embedding")]
        if not voiced:
            continue
        dimension = len(voiced[0]["embedding"])
        total = [0.0] * dimension
        for turn in voiced:
            weight = turn["end"] - turn["start"]
            for i, value in enumerate(_unit(turn["embedding"])):
                total[i] += value * weight
        out.append({
            "id": speaker,
            "turns": [[t["start"], t["end"]] for t in own],
            "talkSeconds": round(sum(t["end"] - t["start"] for t in own), 3),
            "vector": _unit(total),
            "dimension": dimension,
        })
    return out
