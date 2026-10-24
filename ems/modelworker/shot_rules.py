"""From a shot detector's per-frame scores to the cuts a video's chapters start at.

Pure and free of numpy, so the rules are tested without a model. The model and the pictures are
shots.py's; it hands these the scores, each frame's luma, and a way to measure how much the
picture changed around a time.

The rules, each measured on the archive (ein/core#707, wow notes/lessons-media.md):
- a cut is a peak of the score at or over the threshold, with no higher one within `nms` seconds,
  and starts on the frame after the peak: the network marks the old shot's last frame;
- black or white leader (a run of nearly uniform, nearly black or white frames) is a gap: no cut
  inside it, and one where it ends, unless it opens or closes the file;
- a cut must change the picture for good: the colours a second before and after it must differ,
  which a flash, a flicker or a splice blip does not do;
- cuts closer than `merge` seconds are one, the stronger;
- no shot is shorter than `min_shot`: the weaker cut of a short one goes.
"""

from __future__ import annotations

from typing import Callable


def peaks(scores: list[float], times: list[float], threshold: float, nms: float) -> list[tuple[float, float]]:
    """The frames scoring at or over `threshold`, none within `nms` seconds of a higher one, in time
    order, each as the time of the frame after it — the new shot's first — and its score."""
    order = sorted((i for i in range(len(scores)) if scores[i] >= threshold), key=lambda i: -scores[i])
    kept: list[tuple[int, float]] = []
    for i in order:
        if all(abs(times[i] - times[j]) > nms for j, _ in kept):
            kept.append((i, scores[i]))
    return sorted((times[min(i + 1, len(times) - 1)], score) for i, score in kept)


def leader_gaps(means: list[float], spreads: list[float], times: list[float], *, dark: float = 18, light: float = 235, flat: float = 6, least: float = 0.2) -> list[tuple[float, float]]:
    """Runs of blank frames — luma spread under `flat`, mean under `dark` or over `light` — at least `least` seconds long.

    A night scene with a few lights in it is as dark as leader at 48×27, but spreads 11 to 12
    (a phone clip on fry, 2026-10-24): blank means flatter than that.
    """
    gaps: list[tuple[float, float]] = []
    start = None
    for i in range(len(times) + 1):
        blank = i < len(times) and spreads[i] < flat and (means[i] < dark or means[i] > light)
        if blank and start is None:
            start = i
        elif not blank and start is not None:
            end = times[i] if i < len(times) else times[-1]
            if end - times[start] >= least:
                gaps.append((times[start], end))
            start = None
    return gaps


def boundaries(
    candidates: list[tuple[float, float]],
    gaps: list[tuple[float, float]],
    duration: float,
    *,
    change_at: Callable[[float], float] | None,
    min_change: float,
    merge: float,
    min_shot: float,
) -> list[dict]:
    """The cuts kept of `candidates`, as `{t, score, kind}`: `cut` from the detector, `leader` at a gap's end (score None)."""
    kept: list[tuple[float, float, str]] = []
    for t, score in candidates:
        if any(start - 0.5 <= t <= end + 0.5 for start, end in gaps):
            continue
        if change_at is not None and min_change > 0 and change_at(t) < min_change:
            continue
        kept.append((t, score, "cut"))
    for start, end in gaps:
        if start > 0.5 and end < duration - 0.5:
            # A leader's end is certain, so it outranks any detection beside it.
            kept.append((end, float("inf"), "leader"))
    kept.sort()

    merged: list[tuple[float, float, str]] = []
    for t, score, kind in kept:
        if merged and t - merged[-1][0] < merge:
            if score > merged[-1][1]:
                merged[-1] = (t, score, kind)
            continue
        merged.append((t, score, kind))

    while merged:
        edges = [0.0] + [t for t, _, _ in merged] + [duration]
        short = next((i for i in range(len(edges) - 1) if edges[i + 1] - edges[i] < min_shot), None)
        if short is None:
            break
        sides = [j for j in (short - 1, short) if 0 <= j < len(merged)]
        merged.pop(min(sides, key=lambda j: merged[j][1]))

    return [{"t": round(t, 3), "score": None if kind == "leader" else round(score, 4), "kind": kind} for t, score, kind in merged]
