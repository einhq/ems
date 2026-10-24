#!/usr/bin/env python3
"""shots-bench — `video.shots` against listed cuts, and against clips that have none.

A video's chapters are its cuts. The detector and the defaults of its rules were chosen on the
archive's own material (ein/core#707): edited film and tape transfers whose cuts a person
listed, and phone clips that should yield none. This asks the running EMS for the cuts under a
grid of its rules, to measure a change to them the same way.

    make shots-bench SHOTS_BENCH_ARGS="--truth /files-volatile/ab/<id>=12.4,51.08 --negative /files-volatile/cd/<id>"

Paths are as EMS sees them, and are the 1080p renditions the cuts are found in. The arguments:

- `--truth PATH=T1,T2,…` (repeatable): a video and its cuts in seconds;
- `--negative PATH` (repeatable): a video with no cut;
- `--thresholds`, `--changes` (comma-separated): the grid of `threshold` and `min_change` asked for;
- `--strips`: contact strips of ±1.5 s around every listed cut and every cut found with the
  defaults, and a 2 fps sheet of each listed video, to check a list by eye — a person's list is
  approximate, and has been short of a cut;
- `--url`, `--key`: the EMS to ask; `--out`.

A found cut matches a listed one within ±0.5, ±1 and ±2 s, one to one, nearest first. Writes
`report.tsv`, one row per point of the grid, to `--out`, and prints every row.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import urllib.request

TOLERANCES = (0.5, 1.0, 2.0)


def shots(url: str, key: str, path: str, **rules) -> dict:
    body = json.dumps({"file": {"path": path}, **rules}).encode()
    request = urllib.request.Request(f"{url}/v0/video/shots", data=body, headers={"content-type": "application/json", "authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        answer = json.loads(response.read())
    if not answer.get("ok"):
        raise RuntimeError(f"{path}: {answer.get('error')}")
    return answer["result"]


def match(found: list[float], truth: list[float], tolerance: float) -> tuple[int, float | None]:
    """One-to-one matches within `tolerance`, nearest pairs first; the count and their mean error."""
    pairs = sorted((abs(f - g), i, j) for i, f in enumerate(found) for j, g in enumerate(truth) if abs(f - g) <= tolerance)
    used_f, used_g, errors = set(), set(), []
    for error, i, j in pairs:
        if i in used_f or j in used_g:
            continue
        used_f.add(i)
        used_g.add(j)
        errors.append(error)
    return len(errors), (sum(errors) / len(errors) if errors else None)


def strips(path: str, out: str, label: str, times: list[float], duration: float) -> None:
    for t in times:
        target = os.path.join(out, f"strip-{label}-{t:07.2f}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t - 1.5):.3f}", "-t", "3", "-i", path, "-vf", "fps=8,scale=160:-2,tile=24x1", "-frames:v", "1", target], check=False)
    rows = max(1, math.ceil(duration * 2 / 20))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", f"fps=2,scale=96:-2,tile=20x{rows}", "-frames:v", "1", os.path.join(out, f"sheet-{label}.png")], check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", action="append", default=[])
    parser.add_argument("--negative", action="append", default=[])
    parser.add_argument("--thresholds", default="0.4,0.5,0.6,0.7")
    parser.add_argument("--changes", default="0,0.1,0.2")
    parser.add_argument("--strips", action="store_true")
    parser.add_argument("--url", default=f"http://localhost:{os.environ.get('EMS_PORT', '8181')}")
    parser.add_argument("--key", default=os.environ.get("EMS_KEY", ""))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    videos = [{"path": p, "truth": sorted(float(c) for c in cuts.split(",") if c)} for p, _, cuts in (s.rpartition("=") for s in args.truth)]
    videos += [{"path": p, "truth": []} for p in args.negative]
    grid = [(float(t), float(c)) for t in args.thresholds.split(",") for c in args.changes.split(",")]

    found: dict[tuple[float, float], list[list[float]]] = {point: [] for point in grid}
    for n, video in enumerate(videos):
        started = time.monotonic()
        for threshold, change in grid:
            result = shots(args.url, args.key, video["path"], threshold=threshold, min_change=change)
            found[(threshold, change)].append([cut["t"] for cut in result["cuts"]])
        seconds = (time.monotonic() - started) / len(grid)
        print(f"[{n}] {os.path.basename(video['path'])}: {result['duration']:.1f}s, {result['frames']} frames, {seconds:.1f}s a call", flush=True)
        if args.strips and video["truth"]:
            defaults = shots(args.url, args.key, video["path"])
            strips(video["path"], args.out, f"{n}", sorted(set(video["truth"]) | {round(c["t"], 2) for c in defaults["cuts"]}), defaults["duration"])

    rows = []
    for threshold, change in grid:
        tp = {tol: 0 for tol in TOLERANCES}
        errors, found_total, truth_total, clean, negatives = [], 0, 0, 0, 0
        for video, cuts in zip(videos, found[(threshold, change)]):
            if video["truth"]:
                found_total += len(cuts)
                truth_total += len(video["truth"])
                for tol in TOLERANCES:
                    hits, mean = match(cuts, video["truth"], tol)
                    tp[tol] += hits
                    if tol == 1.0 and mean is not None:
                        errors.append(mean)
            else:
                negatives += 1
                clean += 1 if not cuts else 0
        row = {"threshold": threshold, "min_change": change, "found": found_total, "truth": truth_total}
        for tol in TOLERANCES:
            precision = tp[tol] / found_total if found_total else 0.0
            recall = tp[tol] / truth_total if truth_total else 0.0
            row[f"f1@{tol}"] = round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0
        row["mae@1"] = round(sum(errors) / len(errors), 3) if errors else ""
        row["clean_negatives"] = f"{clean}/{negatives}" if negatives else ""
        rows.append(row)

    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "report.tsv"), "w") as handle:
        handle.write("\t".join(keys) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[k]) for k in keys) + "\n")
    for row in rows:
        print("  " + "  ".join(f"{k}={row[k]}" for k in keys))
    print(f"report: {os.path.join(args.out, 'report.tsv')}")


if __name__ == "__main__":
    main()
