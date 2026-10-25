"""The faces of a video, from the detections on its sampled frames — `faces.detect_video`'s rules.

Plain Python on plain lists, so they are tested without the models. A detection is a dict with
`box` (`x0, y0, x1, y1` in pixels of the display frame), `confidence`, `unit` (the face's
vector, unit length) and optionally `pose`; a sample is `{"index", "t", "faces": [detection…]}`.

Detections are followed from sample to sample into tracks, and the tracks of one file grouped
into faces: one face per person seen, as far as the vectors can tell. Two tracks seen in the
same frame are two people and are never grouped. A face is answered with its clearest
detection and one vector, the normalized mean of its best detections.
"""

from __future__ import annotations

import heapq
import math

# Overlap for a detection to continue a track by its box alone.
TRACK_MIN_IOU = 0.3
# Below this a detection never continues a track, however much the boxes overlap: two people
# crossing in front of the camera.
TRACK_FLOOR_SIMILARITY = 0.3
# A person moving fast across the frame, or a long video's samples seconds apart, leave no overlap
# between two sightings; this much likeness continues the track anyway.
TRACK_MIN_SIMILARITY = 0.5
# How long a track may go without a detection and still continue, and a face stay on screen: a
# face turned away or blurred for a moment. Seconds, or samples for a long video sampled more
# sparsely (`max_gap`).
TRACK_MAX_GAP_SECONDS = 2.0
TRACK_MAX_GAP = 2
# Cosine between the mean vectors of two groups of tracks for them to be one face. Lower than a
# photo's match: a face a few dozen pixels high, turned and moving, is far from its own other
# sightings, and the rule that two faces on screen together are two people keeps apart the
# lookalikes that do come this close.
GROUP_MIN_SIMILARITY = 0.4
# The best detections whose mean is a track's or a face's vector.
VECTOR_DETECTIONS = 5
# Shorter side, in pixels, at which a face counts as fully sized for its quality: the size the
# recognition model reads it at.
FULL_SIZE_PIXELS = 112.0
# The most tracks grouped, clearest first. A crowd seen for an hour could give thousands, and
# grouping costs the square of their number; the faint ones dropped are background faces.
MAX_TRACKS = 1500
# How far the box drawn between two kept sightings of a face may stray from one left out of its
# path, as a fraction of the face's own width or height.
PATH_TOLERANCE = 0.1
# The longest a path goes between two kept sightings: a face that stands still is placed anew at
# least this often.
PATH_MAX_STEP = 1.0


def max_gap(interval: float) -> float:
    """Seconds a face may go unseen and still be the same track and on screen."""
    return max(TRACK_MAX_GAP_SECONDS, TRACK_MAX_GAP * interval)


def unit(vector: list[float]) -> list[float]:
    """`vector` scaled to length 1; a zero vector stays zero."""
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm > 0 else list(vector)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def iou(a, b) -> float:
    """Overlap of two `(x0, y0, x1, y1)` boxes as intersection over union."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def quality(detection: dict) -> float:
    """How clear a look at the face a detection is: its confidence, discounted for a face
    smaller than the recognition model reads and for a head turned away."""
    x0, y0, x1, y1 = detection["box"]
    size = min(1.0, min(x1 - x0, y1 - y0) / FULL_SIZE_PIXELS)
    pose = detection.get("pose") or {}
    yaw = pose.get("yaw")
    frontal = max(0.0, math.cos(math.radians(float(yaw)))) if isinstance(yaw, (int, float)) else 1.0
    return float(detection["confidence"]) * size * frontal


def mean_vector(detections: list[dict], count: int = VECTOR_DETECTIONS) -> list[float]:
    """The normalized mean of the `count` clearest detections' vectors."""
    best = sorted(detections, key=quality, reverse=True)[:count]
    total = [0.0] * len(best[0]["unit"])
    for detection in best:
        total = [t + v for t, v in zip(total, detection["unit"])]
    return unit(total)


def follow(samples: list[dict], interval: float) -> list[list[dict]]:
    """Detections followed through the samples, `interval` seconds apart, in order, as tracks:
    lists of detections, each with the sample `index` and `t` it was seen at.

    Greedy and forward-only. Each detection continues at most one track seen within
    `max_gap(interval)`, best pairing first: a box that overlaps the track's last one, or a face
    much like the track's own; never one that looks like somebody else.
    """
    reach = max_gap(interval) + interval + 1e-6
    tracks: list[dict] = []
    for sample in samples:
        index = sample["index"]
        found = [{**d, "index": index, "t": sample["t"]} for d in sample["faces"]]
        live = [ti for ti, track in enumerate(tracks) if sample["t"] - track["last"] <= reach]
        pairs = []
        for ti in live:
            track = tracks[ti]
            for di, detection in enumerate(found):
                likeness = dot(track["vector"], detection["unit"])
                if likeness < TRACK_FLOOR_SIMILARITY:
                    continue
                overlap = iou(track["detections"][-1]["box"], detection["box"])
                if overlap >= TRACK_MIN_IOU or likeness >= TRACK_MIN_SIMILARITY:
                    pairs.append((overlap + likeness, ti, di))
        used_tracks, used_found = set(), set()
        for _score, ti, di in sorted(pairs, reverse=True):
            if ti in used_tracks or di in used_found:
                continue
            track = tracks[ti]
            track["detections"].append(found[di])
            track["sum"] = [s + v for s, v in zip(track["sum"], found[di]["unit"])]
            track["vector"] = unit(track["sum"])
            track["last"] = sample["t"]
            used_tracks.add(ti)
            used_found.add(di)
        for di, detection in enumerate(found):
            if di not in used_found:
                tracks.append({"detections": [detection], "sum": list(detection["unit"]), "vector": list(detection["unit"]), "last": sample["t"]})
    return [track["detections"] for track in tracks]


def dot_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """Every pairwise dot product of `vectors`, with numpy where the worker has it."""
    try:
        import numpy
    except ImportError:
        return [[dot(a, b) for b in vectors] for a in vectors]
    m = numpy.asarray(vectors, dtype=numpy.float64)
    return (m @ m.T).tolist()


def group(tracks: list[list[dict]], min_similarity: float = GROUP_MIN_SIMILARITY) -> list[list[list[dict]]]:
    """The tracks of one file grouped into faces, by the likeness of their mean vectors.

    Two groups join while the cosine between the sums of their tracks' vectors is at least
    `min_similarity`, most alike first, unless some track of one was seen in a frame some track
    of the other was seen in too. A group's sum averages out the noise of its single sightings,
    which the mean of every pairwise cosine between two groups keeps. `between[i][j]` is the dot
    product of two groups' sums and `between[i][i]` a sum's squared length, so a join only adds
    rows: the joined group's row is the sum of the two. Candidate pairs wait in a heap and are
    checked against the groups' versions when taken.
    """
    count = len(tracks)
    between = dot_matrix([mean_vector(track) for track in tracks])
    members = [[i] for i in range(count)]
    seen = [{d["index"] for d in track} for track in tracks]
    version = [0] * count
    live = set(range(count))

    heap: list[tuple[float, int, int, int, int]] = []

    def offer(i: int, j: int) -> None:
        likeness = between[i][j] / math.sqrt(between[i][i] * between[j][j])
        if likeness >= min_similarity and not seen[i] & seen[j]:
            heapq.heappush(heap, (-likeness, i, j, version[i], version[j]))

    for i in range(count):
        for j in range(i + 1, count):
            offer(i, j)

    while heap:
        _likeness, i, j, vi, vj = heapq.heappop(heap)
        if i not in live or j not in live or version[i] != vi or version[j] != vj:
            continue
        live.discard(j)
        members[i].extend(members[j])
        seen[i] |= seen[j]
        version[i] += 1
        between[i][i] += between[j][j] + 2 * between[i][j]
        for k in live:
            if k != i:
                between[i][k] += between[j][k]
                between[k][i] = between[i][k]
        for k in live:
            if k != i:
                offer(min(i, k), max(i, k))
    return [[tracks[m] for m in members[i]] for i in sorted(live)]


def appearances(times: list[float], interval: float, duration: float | None) -> list[list[float]]:
    """The seconds a face is on screen: the times it was seen merged into spans wherever it went
    unseen no longer than `max_gap(interval)`, each padded by half an interval and kept within
    the video."""
    reach = max_gap(interval) + interval + 1e-6
    spans: list[list[float]] = []
    for t in sorted(set(times)):
        if spans and t - spans[-1][1] <= reach:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    end = duration if duration and duration > 0 else math.inf
    return [[round(max(0.0, s - interval / 2), 3), round(min(end, e + interval / 2), 3)] for s, e in spans]


def faces_of(samples: list[dict], interval: float, duration: float | None) -> list[dict]:
    """The faces of a video, first seen first: per face `appearances`, how many `detections`,
    its clearest detection (`detection`, with the `t` it was seen at), its `vector`, and every
    `sighting` in time order. A face's tracks are never seen in the same frame, so there is at
    most one sighting a sampled frame."""
    tracks = follow(samples, interval)
    if len(tracks) > MAX_TRACKS:
        tracks = sorted(tracks, key=lambda track: max(quality(d) for d in track), reverse=True)[:MAX_TRACKS]
    faces = []
    for face_tracks in group(tracks):
        detections = [d for track in face_tracks for d in track]
        clearest = max(detections, key=quality)
        faces.append({
            "appearances": appearances([d["t"] for d in detections], interval, duration),
            "detections": len(detections),
            "detection": clearest,
            "vector": mean_vector(detections),
            "sightings": sorted(detections, key=lambda d: d["t"]),
        })
    return sorted(faces, key=lambda face: (face["appearances"][0][0], face["detection"]["t"]))


def path(points: list[list[float]], interval: float) -> list[list[float]]:
    """A face's sightings `[t, x, y, width, height]`, in time order and `interval` seconds apart
    where it was seen without a break, cut down to the ones a box needs to follow it.

    Each sighting's box is first the median of it and its neighbours', coordinate by coordinate:
    that takes out the detector's jitter on a face standing still and leaves a steady movement as
    it is. A sighting is then kept where the line between the kept ones either side would miss it
    by more than `PATH_TOLERANCE`, and at least every `PATH_MAX_STEP` seconds, so a face moving
    fast keeps every sample and one standing still about one a second. Both ends of a run of
    sightings are kept, so the line never bridges a stretch the face went unseen in.
    """
    runs: list[list[list[float]]] = []
    for point in points:
        if runs and point[0] - runs[-1][-1][0] <= 1.5 * interval + 1e-6:
            runs[-1].append(point)
        else:
            runs.append([point])
    kept: list[list[float]] = []
    for run in runs:
        steady = [list(run[0])] + [
            [run[i][0], *(sorted(run[k][c] for k in (i - 1, i, i + 1))[1] for c in range(1, 5))] for i in range(1, len(run) - 1)
        ] + ([list(run[-1])] if len(run) > 1 else [])
        kept.append(steady[0])
        start = 0
        while start < len(steady) - 1:
            end = start + 1
            while end + 1 < len(steady) and steady[end + 1][0] - steady[start][0] <= PATH_MAX_STEP + 1e-6 and _drawn_within(steady[start], steady[end + 1], steady[start + 1:end + 1]):
                end += 1
            kept.append(steady[end])
            start = end
    return [[point[0], *(round(v, 4) for v in point[1:])] for point in kept]


def _drawn_within(first: list[float], last: list[float], between: list[list[float]]) -> bool:
    """Whether the box slid from `first` to `last` passes within `PATH_TOLERANCE` of each of
    `between`: its centre, width and height."""
    for t, x, y, width, height in between:
        w = (t - first[0]) / (last[0] - first[0])
        at = [first[c] + (last[c] - first[c]) * w for c in range(1, 5)]
        across, down = max(width, 1e-6) * PATH_TOLERANCE, max(height, 1e-6) * PATH_TOLERANCE
        if abs(at[0] + at[2] / 2 - (x + width / 2)) > across or abs(at[1] + at[3] / 2 - (y + height / 2)) > down:
            return False
        if abs(at[2] - width) > across or abs(at[3] - height) > down:
            return False
    return True
