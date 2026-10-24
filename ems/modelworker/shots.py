"""Shot boundaries: TransNetV2's cut score for every frame of a video, and what the rules need beside it.

The frames arrive as a raw rgb24 file at 48×27, the network's own input size, which the front
decodes. They are read a window at a time — 100 frames, 50 apart, the middle 50 of each kept, as
the authors run the network — so a two-hour tape costs a window of memory, not the 700 MB of its
frames. The first and last frames are repeated 25 times to pad the ends, as theirs are.

The weights are the authors' TransNetV2 converted to PyTorch, float32 on the card and the CPU alike.
"""

from __future__ import annotations

import math
import os

WIDTH, HEIGHT = 48, 27
FRAME_BYTES = WIDTH * HEIGHT * 3
WINDOW, STRIDE, PAD = 100, 50, 25
# Windows a forward pass takes at once: two hold 420 MB of the card on fry, eight 1.4 GB, at the
# same speed (2026-10-24).
BATCH = 2
LUMA = (0.299, 0.587, 0.114)


def read_frames(handle, start: int, stop: int):
    """Frames [start, stop) of an open frames file, as a uint8 array of shape (n, 27, 48, 3)."""
    import numpy as np

    handle.seek(start * FRAME_BYTES)
    data = handle.read((stop - start) * FRAME_BYTES)
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH, 3)


def windows(count: int):
    """Each window's source frame for every one of its 100 places, ends padded by repetition."""
    padded = PAD + count + (PAD + STRIDE - (count % STRIDE if count % STRIDE else STRIDE))
    for start in range(0, padded - WINDOW + 1, STRIDE):
        yield [min(max(p - PAD, 0), count - 1) for p in range(start, start + WINDOW)]


class ShotScorer:
    def __init__(self, weights: str, device: str):
        import torch
        from transnetv2 import TransNetV2

        self.torch = torch
        self.device = device
        model = TransNetV2()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model = model.eval().to(device)

    def score_file(self, path: str, count: int) -> tuple[list[float], list[float], list[float]]:
        """The cut score of each of the file's `count` frames, and each frame's luma mean and spread."""
        import numpy as np

        scores: list[float] = []
        means: list[float] = []
        spreads: list[float] = []
        weights = np.array(LUMA, dtype=np.float32)
        pending: list = []

        def flush():
            batch = self.torch.from_numpy(np.ascontiguousarray(np.stack(pending))).to(self.device)
            with self.torch.inference_mode():
                one, _ = self.model(batch)
            for row in self.torch.sigmoid(one).float().cpu().numpy()[:, PAD:PAD + STRIDE, 0]:
                scores.extend(float(s) for s in row)
            pending.clear()

        with open(path, "rb") as handle:
            for places in windows(count):
                first, last = places[0], places[-1]
                frames = read_frames(handle, first, last + 1)
                window = frames[[p - first for p in places]]
                # The frames this window decides, its middle 50, are the ones it measures for luma.
                kept = window[PAD:PAD + STRIDE]
                luma = kept.astype(np.float32) @ weights
                means.extend(float(m) for m in luma.mean(axis=(1, 2)))
                spreads.extend(float(s) for s in luma.std(axis=(1, 2)))
                pending.append(window)
                if len(pending) == BATCH:
                    flush()
            if pending:
                flush()
        return scores[:count], means[:count], spreads[:count]


def histogram(frames):
    """The 8×8×8 RGB histogram of some frames, normalised to sum to one each."""
    import numpy as np

    bins = (frames // 32).astype(np.int32)
    codes = (bins[..., 0] * 64 + bins[..., 1] * 8 + bins[..., 2]).reshape(len(frames), -1)
    out = np.stack([np.bincount(code, minlength=512) for code in codes]).astype(np.float32)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1)


def lasting_change(path: str, times: list[float], t: float) -> float:
    """How far the picture moved across `t`: the Bhattacharyya distance between the median colours of
    [t − 1, t − 0.1] and [t + 0.3, t + 1.3] seconds. Near 0 for a flash or a flicker that returns."""
    import bisect

    import numpy as np

    def span(a: float, b: float):
        lo, hi = bisect.bisect_left(times, a), bisect.bisect_right(times, b)
        if hi <= lo:
            return None
        with open(path, "rb") as handle:
            return np.median(histogram(read_frames(handle, lo, hi)), axis=0)

    before, after = span(t - 1.0, t - 0.1), span(t + 0.3, t + 1.3)
    if before is None or after is None:
        return 1.0
    before, after = before / max(before.sum(), 1e-9), after / max(after.sum(), 1e-9)
    return math.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(before * after)))))


def frame_count(path: str) -> int:
    return os.path.getsize(path) // FRAME_BYTES
