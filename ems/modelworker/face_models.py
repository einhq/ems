"""Faces in a picture: RetinaFace finds them, TransFace turns each into a 512-dim vector, and the
head pose is estimated from the five points the detector gives.

RetinaFace is yakhyo/retinaface-pytorch's ResNet-34 export (MIT, trained on WIDER FACE), run on the
picture scaled down so its longer side is at most `DETECT_EDGE`. TransFace is Alibaba DAMO's ViT
recogniser as ModelScope's `iic/cv_vit_face-recognition` publishes it (MIT), read on the 112x112
crop the five points align to the ArcFace template. Both run in onnxruntime; nothing here imports
torch.

A face is `{"box": [x0, y0, x1, y1], "score", "kps": [[x, y]] * 5, "embedding", "pose"}` in pixels
of the picture it was found in: `kps` are the eyes, the nose tip and the mouth corners, each pair
in the order a face looking into the camera shows them from the picture's left, `embedding` is unit
length and `pose` is None when the points give none.
"""

from __future__ import annotations

import math

import numpy as np

# The longer side a picture is scaled down to before detection. A smaller picture is read at its
# own size.
DETECT_EDGE = 1024
# Detections below this confidence are not reported; the caller's quality gate sits above it.
DETECT_FLOOR = 0.5
NMS_IOU = 0.4
# Anchor grids kept, one per input size: pictures come in a handful of sizes, and one grid at
# DETECT_EDGE is about half a megabyte.
ANCHOR_SIZES_KEPT = 16

# InsightFace's 112x112 template for the five points, which TransFace was trained on.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)

# Where the nose sits between the eye line and the mouth line, as a share of the distance between
# them, for a face neither raised nor lowered, and how fast that share moves with the angle. Fitted
# so the pitch reads like the 3D landmark model's the stored poses came from, sign included: on
# 1,189 faces it agrees with it on facing the camera (|pitch| <= 20, |yaw| <= 25, |roll| <= 20) for
# 88 % (ein/core#887).
PITCH_LEVEL = 0.50
PITCH_SCALE = 0.70


def similarity(src, dst) -> np.ndarray | None:
    """The 2x3 rotation, uniform scale and shift taking `src` points onto `dst` in the least-squares
    sense (Umeyama), or None when `src` has no spread to fit."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_d, dst_d = src - src_mean, dst - dst_mean
    variance = src_d.var(0).sum()
    if not variance > 0:
        return None
    cov = dst_d.T @ src_d / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1] = -1
    rotation = u @ np.diag(d) @ vt
    scale = (s @ d) / variance
    matrix = np.empty((2, 3))
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * rotation @ src_mean
    return matrix


def align(cv2, img, kps) -> np.ndarray | None:
    """The 112x112 BGR crop whose five points sit on ARCFACE_TEMPLATE, black outside the picture."""
    matrix = similarity(kps, ARCFACE_TEMPLATE)
    if matrix is None:
        return None
    return cv2.warpAffine(img, matrix, (112, 112), borderValue=0.0)


def pose(kps) -> dict | None:
    """Head pose in degrees from the five points: `roll` from the eye line, `yaw` from how far the
    nose sits off the middle of the eyes and mouth corners, in half eye-distances, and `pitch` from
    where it sits between the eye line and the mouth line. All 0 is a face looking into the camera;
    a positive yaw has the nose towards the picture's right, a negative pitch towards the mouth.
    None when the points are degenerate."""
    left_eye, right_eye, nose, left_mouth, right_mouth = (np.asarray(p, dtype=np.float64) for p in kps)
    ex, ey = right_eye - left_eye
    eyes_apart = math.hypot(ex, ey)
    if not eyes_apart > 0:
        return None
    c, s = ex / eyes_apart, ey / eyes_apart

    def along(p):
        return (p[0] - left_eye[0]) * c + (p[1] - left_eye[1]) * s

    def down(p):
        return -(p[0] - left_eye[0]) * s + (p[1] - left_eye[1]) * c

    middle = (along(left_eye) + along(right_eye) + along(left_mouth) + along(right_mouth)) / 4
    eye_line = (down(left_eye) + down(right_eye)) / 2
    span = (down(left_mouth) + down(right_mouth)) / 2 - eye_line
    if not span > 0:
        return None
    return {
        "pitch": -math.degrees(math.atan(((down(nose) - eye_line) / span - PITCH_LEVEL) / PITCH_SCALE)),
        "yaw": math.degrees(math.atan((along(nose) - middle) / (eyes_apart / 2))),
        "roll": math.degrees(math.atan2(ey, ex)),
    }


def _providers(device: str, mem_limit_bytes: int) -> list:
    """CUDA first when asked for, CPU always as the fallback. kSameAsRequested stops onnxruntime's
    arena from doubling; the limit is a backstop well above what either model needs."""
    if device != "cuda":
        return ["CPUExecutionProvider"]
    cuda = {"arena_extend_strategy": "kSameAsRequested", "gpu_mem_limit": str(mem_limit_bytes)}
    return [("CUDAExecutionProvider", cuda), "CPUExecutionProvider"]


class RetinaFace:
    """The detector: BGR minus (104, 117, 123), the repository's anchors and decoding, NMS at
    NMS_IOU."""

    MIN_SIZES = ((16, 32), (64, 128), (256, 512))
    STEPS = (8, 16, 32)
    VARIANCE = (0.1, 0.2)

    def __init__(self, path: str, providers: list):
        import cv2
        import onnxruntime

        self.cv2 = cv2
        self.session = onnxruntime.InferenceSession(path, providers=providers)
        self.input = self.session.get_inputs()[0].name
        self._anchors: dict[tuple[int, int], np.ndarray] = {}

    def anchors(self, height: int, width: int) -> np.ndarray:
        if (height, width) not in self._anchors:
            if len(self._anchors) >= ANCHOR_SIZES_KEPT:
                self._anchors.pop(next(iter(self._anchors)))
            out = []
            for sizes, step in zip(self.MIN_SIZES, self.STEPS):
                rows, cols = math.ceil(height / step), math.ceil(width / step)
                ys, xs = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
                cx = ((xs + 0.5) * step / width).reshape(-1)
                cy = ((ys + 0.5) * step / height).reshape(-1)
                per = np.stack([np.stack([cx, cy, np.full_like(cx, m / width), np.full_like(cy, m / height)], 1) for m in sizes], 1)
                out.append(per.reshape(-1, 4))
            self._anchors[(height, width)] = np.concatenate(out).astype(np.float32)
        return self._anchors[(height, width)]

    def __call__(self, img, edge: int = DETECT_EDGE, upscale: bool = False) -> list[dict]:
        """`{"box", "score", "kps"}` per face in `img`, in its pixels. The picture is read with its
        longer side at `edge`, scaled down to it, or also up with `upscale`."""
        cv2 = self.cv2
        height, width = img.shape[:2]
        scale = edge / max(height, width)
        if scale > 1 and not upscale:
            scale = 1.0
        small = img if scale == 1 else cv2.resize(
            img, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        sh, sw = small.shape[:2]
        x = (small.astype(np.float32) - np.float32([104, 117, 123])).transpose(2, 0, 1)[None]
        loc, conf, lmk = (out[0] for out in self.session.run(None, {self.input: x}))
        priors = self.anchors(sh, sw)
        scores = conf[:, 1]
        keep = np.flatnonzero(scores >= DETECT_FLOOR)
        if not len(keep):
            return []
        priors, loc, lmk, scores = priors[keep], loc[keep], lmk[keep], scores[keep]
        v0, v1 = self.VARIANCE
        centre = priors[:, :2] + loc[:, :2] * v0 * priors[:, 2:]
        size = priors[:, 2:] * np.exp(loc[:, 2:] * v1)
        boxes = np.concatenate([centre - size / 2, centre + size / 2], 1) * [sw, sh, sw, sh] / scale
        points = (priors[:, None, :2] + lmk.reshape(-1, 5, 2) * v0 * priors[:, None, 2:]) * [sw, sh] / scale
        rects = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        picked = np.asarray(cv2.dnn.NMSBoxes(rects, scores.tolist(), DETECT_FLOOR, NMS_IOU)).reshape(-1)
        picked = sorted(picked, key=lambda i: -scores[i])
        return [{"box": boxes[i].tolist(), "score": float(scores[i]), "kps": points[i].tolist()} for i in picked]


class TransFace:
    """The recogniser: the aligned crop as RGB scaled to [-1, 1], one face per run (the export takes
    a batch of one). The model has three outputs; the first is the vector."""

    def __init__(self, path: str, providers: list):
        import cv2
        import onnxruntime

        self.cv2 = cv2
        self.session = onnxruntime.InferenceSession(path, providers=providers)
        self.input = self.session.get_inputs()[0].name

    def __call__(self, img, kps) -> np.ndarray | None:
        crop = align(self.cv2, img, kps)
        if crop is None:
            return None
        x = ((crop[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        vector = self.session.run(None, {self.input: np.ascontiguousarray(x)})[0].reshape(-1).astype(np.float64)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else None


class Faces:
    """Both models on one device (`cuda` or `cpu`), as one model family."""

    def __init__(self, detector_path: str, recogniser_path: str, device: str, mem_limit_bytes: int):
        providers = _providers(device, mem_limit_bytes)
        self.detector = RetinaFace(detector_path, providers)
        self.recogniser = TransFace(recogniser_path, providers)

    def get(self, img, edge: int = DETECT_EDGE, upscale: bool = False) -> list[dict]:
        """Every face in `img` (BGR) with its vector and pose; a face whose points do not align is
        left out."""
        faces = []
        for face in self.detector(img, edge, upscale):
            embedding = self.recogniser(img, face["kps"])
            if embedding is None:
                continue
            faces.append({**face, "embedding": embedding, "pose": pose(face["kps"])})
        return faces
