#!/usr/bin/env python3
"""face-compare — face detectors and recognisers side by side, against faces people named.

InsightFace's buffalo_l is the baseline; the candidates are what could replace it under a licence
that allows passing the weights on (ein/core#840). Every detector runs on every picture, and every
recogniser embeds every face each detector found, so a recogniser is judged both on buffalo_l's own
detections and on the detector it would ship with. The ArcFace-style ones (w600k_r50, AuraFace)
and Intel's are aligned on the detector's five landmarks, SFace by its own alignCrop on them,
facenet on the squared box, and dlib on its own five-point predictor in the box.

`--models` holds the weights: `buffalo_l/` as EMS has it, YuNet and SFace from OpenCV Zoo,
`auraface/glintr100.onnx`, `intel/face-reidentification-retail-0095.{xml,bin}`, `dlib/`,
`torch/checkpoints/20180402-114759-vggface2.pt` and MediaPipe's `blaze_face_full_range.tflite`
and `face_landmarker.task`. facenet-pytorch, dlib, openvino and mediapipe are not in EMS's image:
install them beside it (`pip install --target DIR --no-deps`, then PYTHONPATH), and MediaPipe needs
libEGL and libGLESv2.

`run` takes a manifest, one picture per line, as EMS sees it:

    {"file": id, "path": "/files/<id[-2:]>/<id>", "angle": 0, "mirror": false, "taken": "1962-05",
     "gt": [{"region": id, "box": {x, y, width, height}, "person": id, "kind": "legacy"}]}

`box` is a fraction of the display frame, as the archive stores it, or null when only who is in
the picture is known (test content); then a picture showing one person is matched to its largest
detection. `kind` says where the name came from. The old archive's hand-drawn tags went to the
buffalo_l detection inside them (`transfer`, its box is buffalo_l's) or, where it found none, stayed
as they were drawn (`legacy`); together they are every face the old archive named, which is what a
detector's recall is counted over. `manual` faces were named in EiN, on buffalo_l's detections,
and count only towards recognition. Results go to `--out` as they are made and a later run carries
on.

`score` reads them back and prints what the people recognition would see:

- detection: of the faces the old archive named, how many each detector finds at the confidence
  the archive's quality gate asks of buffalo_l, and at the confidence where it finds as many faces
  per picture as buffalo_l does; also apart, of those buffalo_l found and of those it missed;
- recognition: same-person pairs found (TAR) at a fixed rate of different-person pairs passing
  (FAR), over pairs from different pictures;
- assignment: face-assign's vote (k nearest named faces from other pictures, summed similarity,
  winner by a margin) with each face's own name hidden — how often it names the right person,
  the wrong one or nobody, at the thresholds the archive uses for buffalo_l and, for the others, at
  thresholds that let the same share of different-person pairs through;
- head pose: how often the five-point estimate and Face Mesh agree with buffalo_l's 3D landmark
  model on whether a face looks into the camera (person covers prefer those).

Each is split into old pictures (taken before 1990), turned faces (more than 45° by buffalo_l's
pose) and small faces (shorter side under 48 px in the display frame).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "ems", "modelworker"))

# What module-archive's people recognition uses with buffalo_l (face-quality.ts, face-assign.ts).
GATE_CONFIDENCE = 0.7
GATE_MIN_SIZE = 24
VOTE_THRESHOLD = 0.5
ATTACH_THRESHOLD = 0.55
MARGIN = 1.2
K = 24
FACING = {"yaw": 25, "pitch": 20, "roll": 20}  # people/cover.ts
DET_SIZE = 640


# ---- detectors: (img) -> list of {"box": [x0, y0, x1, y1], "score": float, "kps": [[x, y]] * 5}


def _providers():
    import onnxruntime

    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else ["CPUExecutionProvider"]


class Scrfd:
    """SCRFD det_10g as buffalo_l's FaceAnalysis runs it: 640x640, threshold 0.5."""

    def __init__(self, path):
        from insightface.model_zoo import get_model

        self.model = get_model(path, providers=_providers())
        self.model.prepare(ctx_id=0, input_size=(DET_SIZE, DET_SIZE), det_thresh=0.5)

    def __call__(self, img):
        boxes, kpss = self.model.detect(img, max_num=0, metric="default")
        return [{"box": [float(v) for v in b[:4]], "score": float(b[4]), "kps": k.tolist()} for b, k in zip(boxes, kpss)]


class YuNet:
    """OpenCV Zoo's YuNet on the picture scaled so its longer side is `size`."""

    def __init__(self, path, size=DET_SIZE):
        import cv2

        self.cv2, self.size = cv2, size
        self.model = cv2.FaceDetectorYN.create(path, "", (320, 320), score_threshold=0.3, nms_threshold=0.3, top_k=5000)

    def __call__(self, img):
        cv2 = self.cv2
        h, w = img.shape[:2]
        scale = min(1.0, self.size / max(h, w))
        small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
        self.model.setInputSize((small.shape[1], small.shape[0]))
        _, rows = self.model.detect(small)
        out = []
        for r in rows if rows is not None else []:
            x, y, bw, bh = (float(v) / scale for v in r[:4])
            # YuNet's order is the subject's right eye, left eye, nose, right and left mouth corner:
            # in the picture, left to right, the same order as SCRFD's.
            kps = (np.asarray(r[4:14], dtype=np.float64).reshape(5, 2) / scale).tolist()
            out.append({"box": [x, y, x + bw, y + bh], "score": float(r[14]), "kps": kps})
        return out


# ---- recognisers: (img, kps) -> unit vector


class ArcFace:
    """An ArcFace-style ONNX recogniser on InsightFace's 112x112 alignment (w600k_r50, AuraFace)."""

    def __init__(self, path):
        from insightface.model_zoo import get_model

        self.model = get_model(path, providers=_providers())
        self.model.prepare(ctx_id=0)

    def __call__(self, img, kps):
        from insightface.utils import face_align

        crop = face_align.norm_crop(img, landmark=np.asarray(kps, dtype=np.float32), image_size=self.model.input_size[0])
        v = self.model.get_feat(crop).flatten()
        return v / (np.linalg.norm(v) or 1.0)


class SFace:
    """OpenCV Zoo's SFace on its own alignment (FaceRecognizerSF.alignCrop)."""

    def __init__(self, path):
        import cv2

        self.model = cv2.FaceRecognizerSF.create(path, "")

    def __call__(self, img, kps, box=None):
        k = np.asarray(kps, dtype=np.float32).reshape(-1)
        x0, y0, x1, y1 = box
        row = np.concatenate([[x0, y0, x1 - x0, y1 - y0], k, [1.0]]).astype(np.float32)[None, :]
        crop = self.model.alignCrop(img, row)
        v = self.model.feature(crop).flatten()
        return v / (np.linalg.norm(v) or 1.0)


class FaceNet:
    """facenet-pytorch's InceptionResnetV1 trained on VGGFace2: the box squared and cut to 160x160, as
    its MTCNN cuts a face, standardised to (x - 127.5) / 128. No landmark alignment."""

    def __init__(self, path):
        import torch
        from facenet_pytorch import InceptionResnetV1

        os.environ.setdefault("TORCH_HOME", os.path.dirname(os.path.dirname(path)))
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def __call__(self, img, kps, box):
        import cv2

        crop = cv2.resize(_square_crop(img, box, 1.0), (160, 160), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        x = self.torch.from_numpy((crop.astype(np.float32) - 127.5) / 128.0).permute(2, 0, 1)[None].to(self.device)
        with self.torch.inference_mode():
            v = self.model(x).cpu().numpy().flatten()
        return v / (np.linalg.norm(v) or 1.0)


class Dlib:
    """dlib's ResNet recogniser on its own chip: its CC0 five-point predictor in the detector's box,
    then get_face_chip at 150 px."""

    def __init__(self, folder):
        import dlib

        self.dlib = dlib
        self.sp = dlib.shape_predictor(os.path.join(folder, "shape_predictor_5_face_landmarks.dat"))
        self.rec = dlib.face_recognition_model_v1(os.path.join(folder, "dlib_face_recognition_resnet_model_v1.dat"))
        self._rgb = (None, None)

    def __call__(self, img, kps, box):
        if self._rgb[0] is not img:
            self._rgb = (img, np.ascontiguousarray(img[:, :, ::-1]))
        rgb = self._rgb[1]
        rect = self.dlib.rectangle(*(int(round(v)) for v in box))
        chip = self.dlib.get_face_chip(rgb, self.sp(rgb, rect), size=150, padding=0.25)
        # Raw: dlib's vectors are compared by Euclidean distance (0.6 is its same-person bar), and
        # normalising them would throw away the length that distance reads.
        return np.asarray(self.rec.compute_face_descriptor(chip), dtype=np.float32)


class IntelReid:
    """Open Model Zoo's face-reidentification-retail-0095 on OpenVINO: 128x128 BGR, aligned to the
    demo's reference landmarks."""

    REFERENCE = np.array([[0.31556875, 0.4615741], [0.68262291, 0.4615741], [0.50026, 0.6405054],
                          [0.34947187, 0.8246919], [0.65343645, 0.8246919]], dtype=np.float32) * 128

    def __init__(self, path):
        import openvino as ov

        self.model = ov.Core().compile_model(path, "CPU")

    def __call__(self, img, kps):
        import cv2

        matrix, _ = cv2.estimateAffinePartial2D(np.asarray(kps, dtype=np.float32), self.REFERENCE, method=cv2.LMEDS)
        crop = cv2.warpAffine(img, matrix, (128, 128), borderValue=0.0)
        v = np.asarray(self.model(crop.transpose(2, 0, 1)[None].astype(np.float32))[0]).flatten()
        return v / (np.linalg.norm(v) or 1.0)


def _square_crop(img, box, scale):
    """The square around `box`'s centre, its side `scale` times the box's longer side, padded with
    black where it leaves the picture."""
    import cv2

    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * scale
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))
    n = max(1, int(round(side)))
    h, w = img.shape[:2]
    pad = max(0, -a, -b, a + n - w, b + n - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        a, b = a + pad, b + pad
    return img[b:b + n, a:a + n]


class MediaPipe:
    """MediaPipe: its full-range BlazeFace as a detector, and its Face Landmarker (Face Mesh V2) on a
    crop around any detector's box for five alignment points and the head pose. Both are trained on
    Google's own captures, not WIDER FACE."""

    # Face Mesh indices: the subject's right eye's corners, the left eye's, the nose tip, and the
    # mouth's right and left corners — in the picture left to right, as SCRFD orders its points.
    EYES = ((33, 133), (362, 263))
    NOSE, MOUTH = 1, (61, 291)

    def __init__(self, folder):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        self.det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(folder, "blaze_face_full_range.tflite")), min_detection_confidence=0.3))
        self.mesh = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(folder, "face_landmarker.task")), num_faces=1,
            output_facial_transformation_matrixes=True, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))

    def _image(self, bgr):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(bgr[:, :, ::-1]))

    def __call__(self, img):
        out = []
        for d in self.det.detect(self._image(img)).detections:
            bb = d.bounding_box
            box = [float(bb.origin_x), float(bb.origin_y), float(bb.origin_x + bb.width), float(bb.origin_y + bb.height)]
            found = self.landmarks(img, box)
            if found is None:
                continue
            out.append({"box": box, "score": float(d.categories[0].score), "kps": found[0], "pose_mp": found[1]})
        return out

    def landmarks(self, img, box):
        """Five points in the picture and [pitch, yaw, roll], or None when no face is found in the crop."""
        import cv2

        x0, y0, x1, y1 = box
        side = max(x1 - x0, y1 - y0) * 1.8
        crop = _square_crop(img, box, 1.8)
        if crop.size == 0:
            return None
        size = 256
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_LINEAR)
        res = self.mesh.detect(self._image(crop))
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        ox, oy = (x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2
        at = lambda i: (ox + lm[i].x * side, oy + lm[i].y * side)
        mean = lambda a, b: ((at(a)[0] + at(b)[0]) / 2, (at(a)[1] + at(b)[1]) / 2)
        kps = [list(mean(*self.EYES[0])), list(mean(*self.EYES[1])), list(at(self.NOSE)), list(at(self.MOUTH[0])), list(at(self.MOUTH[1]))]
        r = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
        sy = math.hypot(r[0, 0], r[1, 0])
        pose = [math.degrees(math.atan2(r[2, 1], r[2, 2])), math.degrees(math.atan2(-r[2, 0], sy)), math.degrees(math.atan2(r[1, 0], r[0, 0]))]
        return kps, pose


class Pose3d:
    """buffalo_l's 1k3d68: the reference head pose."""

    def __init__(self, path):
        from insightface.model_zoo import get_model

        self.model = get_model(path, providers=_providers())
        self.model.prepare(ctx_id=0)

    def __call__(self, img, box):
        from insightface.app.common import Face

        face = Face(bbox=np.asarray(box, dtype=np.float32))
        self.model.get(img, face)
        return [float(v) for v in face.pose]  # pitch, yaw, roll


def landmark_yaw(kps) -> float | None:
    """face-quality.ts's landmarkYaw: the nose's offset from the midline, in half eye-distances."""
    le, re, nose, lm, rm = (np.asarray(p, dtype=np.float64) for p in kps)
    ex, ey = re - le
    d = math.hypot(ex, ey)
    if not d > 0:
        return None
    c, s = ex / d, ey / d
    along = lambda p: (p[0] - le[0]) * c + (p[1] - le[1]) * s
    mid = (along(le) + along(re) + along(lm) + along(rm)) / 4
    return math.degrees(math.atan((along(nose) - mid) / (d / 2)))


def landmark_pose(kps) -> list[float] | None:
    """pitch, yaw, roll from five points: roll from the eye line, yaw as landmark_yaw, and pitch from
    where the nose sits between the eyes and the mouth (about 0.55 of the way down when level)."""
    le, re, nose, lm, rm = (np.asarray(p, dtype=np.float64) for p in kps)
    ex, ey = re - le
    d = math.hypot(ex, ey)
    if not d > 0:
        return None
    roll = math.degrees(math.atan2(ey, ex))
    c, s = ex / d, ey / d
    down = lambda p: -(p[0] - le[0]) * s + (p[1] - le[1]) * c
    eyes, mouth = (down(le) + down(re)) / 2, (down(lm) + down(rm)) / 2
    span = mouth - eyes
    if not span > 0:
        return None
    pitch = math.degrees(math.atan(((down(nose) - eyes) / span - 0.55) / 0.35))
    return [pitch, landmark_yaw(kps), roll]


# ---- components by name


def components(models: str, names: list[str]):
    det_dir = os.path.join(models, "buffalo_l")
    table = {
        "scrfd": lambda: Scrfd(os.path.join(det_dir, "det_10g.onnx")),
        "yunet": lambda: YuNet(os.path.join(models, "face_detection_yunet_2023mar.onnx")),
        "yunet1280": lambda: YuNet(os.path.join(models, "face_detection_yunet_2023mar.onnx"), 1280),
        "w600k": lambda: ArcFace(os.path.join(det_dir, "w600k_r50.onnx")),
        "auraface": lambda: ArcFace(os.path.join(models, "auraface", "glintr100.onnx")),
        "sface": lambda: SFace(os.path.join(models, "face_recognition_sface_2021dec.onnx")),
        "facenet": lambda: FaceNet(os.path.join(models, "torch", "checkpoints", "20180402-114759-vggface2.pt")),
        "dlib": lambda: Dlib(os.path.join(models, "dlib")),
        "intel": lambda: IntelReid(os.path.join(models, "intel", "face-reidentification-retail-0095.xml")),
        "mediapipe": lambda: MediaPipe(models),
    }
    return {name: table[name]() for name in names}


# ---- run


def frac_to_px(box, w, h):
    return [box["x"] * w, box["y"] * h, (box["x"] + box["width"]) * w, (box["y"] + box["height"]) * h]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(p, box):
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def match(gt_boxes, dets):
    """Each named face's detection, or -1: highest overlap first, one detection per face. A
    hand-drawn box is often looser or tighter than a detector's, so a pair whose centres each fall
    inside the other's box counts as well as an IoU of 0.3."""
    pairs = []
    for gi, g in enumerate(gt_boxes):
        if g is None:
            continue
        gc = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
        for di, d in enumerate(dets):
            b = d["box"]
            o = iou(g, b)
            if o >= 0.3 or (inside(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), g) and inside(gc, b)):
                pairs.append((o, gi, di))
    out, used = [-1] * len(gt_boxes), set()
    for o, gi, di in sorted(pairs, reverse=True):
        if out[gi] < 0 and di not in used:
            out[gi], used = di, used | {di}
    if len(gt_boxes) == 1 and gt_boxes[0] is None and dets:
        out[0] = max(range(len(dets)), key=lambda i: (dets[i]["box"][2] - dets[i]["box"][0]) * (dets[i]["box"][3] - dets[i]["box"][1]))
    return out


def run(args):
    import image_io

    detectors = components(args.models, args.detectors.split(","))
    recognisers = components(args.models, args.recognisers.split(","))
    pose3d = None if args.no_pose else Pose3d(os.path.join(args.models, "buffalo_l", "1k3d68.onnx"))
    mesh = None if args.no_pose else detectors.get("mediapipe") or MediaPipe(args.models)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "results.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["file"] for line in f if line.strip()}
    rows = [json.loads(line) for line in open(args.manifest) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    timing: dict[str, float] = {}
    started = time.time()
    with open(out_path, "a") as out:
        for n, row in enumerate(rows):
            if row["file"] in done:
                continue
            try:
                img = image_io.open_display_bgr(row["path"], row.get("angle", 0), row.get("mirror", False))
            except Exception as exc:  # an unreadable original is skipped, not fatal
                out.write(json.dumps({"file": row["file"], "error": str(exc)[:200]}) + "\n")
                continue
            h, w = img.shape[:2]
            gt_boxes = [frac_to_px(g["box"], w, h) if g.get("box") else None for g in row["gt"]]
            result = {"file": row["file"], "frame": [w, h], "dets": {}, "match": {}, "emb": {}}
            for name, det in detectors.items():
                t = time.time()
                dets = det(img)
                timing[name] = timing.get(name, 0.0) + time.time() - t
                for d in dets:
                    d["yaw5"] = landmark_yaw(d["kps"])
                    d["pose5"] = landmark_pose(d["kps"])
                m = match(gt_boxes, dets)
                # The reference pose for every matched face, from 1k3d68 on this detector's box.
                # Face Mesh's pose too, the candidate that could replace it.
                for di in m if pose3d else []:
                    if di >= 0 and "pose" not in dets[di]:
                        dets[di]["pose"] = pose3d(img, dets[di]["box"])
                    if di >= 0 and "pose_mp" not in dets[di] and name in ("scrfd", "yunet"):
                        found = mesh.landmarks(img, dets[di]["box"])
                        dets[di]["pose_mp"] = found[1] if found else None
                result["dets"][name] = [{k: (np.round(v, 2).tolist() if k in ("box", "kps") else v) for k, v in d.items()} for d in dets]
                result["match"][name] = m
                for rname, rec in recognisers.items():
                    t = time.time()
                    vecs = []
                    for di in m:
                        if di < 0:
                            vecs.append(None)
                            continue
                        d = dets[di]
                        v = rec(img, d["kps"], d["box"]) if isinstance(rec, (SFace, FaceNet, Dlib)) else rec(img, d["kps"])
                        vecs.append(np.round(v, 5).tolist())
                    timing[f"{name}+{rname}"] = timing.get(f"{name}+{rname}", 0.0) + time.time() - t
                    result["emb"][f"{name}+{rname}"] = vecs
            out.write(json.dumps(result) + "\n")
            out.flush()
            if n % 100 == 0:
                rate = (n + 1) / (time.time() - started)
                print(f"{n + 1}/{len(rows)} {rate:.1f}/s " + " ".join(f"{k}={v:.0f}s" for k, v in timing.items()), flush=True)
    with open(os.path.join(args.out, "timing.json"), "w") as f:
        json.dump(timing, f)


# ---- score


def load(args):
    manifest = {r["file"]: r for r in (json.loads(line) for line in open(args.manifest) if line.strip())}
    results = [r for r in (json.loads(line) for line in open(os.path.join(args.out, "results.jsonl")) if line.strip()) if "error" not in r]
    faces = []  # one per named face
    for r in results:
        row = manifest[r["file"]]
        w, h = r["frame"]
        for gi, g in enumerate(row["gt"]):
            box = frac_to_px(g["box"], w, h) if g.get("box") else None
            ref = r["match"].get("scrfd", [-1] * len(row["gt"]))[gi]
            ref_det = r["dets"]["scrfd"][ref] if ref >= 0 and "scrfd" in r["dets"] else None
            faces.append({
                "file": r["file"], "gi": gi, "person": g["person"], "kind": g.get("kind"),
                "old": (row.get("taken") or "9999") < "1990",
                "small": box is not None and min(box[2] - box[0], box[3] - box[1]) < 48,
                "group": len(row["gt"]) >= 4 or (row.get("regions") or 0) >= 4,
                "yaw": abs(ref_det["pose"][1]) if ref_det and ref_det.get("pose") else (abs(g["yaw"]) if g.get("yaw") is not None else None),
                "r": r,
            })
    return results, faces


def gate(det, frame):
    return min(det["box"][2] - det["box"][0], det["box"][3] - det["box"][1]) >= GATE_MIN_SIZE


def score_detection(results, faces, detectors):
    out = {}
    base_per_image = np.mean([sum(1 for d in r["dets"]["scrfd"] if d["score"] >= GATE_CONFIDENCE and gate(d, r["frame"])) for r in results])
    named = [f for f in faces if f["kind"] in ("legacy", "transfer")]
    for name in detectors:
        scores = np.array(sorted((d["score"] for r in results for d in r["dets"][name] if gate(d, r["frame"])), reverse=True))
        # The confidence at which this detector keeps as many faces per picture as SCRFD at the gate.
        n = int(round(base_per_image * len(results)))
        matched_thr = float(scores[min(n, len(scores) - 1)]) if len(scores) else 1.0
        row = {"per_image_at_gate": float(np.mean([sum(1 for d in r["dets"][name] if d["score"] >= GATE_CONFIDENCE and gate(d, r["frame"])) for r in results])),
               "matched_threshold": matched_thr}
        for label, thr in (("gate", GATE_CONFIDENCE), ("matched", matched_thr)):
            for slice_name, subset in (("all", named), ("found", [f for f in named if f["kind"] == "transfer"]),
                                       ("missed", [f for f in named if f["kind"] == "legacy"]), ("old", [f for f in named if f["old"]]),
                                       ("small", [f for f in named if f["small"]]), ("group", [f for f in named if f["group"]]),
                                       ("turned", [f for f in named if f["yaw"] is not None and f["yaw"] > 45])):
                hit = 0
                for f in subset:
                    di = f["r"]["match"][name][f["gi"]]
                    if di >= 0:
                        d = f["r"]["dets"][name][di]
                        hit += d["score"] >= thr and gate(d, f["r"]["frame"])
                row[f"recall_{label}_{slice_name}"] = (hit / len(subset), len(subset)) if subset else None
        out[name] = row
    return out


def similarities(vecs, combo):
    """Cosine similarity, or for dlib 1 - Euclidean distance, so that higher is more alike for both."""
    if combo.endswith("+dlib"):
        sq = (vecs * vecs).sum(axis=1)
        return 1.0 - np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (vecs @ vecs.T), 0.0))
    return vecs @ vecs.T


def pair_stats(vecs, persons, files, combo):
    """Similarities of every pair of faces from different pictures, split into same and different person."""
    sims = similarities(vecs, combo)
    iu = np.triu_indices(len(vecs), 1)
    s = sims[iu]
    same = persons[iu[0]] == persons[iu[1]]
    other_file = files[iu[0]] != files[iu[1]]
    return s[same & other_file], s[~same & other_file], sims


def tar_at_far(genuine, impostor, far):
    thr = np.quantile(impostor, 1 - far) if len(impostor) else 1.0
    return float(np.mean(genuine >= thr)), float(thr)


def assign(sims, persons, files, vote, attach, probe_mask):
    """face-assign's vote, every face a probe in turn against the others named in other pictures."""
    right = wrong = none = 0
    for i in np.flatnonzero(probe_mask):
        row = sims[i].copy()
        row[files == files[i]] = -1
        top = np.argpartition(-row, min(K, len(row) - 1))[:K]
        tally: dict = {}
        for j in top:
            if row[j] >= vote:
                tally[persons[j]] = tally.get(persons[j], 0.0) + row[j]
        if not tally:
            none += 1
            continue
        ranked = sorted(tally.items(), key=lambda kv: -kv[1])
        best, weight = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if weight < attach or (second > 0 and weight < second * MARGIN):
            none += 1
        elif best == persons[i]:
            right += 1
        else:
            wrong += 1
    n = max(1, right + wrong + none)
    return {"right": right / n, "wrong": wrong / n, "none": none / n, "n": right + wrong + none}


def score_recognition(faces, combos, common_only):
    out = {}
    # The faces every combination embedded, so each is scored on the same faces.
    keep = [f for f in faces if all(f["r"]["emb"].get(c, [None] * (f["gi"] + 1))[f["gi"]] is not None for c in combos)] if common_only else None
    base_far = {}
    # buffalo_l first: its pairs passing the archive's thresholds set the others' thresholds.
    for combo in sorted(combos, key=lambda c: c != "scrfd+w600k"):
        subset = keep if keep is not None else [f for f in faces if f["r"]["emb"].get(combo) and f["r"]["emb"][combo][f["gi"]] is not None]
        vecs = np.asarray([f["r"]["emb"][combo][f["gi"]] for f in subset], dtype=np.float32)
        persons = np.asarray([f["person"] for f in subset])
        files = np.asarray([f["file"] for f in subset])
        genuine, impostor, sims = pair_stats(vecs, persons, files, combo)
        row = {"faces": len(subset), "persons": len(set(persons))}
        for far in (1e-3, 1e-4, 1e-5):
            row[f"tar@{far:g}"], row[f"thr@{far:g}"] = tar_at_far(genuine, impostor, far)
        if combo == "scrfd+w600k":
            base_far = {"vote": float(np.mean(impostor >= VOTE_THRESHOLD)), "attach": float(np.mean(impostor >= ATTACH_THRESHOLD))}
        far_vote = base_far.get("vote", float(np.mean(impostor >= VOTE_THRESHOLD)))
        far_attach = base_far.get("attach", float(np.mean(impostor >= ATTACH_THRESHOLD)))
        vote = VOTE_THRESHOLD if combo == "scrfd+w600k" else float(np.quantile(impostor, 1 - far_vote))
        attach = ATTACH_THRESHOLD if combo == "scrfd+w600k" else float(np.quantile(impostor, 1 - far_attach))
        row["vote_threshold"], row["attach_threshold"], row["far_at_vote"] = vote, attach, float(np.mean(impostor >= vote))
        for slice_name, mask in (("all", np.ones(len(subset), bool)), ("old", np.asarray([f["old"] for f in subset])),
                                 ("turned", np.asarray([f["yaw"] is not None and f["yaw"] > 45 for f in subset])),
                                 ("small", np.asarray([f["small"] for f in subset]))):
            row[f"assign_{slice_name}"] = assign(sims, persons, files, vote, attach, mask) if mask.any() else None
        # Old faces against everyone, as pairs: does the model hold a person across decades?
        old = np.asarray([f["old"] for f in subset])
        if old.any():
            iu = np.triu_indices(len(subset), 1)
            either_old = old[iu[0]] | old[iu[1]]
            same = persons[iu[0]] == persons[iu[1]]
            other_file = files[iu[0]] != files[iu[1]]
            s = sims[iu]
            g_old, i_old = s[same & other_file & either_old], s[~same & other_file & either_old]
            row["tar_old@1e-4"] = float(np.mean(g_old >= row["thr@0.0001"])) if len(g_old) else None
        out[combo] = row
    return out


def score_pose(results, detectors):
    """Each candidate pose against 1k3d68's on the same box. The error is between magnitudes, since
    the sign conventions differ and covers ask only how far a face is turned."""
    out = {}
    facing = lambda p: abs(p[0]) <= FACING["pitch"] and abs(p[1]) <= FACING["yaw"] and abs(p[2]) <= FACING["roll"]
    for name in detectors:
        for source in ("pose5", "pose_mp"):
            agree = n = missing = 0
            err = []
            for r in results:
                for d in r["dets"][name]:
                    if not d.get("pose"):
                        continue
                    if not d.get(source):
                        missing += 1
                        continue
                    n += 1
                    agree += facing(d["pose"]) == facing(d[source])
                    err.append([abs(abs(a) - abs(b)) for a, b in zip(d["pose"], d[source])])
            e = np.asarray(err) if err else np.zeros((0, 3))
            out[f"{name}:{source}"] = {"n": n, "missing": missing, "facing_agree": agree / max(1, n),
                                       "mae_pitch_yaw_roll": e.mean(axis=0).round(1).tolist() if n else None}
    return out


def score(args):
    results, faces = load(args)
    detectors = list(results[0]["dets"].keys())
    combos = list(results[0]["emb"].keys())
    report = {
        "pictures": len(results), "faces": len(faces), "persons": len({f["person"] for f in faces}),
        "detection": score_detection(results, faces, detectors),
        # Recognisers on SCRFD's own detections and landmarks: the recogniser alone.
        "recognition_on_scrfd": score_recognition(faces, [c for c in combos if c.startswith("scrfd+")], common_only=False),
        # Every combination on the faces all of them found.
        "recognition_common": score_recognition(faces, combos, common_only=True),
        "pose": score_pose(results, detectors),
    }
    timing_path = os.path.join(args.out, "timing.json")
    if os.path.exists(timing_path):
        report["seconds_per_picture"] = {k: v / len(results) for k, v in json.load(open(timing_path)).items()}
    print(json.dumps(report, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "score"])
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models", default="/opt/insightface/models/..", help="directory with buffalo_l/ and the candidates' weights")
    p.add_argument("--detectors", default="scrfd,yunet")
    p.add_argument("--recognisers", default="w600k,auraface,sface")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-pose", action="store_true", help="skip the head pose models, for a run that adds recognisers")
    args = p.parse_args()
    run(args) if args.command == "run" else score(args)


if __name__ == "__main__":
    main()
