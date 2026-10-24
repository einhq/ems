"""Text in pictures: lines of text, each with where it stands and how sure the reader is.

The photo engine is PP-OCRv6 (det + rec, one multilingual model that reads Swedish å/ä/ö without
a language switch) run through RapidOCR on onnxruntime, with RapidOCR's 0°/180° line classifier
between the two so upside-down text reads. It finds scene text — a sign, a shop front, a caption
under a print — which Tesseract, made for pages, misses or answers with noise.

ONE PRECISION ON EVERY HOST. float32 on the card and on the CPU alike, as for the other ONNX
families (image_embed.py), so a CPU host reads the same lines.

HANDWRITING IS A SECOND READING OF A LINE, NOT A ROUTE FOR A PAGE. A line PP-OCRv6 reads below the
floor is cut out again and read by the National Archives of Finland's recogniser
(`multicentury-ppocrv6-medium`, PP-OCRv6_medium_rec fine-tuned on 1.4 M mostly handwritten lines
of 16th-20th century Swedish and Finnish), and kept if that reading reaches the floor. On the
archive's photo-album cards and letters (ein/core#352) it read handwriting and worn print
where PP-OCRv6 read noise, and stayed below the floor on ornaments and patterns; read whole, it
mixes the case of printed lines, so it never replaces a line PP-OCRv6 reads well. A line it keeps
is sent on with its crop, for a transcriber (htr.py) to write its text: this recogniser is the
judge of whether a line is writing, the transcriber the better reader of it. PP-OCRv6 is sometimes
sure and wrong on handwriting (a letter's date line "Ystad den 3-11-41" read at 0.8 or more as
"arbug lou 31-12-10"), so on a picture that is writing — most of its kept lines, and at least
`MIN_WRITING_LINES`, came through the recogniser — every other kept line goes to the transcriber as
well. A photo card's few handwritten names and dates stay under that count, so the printed imprint
beside them keeps PP-OCRv6's reading.

WHICH TRANSCRIBER, IF ANY, DEPENDS ON THE LANGUAGE (`compose`, ein/core#670). The
recogniser and the transcriber above know Swedish and Finnish. English writing scores under the
recogniser's floor, so on a picture in a language that has a transcriber of its own the photo
engine's own doubt is the judge instead (`is_writing_by_certainty`), and on such a page every line
it reads at `WRITING_LINE_SCORE` or more goes to that transcriber. In any other language nothing is
transcribed: a line the recogniser kept keeps its reading.

The models are files baked into the image (`root`), never downloaded at run time: RapidOCR
fetches a model it is not given a path for, so every one is passed by path, the line classifier
from RapidOCR's own wheel.
"""

from __future__ import annotations

import os

# A line is kept when the recogniser's mean character confidence reaches this and it holds two
# letters or digits. Set on 60 archive pictures (ein/core#352): below 0.8 a shirt's pattern,
# a window's lattice and graffiti read as words ("DOSE | BODE"), and a sign's own lines score 0.83
# and up; a lone glyph is no word to search for.
MIN_LINE_SCORE = float(os.environ.get("CFG_OCR_MIN_LINE_SCORE", "0.8"))
MIN_LINE_CHARS = 2

# A picture is writing when at least this many of its kept lines came through the handwriting
# recogniser and they are more than half of them. The archive's letters keep 10-24 lines a page,
# nearly all through it; its photo cards hold a name or a date or two.
MIN_WRITING_LINES = 4

# In a language the recogniser does not know, a picture is writing when fewer than half of at least
# MIN_CERTAIN_LINES detected lines read at CERTAIN_SCORE or above. Print reads at it: on every
# printed picture of 60 archive photos, and on a typed letter, 62 % of the lines and more did; on 26
# handwritten pages, English and Swedish, half and less (ein/core#670).
CERTAIN_SCORE = 0.95
MIN_CERTAIN_LINES = 3
# On such a page a line goes to the transcriber when PP-OCRv6 reads it at this or above, kept by the
# floor or not: on 16 English handwritten pages and a typed one, the English transcriber then wrote
# 0.53 of their words, against 0.46 on the lines the floor keeps.
WRITING_LINE_SCORE = 0.6


def _cuda_options(mem_limit_bytes: int) -> dict:
    """The CUDA provider's options, bounded like every ONNX family's (see image_embed.providers).

    HEURISTIC, not RapidOCR's EXHAUSTIVE: the detector's input is each picture's own size, and an
    exhaustive search benchmarks every convolution again for every new shape.
    """
    return {
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": mem_limit_bytes,
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": True,
    }


def _shrink_arenas_after_each_run(stages) -> None:
    """Give each session's CUDA arena back down to its first chunk after every run.

    Every picture is a new input shape, and the arena keeps what it grew to: on fry a run of 20
    album scans took the detector past its 2 GB cap on the 17th, a picture that needs 1.3 GB in a
    fresh session. RapidOCR runs its sessions with no run options, so they are added here; `stages`
    are its loaders, which build a session on first use, so they are built now.
    """
    import functools

    import onnxruntime as ort

    options = ort.RunOptions()
    options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    for load in stages:
        session = load().session.session
        session.run = functools.partial(type(session).run, session, run_options=options)


class HandReader:
    """The handwriting recogniser on one cut-out line: a BGR array in, its text and score out.

    `root` holds the model as its authors publish it — `inference.onnx` and `inference.yml`, whose
    `character_dict` the image writes to `keys.txt` with the space its README says to add."""

    def __init__(self, root: str, device: str, mem_limit_bytes: int, threads: int):
        import yaml
        from rapidocr import RapidOCR

        with open(os.path.join(root, "inference.yml")) as handle:
            config = yaml.safe_load(handle)
        shape = next(op["RecResizeImg"]["image_shape"] for op in config["PreProcess"]["transform_ops"] if "RecResizeImg" in op)
        self.model = os.path.basename(os.path.normpath(root))
        self.engine = RapidOCR(params={
            "Global.log_level": "warning",
            "Global.text_score": 0.0,
            "Global.use_det": False,
            "Global.use_cls": False,
            "Rec.model_path": os.path.join(root, "inference.onnx"),
            "Rec.rec_keys_path": os.path.join(root, "keys.txt"),
            "Rec.rec_img_shape": [3, shape[1], 320],
            "EngineConfig.onnxruntime.use_cuda": device == "cuda",
            "EngineConfig.onnxruntime.cuda_ep_cfg": _cuda_options(mem_limit_bytes),
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
        })
        if device == "cuda":
            _shrink_arenas_after_each_run([self.engine._load_rec_model])

    def read(self, line) -> tuple[str, float]:
        out = self.engine(line)
        texts = getattr(out, "txts", None) or ()
        scores = getattr(out, "scores", None) or ()
        return (str(texts[0]).strip(), float(scores[0])) if texts else ("", 0.0)


class PhotoReader:
    """PP-OCRv6 at one tier (`small` or `medium`), with the handwriting recogniser in `hand_root`
    for its doubtful lines when there is one: PIL images in, lines out."""

    def __init__(self, root: str, tier: str, device: str, mem_limit_bytes: int, max_side: int, hand_root: str | None = None):
        from pathlib import Path

        import rapidocr
        from rapidocr import RapidOCR

        threads = os.environ.get("CFG_MODEL_THREADS", "").strip()
        thread_count = int(threads) if threads.isdigit() else -1
        classifier = Path(rapidocr.__file__).parent / "models" / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
        self.device = device
        self.hand = HandReader(hand_root, device, mem_limit_bytes, thread_count) if hand_root else None
        self.model = f"PP-OCRv6_{tier}" + (f"+{self.hand.model}" if self.hand else "")
        self.engine = RapidOCR(params={
            "Global.log_level": "warning",
            # Every line comes back; the floor is applied in `read`, where it is one setting.
            "Global.text_score": 0.0,
            "Global.max_side_len": max_side,
            # The detector sees the picture with its long side at most `max_side`, never scaled up:
            # its memory grows with the area, and RapidOCR's default (the short side raised to 736)
            # takes a panorama or a square scan past 4 MP, 4 GB of arena on the card.
            "Det.limit_type": "max",
            "Det.limit_side_len": max_side,
            "Global.model_root_dir": root,
            "Det.model_path": os.path.join(root, f"PP-OCRv6_det_{tier}.onnx"),
            "Rec.model_path": os.path.join(root, f"PP-OCRv6_rec_{tier}.onnx"),
            "Cls.model_path": str(classifier),
            "EngineConfig.onnxruntime.use_cuda": device == "cuda",
            "EngineConfig.onnxruntime.cuda_ep_cfg": _cuda_options(mem_limit_bytes),
            "EngineConfig.onnxruntime.intra_op_num_threads": thread_count,
        })
        if device == "cuda":
            _shrink_arenas_after_each_run([self.engine._load_det_model, self.engine._load_cls_model, self.engine._load_rec_model])

    def read(self, image, judge: str | None = "recogniser") -> dict:
        """`compose(self.detect(image), judge)`: the reading of a picture whose language routes it
        to `judge`."""
        return compose(self.detect(image), judge)

    def detect(self, image) -> dict:
        """Every line PP-OCRv6 finds in one picture in its display frame: `{width, height, detected}`.

        Each detected line is `{text, score, box, crop}` as PP-OCRv6 reads it, plus `hand` — the
        handwriting recogniser's (text, score) — when it read the line below the floor. `box` is the
        line's four corners, clockwise from its top left as the text runs, as fractions of the
        picture's width and height; `crop` the line cut out along them, as a BGR array. Lines come
        in reading order, top to bottom.
        """
        image = image.convert("RGB")
        width, height = image.size
        out = self.engine(image)
        boxes = out.boxes if getattr(out, "boxes", None) is not None else []
        texts = getattr(out, "txts", None) or ()
        scores = getattr(out, "scores", None) or ()
        pixels: dict = {}
        detected = []
        for box, text, score in zip(boxes, texts, scores):
            line = {"text": str(text).strip(), "score": float(score), "hand": None}
            if line["score"] < MIN_LINE_SCORE and self.hand is not None:
                hand_text, hand_score, line["crop"] = self._read_hand(image, box, pixels)
                line["hand"] = (hand_text, hand_score)
            else:
                line["crop"] = self._crop(image, box, pixels)
            line["box"] = [[round(float(x) / width, 5), round(float(y) / height, 5)] for x, y in box]
            detected.append(line)
        return {"width": width, "height": height, "detected": detected}

    def _read_hand(self, image, box, pixels: dict):
        """The handwriting recogniser's reading of the line at `box` (text, score, the crop it read),
        cut out of `image` along its corners."""
        crop = self._crop(image, box, pixels)
        return (*self.hand.read(crop), crop)

    def _crop(self, image, box, pixels: dict):
        """The line at `box` cut out of `image` along its corners, as a BGR array. `pixels` keeps
        the picture as one BGR array for the rest of its lines."""
        import numpy as np
        from rapidocr.utils.process_img import get_rotate_crop_image

        if "bgr" not in pixels:
            pixels["bgr"] = np.asarray(image)[:, :, ::-1].copy()
        return get_rotate_crop_image(pixels["bgr"], np.array(box, dtype=np.float32))


def _has_word(text: str) -> bool:
    return sum(c.isalnum() for c in text) >= MIN_LINE_CHARS


def _kept(line: dict) -> tuple[str, float, bool] | None:
    """The (text, score, through the recogniser) a detected line is kept with by the floor, or None."""
    if line["score"] >= MIN_LINE_SCORE:
        return (line["text"], line["score"], False) if _has_word(line["text"]) else None
    if line["hand"] is not None and line["hand"][1] >= MIN_LINE_SCORE and _has_word(line["hand"][0]):
        return (line["hand"][0], line["hand"][1], True)
    return None


def kept_text(reading: dict) -> str:
    """The text of the lines the floor keeps, joined by newlines: what the language is told from."""
    return "\n".join(kept[0] for kept in map(_kept, reading["detected"]) if kept)


def is_writing_by_certainty(detected: list) -> bool:
    """Whether a picture is writing by how sure PP-OCRv6 is of it (see CERTAIN_SCORE)."""
    certain = sum(line["score"] >= CERTAIN_SCORE for line in detected)
    return len(detected) >= MIN_CERTAIN_LINES and certain * 2 < len(detected)


def compose(reading: dict, judge: str | None) -> dict:
    """The answer to `ocr.read` before any transcriber has run, and the lines to send to one.

    `judge` is what decides that a picture is writing: `recogniser` (its transcriber knows the
    language the recogniser does), `certainty` (its transcriber is for another language) or None (no
    transcriber). Answers `{text, lines: [{text, box, score}], width, height, transcribe, tentative}`:
    `transcribe` lists (index in `lines`, crop) for the transcriber to write, never to send;
    `tentative` the indices of lines kept only if the transcriber writes a word there.
    """
    detected = reading["detected"]
    lines: list[dict] = []
    transcribe: list = []
    tentative: list[int] = []
    if judge == "certainty" and is_writing_by_certainty(detected):
        for line in detected:
            kept = _kept(line)
            if kept is None and line["score"] < WRITING_LINE_SCORE:
                continue
            text, score = kept[:2] if kept else (line["text"], line["score"])
            if kept is None:
                tentative.append(len(lines))
            transcribe.append((len(lines), line["crop"]))
            lines.append({"text": text, "box": line["box"], "score": round(float(score), 4)})
    else:
        confident = []
        for line in detected:
            kept = _kept(line)
            if kept is None:
                continue
            text, score, through_hand = kept
            (transcribe if through_hand else confident).append((len(lines), line["crop"]))
            lines.append({"text": text, "box": line["box"], "score": round(float(score), 4)})
        if judge is None:
            transcribe = []
        elif judge == "recogniser" and len(transcribe) >= MIN_WRITING_LINES and len(transcribe) > len(confident):
            transcribe = sorted(transcribe + confident, key=lambda pair: pair[0])
    return {"text": "\n".join(line["text"] for line in lines), "lines": lines, "width": reading["width"],
            "height": reading["height"], "transcribe": transcribe, "tentative": tentative}


def write(reading: dict, texts: list[str]) -> dict:
    """`reading` (from `compose`) with the transcriber's `texts` for its `transcribe` lines, in order:
    a line keeps its reading where the transcriber wrote nothing, and a tentative line without a word
    is dropped. Answers `{text, lines, width, height}`."""
    lines = [dict(line) for line in reading["lines"]]
    for (index, _), text in zip(reading["transcribe"], texts):
        if text:
            lines[index]["text"] = text
    dropped = {index for index in reading["tentative"] if not _has_word(lines[index]["text"])}
    lines = [line for index, line in enumerate(lines) if index not in dropped]
    return {"text": "\n".join(line["text"] for line in lines), "lines": lines, "width": reading["width"], "height": reading["height"]}
