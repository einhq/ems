#!/usr/bin/env python3
"""ocr-spotcheck — OCR engines on the same pictures, side by side.

The OCR engines are chosen from published results; before one replaces another it reads ~20
archive pictures of its kind beside it, and an engine visibly worse on our material is swapped
for the runner-up (ein/core#352, wow plans/ocr-tiers.md). This prints each picture's
readings as it goes and writes `report.html` with a preview of each picture beside them, every
line with its score where the engine gives one, so a floor can be picked by reading which lines
are noise.

    make ocr-spotcheck OCR_SPOTCHECK_ARGS="--engines medium,vl-spotting"

`--list` is tab-separated: path (as EMS sees it), angle, mirror (0/1), and a label the report
shows. The engines:

- `medium`, `small`: PP-OCRv6 at that tier, every line whatever its score;
- `ocr-read`: what `ocr.read` answers — PP-OCRv6 medium at its floor, its doubtful lines read again
  by the handwriting recogniser;
- `vl-ocr`, `vl-spotting`: PaddleOCR-VL-1.6 through transformers, its `OCR:` and `Spotting:`
  tasks over the whole picture;
- `hand-fi`: PP-OCRv6 medium's detector with the National Archives of Finland's handwriting
  recogniser (`Kansallisarkisto/multicentury-ppocrv6-medium`);
- `hand-trocr`: the Swedish National Archives' TrOCR (`trocr-base-handwritten-hist-swe-2`) over
  the lines PP-OCRv6 medium's detector finds.

Weights an engine fetches go to `HF_HUB_CACHE` on first use, outside the image.

Seconds are per picture, after a warm-up.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ems", "modelworker"))

PREVIEW_EDGE = 900
VL_MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
# The model card's pixel budgets: spotting reads at up to 1.6 MP, the other tasks at 1 MP.
VL_MAX_PIXELS = {"ocr": 1280 * 28 * 28, "spotting": 2048 * 28 * 28}
VL_PROMPTS = {"ocr": "OCR:", "spotting": "Spotting:"}
# A dense page is a few thousand tokens; the card's 512 cuts it off.
VL_MAX_NEW_TOKENS = 4096
LOCATION = re.compile(r"<\|LOC_\d+\|>")


class VisionLanguage:
    """PaddleOCR-VL-1.6 on one task, as its model card runs it through transformers."""

    def __init__(self, task: str, device: str, shared: dict):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if "model" not in shared:
            shared["model"] = AutoModelForImageTextToText.from_pretrained(VL_MODEL, dtype=torch.bfloat16).to(device).eval()
            shared["processor"] = AutoProcessor.from_pretrained(VL_MODEL)
        self.model, self.processor, self.task, self.torch = shared["model"], shared["processor"], task, torch

    def read(self, image) -> dict:
        from PIL import Image

        image = image.convert("RGB")
        width, height = image.size
        # The model card doubles a small picture before spotting.
        if self.task == "spotting" and width < 1500 and height < 1500:
            image = image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": VL_PROMPTS[self.task]}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": self.processor.image_processor.size.shortest_edge, "longest_edge": VL_MAX_PIXELS[self.task]}},
        ).to(self.model.device)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=VL_MAX_NEW_TOKENS)
        raw = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
        return {"text": LOCATION.sub(" ", raw).strip(), "raw": raw[:4000]}


FINNISH_REC = "Kansallisarkisto/multicentury-ppocrv6-medium"
FINNISH_REC_REVISION = "01a6a5413959c8cff053e2aa60485de14b507f82"
TROCR = "Riksarkivet/trocr-base-handwritten-hist-swe-2"
TROCR_REVISION = "aa79fcb1850bf3155ebc442570d6c6bfc0ac8100"


class FinnishHands:
    """PP-OCRv6 medium's detector with the National Archives of Finland's recogniser, a
    PP-OCRv6_medium_rec fine-tuned on 1.4 M lines of 16th-20th century Swedish and Finnish, mostly
    handwritten. Its dictionary is the `character_dict` of its inference.yml, plus a space."""

    def __init__(self, root: str, device: str, max_side: int, out: str):
        import rapidocr
        import yaml
        from huggingface_hub import hf_hub_download
        from rapidocr import RapidOCR

        import text_reader

        onnx = hf_hub_download(FINNISH_REC, "inference.onnx", revision=FINNISH_REC_REVISION)
        config = yaml.safe_load(open(hf_hub_download(FINNISH_REC, "inference.yml", revision=FINNISH_REC_REVISION)))
        keys = os.path.join(out, "finnish-keys.txt")
        with open(keys, "w") as handle:
            handle.write("\n".join(config["PostProcess"]["character_dict"]) + "\n ")
        height = config["PreProcess"]["transform_ops"][2]["RecResizeImg"]["image_shape"][1]
        classifier = os.path.join(os.path.dirname(rapidocr.__file__), "models", "ch_ppocr_mobile_v2.0_cls_mobile.onnx")
        self.engine = RapidOCR(params={
            "Global.log_level": "warning",
            "Global.text_score": 0.0,
            "Global.max_side_len": max_side,
            "Det.limit_type": "max",
            "Det.limit_side_len": max_side,
            "Global.model_root_dir": root,
            "Det.model_path": os.path.join(root, "PP-OCRv6_det_medium.onnx"),
            "Cls.model_path": classifier,
            "Rec.model_path": onnx,
            "Rec.rec_keys_path": keys,
            "Rec.rec_img_shape": [3, height, 320],
            "EngineConfig.onnxruntime.use_cuda": device == "cuda",
            "EngineConfig.onnxruntime.cuda_ep_cfg": text_reader._cuda_options(2 << 30),
        })

    def read(self, image) -> dict:
        image = image.convert("RGB")
        width, height = image.size
        out = self.engine(image)
        boxes = out.boxes if getattr(out, "boxes", None) is not None else []
        lines = [
            {"text": str(text).strip(), "score": round(float(score), 4), "box": [[round(float(x) / width, 5), round(float(y) / height, 5)] for x, y in box]}
            for box, text, score in zip(boxes, getattr(out, "txts", None) or (), getattr(out, "scores", None) or ())
            if str(text).strip()
        ]
        return {"lines": lines}


def load_trocr(device: str):
    """The Swedish TrOCR on `device`, its position table built where it runs (htr.materialize_positions)."""
    from transformers import VisionEncoderDecoderModel

    from htr import materialize_positions

    model = VisionEncoderDecoderModel.from_pretrained(TROCR, revision=TROCR_REVISION).to(device).eval()
    materialize_positions(model, device)
    return model


class SwedishHands:
    """The Swedish National Archives' TrOCR (`trocr-base-handwritten-hist-swe-2`, 17th-19th century
    running hands) over the lines PP-OCRv6 medium's detector finds, each cut out by its bounding box."""

    def __init__(self, root: str, device: str, max_side: int):
        import torch
        from transformers import TrOCRProcessor

        import text_reader

        self.torch, self.device = torch, device
        self.detector = text_reader.PhotoReader(root, "medium", device, 2 << 30, max_side)
        self.processor = TrOCRProcessor.from_pretrained(TROCR, revision=TROCR_REVISION)
        self.model = load_trocr(device)

    def read(self, image) -> dict:
        image = image.convert("RGB")
        width, height = image.size
        detected = self.detector.engine(image, use_rec=False, use_cls=False)
        boxes = list(detected.boxes) if getattr(detected, "boxes", None) is not None else []
        crops, kept = [], []
        for box in boxes:
            xs, ys = [float(x) for x, _ in box], [float(y) for _, y in box]
            left, top, right, bottom = max(0, min(xs) - 4), max(0, min(ys) - 4), min(width, max(xs) + 4), min(height, max(ys) + 4)
            if right - left < 8 or bottom - top < 8:
                continue
            crops.append(image.crop((left, top, right, bottom)))
            kept.append(box)
        lines = []
        for at in range(0, len(crops), 16):
            pixels = self.processor(images=crops[at:at + 16], return_tensors="pt").pixel_values.to(self.device)
            with self.torch.inference_mode():
                generated = self.model.generate(pixels, max_new_tokens=96, num_beams=4, output_scores=True, return_dict_in_generate=True)
            texts = self.processor.batch_decode(generated.sequences, skip_special_tokens=True)
            scores = generated.sequences_scores.exp().tolist() if getattr(generated, "sequences_scores", None) is not None else [1.0] * len(texts)
            for text, score, box in zip(texts, scores, kept[at:at + 16]):
                if text.strip():
                    lines.append({"text": text.strip(), "score": round(float(score), 4),
                                  "box": [[round(float(x) / width, 5), round(float(y) / height, 5)] for x, y in box]})
        return {"lines": lines}


class AsServed:
    """`ocr.read` as EMS serves it: the floor in force, so its doubtful lines go to the
    handwriting recogniser (the other PP-OCRv6 columns show every line, the floor at 0)."""

    def __init__(self, reader):
        self.reader = reader

    def read(self, image) -> dict:
        import text_reader

        shown, text_reader.MIN_LINE_SCORE = text_reader.MIN_LINE_SCORE, float(os.environ.get("CFG_OCR_MIN_LINE_SCORE", "0.8"))
        try:
            return self.reader.read(image)
        finally:
            text_reader.MIN_LINE_SCORE = shown


def preview(image) -> str:
    small = image.copy()
    small.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE))
    buffer = io.BytesIO()
    small.convert("RGB").save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


def column_html(name: str, reading: dict) -> str:
    if "lines" in reading:
        rows = "".join(f"<li><span class=s>{line['score']:.2f}</span> {html.escape(line['text'])}</li>" for line in reading["lines"])
        body = f"<ol>{rows}</ol>" if rows else "<i>no text</i>"
    else:
        body = f"<pre>{html.escape(reading.get('text', '')) or '<i>no text</i>'}</pre>"
    return f"<div><h3>{html.escape(name)} · {reading['seconds']} s</h3>{body}</div>"


def text_of(reading: dict, floor: float = 0.8) -> str:
    if "lines" in reading:
        return " | ".join(line["text"] for line in reading["lines"] if line["score"] >= floor)
    return reading.get("text", "").replace("\n", " | ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--root", default=os.environ.get("CFG_OCR_ROOT", "/opt/ocr"))
    parser.add_argument("--max-side", type=int, default=1600, help="the longest side PP-OCRv6's detector sees")
    parser.add_argument("--hand-root", default=os.environ.get("CFG_OCR_HAND_ROOT", "/opt/ocr/multicentury-ppocrv6-medium"))
    parser.add_argument("--engines", default="medium,small", help="comma-separated: medium small ocr-read vl-ocr vl-spotting hand-fi hand-trocr")
    args = parser.parse_args()

    import image_io
    import text_reader
    from text_reader import PhotoReader

    # Every line, so the report shows what a floor would drop.
    text_reader.MIN_LINE_SCORE = 0.0
    names = [name.strip() for name in args.engines.split(",") if name.strip()]
    shared: dict = {}
    engines = {}
    for name in names:
        if name in ("medium", "small"):
            engines[name] = PhotoReader(args.root, name, args.device, 2 << 30, args.max_side)
        elif name == "ocr-read":
            engines[name] = AsServed(PhotoReader(args.root, "medium", args.device, 2 << 30, args.max_side, args.hand_root))
        elif name in ("vl-ocr", "vl-spotting"):
            engines[name] = VisionLanguage(name.removeprefix("vl-"), args.device, shared)
        elif name == "hand-fi":
            os.makedirs(args.out, exist_ok=True)
            engines[name] = FinnishHands(args.root, args.device, args.max_side, args.out)
        elif name == "hand-trocr":
            engines[name] = SwedishHands(args.root, args.device, args.max_side)
        else:
            parser.error(f"unknown engine {name}")
    os.makedirs(args.out, exist_ok=True)

    rows = []
    with open(args.list) as handle:
        for raw in handle:
            if raw.strip():
                path, angle, mirror, label = (raw.rstrip("\n").split("\t") + ["", "", ""])[:4]
                rows.append((path, int(angle or 0), mirror == "1", label))

    results = []
    warm = False
    with open(os.path.join(args.out, "results.jsonl"), "w") as jsonl:
        for path, angle, mirror, label in rows:
            try:
                image = image_io.open_display(path, angle, mirror)
            except Exception as exc:  # noqa: BLE001 - a picture that will not open is reported, not fatal
                print(f"skip {label}: {exc}", flush=True)
                continue
            if not warm:
                for name, engine in engines.items():
                    try:
                        engine.read(image)
                    except Exception as exc:  # noqa: BLE001 - its column reports it on every picture
                        print(f"warm-up of {name} failed: {exc}", flush=True)
                warm = True
            entry = {"path": path, "label": label, "size": image.size}
            for name in names:
                start = time.perf_counter()
                try:
                    reading = engines[name].read(image)
                    reading = {key: value for key, value in reading.items() if key in ("text", "lines", "raw")}
                except Exception as exc:  # noqa: BLE001 - one engine failing a picture is reported in its column
                    reading = {"text": "", "error": str(exc)[-300:]}
                reading["seconds"] = round(time.perf_counter() - start, 3)
                entry[name] = reading
            jsonl.write(json.dumps(entry, ensure_ascii=False) + "\n")
            jsonl.flush()
            print(label + "".join(f"\n  {name} ({entry[name]['seconds']}s): {text_of(entry[name])[:240]}" for name in names), flush=True)
            entry["preview"] = preview(image)
            results.append(entry)

    cards = [
        f"<section><h2>{html.escape(entry['label'])}</h2><img src='data:image/jpeg;base64,{entry['preview']}'>"
        f"<div class=cols>{''.join(column_html(name, entry[name]) for name in names)}</div></section>"
        for entry in results
    ]
    page = (
        "<!doctype html><meta charset=utf-8><title>OCR spot-check</title><style>body{font:14px system-ui;margin:16px}"
        f"img{{max-width:100%;max-height:600px}}.cols{{display:grid;grid-template-columns:repeat({len(names)},1fr);gap:12px}}"
        ".s{color:#888;font-variant-numeric:tabular-nums}section{border-top:1px solid #ccc;padding:12px 0}pre{white-space:pre-wrap}</style>"
        + "".join(cards)
    )
    with open(os.path.join(args.out, "report.html"), "w") as handle:
        handle.write(page)
    print(f"wrote {len(results)} pictures to {args.out}/report.html", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
