"""SigLIP2 image-text embeddings from an Immich ONNX export.

WHY IMMICH'S EXPORT. It is the file set Immich's own published benchmarks ran, split into a
visual and a textual ONNX model with the preprocessing written down beside them, and onnxruntime
is already in this image for InsightFace — no open_clip, no timm.

ONE PRECISION ON EVERY HOST. float32, on the card and on the CPU alike. A CPU-only host must
write the same vectors as a host with a card so a database moves between them without a
re-embed, and onnxruntime's CPU provider does not run float16 well. The card is fast enough at
float32 for a whole-archive backfill.

THE TWO TOWERS ARE SEPARATE FAMILIES. `siglip-text` embeds search queries on the 2-second
interactive path, is pinned, and runs on the CPU by default (model_registry.IMAGE_EMBED_TEXT_DEVICE);
`siglip-visual` embeds pictures
in batch on the card and is unloaded when idle to give its ~2 GB back.

The preprocessing mirrors immich_ml: squash-resize to the configured size (bicubic), scale to
[0,1], normalize by the configured mean/std. Text is cleaned as the config says (SigLIP2:
canonicalize — lower case, no punctuation), padded and truncated to the context length.
"""

from __future__ import annotations

import html
import json
import os
import re
import string


def _canonicalize(text: str) -> str:
    # open_clip's canonicalize_text, which the export's `clean: canonicalize` asks for.
    text = text.replace("_", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text.lower()).strip()


def _whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def l2norm(x):
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def providers(device: str, mem_limit_bytes: int) -> list:
    """CUDA first when asked for, CPU always as the fallback.

    The arena is bounded the way InsightFace's is (see model_registry.insightface): onnxruntime's
    default doubling never returns memory, and this card is shared.
    """
    if device != "cuda":
        return ["CPUExecutionProvider"]
    cuda = {"arena_extend_strategy": "kSameAsRequested", "gpu_mem_limit": str(mem_limit_bytes)}
    return [("CUDAExecutionProvider", cuda), "CPUExecutionProvider"]


def session_options():
    """onnxruntime's options: CFG_MODEL_THREADS intra-op threads when set (the query worker).

    By default a session takes every core, and a short query split across all of them waits on
    whichever thread the batch work has descheduled — see make_workers in ems/app.py.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    threads = os.environ.get("CFG_MODEL_THREADS", "").strip()
    if threads.isdigit() and int(threads) > 0:
        options.intra_op_num_threads = int(threads)
    return options


class Visual:
    """The picture tower: PIL images in, normalized vectors out."""

    def __init__(self, root: str, device: str, mem_limit_bytes: int):
        import onnxruntime as ort

        with open(os.path.join(root, "visual", "preprocess_cfg.json")) as fh:
            self.pre = json.load(fh)
        self.session = ort.InferenceSession(os.path.join(root, "visual", "model.onnx"), sess_options=session_options(), providers=providers(device, mem_limit_bytes))
        self.input = self.session.get_inputs()[0].name
        self.size = tuple(self.pre["size"])
        self.batchable = True

    def pixels(self, img):
        import numpy as np
        from PIL import Image

        img = img.convert("RGB")
        if img.size != self.size:
            img = img.resize(self.size, Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.array(self.pre["mean"], dtype=np.float32)) / np.array(self.pre["std"], dtype=np.float32)
        return arr.transpose(2, 0, 1)

    def embed(self, images) -> list[list[float]]:
        import numpy as np

        batch = np.stack([self.pixels(img) for img in images])
        if self.batchable:
            try:
                return l2norm(self.session.run(None, {self.input: batch})[0]).tolist()
            except Exception:
                # An export with a fixed batch dimension of 1: fall back once and remember.
                self.batchable = False
        out = [self.session.run(None, {self.input: batch[i:i + 1]})[0] for i in range(len(batch))]
        return l2norm(np.concatenate(out)).tolist()


class Textual:
    """The text tower: query strings in, vectors in the same space as the pictures'."""

    def __init__(self, root: str, device: str, mem_limit_bytes: int):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        with open(os.path.join(root, "config.json")) as fh:
            cfg = json.load(fh)
        with open(os.path.join(root, "textual", "tokenizer_config.json")) as fh:
            tok_cfg = json.load(fh)
        text_cfg = cfg.get("text_cfg", {})
        context = int(text_cfg.get("context_length", 77))
        self.canonicalize = (text_cfg.get("tokenizer_kwargs") or {}).get("clean") == "canonicalize"
        self.tokenizer = Tokenizer.from_file(os.path.join(root, "textual", "tokenizer.json"))
        pad = tok_cfg["pad_token"]
        self.tokenizer.enable_padding(length=context, pad_token=pad, pad_id=self.tokenizer.token_to_id(pad))
        self.tokenizer.enable_truncation(max_length=context)
        self.session = ort.InferenceSession(os.path.join(root, "textual", "model.onnx"), sess_options=session_options(), providers=providers(device, mem_limit_bytes))
        self.input = self.session.get_inputs()[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        prepared = [_canonicalize(t) if self.canonicalize else _whitespace_clean(t) for t in texts]
        dtype = np.int64 if "int64" in self.input.type else np.int32
        ids = np.array([e.ids for e in self.tokenizer.encode_batch(prepared)], dtype=dtype)
        out = [self.session.run(None, {self.input.name: ids[i:i + 1]})[0] for i in range(len(ids))]
        return l2norm(np.concatenate(out)).tolist()
