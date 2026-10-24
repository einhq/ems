"""Handwritten lines to text: a TrOCR-large per language of writing.

- `multicentury-htr-model`, the National Archives of Finland's TrOCR-large (558 M parameters)
  fine-tuned on 913 000 handwritten and typed lines of 16th-20th century Swedish and Finnish, at
  192x1024 pixels a line (2.8 % character error on its own test set). On two of the archive's
  handwritten letters it halved the word errors of the PP-OCRv6 handwriting recogniser (36 % to
  18 %, 46 % to 27 %; ein/core#352). It was trained at a size its config does not name, so
  every call interpolates the encoder's position encoding.
- `trocr-large-handwritten`, Microsoft's TrOCR-large fine-tuned on IAM's modern English
  handwriting, at its own 384x384. Over every line of 17 English pages of 1775-1918, letters and
  postcards and a typed letter, it wrote 0.53 of the words where the Swedish and Finnish model
  wrote 0.31 (ein/core#670).

Each writes the text of the lines text_reader sends it. float32 on the card and on the CPU alike,
as everywhere here.
"""

from __future__ import annotations

# Lines per generate call: 18 lines at once peaked at 3.1 GB on the card, one at a time at 2.25 GB.
BATCH = 4
MAX_NEW_TOKENS = 96
BEAMS = 4


def materialize_positions(model, device: str) -> None:
    """Build TrOCR's sinusoidal position table where the model runs.

    transformers 5 constructs a model on the meta device, and the table is a plain attribute
    rather than a buffer, so loading never fills it: the first decode fails with "Cannot copy out
    of meta tensor".
    """
    from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding

    for module in model.modules():
        if isinstance(module, TrOCRSinusoidalPositionalEmbedding):
            module.weights = module.get_embedding(module.weights.size(0), module.embedding_dim, module.padding_idx).to(device)


class LineTranscriber:
    """Cut-out lines (BGR arrays) in, their text out, in order.

    `size` is the (height, width) a line is read at when the model was trained at another than its
    config says; None reads at the config's own.
    """

    def __init__(self, root: str, device: str, size: tuple[int, int] | None = None):
        import os

        import torch
        from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

        self.torch = torch
        self.device = device
        self.model_name = os.path.basename(os.path.normpath(root))
        resize = {"use_fast": True, "do_resize": True, "size": {"height": size[0], "width": size[1]}} if size else {}
        self.images = AutoImageProcessor.from_pretrained(root, **resize)
        self.tokenizer = AutoTokenizer.from_pretrained(root)
        self.interpolate = size is not None
        self.model = VisionEncoderDecoderModel.from_pretrained(root).to(device).eval()
        materialize_positions(self.model, device)

    def read(self, lines) -> list[str]:
        texts: list[str] = []
        for at in range(0, len(lines), BATCH):
            batch = [line[:, :, ::-1].copy() for line in lines[at:at + BATCH]]
            pixels = self.images(images=batch, return_tensors="pt").pixel_values.to(self.device)
            with self.torch.inference_mode():
                generated = self.model.generate(pixels, max_new_tokens=MAX_NEW_TOKENS, num_beams=BEAMS, interpolate_pos_encoding=self.interpolate)
            texts += [text.strip() for text in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]
        return texts
