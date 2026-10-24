"""SSCD copy-detection descriptors: a picture in, a 512-dimension unit vector out.

Two pictures are copies of one another when their descriptors' cosine is high: re-encodes,
resizes, crops, colour edits, overlays, scans of a print and screenshots stay close, while two
different photos of the same scene stay apart — what a perceptual hash and an image-text
embedding each get wrong one way.

The model is Meta's `sscd_disc_mixup` (ResNet50, MIT licence) as its authors publish it, a
TorchScript file with the pooling and the L2 normalisation built in. The preprocessing is theirs
too: the short edge resized to 288, ImageNet mean and std. float32 on the card and on the CPU
alike, so a descriptor does not depend on where it was made.
"""

from __future__ import annotations

SHORT_EDGE = 288
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Descriptor:
    def __init__(self, path: str, device: str):
        import torch

        self.torch = torch
        self.device = device
        self.model = torch.jit.load(path, map_location=device).eval()

    def tensor(self, img):
        import numpy as np
        from PIL import Image

        img = img.convert("RGB")
        # torchvision's Resize(288) on a PIL image, as the authors ran it: the long edge truncated.
        w, h = img.size
        long = max(1, int(SHORT_EDGE * max(w, h) / min(w, h)))
        img = img.resize((SHORT_EDGE, long) if w <= h else (long, SHORT_EDGE), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
        return self.torch.from_numpy(arr.transpose(2, 0, 1).copy())

    def describe(self, images) -> list[list[float]]:
        # One at a time: pictures keep their own aspect, so a batch would need padding the model
        # was not trained with.
        out = []
        with self.torch.inference_mode():
            for img in images:
                vector = self.model(self.tensor(img).unsqueeze(0).to(self.device))[0]
                out.append(vector.float().cpu().tolist())
        return out
