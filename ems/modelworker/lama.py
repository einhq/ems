"""LaMa inpainting: a picture and a mask in, the picture with the masked part painted out.

The model is big-lama (Suvorov et al., Apache-2.0) as IOPaint exports it: one TorchScript file that
takes the picture as RGB in 0..1 and the mask as 0 or 1, both with sides a multiple of 8, paints
the masked part and gives the rest back as it came. The sides are padded by mirroring the picture,
as IOPaint pads them, and the padding is cut off again. float32 on the card and on the CPU alike.
"""

from __future__ import annotations

MULTIPLE = 8


class Inpainter:
    def __init__(self, path: str, device: str):
        import torch

        self.torch = torch
        self.device = device
        self.model = torch.jit.load(path, map_location=device).eval()

    def fill(self, picture, mask):
        """`picture` (height × width × 3, RGB float32 in 0..1) painted where `mask` (height × width,
        boolean) is set; the same shape and range."""
        import numpy as np

        height, width = mask.shape
        pad = ((0, -height % MULTIPLE), (0, -width % MULTIPLE))
        picture = np.pad(picture, (*pad, (0, 0)), mode="symmetric")
        mask = np.pad(mask.astype(np.float32), pad, mode="symmetric")
        torch = self.torch
        with torch.inference_mode():
            image = torch.from_numpy(np.ascontiguousarray(picture.transpose(2, 0, 1)))[None].to(self.device)
            painted = self.model(image, torch.from_numpy(mask)[None, None].to(self.device))[0]
            return painted.clamp(0, 1).permute(1, 2, 0)[:height, :width].float().cpu().numpy()
