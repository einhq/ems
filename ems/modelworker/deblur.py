"""Deblurring: a picture in, the same picture sharper out.

The model is Restormer (Zamir et al., MIT), the network of restormer.py with its authors' weights
from the repository's v1.0 release, loaded as plain tensors: `motion_deblurring`, trained on
GoPro's camera shake, and `single_image_defocus_deblurring`, trained on DPDD's out-of-focus
photographs. It takes RGB in 0..1, a size a multiple of 8, and answers the change it makes on top
of the picture; a grey picture is shown to it as three equal channels.

Only the detail the model adds is kept: its change less that change blurred by a Gaussian of
DETAIL pixels. The defocus weights also shift a picture's tones by 4 to 10 8-bit steps, which on
photographs blurred for a bench made them further from the sharp picture than before; their detail
alone brought them nearer (ein/core#1241).

A picture goes through it in tiles of at most TILE pixels a side, each with MARGIN pixels of the
picture around it whose answer is dropped, which bounds what it holds on the card whatever the
picture's size. float32 on the card and on the CPU alike.
"""

from __future__ import annotations

TILE = 384
MARGIN = 48
DETAIL = 6.0


def detail(change):
    """The fine part of `change` (height × width × 3, float32): less what a Gaussian of DETAIL keeps."""
    import cv2

    return change - cv2.GaussianBlur(change, (0, 0), DETAIL)


def tiles(height: int, width: int):
    """`(core, shown)` boxes, `(y0, y1, x0, x1)` each: the cores share out a picture of that size,
    and each is shown with MARGIN pixels of the picture around it."""
    core = TILE - 2 * MARGIN
    for y in range(0, height, core):
        for x in range(0, width, core):
            shown = (max(y - MARGIN, 0), min(y + core + MARGIN, height), max(x - MARGIN, 0), min(x + core + MARGIN, width))
            yield (y, min(y + core, height), x, min(x + core, width)), shown


class Deblurrer:
    def __init__(self, path: str, device: str):
        import torch
        from restormer import Restormer

        self.torch = torch
        self.device = device
        model = Restormer()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["params"], strict=True)
        self.model = model.to(device).eval()

    def run(self, rgb):
        """`rgb` (height × width × 3, float32 in 0..1) deblurred; the same shape and range."""
        import numpy as np

        torch = self.torch
        height, width = rgb.shape[:2]
        # Padded with its edge to a multiple of 8, as the network halves the picture three times.
        padded = np.pad(rgb, ((0, -height % 8), (0, -width % 8), (0, 0)), mode="edge")
        with torch.inference_mode():
            tile = torch.from_numpy(padded.transpose(2, 0, 1).copy())[None].to(self.device)
            return self.model(tile)[0, :, :height, :width].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
