"""Denoising: a picture in, the same picture with less noise out.

The model is SCUNet (Zhang et al., Apache-2.0), the network of scunet.py with its authors'
weights from KAIR's v1.0 release, loaded as plain tensors: `scunet_color_real_psnr`, trained on
real noise for fidelity rather than for looks. It takes RGB in 0..1 and pads to a multiple of 64
itself; a grey picture is shown to it as three equal channels.

A picture goes through it in tiles of at most TILE pixels a side, each with MARGIN pixels of the
picture around it whose answer is dropped, which bounds what it holds on the card whatever the
picture's size. float32 on the card and on the CPU alike.
"""

from __future__ import annotations

TILE = 512
MARGIN = 64


def tiles(height: int, width: int):
    """`(core, shown)` boxes, `(y0, y1, x0, x1)` each: the cores share out a picture of that size,
    and each is shown with MARGIN pixels of the picture around it."""
    core = TILE - 2 * MARGIN
    for y in range(0, height, core):
        for x in range(0, width, core):
            shown = (max(y - MARGIN, 0), min(y + core + MARGIN, height), max(x - MARGIN, 0), min(x + core + MARGIN, width))
            yield (y, min(y + core, height), x, min(x + core, width)), shown


class Denoiser:
    def __init__(self, path: str, device: str):
        import torch
        from scunet import SCUNet

        self.torch = torch
        self.device = device
        model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.model = model.to(device).eval()

    def run(self, rgb):
        """`rgb` (height × width × 3, float32 in 0..1) denoised; the same shape and range."""
        torch = self.torch
        with torch.inference_mode():
            tile = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(self.device)
            return self.model(tile)[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
