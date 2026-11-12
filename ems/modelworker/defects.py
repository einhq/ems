"""Scratch and dust detection: a grey picture in, each pixel's probability of being a defect out.

The model is the scratch detector of "Bringing Old Photos Back to Life" (Microsoft, MIT), the UNet
of bopb_detection.py with its authors' checkpoint, loaded as plain tensors. Its preprocessing is
theirs: grey in 0..1, scaled to -1..1; a sigmoid over its output. It was trained on defects lighter
than the picture around them, so darker ones are looked for in the inverted picture.

The network keeps 64 channels at the picture's own size, so a picture goes through it in tiles of
at most TILE pixels a side, each with MARGIN pixels of the picture around it whose answer is
dropped, which bounds what it holds on the card whatever the picture's size. float32 on the card
and on the CPU alike.
"""

from __future__ import annotations

TILE = 768
MARGIN = 128
# The network halves the picture four times.
MULTIPLE = 16


class Detector:
    def __init__(self, path: str, device: str):
        import torch
        from bopb_detection import UNet

        self.torch = torch
        self.device = device
        model = UNet(in_channels=1, out_channels=1, depth=4, conv_num=2, wf=6, padding=True, batch_norm=True,
                     up_mode="upsample", with_tanh=False, sync_bn=True, antialiasing=True)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state"])
        self.model = model.to(device).eval()

    def probabilities(self, grey, dark: bool = False):
        """Each pixel's probability of being a defect, for `grey` (height × width, float32 in 0..1):
        a defect lighter than its surroundings, and with `dark` darker ones too (the larger of the
        two answers)."""
        found = self._light(grey)
        if dark:
            import numpy as np

            found = np.maximum(found, self._light(1.0 - grey))
        return found

    def _light(self, grey):
        import numpy as np

        height, width = grey.shape
        core = TILE - 2 * MARGIN
        out = np.zeros((height, width), np.float32)
        for y in range(0, height, core):
            for x in range(0, width, core):
                y0, x0 = max(y - MARGIN, 0), max(x - MARGIN, 0)
                y1, x1 = min(y + core + MARGIN, height), min(x + core + MARGIN, width)
                found = self._run(grey[y0:y1, x0:x1])
                cy1, cx1 = min(y + core, height), min(x + core, width)
                out[y:cy1, x:cx1] = found[y - y0 : cy1 - y0, x - x0 : cx1 - x0]
        return out

    def _run(self, tile):
        import numpy as np

        torch = self.torch
        height, width = tile.shape
        pad = ((0, -height % MULTIPLE), (0, -width % MULTIPLE))
        # Reflection needs more picture than it pads; a sliver at an edge is repeated instead.
        mode = "reflect" if min(height, width) > MULTIPLE else "edge"
        tile = np.pad((tile - 0.5) / 0.5, pad, mode=mode).astype(np.float32)
        with torch.inference_mode():
            found = torch.sigmoid(self.model(torch.from_numpy(tile)[None, None].to(self.device)))[0, 0]
            return found[:height, :width].float().cpu().numpy()
