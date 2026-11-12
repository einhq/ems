"""Face restoration: a face in a picture redrawn by GFPGAN, and put back where it was.

The model is GFPGAN v1.4 (Tencent ARC), the clean network of gfpgan_arch.py with its authors' EMA
weights, loaded as plain tensors. It is generative: it draws a face from what it learnt of faces,
and the detail it adds is invented. Its noise is the fixed one stored with the weights, not drawn
anew, so the same face gives the same answer.

A face is aligned on its five points (the eyes, the nose tip, the mouth corners, as the face
detector finds them) to the 512 px template GFPGAN was trained on, as facexlib aligns it, redrawn
there, and turned back into the picture inside a mask of the aligned square whose edge is eroded
and feathered by the face's size, as facexlib pastes it. Every pixel outside that mask is the
picture's own. float32 on the card and on the CPU alike.
"""

from __future__ import annotations

SIZE = 512
# facexlib's five-point template for a 512 px face.
TEMPLATE = (
    (192.98138, 239.94708),
    (318.90277, 240.1936),
    (256.63416, 314.01935),
    (201.26117, 371.41043),
    (313.08905, 371.15118),
)
# The grey the aligned square is filled with past the picture's edge, as facexlib fills it.
BORDER = (135 / 255, 133 / 255, 132 / 255)


def align(rgb, landmarks):
    """The face at `landmarks` (five `[x, y]` in pixels of `rgb`), as a SIZE × SIZE RGB square in
    0..1, and the 2 × 3 matrix that took it there."""
    import cv2
    import numpy as np

    matrix, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, np.float32), np.asarray(TEMPLATE, np.float32), method=cv2.LMEDS)
    face = cv2.warpAffine(rgb, matrix, (SIZE, SIZE), borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER)
    return face, matrix


def region(matrix, height: int, width: int) -> tuple[int, int, int, int]:
    """`(y0, y1, x0, x1)`: the part of a picture of that size the face aligned by `matrix` goes
    back into, a pixel wider on each side."""
    import cv2
    import numpy as np

    corners = np.array([[[0, 0], [SIZE, 0], [0, SIZE], [SIZE, SIZE]]], np.float32)
    back = cv2.transform(corners, cv2.invertAffineTransform(matrix))[0]
    x0, y0 = np.floor(back.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(back.max(axis=0)).astype(int) + 1
    return max(int(y0), 0), min(int(y1), height), max(int(x0), 0), min(int(x1), width)


def _inverse(matrix, y0: int, x0: int):
    """The matrix that turns the aligned square back into the part of the picture from `(x0, y0)`."""
    import cv2

    inverse = cv2.invertAffineTransform(matrix)
    inverse[:, 2] -= (x0, y0)
    return inverse


def paste_mask(matrix, box: tuple[int, int, int, int]):
    """How much of a face aligned by `matrix` goes back into each pixel of `box` (`region`'s): 1
    inside, falling to 0 over a feathered edge, 0 outside."""
    import cv2
    import numpy as np

    y0, y1, x0, x1 = box
    square = cv2.warpAffine(np.ones((SIZE, SIZE), np.float32), _inverse(matrix, y0, x0), (x1 - x0, y1 - y0))
    square = cv2.erode(square, np.ones((2, 2), np.uint8))
    edge = int(float(square.sum()) ** 0.5) // 20
    if edge < 1:
        return square
    centre = cv2.erode(square, np.ones((edge * 2, edge * 2), np.uint8))
    return cv2.GaussianBlur(centre, (edge * 2 + 1, edge * 2 + 1), 0)


def paste(restored, matrix, height: int, width: int, strength: float):
    """`restored` (the aligned square) turned back by `matrix` into a picture of that size: the
    `region` it lands in, the face there, and the weight of each of its pixels, the paste mask times
    `strength`."""
    import cv2

    box = region(matrix, height, width)
    y0, y1, x0, x1 = box
    back = cv2.warpAffine(restored, _inverse(matrix, y0, x0), (x1 - x0, y1 - y0))
    return box, back, paste_mask(matrix, box) * strength


class Restorer:
    def __init__(self, path: str, device: str):
        import torch
        from gfpgan_arch import GFPGANv1Clean

        self.torch = torch
        self.device = device
        model = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None, fix_decoder=False,
                              num_mlp=8, input_is_latent=True, different_w=True, narrow=1, sft_half=True)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["params_ema"], strict=True)
        self.model = model.to(device).eval()

    def restore(self, face):
        """`face` (SIZE × SIZE × 3, RGB float32 in 0..1, aligned) redrawn; the same shape and range."""
        torch = self.torch
        with torch.inference_mode():
            tensor = torch.from_numpy((face.transpose(2, 0, 1) - 0.5) / 0.5).float()[None].to(self.device)
            out = self.model(tensor, return_rgb=False, randomize_noise=False)[0]
            return ((out[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).float().cpu().numpy()
