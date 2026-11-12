# EMS, the EiN Media Service: what this image passes on

EMS's own code is under the MIT licence, in /opt/ems/LICENSE. The image also carries models and tools made by others, each under its own
licence. Every one of them, its maker, its model card and its licence, is listed at

https://einhq.eu/docs/ems/#models-tools-and-their-licences

which EiN also shows offline, among its docs.

The image carries each model's weights as their maker publishes them, unmodified. Where the page
names an export or a conversion (SigLIP 2 as Immich exports it, TransNetV2 converted to PyTorch,
big-lama as IOPaint exports it),
it carries that export or conversion as its maker publishes it. The image adds two files derived
from published ones: /opt/ocr/multicentury-ppocrv6-medium/keys.txt, the character list of that
model's inference.yml written out on its own, and tokenizer.json beside TrOCR large handwritten,
its tokenizer in the format transformers reads.

The models are provided as is, without warranties of any kind, as their licences say.

## Under Creative Commons

- NVIDIA's TitaNet large, https://huggingface.co/nvidia/speakerverification_en_titanet_large, in
  /opt/huggingface: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/legalcode
- NVIDIA's Streaming Sortformer diar_streaming_sortformer_4spk-v2,
  https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2, in /opt/huggingface:
  CC BY 4.0, https://creativecommons.org/licenses/by/4.0/legalcode

## Under the MIT licence

The licence's text, with the copyright notice it asks to be kept, is MIT, beside this file.

- RetinaFace ResNet-34 from yakhyo/retinaface-pytorch, its v0.0.1 release,
  https://github.com/yakhyo/retinaface-pytorch, in /opt/faces/retinaface_r34.onnx
- Alibaba DAMO Academy's TransFace, https://www.modelscope.cn/models/iic/cv_vit_face-recognition,
  in /opt/faces/transface.onnx: MIT as its model card gives it, which names no copyright holder
- BAAI's bge-m3, https://huggingface.co/BAAI/bge-m3, in /opt/huggingface: MIT as its model card
  gives it, which names no copyright holder

## Under Apache-2.0

The licence's text is Apache-2.0, beside this file.

- Google's SigLIP 2 So400m/16-384, as Immich exports it,
  https://huggingface.co/immich-app/ViT-SO400M-16-SigLIP2-384__webli, in /opt/image-embed
- PaddleOCR's PP-OCRv6, https://github.com/PaddlePaddle/PaddleOCR, as RapidOCR publishes it, in
  /opt/ocr
- The National Archives of Finland's multicentury-ppocrv6-medium,
  https://huggingface.co/Kansallisarkisto/multicentury-ppocrv6-medium, in /opt/ocr
- The National Archives of Finland's multicentury-htr-model,
  https://huggingface.co/Kansallisarkisto/multicentury-htr-model, in /opt/htr
- TransNetV2's weights converted to PyTorch, https://huggingface.co/Sn4kehead/TransNetV2, in
  /opt/transnet
- Samsung AI Center's big-lama, https://github.com/advimman/lama, as IOPaint exports it to
  TorchScript, https://github.com/Sanster/models/releases/tag/add_big_lama, in /opt/inpaint

The Python packages in /opt/venv carry their own licences in their dist-info folders.

## Under the GPL

The image's ffmpeg, in /opt/ffmpeg, is BtbN's LGPL build of FFmpeg 9.0.1
(https://github.com/BtbN/FFmpeg-Builds, its build ffmpeg-n9.0.1-11-ge47273f4d9-linux64-lgpl-9.0);
its source is that tag's FFmpeg, https://github.com/FFmpeg/FFmpeg/tree/e47273f4d9, and BtbN's
scripts, which also build the libraries in it, among them Chromaprint, which EMS fingerprints
audio with. The image does not modify it. The image's GPL code is in Debian's packages, as Debian
below says.

## Debian

The image is built on Debian 13 (trixie) and carries Debian's packages unmodified, each under its
own licence, many of them the GPL or the LGPL; each package's is in
/usr/share/doc/<package>/copyright. `dpkg-query -W` in the image lists every package with its
version. Their source is Debian's, at https://sources.debian.org/ and, for the exact version, at
https://snapshot.debian.org/.
