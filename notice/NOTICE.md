# EMS, the EiN Media Service: what this image passes on

EMS's own code is under the MIT licence, in /opt/ems/LICENSE. The image
also carries models and tools made by others, each under its own licence. Every one of them, its
maker, its model card and its licence, is listed at

https://einhq.eu/docs/ems/#models-tools-and-their-licences

which EiN also shows offline, among its docs.

The image carries each model's weights as their maker publishes them, unmodified. Where the page
names an export or a conversion (SigLIP 2 as Immich exports it, TransNetV2 converted to PyTorch),
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
- Jina AI's jina-embeddings-v5-text-small, https://huggingface.co/jinaai/jina-embeddings-v5-text-small,
  in /opt/huggingface: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/legalcode

CC BY-NC 4.0 allows no commercial use.

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

The Python packages in /opt/venv carry their own licences in their dist-info folders.

## Under the GPL

The image's ffmpeg, in /opt/ffmpeg, is BtbN's LGPL build of FFmpeg 9.0.1
(https://github.com/BtbN/FFmpeg-Builds, its build ffmpeg-n9.0.1-11-ge47273f4d9-linux64-lgpl-9.0);
its source is that tag's FFmpeg, https://github.com/FFmpeg/FFmpeg/tree/e47273f4d9, and BtbN's
scripts. Two parts of the image carry GPL code all the same, and this is where the source of each
is:

- The Debian packages under /usr: the image is built on Debian 13 (trixie) and carries its
  packages unmodified. `dpkg-query -W` in the image lists each with its version; its source is
  Debian's, at https://sources.debian.org/ and, for the exact version, at
  https://snapshot.debian.org/. Among them are libavcodec61 and its siblings (FFmpeg 7.1.5,
  7:7.1.5-0+deb13u1), libx264-164, libx265-215, libxvidcore4 and libpostproc58, which
  chromaprint's fpcalc links; EMS uses fpcalc to fingerprint audio.
- PyAV 18.1.0, in /opt/venv (av.libs), which faster-whisper decodes audio with. Its wheel bundles
  FFmpeg 8.1.2 built with x264 and x265, as PyAV's own build publishes it; the source is PyAV's,
  https://pypi.org/project/av/18.1.0/#files, and its FFmpeg build's,
  https://github.com/PyAV-Org/pyav-ffmpeg.

The image modifies none of them.
