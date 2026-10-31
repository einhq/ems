# EMS, the EiN Media Service: what this image passes on

EMS's own code is under the MIT licence, in /opt/ems/LICENSE. The image
also carries models and tools made by others, each under its own licence. Every one of them, its
maker, its model card and its licence, is listed at

https://einhq.eu/docs/ems/#models-and-their-licences

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
- NVIDIA's Sortformer diar_sortformer_4spk-v1, https://huggingface.co/nvidia/diar_sortformer_4spk-v1,
  in /opt/huggingface: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/legalcode
- Jina AI's jina-embeddings-v5-text-small, https://huggingface.co/jinaai/jina-embeddings-v5-text-small,
  in /opt/huggingface: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/legalcode

CC BY-NC 4.0 allows no commercial use.

## Under Apache-2.0

The licence's text is Apache-2.0, beside this file.

- Google's SigLIP 2 So400m/16-384, as Immich exports it,
  https://huggingface.co/immich-app/ViT-SO400M-16-SigLIP2-384__webli, in /opt/image-embed
- Qwen3-0.6B-Base, https://huggingface.co/Qwen/Qwen3-0.6B-Base, which jina-embeddings-v5-text-small
  is built on
- PaddleOCR's PP-OCRv6, https://github.com/PaddlePaddle/PaddleOCR, as RapidOCR publishes it, in
  /opt/ocr
- The National Archives of Finland's multicentury-ppocrv6-medium,
  https://huggingface.co/Kansallisarkisto/multicentury-ppocrv6-medium, in /opt/ocr
- The National Archives of Finland's multicentury-htr-model,
  https://huggingface.co/Kansallisarkisto/multicentury-htr-model, in /opt/htr
- TransNetV2's weights converted to PyTorch, https://huggingface.co/Sn4kehead/TransNetV2, in
  /opt/transnet

The Python packages in /opt/venv carry their own licences in their dist-info folders.
