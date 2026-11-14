# EMS — EiN Media Service
#
# Every media tool and every model the archive needs, behind one HTTP API. See docs/api.md for
# the contract and README.md for running it.
#
# Build:
#   podman build -t ems -f Containerfile .
#
# Run (GPU, mounting its callers' files with sshfs, docs/api.md 3.1):
#   podman run --rm --device nvidia.com/gpu=all --device /dev/fuse --cap-add SYS_ADMIN --network=host \
#     -e EMS_KEY=let-me-in ems
#
# podman's build cache is keyed on the parent layer plus the instruction, so an edit rebuilds
# every layer after it. The ~30 GB of wheels and baked weights come first, the pins that move
# (ffmpeg, then EMS's own code) last.

FROM docker.io/library/debian:trixie-20261027-slim

# CUDA wheel index for torch, and effectively for the whole image: torch's bundled nvidia-*
# packages are what every other GPU library resolves against at runtime.
#
# cu130 (CUDA 13) is what the GPU stack needs. cu124 has no Blackwell kernels at all — torch dies at the first kernel launch on sm_120 with "no kernel
# image is available for execution on the device". Beyond that, the two ML runtimes disagree
# about which CUDA major they want, and BOTH failure modes are silent:
#
#   - onnxruntime-gpu (the face models, the picture embedder, the text reader) needs CUDA 13
#     (libcublasLt.so.13) from 1.24 onward. Given
#     CUDA 12 it drops to CPUExecutionProvider with only a warning. Pinning it back to the
#     last CUDA 12 line (1.23.0) is NOT a way out: that build binds CUDAExecutionProvider
#     happily and then fails on the first inference with cudaErrorNoKernelImageForDevice,
#     because it carries no sm_120 kernels.
#   - CTranslate2 (faster-whisper) links libcublas.so.12 and resolves it lazily at the first
#     decode, so on a pure CUDA 13 stack the model loads fine and only transcription fails
#     with "Library libcublas.so.12 is not found or cannot be loaded".
#
# The resolution is to ship both cuBLAS majors — see nvidia-cublas-cu12 below. The sonames
# differ (.so.12 vs .so.13) so they coexist in the ldconfig cache, and only the CUDA 13 cuDNN
# is installed, so there is no libcudnn.so.9 ambiguity.
#
# Anything that changes here must be re-checked with `make gpu-check`, which runs real
# inference per model — every failure above passes a load-only smoke test.
#
# Override per host: podman build --build-arg TORCH_CUDA=cu126 (or make build TORCH_CUDA=...).
ARG TORCH_CUDA=cu130

# faster-whisper model baked into the image. Must match model_registry.WHISPER_MODEL
# (env CFG_WHISPER_MODEL) or the first transcription downloads it at runtime, inside a
# request, with no network guarantee. See model_registry for why large-v3.
ARG WHISPER_MODEL=large-v3

# libheif-plugin-aomenc is what makes ImageMagick able to WRITE AVIF, which every image
# rendition depends on. Debian splits libheif's codecs into plugin packages and
# `imagemagick` pulls in only the decoders (dav1d, libde265), so without this the failure is
# quiet and late: `convert -list format` reports AVIF as `r--` and every encode dies with
# "no encode delegate for this image format `AVIF'". Installing svtenc/rav1e alongside it
# changes nothing — libheif keeps aomenc's priority and the output is byte-identical.
#
# curl + xz-utils are for the ffmpeg fetch below, ca-certificates for every download; tzdata is
# what TZ names; tini is PID 1 (ENTRYPOINT below). The slim base has none of them.
#
# libreoffice-writer-nogui converts Word, RTF, OpenDocument and text files to PDF for
# `document.renditions`, run as its own process; fonts-liberation gives it fonts metric-compatible
# with Times, Arial and Courier, so a page breaks where the author's did.
#
# No package here brings Debian's FFmpeg libraries: the image's one FFmpeg is the LGPL build
# below, which also fingerprints audio, with its chromaprint muxer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libimage-exiftool-perl \
    imagemagick \
    libheif-plugin-aomenc \
    curl \
    xz-utils \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    tzdata \
    tini \
    libreoffice-writer-nogui \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch and torchvision come from the $TORCH_CUDA wheel index (see the ARG above). These wheels
# bundle their own CUDA runtime + cuDNN 9 as nvidia-* packages, so no CUDA base image is needed —
# only the host driver, exposed via the NVIDIA container toolkit (podman --device nvidia.com/gpu=all).
# onnxruntime-gpu and CTranslate2 (faster-whisper) reuse those bundled libs at
# runtime via the dynamic linker (registered with ldconfig below). With no GPU device passed,
# everything falls back to CPU automatically — see the model registry's device selection.
#
# This MUST stay a separate pip invocation with a single --index-url. pip resolves the highest
# version across all indexes, so with pypi.org as an --extra-index-url PyPI's plain torch — a
# newer CUDA build than this image wants — outranks the pinned one and TORCH_CUDA silently
# does nothing. With no second index to outbid it, the CUDA index's newest matching wheel is
# what gets installed. torchvision comes along here so it matches torch's CUDA build rather
# than being resolved from PyPI as a transitive dependency.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
    torch \
    torchvision

# Everything else from PyPI. torch/torchvision are already satisfied above, so pip keeps the
# CUDA-index builds instead of resolving its own.
#
# onnxruntime-gpu is pinned so its CUDA major cannot drift away from $TORCH_CUDA: 1.28 wants
# CUDA 13, matching cu130. Moving one means re-checking the other with `make gpu-check`.
#
# nvidia-cublas-cu12 is here solely for CTranslate2 (faster-whisper), which links
# libcublas.so.12 while the rest of the image is CUDA 13. It installs alongside the CUDA 13
# cuBLAS rather than replacing it — different sonames, same directory.
#
# transformers is pinned to an exact release rather than a floor, because this is the layer that
# decides whether every model in the image still loads; a bump is checked with `make gpu-check`.
#
# sentence-transformers is pinned for a different and sharper reason: the text embedder's output
# must not move. Every text vector in the caller's index came from it, so a change in its pooling
# or normalisation silently invalidates the whole search index without failing anything.
#
# faster-whisper is pinned at the version the stored transcripts were made with, so a rebuild of
# this layer cannot move it. It goes in without its dependencies, which are listed here instead,
# all but two: PyAV, whose wheel bundles an FFmpeg built with x264 and x265 and which it decodes a
# path with (EMS hands it samples, and model_registry.faster_whisper imports it without PyAV), and
# the CPU `onnxruntime`, whose module of the same name would shadow onnxruntime-gpu's and leave
# every ONNX model on the CPU.
RUN pip install --no-cache-dir \
    onnxruntime-gpu==1.28.0 \
    nvidia-cublas-cu12 \
    opencv-python-headless \
    Pillow \
    pi-heif \
    rawpy \
    pypdfium2 \
    python-docx \
    odfpy \
    transformers==5.16.1 \
    sentence-transformers==5.7.0 \
    "ctranslate2>=4.0,<5" \
    "huggingface-hub>=0.21" \
    "tokenizers>=0.13,<1" \
    tqdm \
    && pip install --no-cache-dir --no-deps faster-whisper==1.2.1

# Expose the torch-bundled NVIDIA libs (cuDNN, cuBLAS, cuda_runtime, ...) to the dynamic
# linker so onnxruntime-gpu and CTranslate2 (faster-whisper) can find them at runtime. torch
# loads its own copies via RPATH, but those two resolve via the system linker. Register the
# nvidia/*/lib dirs through ldconfig (robust against the python minor version in the path).
#
# This MUST precede the onnxruntime check: it imports onnxruntime-gpu, which resolves libcudart
# via the system linker. Run it first and the import fails with "libcudart.so.13: cannot open
# shared object file" even though the lib is present.
RUN python3 -c 'import glob, os, nvidia; print("\n".join(sorted({d for base in nvidia.__path__ for d in glob.glob(os.path.join(base, "*", "lib"))})))' \
        > /etc/ld.so.conf.d/nvidia-python.conf \
    && ldconfig \
    && python3 -c "import onnxruntime as ort; ps = ort.get_available_providers(); \
        print('onnxruntime providers:', ps); \
        assert 'CUDAExecutionProvider' in ps, 'a CPU onnxruntime shadows onnxruntime-gpu: %r' % (ps,)"

# The models baked from here on are passed on with the image, unmodified, and some licences ask
# something of that: CC BY and CC BY-NC a credit and a link to the licence, Apache-2.0 a copy of
# it. notice/NOTICE.md, copied in at the end as /usr/share/doc/ein-ems/ so an edit to it rebuilds
# no model layer, gives those and points to einhq.eu/docs/ems/#models-tools-and-their-licences, which
# lists every model. A model added, dropped or swapped here changes both, in the same session.
#
# The face models (ems/modelworker/face_models.py), both MIT: RetinaFace's ResNet-34 export from
# yakhyo/retinaface-pytorch's v0.0.1 release (~89 MB), which finds the faces and their five
# points, and TransFace from ModelScope's iic/cv_vit_face-recognition at its v1.2 tag (~498 MB),
# which makes each face's vector. Checked against their sha256, as the copy detector is, so a
# rebuild cannot change the vectors under stored ones.
ARG RETINAFACE_SHA256=bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630
ARG TRANSFACE_REVISION=v1.2
ARG TRANSFACE_SHA256=b8e8af9be676f6817d8529e58f5b6d265576e9342ffa0401ea5490bc4b0c127c
RUN mkdir -p /opt/faces \
    && curl -fsSL -o /opt/faces/retinaface_r34.onnx \
        "https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.onnx" \
    && echo "${RETINAFACE_SHA256}  /opt/faces/retinaface_r34.onnx" | sha256sum -c - \
    && curl -fsSL -o /opt/faces/transface.onnx \
        "https://www.modelscope.cn/models/iic/cv_vit_face-recognition/resolve/${TRANSFACE_REVISION}/model.onnx" \
    && echo "${TRANSFACE_SHA256}  /opt/faces/transface.onnx" | sha256sum -c -
ENV CFG_FACES_ROOT=/opt/faces

# Where every Hugging Face model — whisper, the text embedder and NeMo's checkpoints — is baked
# and later read from. This MUST precede the bake layers below.
#
# Without it the cache follows $HOME, and $HOME need not be the same at build time as at run
# time: the build runs as root, and a model server running as another user would find nothing
# and re-download every model from the Hub inside a request, on every container recreate.
#
# /opt rather than either home directory: it belongs to neither user, and it is the same shape
# as /opt/faces.
ENV HF_HOME=/opt/huggingface

ENV CFG_WHISPER_MODEL=${WHISPER_MODEL}

# Prewarm whisper *and its VAD*. Constructing WhisperModel downloads the weights only; the
# Silero VAD is a separate ONNX asset that faster-whisper fetches on the first transcribe
# asking for it — which in this system is inside a request, with no network guarantee.
# That is the same failure the WHISPER_MODEL note warns about, one layer down. Running a
# real transcribe here pulls it at build time instead. Over a second of silence, so it costs
# nothing: the VAD finds no speech and no decode ever runs. EMS's code is not in the image yet,
# so the empty `av` model_registry.faster_whisper imports faster-whisper with is lent here too.
RUN python3 -c "import sys, types; sys.modules['av'] = types.ModuleType('av'); \
    import numpy; \
    from faster_whisper import WhisperModel; \
    model = WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8'); \
    segments, _info = model.transcribe(numpy.zeros(16000, numpy.float32), vad_filter=True, condition_on_previous_text=False); \
    list(segments); \
    print('whisper and vad prewarmed')"

# ── Speaker diarization + voiceprint (NeMo) and active-speaker detection (LR-ASD) ─────────────
#
# This block is deliberately LAST among the pip/model layers and carries its own risk note, because nemo_toolkit is not one clean library: the
# [asr] extra hard-depends on a training stack — lightning (<=2.4), hydra-core (<=1.3.2),
# omegaconf, webdataset, wandb, torchmetrics, sentencepiece, nv_one_logger_* — layered on top
# of the curated torch(cu130) / transformers==5.16.1 / onnxruntime-gpu==1.28.0 / CTranslate2
# graph above. Accepted deliberately: pyannote's weights are gated, and SpeechBrain has no
# ready diarization recipe.
#
# Rules this install follows so it cannot disturb the stack above:
#   - SEPARATE pip invocation with NO --index-url / --extra-index-url, so pip keeps the cu130
#     torch already satisfied instead of resolving torch==2.12.0+cu132 from NeMo's `cu13`
#     extra (that extra is opt-in and NOT requested here).
#   - transformers is left unpinned by NeMo, so the exact 5.16.1 from above wins. NeMo 3.0.0
#     is not known to be tested against transformers 5.x — a NeMo import breaking on it is the
#     first thing to bisect.
#   - `make gpu-check` (torch / the face models / the embedders / whisper / NeMo / LR-ASD must
#     all still load and compute) is mandatory after any change here.
RUN pip install --no-cache-dir \
    "nemo_toolkit[asr]==3.0.0" \
    python_speech_features

# Bake the two checkpoints so the first diarize / voiceprint in a request does not pull
# them from the Hub with no network guarantee — the same rule as WHISPER_MODEL et al. Each is
# one .nemo file on the Hub, named after its repository, fetched at a pinned commit into the
# Hugging Face cache under HF_HOME (/opt/huggingface/hub/models--nvidia--…), where the worker
# finds it by the same commit: a rebuild cannot change the speaker turns or the voiceprints
# under stored ones. Streaming Sortformer v2 = speaker-turn segmentation (handles overlapped
# speech); TitaNet-large = the 192-dim voiceprint embedding. Both are CC BY 4.0 and download
# without a token.
#
# Loading them checks that NeMo still has the classes their model cards name; if NeMo moves
# them, this layer fails first.
ARG SORTFORMER_MODEL=nvidia/diar_streaming_sortformer_4spk-v2
ARG SORTFORMER_REVISION=84edd514b8ef68004c10086918cd62f2148cbd59
ARG TITANET_MODEL=nvidia/speakerverification_en_titanet_large
ARG TITANET_REVISION=0dc382f40121a5fbd34db10a2bb04d826c2be6a8
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    from nemo.collections.asr.models import SortformerEncLabelModel, EncDecSpeakerLabelModel; \
    nemo = lambda repo, revision: hf_hub_download(repo, repo.split('/')[-1] + '.nemo', revision=revision); \
    SortformerEncLabelModel.restore_from(nemo('${SORTFORMER_MODEL}', '${SORTFORMER_REVISION}'), map_location='cpu'); \
    EncDecSpeakerLabelModel.restore_from(nemo('${TITANET_MODEL}', '${TITANET_REVISION}'), map_location='cpu')"
ENV CFG_SORTFORMER_MODEL=${SORTFORMER_MODEL}
ENV CFG_SORTFORMER_REVISION=${SORTFORMER_REVISION}
ENV CFG_TITANET_MODEL=${TITANET_MODEL}
ENV CFG_TITANET_REVISION=${TITANET_REVISION}

# LR-ASD (active speaker detection, IJCV 2025). MIT-licensed; the two ~3.4MB weight files are
# committed in the repo, so there is nothing to download separately and no gate. Fetched as a
# pinned-commit tarball via curl, as ffmpeg is — `git` is not in the image and adding it to the
# apt layer above would invalidate every model bake after it.
# Only the core model (model/*.py + the weights) is used at runtime; the repo's own S3FD face
# detector and scenedetect demo path are not — face tracks come from the face models instead.
ARG LR_ASD_COMMIT=1b6dcd2d8fc2895683de6508ec6294ec47d388ca
RUN curl -fsSL "https://github.com/Junhua-Liao/LR-ASD/archive/${LR_ASD_COMMIT}.tar.gz" \
    | tar -xz -C /opt \
    && mv "/opt/LR-ASD-${LR_ASD_COMMIT}" /opt/lr-asd \
    && test -f /opt/lr-asd/weight/finetuning_TalkSet.model
ENV LR_ASD_HOME=/opt/lr-asd

# The image-text embedder (vision.embed, vision.embed_text): SigLIP2 So400m/16-384 as Immich
# exports it (ems/modelworker/image_embed.py has why). Baked as
# a plain directory, not into the hub cache: the cache stores files as symlinks into blobs/, and
# onnxruntime refuses external weight data that resolves outside the model's folder. The rknpu
# builds (Rockchip NPUs) are 4 GB of files this image never reads. Pinned to a commit so a rebuild
# cannot change the vectors under stored ones. ~4.3 GB.
ARG IMAGE_EMBED_MODEL=ViT-SO400M-16-SigLIP2-384__webli
ARG IMAGE_EMBED_REVISION=19baa26af70bd3639ca0ca17d1560cb8056dd983
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('immich-app/${IMAGE_EMBED_MODEL}', revision='${IMAGE_EMBED_REVISION}', \
        local_dir='/opt/image-embed/${IMAGE_EMBED_MODEL}', \
        allow_patterns=['config.json', 'visual/*', 'textual/*'], ignore_patterns=['*/rknpu/*'])" \
    && rm -rf /opt/image-embed/${IMAGE_EMBED_MODEL}/.cache \
    && test -f /opt/image-embed/${IMAGE_EMBED_MODEL}/visual/model.onnx
ENV CFG_IMAGE_EMBED_MODEL=${IMAGE_EMBED_MODEL}
ENV CFG_IMAGE_EMBED_ROOT=/opt/image-embed

# The copy detector (fingerprint.compute's sscd-* kinds): Meta's SSCD sscd_disc_mixup as its
# authors publish it, a standalone TorchScript file (ems/modelworker/copy_detect.py). Checked
# against its sha256 so a rebuild cannot change the descriptors under stored ones. ~99 MB.
ARG COPY_DETECT_MODEL=sscd_disc_mixup
ARG COPY_DETECT_SHA256=9f26bd4c848cc19b73d2ae92eea6e04886f61a7b764ceb7a13aeee62e6a6db56
RUN mkdir -p /opt/copy-detect \
    && curl -fsSL -o /opt/copy-detect/${COPY_DETECT_MODEL}.torchscript.pt \
        "https://dl.fbaipublicfiles.com/sscd-copy-detection/${COPY_DETECT_MODEL}.torchscript.pt" \
    && echo "${COPY_DETECT_SHA256}  /opt/copy-detect/${COPY_DETECT_MODEL}.torchscript.pt" | sha256sum -c -
ENV CFG_COPY_DETECT_MODEL=${COPY_DETECT_MODEL}
ENV CFG_COPY_DETECT_ROOT=/opt/copy-detect

# The shot detector (video.shots): TransNetV2's weights (MIT, Souček and Lokoč) converted from
# the authors' TensorFlow release to PyTorch, for the module vendored as
# ems/modelworker/transnetv2.py. Checked against their sha256, as the copy detector is. ~31 MB.
ARG TRANSNET_SHA256=834b10f25ae9e1b4e4f2652fe2843bd2b1388057a435d68b7c52635578fcc04d
RUN mkdir -p /opt/transnet \
    && curl -fsSL -o /opt/transnet/transnetv2-pytorch-weights.pth \
        "https://huggingface.co/Sn4kehead/TransNetV2/resolve/main/transnetv2-pytorch-weights.pth" \
    && echo "${TRANSNET_SHA256}  /opt/transnet/transnetv2-pytorch-weights.pth" | sha256sum -c -
ENV CFG_TRANSNET_ROOT=/opt/transnet

# The text embedder (text.embed with `model: current`): BAAI/bge-m3 (MIT), at a pinned revision so
# a rebuild cannot change stored vectors, read from that revision's own weights file
# (use_safetensors=False, as model_registry loads it). Loading it here fetches the weights and
# checks the dimension the caller's vector table is sized for. ~2.3 GB.
ARG TEXT_EMBED_MODEL=BAAI/bge-m3
ARG TEXT_EMBED_REVISION=5617a9f61b028005a4858fdac845db406aefb181
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
    m = SentenceTransformer('${TEXT_EMBED_MODEL}', revision='${TEXT_EMBED_REVISION}', device='cpu', \
        model_kwargs={'use_safetensors': False}); \
    v = m.encode(['en segelbåt'], prompt_name='query', normalize_embeddings=True); \
    assert v.shape == (1, 1024), v.shape"
ENV CFG_TEXT_EMBED_MODEL=${TEXT_EMBED_MODEL}
ENV CFG_TEXT_EMBED_REVISION=${TEXT_EMBED_REVISION}

# The text reader (ocr.read, ems/modelworker/text_reader.py): PP-OCRv6 through RapidOCR. rapidocr
# declares no onnxruntime of its own, so the GPU wheel above serves it, and everything else it
# wants (opencv, numpy, Pillow) is already here and stays. The ONNX files of both tiers are baked
# at the sha256s RapidOCR's own model list pins, so it never downloads one inside a request and a
# rebuild cannot change what is read: medium is the engine, small the runner-up the spot-check
# compares (ein/core#352). ~170 MB.
ARG RAPIDOCR_VERSION=3.10.0
ARG OCR_MODELS="det/PP-OCRv6_det_small.onnx:090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f \
    det/PP-OCRv6_det_medium.onnx:92078b7355007ccfffcd4c8cd441a3afd4538904d06881b29a155e1e679907c2 \
    rec/PP-OCRv6_rec_small.onnx:6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884 \
    rec/PP-OCRv6_rec_medium.onnx:eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b"
RUN pip install --no-cache-dir rapidocr==${RAPIDOCR_VERSION} \
    && mkdir -p /opt/ocr \
    && for entry in ${OCR_MODELS}; do \
        file="${entry%%:*}"; sum="${entry##*:}"; \
        curl -fsSL -o "/opt/ocr/$(basename "$file")" \
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v${RAPIDOCR_VERSION}/onnx/PP-OCRv6/${file}" \
        && echo "${sum}  /opt/ocr/$(basename "$file")" | sha256sum -c - || exit 1; \
    done
ENV CFG_OCR_ROOT=/opt/ocr

# The handwriting recogniser the photo engine reads its doubtful lines with again
# (text_reader.HandReader): the National Archives of Finland's PP-OCRv6_medium_rec fine-tune, as its
# authors publish it, at a pinned revision. Its dictionary is `character_dict` in inference.yml,
# written out with the space its README says to add. Apache-2.0, ~77 MB.
ARG OCR_HAND_MODEL=Kansallisarkisto/multicentury-ppocrv6-medium
ARG OCR_HAND_REVISION=01a6a5413959c8cff053e2aa60485de14b507f82
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('${OCR_HAND_MODEL}', revision='${OCR_HAND_REVISION}', \
        local_dir='/opt/ocr/multicentury-ppocrv6-medium', allow_patterns=['inference.onnx', 'inference.yml'])" \
    && rm -rf /opt/ocr/multicentury-ppocrv6-medium/.cache \
    && python3 -c "import yaml; \
    keys = yaml.safe_load(open('/opt/ocr/multicentury-ppocrv6-medium/inference.yml'))['PostProcess']['character_dict']; \
    open('/opt/ocr/multicentury-ppocrv6-medium/keys.txt', 'w').write('\n'.join(keys) + '\n ')" \
    && test -s /opt/ocr/multicentury-ppocrv6-medium/inference.onnx
ENV CFG_OCR_HAND_ROOT=/opt/ocr/multicentury-ppocrv6-medium

# The handwriting transcriber (ems/modelworker/htr.py): the National Archives of Finland's TrOCR-large
# fine-tune, as its authors publish it, at a pinned revision. It writes the lines the recogniser
# above judges to be writing. Apache-2.0, ~2.2 GB.
ARG HTR_MODEL=Kansallisarkisto/multicentury-htr-model
ARG HTR_REVISION=fe70dcf30a5856279c616b60271bc9c22d4d2ed6
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('${HTR_MODEL}', revision='${HTR_REVISION}', local_dir='/opt/htr/multicentury-htr-model', \
        ignore_patterns=['README.md', '.gitattributes'])" \
    && rm -rf /opt/htr/multicentury-htr-model/.cache \
    && test -s /opt/htr/multicentury-htr-model/pytorch_model.bin
ENV CFG_HTR_ROOT=/opt/htr/multicentury-htr-model

# The English handwriting transcriber (htr.py): Microsoft's TrOCR large handwritten, at a pinned
# revision. Its RoBERTa tokenizer comes as vocab.json and merges.txt, which transformers 5 turns into
# a fast tokenizer only through sentencepiece; it is written out as tokenizer.json here instead. The
# model card names no licence for the weights; TrOCR's code is MIT. ~2.2 GB.
ARG HTR_EN_MODEL=microsoft/trocr-large-handwritten
ARG HTR_EN_REVISION=e68501f437cd2587ae5d68ee457964cac824ddee
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('${HTR_EN_MODEL}', revision='${HTR_EN_REVISION}', local_dir='/opt/htr/trocr-large-handwritten', \
        ignore_patterns=['README.md', '.gitattributes'])" \
    && rm -rf /opt/htr/trocr-large-handwritten/.cache \
    && python3 -c "from transformers import AutoTokenizer, RobertaTokenizer; \
    RobertaTokenizer.from_pretrained('/opt/htr/trocr-large-handwritten').save_pretrained('/opt/htr/trocr-large-handwritten'); \
    t = AutoTokenizer.from_pretrained('/opt/htr/trocr-large-handwritten'); \
    assert t.decode(t('Dear Bert').input_ids, skip_special_tokens=True) == 'Dear Bert'" \
    && test -s /opt/htr/trocr-large-handwritten/pytorch_model.bin
ENV CFG_HTR_EN_ROOT=/opt/htr/trocr-large-handwritten

# The language of read text (text_language.py): lingua, every language's models in its wheel. ~290 MB.
RUN pip install --no-cache-dir lingua-language-detector==2.2.0

# The inpainter (image.inpaint's lama, ems/modelworker/lama.py): big-lama (Suvorov et al.,
# Apache-2.0) as IOPaint exports it, one TorchScript file. Checked against its sha256, as the copy
# detector is, so a rebuild cannot change what it paints. ~206 MB.
ARG INPAINT_MODEL=big-lama
ARG INPAINT_SHA256=344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9
RUN mkdir -p /opt/inpaint \
    && curl -fsSL -o /opt/inpaint/${INPAINT_MODEL}.pt \
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt" \
    && echo "${INPAINT_SHA256}  /opt/inpaint/${INPAINT_MODEL}.pt" | sha256sum -c -
ENV CFG_INPAINT_MODEL=${INPAINT_MODEL}
ENV CFG_INPAINT_ROOT=/opt/inpaint

# The defect detector (image.defects, ems/modelworker/defects.py): the scratch detector of
# Microsoft's Bringing Old Photos Back to Life (MIT), its checkpoint as the authors publish it in
# the v1.0 release's global_checkpoints.zip. The zip is checked against its sha256 and only the
# detector's file kept. ~450 MB kept, a 2 GB download.
ARG DEFECTS_ZIP_SHA256=3ff4c52a3a65d71e458ff24b1d7339b326d216ef793c2027761e11efb30f7882
RUN mkdir -p /opt/defects \
    && curl -fsSL -o /tmp/global_checkpoints.zip \
        "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip" \
    && echo "${DEFECTS_ZIP_SHA256}  /tmp/global_checkpoints.zip" | sha256sum -c - \
    && python3 -c "import zipfile; \
        open('/opt/defects/FT_Epoch_latest.pt', 'wb').write(zipfile.ZipFile('/tmp/global_checkpoints.zip').read('checkpoints/detection/FT_Epoch_latest.pt'))" \
    && rm /tmp/global_checkpoints.zip \
    && echo "b2d7ab04e9b3885c6b1991bb7a0b823129dd6e3ac078a9fd059ebd2a7ba59a95  /opt/defects/FT_Epoch_latest.pt" | sha256sum -c -
ENV CFG_DEFECTS_ROOT=/opt/defects

# The denoiser (image.denoise, ems/modelworker/denoise.py): SCUNet (Apache-2.0), its authors' weights
# from KAIR's v1.0 release, one a noise the request picks, each checked against its sha256. The
# network (scunet.py) needs einops. ~216 MB.
ARG DENOISE_WEIGHTS="scunet_color_15:fa3a95efb4add693a78917e70757a3d535c5a8c905ace9f93ba7e5897351e1b2 \
    scunet_color_25:6b4e572fe69b1530aade8b7856b18b9e6ddf9cf2bd87c21bf045b51662096320 \
    scunet_color_real_psnr:fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e"
RUN pip install --no-cache-dir einops==0.8.2 \
    && mkdir -p /opt/denoise \
    && for entry in ${DENOISE_WEIGHTS}; do \
        name="${entry%%:*}"; sum="${entry##*:}"; \
        curl -fsSL -o "/opt/denoise/$name.pth" "https://github.com/cszn/KAIR/releases/download/v1.0/$name.pth" \
        && echo "$sum  /opt/denoise/$name.pth" | sha256sum -c - || exit 1; \
    done
ENV CFG_DENOISE_ROOT=/opt/denoise

# The deblurrer (image.deblur, ems/modelworker/deblur.py): Restormer (MIT), its authors' weights from
# the repository's v1.0 release, one a blur the request names (camera shake, out of focus), each
# checked against its sha256. The network (restormer.py) needs einops, installed above. ~210 MB.
ARG DEBLUR_WEIGHTS="motion_deblurring:194e38fb5b607c9dc5a5b3e08e65b2e79ee2bf0ef5048e0612f6b2ff2f79da31 \
    single_image_defocus_deblurring:7dce451f33f8f5e0faf7c4e3996e5dcc1bd425ecd1ada99b0f9750e490fd4c9e"
RUN mkdir -p /opt/deblur \
    && for entry in ${DEBLUR_WEIGHTS}; do \
        name="${entry%%:*}"; sum="${entry##*:}"; \
        curl -fsSL -o "/opt/deblur/$name.pth" "https://github.com/swz30/Restormer/releases/download/v1.0/$name.pth" \
        && echo "$sum  /opt/deblur/$name.pth" | sha256sum -c - || exit 1; \
    done
ENV CFG_DEBLUR_ROOT=/opt/deblur

# The face restorer (image.restore_faces, ems/modelworker/faces_restore.py): GFPGAN v1.4, Tencent
# ARC's weights from GFPGAN's v1.3.0 release, checked against their sha256. GFPGAN's code is
# Apache-2.0; its LICENSE also names third-party parts under non-commercial terms, which EiN's list
# of models and their licences says. ~350 MB.
ARG GFPGAN_SHA256=e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad
RUN mkdir -p /opt/gfpgan \
    && curl -fsSL -o /opt/gfpgan/GFPGANv1.4.pth "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
    && echo "${GFPGAN_SHA256}  /opt/gfpgan/GFPGANv1.4.pth" | sha256sum -c -
ENV CFG_GFPGAN_ROOT=/opt/gfpgan

# ffmpeg 9.0.1, from BtbN's static LGPL builds. The image installs no distro ffmpeg.
#
# The LGPL build, not the GPL one: EMS encodes AV1 only, so it needs none of what --enable-gpl
# adds (libx264, libx265, xvid and GPL-only filters such as vidstab, frei0r and hqdn3d, none of
# which EMS calls), and an image carrying GPL binaries would have to carry their source too.
#
# The base image cannot supply this: Debian stable (trixie) carries ffmpeg 7.1.x, 8.1 is only in
# sid and 9.0 only in experimental — so no base-image bump gets us here, and apt-pinning
# experimental would drag newer libs under a stable userland.
#
# What 9 buys is transpose_cuda (absent from 8.1). It keeps a *rotated* video's whole filter
# chain on the GPU: yadif_cuda,scale_cuda,transpose_cuda -> av1_nvenc with no hwdownload and no
# CPU hop. See ems/tools/encode.py.
#
# This follows upstream's n9.0 release branch, not git master. Pin via the dated release tag,
# whose asset name records the exact release-branch commit; the `latest` tag's `n9.0-latest`
# asset is a moving target. BtbN keeps only month-end dated builds long-term, so prefer one of
# those whenever the branch has one. Debian's own ffmpeg is a GPL build, so it is no way back.
#
# HOST DRIVER REQUIREMENT: this build wants nvenc API 13.1, i.e. **NVIDIA driver >= 610.00**.
# On an older driver EVERY nvenc encoder fails to open with "Driver does not support the
# required nvenc API version", even where Debian's ffmpeg would encode. Nothing breaks (the
# transcode falls back to the CPU chain) but the GPU encode is silently unused, so check
# `make gpu-check` after any driver change.
#
# The tools are spawned by name (ems/tools/run.py), so one PATH entry keeps every caller on this
# binary.
ARG FFMPEG_BUILD=autobuild-2026-08-31-13-27
ARG FFMPEG_ASSET=ffmpeg-n9.0.1-11-ge47273f4d9-linux64-lgpl-9.0
RUN curl -fsSL "https://github.com/BtbN/FFmpeg-Builds/releases/download/${FFMPEG_BUILD}/${FFMPEG_ASSET}.tar.xz" \
    | tar -xJ -C /opt \
    && mv "/opt/${FFMPEG_ASSET}" /opt/ffmpeg \
    && rm -f /opt/ffmpeg/bin/ffplay \
    && /opt/ffmpeg/bin/ffmpeg -hide_banner -filters | grep -q transpose_cuda \
    && ! /opt/ffmpeg/bin/ffmpeg -hide_banner -buildconf | grep -q -- --enable-gpl
ENV PATH="/opt/ffmpeg/bin:$PATH"

# ── EMS ─────────────────────────────────────────────────────────────────────────────────────

# The HTTP front. Pinned floors only; the model stack above is what decides compatibility.
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30"

# sshfs, which mounts a caller's files (ems/remote.py), and socat, which carries each of its
# connections (ems/files_connect.py); the container needs /dev/fuse and SYS_ADMIN for it.
RUN apt-get update && apt-get install -y --no-install-recommends sshfs socat \
    && rm -rf /var/lib/apt/lists/*

# Nothing installed above brought back PyAV, the CPU onnxruntime or Debian's FFmpeg libraries.
RUN ! pip show av 2>/dev/null && ! pip show onnxruntime 2>/dev/null \
    && ! ldconfig -p | grep -q libavcodec

WORKDIR /opt/ems
COPY ems ems
COPY scripts scripts
COPY docs docs
COPY pyproject.toml README.md LICENSE ./
COPY notice /usr/share/doc/ein-ems

# Compile once so the first request does not.
RUN python3 -m compileall -q ems scripts

# Defaults an operator overrides per deployment.
ENV EMS_PORT=8181
ENV EMS_HOST=0.0.0.0
ENV EMS_SCRATCH=/tmp/ems
ENV EMS_KEYS_FILE=/etc/ems.keys
# The model worker's own knobs, in the EMS namespace (worker.py maps EMS_* → CFG_*). These
# repeat the CFG_* values baked above so `capabilities` reports them without consulting both.
ENV EMS_WHISPER_MODEL=${WHISPER_MODEL}
ENV EMS_IMAGE_EMBED_MODEL=${IMAGE_EMBED_MODEL}
ENV EMS_TEXT_EMBED_MODEL=${TEXT_EMBED_MODEL}
# ImageMagick parallelises one `magick` across cores via OpenMP (default: all of them). Left
# alone, every rendition in the tool lane (EMS_LIMIT_TOOLS wide) spawns that many threads and
# oversubscribes the box. Per-image OpenMP scaling is sublinear, so N independent images at one
# thread each beats one image at N threads — and one thread per slot is what lets EMS_LIMIT_TOOLS
# be sized against cores.
ENV MAGICK_THREAD_LIMIT=1
# The transcodes' knobs (see ems/tools/encode.py): the CPU decode's projected footprint per window,
# the constant it is projected from, how many windows one encode may become. EMS_VIDEO_ENCODER
# is empty on purpose — av1_nvenc when the card and this ffmpeg offer it, libsvtav1 otherwise.
ENV EMS_CPU_ENCODE_MAX_MB=4000
ENV EMS_CPU_ENCODE_MB_PER_1K_FRAMES=170
ENV EMS_CPU_ENCODE_MAX_SEGMENTS=64

EXPOSE 8181

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "-m", "ems"]
