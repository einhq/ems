# EMS — EiN Media Service
#
# Every media tool and every model the archive needs, behind one HTTP API. See docs/api.md for
# the contract and README.md for running it.
#
# Build:
#   podman build -t ems -f Containerfile .
#
# Run (GPU, with the roots the caller shares — identical paths on both sides):
#   podman run --rm --device nvidia.com/gpu=all --network=host \
#     -e EMS_KEY=let-me-in -e EMS_ROOTS=/files:ro,/old:ro,/files-volatile:rw,/files-tmp:rw \
#     -v /srv/files:/files:ro -v /srv/old:/old:ro -v /srv/files-volatile:/files-volatile -v /srv/files-tmp:/files-tmp \
#     ems
#
# podman's build cache is keyed on the parent layer plus the instruction, so an edit rebuilds
# every layer after it. The ~30 GB of wheels and baked weights come first, the pins that move
# (ffmpeg, then EMS's own code) last.

FROM docker.io/library/debian:trixie-20261005-slim

# CUDA wheel index for torch, and effectively for the whole image: torch's bundled nvidia-*
# packages are what every other GPU library resolves against at runtime.
#
# cu130 (CUDA 13) is what the GPU stack needs. cu124 has no Blackwell kernels at all — torch dies at the first kernel launch on sm_120 with "no kernel
# image is available for execution on the device". Beyond that, the two ML runtimes disagree
# about which CUDA major they want, and BOTH failure modes are silent:
#
#   - onnxruntime-gpu (InsightFace) needs CUDA 13 (libcublasLt.so.13) from 1.24 onward. Given
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
RUN apt-get update && apt-get install -y --no-install-recommends \
    libimage-exiftool-perl \
    imagemagick \
    libheif-plugin-aomenc \
    ffmpeg \
    libchromaprint-tools \
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
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch and torchvision come from the $TORCH_CUDA wheel index (see the ARG above). These wheels
# bundle their own CUDA runtime + cuDNN 9 as nvidia-* packages, so no CUDA base image is needed —
# only the host driver, exposed via the NVIDIA container toolkit (podman --device nvidia.com/gpu=all).
# onnxruntime-gpu (InsightFace) and CTranslate2 (faster-whisper) reuse those bundled libs at
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
# accelerate is here for peft (the text embedder's LoRA adapters), which is installed without its
# dependencies below.
#
# insightface, faster-whisper and av are pinned at the versions the stored face vectors and
# transcripts were made with, so a rebuild of this layer cannot move them: av 19 dropped the
# `metadata_errors` argument faster-whisper 1.2.1 passes to it, and failed the whisper bake below.
RUN pip install --no-cache-dir \
    insightface==2.0 \
    onnxruntime-gpu==1.28.0 \
    nvidia-cublas-cu12 \
    opencv-python-headless \
    Pillow \
    pillow-heif \
    rawpy \
    PyMuPDF \
    python-docx \
    odfpy \
    transformers==5.16.1 \
    accelerate \
    sentence-transformers==5.7.0 \
    faster-whisper==1.2.1 \
    av==18.1.0

# insightface and faster-whisper both declare a hard dependency on `onnxruntime` (the CPU
# wheel), so pip installs it alongside onnxruntime-gpu. Both provide the same `onnxruntime`
# module and the CPU wheel wins at import time — CUDAExecutionProvider then never shows up in
# get_available_providers() and InsightFace silently runs on CPU even with the GPU passed
# through. Drop the CPU wheel (two layers down); onnxruntime-gpu satisfies the same import.
#
# Expose the torch-bundled NVIDIA libs (cuDNN, cuBLAS, cuda_runtime, ...) to the dynamic
# linker so onnxruntime-gpu and CTranslate2 (faster-whisper) can find them at runtime. torch
# loads its own copies via RPATH, but those two resolve via the system linker. Register the
# nvidia/*/lib dirs through ldconfig (robust against the python minor version in the path).
#
# This MUST precede the onnxruntime check below: that step imports onnxruntime-gpu, which
# resolves libcudart via the system linker. Run it first and the import fails with
# "libcudart.so.13: cannot open shared object file" even though the lib is present.
RUN python3 -c 'import glob, os, nvidia; print("\n".join(sorted({d for base in nvidia.__path__ for d in glob.glob(os.path.join(base, "*", "lib"))})))' \
        > /etc/ld.so.conf.d/nvidia-python.conf \
    && ldconfig

# Both wheels install into the same `onnxruntime` package directory, and each one's RECORD
# lists those shared files — so uninstalling the CPU wheel deletes files onnxruntime-gpu is
# still using, leaving an `onnxruntime` that imports but has no get_available_providers().
# Reinstalling the GPU wheel afterwards puts them back; --no-deps keeps insightface's
# dependency on the CPU wheel from dragging it straight back in.
RUN pip uninstall -y onnxruntime \
    && pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu==1.28.0 \
    && python3 -c "import onnxruntime as ort; ps = ort.get_available_providers(); \
        print('onnxruntime providers:', ps); \
        assert 'CUDAExecutionProvider' in ps, 'CPU wheel still shadows onnxruntime-gpu: %r' % (ps,)"

RUN python3 -c "from insightface.app import FaceAnalysis; \
    app = FaceAnalysis(name='buffalo_l', root='/opt/insightface', providers=['CPUExecutionProvider']); \
    app.prepare(ctx_id=0, det_size=(640, 640))"

ENV INSIGHTFACE_HOME=/opt/insightface

# Where every Hugging Face model — whisper and the text embedder — is baked
# and later read from. This MUST precede the bake layers below, and it is set for exactly
# the reason INSIGHTFACE_HOME above is.
#
# Without it the cache follows $HOME, and $HOME need not be the same at build time as at run
# time: the build runs as root, and a model server running as another user would find nothing
# and re-download every model from the Hub inside a request, on every container recreate.
#
# /opt rather than either home directory: it belongs to neither user, and it is the same shape
# as /opt/insightface.
ENV HF_HOME=/opt/huggingface

ENV CFG_WHISPER_MODEL=${WHISPER_MODEL}

# Prewarm whisper *and its VAD*. Constructing WhisperModel downloads the weights only; the
# Silero VAD is a separate ONNX asset that faster-whisper fetches on the first transcribe
# asking for it — which in this system is inside a request, with no network guarantee.
# That is the same failure the WHISPER_MODEL note warns about, one layer down. Running a
# real transcribe here pulls it at build time instead. Over generated silence, so it costs
# nothing: the VAD finds no speech and no decode ever runs.
RUN python3 -c "import os, tempfile, wave; \
    from faster_whisper import WhisperModel; \
    model = WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8'); \
    path = os.path.join(tempfile.mkdtemp(), 'silence.wav'); \
    handle = wave.open(path, 'wb'); \
    handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000); \
    handle.writeframes(b'\x00' * 32000); handle.close(); \
    segments, _info = model.transcribe(path, vad_filter=True, condition_on_previous_text=False); \
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
#   - `make gpu-check` (torch / InsightFace / the embedders / whisper / NeMo / LR-ASD must
#     all still load and compute) is mandatory after any change here.
RUN pip install --no-cache-dir \
    "nemo_toolkit[asr]==3.0.0" \
    python_speech_features

# NeMo caches pretrained checkpoints under $HOME by default — the same build-time vs run-time
# HOME split that HF_HOME and INSIGHTFACE_HOME above exist to solve. Point it at /opt.
ENV NEMO_CACHE_DIR=/opt/nemo

# Bake the two checkpoints so the first diarize / voiceprint in a request does not pull
# them from NGC / the Hub with no network guarantee — the same rule as WHISPER_MODEL et al.
# Sortformer = speaker-turn segmentation (handles overlapped speech); TitaNet-large = the
# 192-dim voiceprint embedding. Sortformer is CC BY-NC 4.0 (non-commercial), TitaNet is
# CC BY 4.0; both download without a token.
#
# Class names follow the NeMo model cards for nvidia/diar_sortformer_4spk-v1 and
# nvidia/speakerverification_en_titanet_large; if NeMo moves them, this layer fails first.
RUN python3 -c "from nemo.collections.asr.models import SortformerEncLabelModel, EncDecSpeakerLabelModel; \
    SortformerEncLabelModel.from_pretrained('nvidia/diar_sortformer_4spk-v1'); \
    EncDecSpeakerLabelModel.from_pretrained('nvidia/speakerverification_en_titanet_large')"

# LR-ASD (active speaker detection, IJCV 2025). MIT-licensed; the two ~3.4MB weight files are
# committed in the repo, so there is nothing to download separately and no gate. Fetched as a
# pinned-commit tarball via curl, as ffmpeg is — `git` is not in the image and adding it to the
# apt layer above would invalidate every model bake after it.
# Only the core model (model/*.py + the weights) is used at runtime; the repo's own S3FD face
# detector and scenedetect demo path are not — face tracks come from InsightFace instead.
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

# The text embedder (text.embed with `model: current`): jina-embeddings-v5-text-small. Its remote
# code puts LoRA task adapters on a Qwen3-0.6B, so it needs peft, which nothing else here uses;
# --no-deps because everything peft wants (torch, transformers, accelerate, safetensors) is
# already pinned above and must not move. peft and the revision are pinned so a rebuild cannot
# change stored vectors. Loading it
# here fetches the remote code as well as the weights, and checks the adapter and the dimension
# the caller's vector table is sized for. ~1.2 GB.
ARG TEXT_EMBED_MODEL=jinaai/jina-embeddings-v5-text-small
ARG TEXT_EMBED_REVISION=dd76d535f5447ca3897a9c893fb1e612ead98192
RUN pip install --no-cache-dir --no-deps peft==0.21.0 \
    && python3 -c "from sentence_transformers import SentenceTransformer; \
    m = SentenceTransformer('${TEXT_EMBED_MODEL}', revision='${TEXT_EMBED_REVISION}', trust_remote_code=True, \
        device='cpu', model_kwargs={'default_task': 'retrieval'}); \
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

# The English handwriting transcriber (htr.py): Microsoft's TrOCR-large fine-tuned on IAM, at a pinned
# revision. Its RoBERTa tokenizer comes as vocab.json and merges.txt, which transformers 5 turns into
# a fast tokenizer only through sentencepiece; it is written out as tokenizer.json here instead. The
# model card names no licence (TrOCR's code is MIT; IAM's terms are non-commercial research). ~2.2 GB.
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

# ffmpeg 9.0.1, from BtbN's static builds, shadowing the distro package on PATH.
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
# those whenever the branch has one, or return to the distro package once backports ships 9.x.
#
# HOST DRIVER REQUIREMENT: this build wants nvenc API 13.1, i.e. **NVIDIA driver >= 610.00**.
# On an older driver EVERY nvenc encoder fails to open with "Driver does not support the
# required nvenc API version", even where the distro ffmpeg encodes. Nothing breaks (the
# transcode falls back to the CPU chain) but the GPU encode is silently unused, so check
# `make gpu-check` after any driver change.
#
# The distro ffmpeg stays installed and PATH decides which one runs: the tools are spawned by
# name (ems/tools/run.py), so one PATH entry keeps every caller on the same binary.
ARG FFMPEG_BUILD=autobuild-2026-08-31-13-27
ARG FFMPEG_ASSET=ffmpeg-n9.0.1-11-ge47273f4d9-linux64-gpl-9.0
RUN curl -fsSL "https://github.com/BtbN/FFmpeg-Builds/releases/download/${FFMPEG_BUILD}/${FFMPEG_ASSET}.tar.xz" \
    | tar -xJ -C /opt \
    && mv "/opt/${FFMPEG_ASSET}" /opt/ffmpeg \
    && rm -f /opt/ffmpeg/bin/ffplay \
    && /opt/ffmpeg/bin/ffmpeg -hide_banner -filters | grep -q transpose_cuda
ENV PATH="/opt/ffmpeg/bin:$PATH"

# ── EMS ─────────────────────────────────────────────────────────────────────────────────────

# The HTTP front. Pinned floors only; the model stack above is what decides compatibility.
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30"

WORKDIR /opt/ems
COPY ems ems
COPY scripts scripts
COPY docs docs
COPY pyproject.toml README.md ./

# Compile once so the first request does not.
RUN python3 -m compileall -q ems scripts

# Defaults an operator overrides per deployment. EMS_ROOTS is empty on purpose: a container
# started without naming what it may read refuses every file request instead of serving the
# disk it happens to see.
ENV EMS_PORT=8181
ENV EMS_HOST=0.0.0.0
ENV EMS_ROOTS=""
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
