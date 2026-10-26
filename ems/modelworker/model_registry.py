#!/usr/bin/env python3
"""
Model cache for the model server.

Models stay loaded across requests. Idle families are evicted by `evict_idle()` so an
audio-free crawl stops paying for whisper and VRAM stays bounded.

Everything imports lazily inside the getters — importing this module must stay cheap.
"""

import gc
import heapq
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from contextlib import contextmanager

warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('ORT_LOGGING_LEVEL', '3')

# Unload a resident family after this long without a request.
#
# Long enough that a working crawl never evicts — a cold load costs more than the memory it frees
# for a moment — and short enough that a machine left alone overnight gives the card back. VRAM
# pressure is handled by measurement rather than by a clock: `reset_insightface_if_low` releases
# onnxruntime's arenas, and a family that does not fit the card runs on the CPU (`place`).
IDLE_EVICT_SECONDS = float(os.environ.get('CFG_MODEL_IDLE_EVICT', '21600'))

# Families that are never evicted, and are preloaded by the resident server.
#
# The query-side models sit on the *interactive* path — every search embeds its query with
# them (`text-query`, the text embedder's query family; `siglip-text`, the picture space's text
# tower). Evicting one means the next search pays a reload of tens of seconds, which is the
# difference between search feeling instant and feeling broken. whisper and the other batch
# families are worth evicting; these are not.
PINNED_FAMILIES = {
    name.strip() for name in os.environ.get('CFG_MODEL_PINNED', 'text-query').split(',') if name.strip()
}

# Which faster-whisper model to load. MUST match the model baked into the image (the
# Containerfile's ARG WHISPER_MODEL) or the first transcription downloads it at runtime, with
# no network guarantee.
#
# large-v3, not turbo or smaller: on Swedish home video it recovers whole sentences and first
# names where turbo is merely recognisable, at about 1.6x turbo's time. It is not pinned, so it
# is evicted after an idle spell and costs nothing on an audio-free crawl. Do not downgrade it
# to free VRAM.
WHISPER_MODEL = os.environ.get('CFG_WHISPER_MODEL', 'large-v3')

# Language whisper is told to expect, as an ISO-639-1 code. Empty means detect it, the way
# WHISPER_LANGUAGES below describes.
#
# This is a pin: every file is decoded as this language, detection never runs. Right for a
# corpus in one language only; a family archive with holiday clips is not one, and a pin
# decodes the English of a Las Vegas clip into Swedish-shaped nonsense. Prefer
# WHISPER_LANGUAGES for that case.
WHISPER_LANGUAGE = os.environ.get('CFG_WHISPER_LANGUAGE', '').strip()

# Languages the archive is expected to hold, as ISO-639-1 codes separated by commas, most
# common first, used when WHISPER_LANGUAGE is empty. Empty means detection alone decides.
#
# Whisper's own detection reads one 30-second window and decides for the *whole file*, and
# on a Swedish home archive it is confidently wrong often enough to matter: a clip that
# opens on wind noise, a doorbell or a single loanword is detected as Norwegian or Danish
# with high confidence, and every sentence after it is decoded as that language (`ikke` and
# `hadde` for `inte` and `hade`). Told the wrong language, whisper may even *translate*, and
# the output still reads as speech and goes into search.
#
# So detection here (inference_ops.language_votes and choose_language) reads
# WHISPER_LANGUAGE_WINDOWS windows spread across the speech the VAD found, not the first 30
# seconds of the file, and averages them weighted by how sure whisper was of each. An
# expected language then claims its own probability plus that of the neighbours whisper
# confuses it with (Norwegian, Nynorsk and Danish for Swedish; inference_ops.SIBLINGS), and
# wins when the claim reaches WHISPER_LANGUAGE_FLOOR. When nothing is claimed and the top
# guess is under 0.5, whisper has not detected anything, and the first language listed —
# the archive's main one — applies. A genuinely foreign clip, detected confidently, leaves
# every claim near zero and is decoded as what it is.
#
# What this cannot do is tell a Norwegian relative from a Swedish one when only `sv` is
# listed: the Norwegian is decoded as Swedish. List the languages that really occur, in
# order — a neighbour's mass goes to the first listed language it belongs to.
WHISPER_LANGUAGES = tuple(
    code.strip() for code in os.environ.get('CFG_WHISPER_LANGUAGES', '').split(',') if code.strip()
)

# The claim an expected language needs to be chosen over the top guess. Low, because a
# foreign clip's claim is near zero; kept above that so a foreign clip stays foreign.
WHISPER_LANGUAGE_FLOOR = float(os.environ.get('CFG_WHISPER_LANGUAGE_FLOOR', '0.1'))

# How many 30-second windows of speech the detection reads. Each is one encoder pass, the
# same cost as transcribing that window, so a short clip pays for it twice and a long one
# hardly notices; the windows are spread over the whole clip so an opening on noise, or a
# guest speaking at the start, is one vote among several rather than the verdict.
WHISPER_LANGUAGE_WINDOWS = max(1, int(os.environ.get('CFG_WHISPER_LANGUAGE_WINDOWS', '3')))

# Whether to align each word to its own timestamp, rather than each segment.
#
# Whisper segments on pauses and a diarizer on voices, so one segment can hold the end of one
# person's sentence and the start of another's; word timings let a caller cut it at the word
# nearest the change. Off by default because it is a second pass over the decoder's
# cross-attention for every segment; a request can ask for it (`word_timestamps`).
WHISPER_WORD_TIMESTAMPS = os.environ.get('CFG_WHISPER_WORD_TIMESTAMPS', '') == '1'

# What whisper decodes with, by device. A host runs it in one place only — on the card if it has
# one, waiting there for room rather than moving to the CPU, and on the CPU if it has none — so a
# transcript never depends on what else was on the card; the place is in its signature
# (ein/core#515). One place, not one precision on both: on fry's Swedish home video int8 on
# the card and int8 on the CPU differed in 7% of words, int8 on the card from float16 in 8%, and
# float32 on the CPU, float16's twin, held 11 GB of RAM at well below real time
# (`whisper-coverage --compute`, wow notes/host-and-runtime-limits.md).
WHISPER_COMPUTE = {
    "cuda": os.environ.get('CFG_WHISPER_CARD_COMPUTE', 'float16'),
    "cpu": os.environ.get('CFG_WHISPER_CPU_COMPUTE', 'int8'),
}

# The Silero VAD in front of whisper: what counts as speech, and what is cut away.
#
# The VAD is what stops whisper hallucinating over silence — an archive of phone clips is
# mostly silence, and left to itself the decoder fills it with invented speech and
# repetition loops. But faster-whisper does not merely skip the silence: it cuts every gap
# longer than MIN_SILENCE out of the audio, concatenates what is left into one stream, and
# decodes *that*. So the setting decides what the decoder hears, and a tight one feeds it
# sentences jammed together with no breath between them.
#
# The three are set together. A tight setting (threshold 0.5, pad 400ms, silence 500ms) drops
# audible speech, and loosening any one of them alone lets whisper invent fluent nonsense over
# a near-silent clip. Together they keep the speech and hold such a clip to a line, and that
# residue is what WHISPER_NO_SPEECH_PROB_MAX below removes.
#
#   threshold   Silero's speech probability a frame needs. Outdoor audio, wind, water and a
#               child's voice a few metres from a phone are exactly what 0.5 rejects.
#   pad         Silence kept on each side of a region, so word onsets are not clipped and the
#               decoder gets a natural edge rather than a cut.
#   silence     How long a gap has to be before it is cut. 2000ms is the library's own
#               default; at 500ms every pause between sentences goes too.
#
# Do not lower `silence` or raise `threshold` "to be safe": each makes the hallucination
# problem *worse*, not better — a short fragment of borderline audio is what whisper invents
# over, and both settings produce more of them.
WHISPER_VAD_THRESHOLD = float(os.environ.get('CFG_WHISPER_VAD_THRESHOLD', '0.2'))
WHISPER_VAD_SPEECH_PAD_MS = int(os.environ.get('CFG_WHISPER_VAD_SPEECH_PAD_MS', '800'))
WHISPER_VAD_MIN_SILENCE_MS = int(os.environ.get('CFG_WHISPER_VAD_MIN_SILENCE_MS', '2000'))


def whisper_vad_parameters() -> dict:
    """The `vad_parameters` the pipeline decodes with; one place so a harness measures the same thing."""
    return {
        'threshold': WHISPER_VAD_THRESHOLD,
        'speech_pad_ms': WHISPER_VAD_SPEECH_PAD_MS,
        'min_silence_duration_ms': WHISPER_VAD_MIN_SILENCE_MS,
    }


# Segments the decoder itself rates this likely to be non-speech are dropped.
#
# Whisper scores every window for "there is no speech here" before decoding it, and the
# reference implementation only acts on that score when the decode was also low-confidence.
# A hallucination is the opposite case: a confident, fluent sentence over a window the model
# rated likely no-speech. Invented lines — fluent nonsense, the "Takk for at du så på!"
# video-credit class — score high on it and genuine lines, sung ones included, lower; 0.8 sits
# between them on the side that keeps speech, and the odd credit line is the accepted residue.
# Set to 1 to disable.
WHISPER_NO_SPEECH_PROB_MAX = float(os.environ.get('CFG_WHISPER_NO_SPEECH_PROB_MAX', '0.8'))

# Speaker diarization: who is talking, and between which seconds. Same baked-model rule as
# WHISPER_MODEL (the Containerfile). Sortformer is end-to-end and handles overlapping speech.
SORTFORMER_MODEL = os.environ.get('CFG_SORTFORMER_MODEL', 'nvidia/diar_sortformer_4spk-v1')

# How much audio goes into one diarization call, in seconds.
#
# This is not a tuning knob, it is a hard model limit. Sortformer's own training config
# declares `session_len_sec: 90`, and it is not linear in input length: handed a whole
# recording it allocates proportionally to the square of the sequence, and a recording of a
# few minutes asks for more than a 16GB card holds. At 90s a window peaks under 1GB.
#
# Raising this does not buy accuracy; it buys an OOM. The cost of windowing is that speaker
# indices are local to each call, which `diarize` reports rather than hides — see there.
DIARIZE_WINDOW_SECONDS = float(os.environ.get('CFG_DIARIZE_WINDOW_SECONDS', '90'))

# Voiceprint embedding, for matching a speech turn to a person the way a face embedding
# matches a photo to one. Same baked-model rule as WHISPER_MODEL.
TITANET_MODEL = os.environ.get(
    'CFG_TITANET_MODEL', 'nvidia/speakerverification_en_titanet_large',
)

# 'auto' (default) puts the voiceprint model on the GPU when there is one; 'cpu' keeps it off.
#
# The model is small, but a recording's turns arrive as one batch of a thousand, several
# recordings at once, and on a loaded host the CPU is an order of magnitude slower at it than
# the card — where it costs ~100MB of VRAM and nothing else, since inference serialises per
# family anyway.
TITANET_DEVICE = os.environ.get('CFG_TITANET_DEVICE', 'auto')

# Active speaker detection: whether a visible face is the one making the sound.
#
# Not a Hugging Face model — LR-ASD is vendored into the image as a source tree with its
# weights committed alongside (see the Containerfile), so this is a directory rather than a
# model id. `finetuning_TalkSet.model` over `pretrain_AVA.model`: AVA is broadcast footage,
# TalkSet is the fine-tune the authors ship for video in the wild, which is what home
# recordings are.
LR_ASD_HOME = os.environ.get('LR_ASD_HOME', '/opt/lr-asd')
LR_ASD_WEIGHTS = os.environ.get('CFG_LR_ASD_WEIGHTS', 'finetuning_TalkSet.model')

# 'auto' (default) puts it on the GPU when there is one, 'cpu' forces it off the card.
#
# The model is small (0.8M params) but it runs once per face track per turn, and the card is
# about 40x faster at it than the CPU.
LR_ASD_DEVICE = os.environ.get('CFG_LR_ASD_DEVICE', 'auto')

# Ceiling for InsightFace's onnxruntime CUDA arena. Group photos are the driver — det_size is
# fixed but the recognition batch scales with the number of faces — and half this exhausts the
# arena mid-inference ("Available memory of … is smaller than requested bytes of …"). 6144
# leaves room on a 16GB card for the embedders and the CUDA contexts. See insightface() for why
# a ceiling is needed at all.
INSIGHTFACE_GPU_MEM_LIMIT_BYTES = int(
    os.environ.get('CFG_INSIGHTFACE_GPU_MEM_MB', '6144')
) * 1024 * 1024

# The image-text embedder (see image_embed.py): an Immich ONNX export baked as a plain directory
# (the hub cache's symlinks into blobs/ make onnxruntime refuse the external weight files). The
# directory name is the producer, so changing the model changes every stored vector's name.
IMAGE_EMBED_MODEL = os.environ.get('CFG_IMAGE_EMBED_MODEL', 'ViT-SO400M-16-SigLIP2-384__webli')
IMAGE_EMBED_ROOT = os.environ.get('CFG_IMAGE_EMBED_ROOT', '/opt/image-embed')
# auto = the card when there is one. The vectors are float32 either way, so the device does not
# change what is stored (image_embed.py, "one precision on every host").
IMAGE_EMBED_DEVICE = os.environ.get('CFG_IMAGE_EMBED_DEVICE', 'auto')
# The text tower's own device, and the CPU by default. It embeds one search query at a time, a
# few hundred milliseconds on the CPU, while on the card it would hold ~2.8 GB for as long as it
# is pinned — room the batch families need. The CPU's float32 vectors match the card's to
# within rounding, so a query embedded here still searches pictures embedded there.
IMAGE_EMBED_TEXT_DEVICE = os.environ.get('CFG_IMAGE_EMBED_TEXT_DEVICE', 'cpu')
# Each tower's onnxruntime CUDA arena. The visual export is 1.7 GB of float32 weights and a batch
# of 32 at 384 px peaks at ~2.5 GB of arena; the textual one is 1.2 GB.
IMAGE_EMBED_GPU_MEM_LIMIT_BYTES = int(os.environ.get('CFG_IMAGE_EMBED_GPU_MEM_MB', '3072')) * 1024 * 1024
# Free card memory the picture tower needs to load: it holds 1.9 GB loaded, 2.1 GB after a batch.
SIGLIP_VISUAL_MIN_FREE_MB = float(os.environ.get('CFG_SIGLIP_VISUAL_MIN_FREE_MB', '2600'))

# The copy detector (`fingerprint.compute`'s `sscd-*` kinds, see copy_detect.py): one TorchScript
# file, baked at a pinned sha256. Its name is the producer, so changing it changes every stored
# descriptor's name. auto = the card when it has room; the vectors are float32 either way.
COPY_DETECT_MODEL = os.environ.get('CFG_COPY_DETECT_MODEL', 'sscd_disc_mixup')
COPY_DETECT_ROOT = os.environ.get('CFG_COPY_DETECT_ROOT', '/opt/copy-detect')
COPY_DETECT_DEVICE = os.environ.get('CFG_COPY_DETECT_DEVICE', 'auto')
# A ResNet50 (98 MB of float32 weights) on one picture at 288 px.
SSCD_MIN_FREE_MB = float(os.environ.get('CFG_SSCD_MIN_FREE_MB', '600'))

# The shot detector (`video.shots`, see shots.py): TransNetV2's weights converted to PyTorch, one
# file baked at a pinned sha256. auto = the card when it has room; float32 either way.
TRANSNET_ROOT = os.environ.get('CFG_TRANSNET_ROOT', '/opt/transnet')
TRANSNET_DEVICE = os.environ.get('CFG_TRANSNET_DEVICE', 'auto')
TRANSNET_WEIGHTS = 'transnetv2-pytorch-weights.pth'
# 2.5 M parameters over batches of two 100-frame windows at 48×27 (shots.BATCH): 420 MB reserved
# on fry (2026-10-24).
TRANSNET_MIN_FREE_MB = float(os.environ.get('CFG_TRANSNET_MIN_FREE_MB', '600'))

# The text reader (`ocr.read`, see text_reader.py): PP-OCRv6 det + rec through RapidOCR, ONNX files
# baked at pinned sha256s under OCR_ROOT. The tier is the producer, so changing it changes what
# every stored reading is named. auto = the card when it has room; float32 either way.
OCR_ROOT = os.environ.get('CFG_OCR_ROOT', '/opt/ocr')
OCR_PHOTO_TIER = os.environ.get('CFG_OCR_PHOTO_TIER', 'medium')
# The handwriting recogniser for lines the photo engine reads below its floor (text_reader.HandReader),
# a directory as its authors publish it. Absent, there is no second reading; present, its name joins
# the producer, as `capabilities.models.ocr` says.
OCR_HAND_ROOT = os.environ.get('CFG_OCR_HAND_ROOT', '/opt/ocr/multicentury-ppocrv6-medium')
# The transcriber that writes the text of the lines the handwriting recogniser keeps (htr.py): a
# family of its own (`htr`), loaded when a picture has such lines and given back when idle, so its
# 2.2 GB do not weigh on where the photo engine is placed. Absent, those lines keep the
# recogniser's reading; present, its name joins the producer.
HTR_ROOT = os.environ.get('CFG_HTR_ROOT', '/opt/htr/multicentury-htr-model')
HTR_DEVICE = os.environ.get('CFG_HTR_DEVICE', 'auto')
# 558 M parameters of float32 (2.1 GB on fry's card) and a batch of four lines.
HTR_MIN_FREE_MB = float(os.environ.get('CFG_HTR_MIN_FREE_MB', '2800'))
# The line size the transcriber was trained at, which its config does not say (htr.py).
HTR_LINE_SIZE = (192, 1024)
# The English transcriber (htr.py), a family of its own (`htr-en`) for the same reasons and as large.
# Absent, an English page is read as one in a language without a transcriber.
HTR_EN_ROOT = os.environ.get('CFG_HTR_EN_ROOT', '/opt/htr/trocr-large-handwritten')
HTR_EN_DEVICE = os.environ.get('CFG_HTR_EN_DEVICE', 'auto')
HTR_EN_MIN_FREE_MB = float(os.environ.get('CFG_HTR_EN_MIN_FREE_MB', '2800'))
# The languages each transcriber writes, as the language identifier names them (text_language.py).
# A picture whose language is unknown goes to the first, as if it were Swedish: the archive's.
HTR_LANGUAGES = {"htr": ("sv", "fi"), "htr-en": ("en",)}
OCR_DEVICE = os.environ.get('CFG_OCR_DEVICE', 'auto')
# The longest side the detector sees; a larger picture is scaled down to it. Its arena grows with
# the area: on fry a 2000x1500 picture peaked at 2 GB and a 2000x2000 one went past it.
OCR_MAX_SIDE = int(os.environ.get('CFG_OCR_MAX_SIDE', '1600'))
# Each of its three sessions' CUDA arena (detector, line classifier, recogniser); the detector's
# is the one that fills.
OCR_GPU_MEM_LIMIT_BYTES = int(os.environ.get('CFG_OCR_GPU_MEM_MB', '2048')) * 1024 * 1024
# The detector's full arena, the two recognisers' and the classifier's few hundred MB, and a margin.
OCR_MIN_FREE_MB = float(os.environ.get('CFG_OCR_MIN_FREE_MB', '3000'))

# The text embedder (`text.embed` with `model: current`): jina-embeddings-v5-text-small, a Qwen3-0.6B
# with LoRA task adapters, loaded through its own remote code at a pinned revision so a rebuild
# cannot change the vectors under stored ones; the retrieval adapter, "Query: " / "Document: "
# prompts, last-token pooling, normalized, 1024 dims. MUST match the Containerfile's bake.
TEXT_EMBED_MODEL = os.environ.get('CFG_TEXT_EMBED_MODEL', 'jinaai/jina-embeddings-v5-text-small')
TEXT_EMBED_REVISION = os.environ.get('CFG_TEXT_EMBED_REVISION', 'dd76d535f5447ca3897a9c893fb1e612ead98192')
# Longest input in tokens. The model reads 32k; past this a text is refused rather than silently
# cut, because the caller chunks and a cut chunk is a lie about what was embedded. 8192 keeps a
# batch's attention within a few hundred MB on the card.
TEXT_EMBED_MAX_TOKENS = int(os.environ.get('CFG_TEXT_EMBED_MAX_TOKENS', '8192'))
# Documents (transcripts, OCR text, the command index) go to the card when there is
# one, in float16; the family is not pinned and gives the card back when idle.
TEXT_EMBED_DEVICE = os.environ.get('CFG_TEXT_EMBED_DEVICE', 'auto')
# Search queries have a family of their own, on the CPU by default, for the reason siglip-text
# does: it is on the interactive path, so it is pinned, and pinned on the card it would hold
# ~1.2 GB against the batch families. Its own family also means a query never waits behind a
# backfill batch holding the document family's lock. float32 there; the vectors match the card's
# float16 ones to within the stored halfvec's precision.
TEXT_EMBED_QUERY_DEVICE = os.environ.get('CFG_TEXT_EMBED_QUERY_DEVICE', 'cpu')
# Free card memory the document family needs to load: 1.2 GB of float16 weights plus a batch of
# 16 at up to 8192 tokens. An estimate, not a measurement.
TEXT_EMBED_MIN_FREE_MB = float(os.environ.get('CFG_TEXT_EMBED_MIN_FREE_MB', '2500'))

# Free-VRAM floor below which InsightFace's sessions are recreated to release their arenas
# (see reset_insightface_if_low): the arena is what fills the card during a crawl, so the reset
# is usually the whole fix.
INSIGHTFACE_RESET_FREE_MB = float(os.environ.get('CFG_INSIGHTFACE_RESET_FREE_MB', '2500'))

# A family untouched this long, and not in use, is unloaded to make room for another on the card
# before that one is sent to the CPU (`_short_of_room`).
#
# Short, because a caller that rotates between kinds of work starts the next kind seconds after
# its last request of the previous one. A family whose lock is held or waited on is in use
# whatever its clock says, so the grace only has to cover the gap between one request and the
# next: a picture embedded every second is never unloaded. Pinned families stay, as everywhere.
ROOM_IDLE_SECONDS = float(os.environ.get('CFG_ROOM_IDLE', '30'))

# How often a family the CPU took for want of room on the card looks whether the card has room
# for it now (`_back_to_card`). Not every request: a look that is short of room lets torch give
# back what it holds, which is not free to do per picture.
CARD_RETURN_SECONDS = float(os.environ.get('CFG_CARD_RETURN_SECONDS', '60'))
_card_return_at: dict[str, float] = {}


class CardBusy(RuntimeError):
    """The card has no room now for a family that runs only there (whisper on a host with a card),
    so the caller should retry later."""


def _family_of(key: str) -> str:
    """Cache keys are `family` or `family:device[:compute]` — take the family."""
    return key.split(':', 1)[0]


# Set while the pinned families are loaded or there is nothing to wait for; the resident
# server clears it for its startup preload. Every other family waits for it before loading.
#
# Loading is serialized (`_load_lock`, below), and the preload runs beside the socket, so
# without this the first request after a restart could take the lock for a batch family's
# cold load while every search query timed out behind it. Pinned means on the
# interactive path; it goes first.
pinned_ready = threading.Event()
pinned_ready.set()
PINNED_READY_WAIT_SECONDS = 300.0

_cache: dict[str, object] = {}
_last_used: dict[str, float] = {}
_lock = threading.Lock()

# What `health` says about a family beyond "resident or not", all guarded by `_lock` and
# wall-clock because a person reads them. A cold load takes up to minutes, and without
# `_loading` it is indistinguishable from a worker that has hung; `_unloading` covers the
# reclaim after an eviction, which is what decides when the card is actually free again.
# `_devices` is where a key's model really went when the key does not say so (`insightface`
# picks its provider at load time).
_loading: dict[str, float] = {}
_unloading: dict[str, float] = {}
_loaded_at: dict[str, float] = {}
_devices: dict[str, str] = {}
# What a key cost when it last loaded, kept after it is unloaded so a family that is not resident
# still says how big it is: `vram_mb` is the card's free memory before less after (device-wide, so
# onnxruntime and CTranslate2 count, and the first CUDA load pays for the context too), `ram_mb`
# the process's RSS the same way, and `load_seconds` how long it took. Approximate by nature —
# another family's inference running beside the load moves both numbers — and measured, not
# looked up, because nothing else knows what a quantized checkpoint really costs here.
_sizes: dict[str, dict] = {}

class RequestAbandoned(TimeoutError):
    """A request dropped because its caller no longer waits for it: an outcome, not a fault."""


_request = threading.local()

# How often a request waiting for its family looks at whether its caller still wants it.
ABANDON_POLL_SECONDS = 0.1


@contextmanager
def request_scope(expires_at: float | None, hung_up=None):
    """The request this thread serves: its deadline (epoch seconds) and a check for its caller gone.

    A family lock taken inside the scope gives up on a request nobody is waiting for, so work
    queued behind other work is dropped instead of run for no one. Threads outside a scope
    (preload, eviction) always wait.
    """
    _request.expires_at, _request.hung_up = expires_at, hung_up
    try:
        yield
    finally:
        _request.expires_at, _request.hung_up = None, None


def abandoned() -> str | None:
    """Why this thread's request is no longer wanted, or None while its caller waits."""
    expires_at = getattr(_request, "expires_at", None)
    if expires_at is not None and time.time() >= float(expires_at):
        return "request expired"
    hung_up = getattr(_request, "hung_up", None)
    if hung_up is not None and hung_up():
        return "caller hung up"
    return None


class FamilyLock:
    """A per-family lock with stable ordering inside small explicit priorities.

    A request whose caller gave up (see `request_scope`) leaves the queue with RequestAbandoned,
    and one that reaches the front after that is refused instead of run. Work already holding
    the lock runs to its end: a forward pass cannot be stopped partway.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._held = False
        self._ticket = 0
        self._waiters: list[tuple[int, int, object]] = []

    def acquire(self, priority: int = 0) -> None:
        token = object()
        with self._condition:
            ticket = self._ticket
            self._ticket += 1
            entry = (-priority, ticket, token)
            heapq.heappush(self._waiters, entry)
            while self._held or self._waiters[0][2] is not token:
                reason = abandoned()
                if reason:
                    self._waiters.remove(entry)
                    heapq.heapify(self._waiters)
                    self._condition.notify_all()
                    raise RequestAbandoned(f"{reason} while waiting for the model")
                self._condition.wait(ABANDON_POLL_SECONDS)
            heapq.heappop(self._waiters)
            self._held = True
        reason = abandoned()
        if reason:
            self.release()
            raise RequestAbandoned(f"{reason} before the model ran")

    def busy(self) -> bool:
        """Held, or wanted: a family in this state is in use and is not to be unloaded."""
        with self._condition:
            return self._held or bool(self._waiters)

    def snapshot(self) -> tuple[bool, int]:
        """Whether a request holds the family right now, and how many wait behind it."""
        with self._condition:
            return self._held, len(self._waiters)

    def release(self) -> None:
        with self._condition:
            if not self._held:
                raise RuntimeError("release unlocked FamilyLock")
            self._held = False
            self._condition.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.release()

    @contextmanager
    def hold(self, priority: int = 0):
        self.acquire(priority)
        try:
            yield self
        finally:
            self.release()


# One inference lock per family. GPU work serialises anyway and the card cannot hold parallel
# runs of one family, so concurrent clients queue rather than thrash VRAM. Interactive work may
# choose who acquires next, but never interrupts a run already going.
_family_locks: dict[str, FamilyLock] = {}

# One *construction* lock across all families. Model loading is not thread-safe, whatever the
# family: `transformers.from_pretrained` sets the process-global `torch.set_default_dtype` to
# its `torch_dtype` for the whole of model construction and restores it at the end
# (modeling_utils.py, `_set_default_torch_dtype` -> `set_default_dtype(dtype_orig)`), with no
# lock of its own. Two families loading at once therefore corrupt each other's default dtype:
# a whisper or text-embedder load finishing mid-construction of another family restores the default
# to float32, and every submodule built after that instant is float32 while its inference path
# feeds it half precision -- `Input type (c10::Half) and bias type (float) should be the same`,
# intermittent, surviving in the cache until that copy is evicted.
#
# The window is real because `_get` loads outside `_lock` and each family has its own thread,
# so a search query can land inside another family's load. Serialising loads costs a queued moment on a
# cold family and nothing at all on a cache hit; concurrent loads contend on VRAM
# and PCIe anyway. Distinct from `_lock`: `health` and cache lookups never wait on a load.
_load_lock = threading.Lock()

# libc handle for malloc_trim (glibc only). False once we know it is unavailable.
_libc: object = None

# One long-lived inference thread per family. See run_on_model_thread.
_family_executors: dict[str, object] = {}

# Pseudo-family for reclaim work, which belongs to no model but still calls into CUDA.
_MAINTENANCE = "maintenance"

def family_lock(family: str) -> FamilyLock:
    with _lock:
        return _family_locks.setdefault(family, FamilyLock())


def run_on_model_thread(family: str, fn, *args, **kwargs):
    """Run `fn` on that family's one long-lived thread.

    **No model may ever run on a request thread.** The model server is a
    ThreadingUnixStreamServer, so it hands every request its own OS thread, and both CUDA
    stacks under us attach per-thread state to whatever thread touches them and never release
    it when that thread exits. `malloc_trim` cannot touch any of it: the bytes are still
    *allocated* (glibc `uordblks`), not freed-and-retained (`fordblks`).

    onnxruntime's CUDA provider builds a PerThreadContext (cuBLAS/cuDNN handles, a stream,
    host staging buffers) per thread, a few MB. The torch side is not the same mechanism and is
    much larger, tens of MB per new thread; a plain forward does not carry it, the beam-search
    `generate` path does. `generate` is therefore assumed to leak for *every* model, which is
    why this is applied per family. At a crawl's rate that is gigabytes an hour, until the
    container's OOM killer takes the server.

    Model *loading* has to happen on this thread too, not just inference — the thread that
    builds a session or moves weights onto the card acquires the per-thread state exactly like
    the one that runs it.

    One executor per family rather than one shared: `family_lock` already serialises within a
    family, so a per-family thread costs nothing and never queues behind more than the single
    call that lock admits — while a shared thread would put a face request behind a minutes-long
    transcription.
    """
    # Not only the load waits for the pins (see `_get`): so does everything a family does first,
    # its device probe and first imports included. Two threads importing numpy or torch for the
    # first time at once can leave the module half-initialized for the life of the process —
    # "import numpy failed" and "cannot load module more than once per process" on every
    # request after it.
    if family not in PINNED_FAMILIES:
        pinned_ready.wait(PINNED_READY_WAIT_SECONDS)

    with _lock:
        executor = _family_executors.get(family)
        if executor is None:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"model-{family}")
            _family_executors[family] = executor

    return executor.submit(fn, *args, **kwargs).result()


def select_device() -> str:
    """'cuda' when torch sees a usable GPU, else 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _malloc_trim() -> None:
    """Hand each glibc arena's freed high-water memory back to the OS. No-op off glibc.

    The heavy inference buffers are native (numpy/torch/onnxruntime), so gc frees almost
    nothing; glibc keeps the freed bytes at each arena's high-water mark and only returns
    them on an explicit trim. Without this, the resident server's RSS only ever grows — up
    to the largest transient working set it ever saw — until it trips the container memory cap
    and OOMs.
    """
    global _libc
    if _libc is None:
        import ctypes
        try:
            _libc = ctypes.CDLL("libc.so.6")
        except OSError:
            _libc = False
    if _libc:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


def _empty_cuda_cache() -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def cleanup_torch() -> None:
    """Reclaim memory after model use: gc, free cached VRAM, and trim glibc arenas.

    Only touches torch if it is already imported, so a faces-only (onnxruntime) server
    never pays to load torch just to reclaim.

    The CUDA part goes on a thread of its own. This runs per request, from the request
    thread, and `empty_cache` is a CUDA call like any other — reclaiming on the throwaway
    thread would attach to it exactly the per-thread state run_on_model_thread exists to
    avoid. gc and the trim are thread-agnostic and stay on the caller.
    """
    gc.collect()
    if sys.modules.get("torch") is not None:
        # A cleanup that came *from* the maintenance thread would submit to the single worker
        # it is already running on and wait on itself, so run it inline there.
        if threading.current_thread().name.startswith(f"model-{_MAINTENANCE}"):
            _empty_cuda_cache()
        else:
            run_on_model_thread(_MAINTENANCE, _empty_cuda_cache)
        _forget_own_card_reading()
    _malloc_trim()


def _peek(key: str):
    """Cached model for `key`, touching its idle timer, or None."""
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            _last_used[key] = time.time()
        return cached


def _get(key: str, loader):
    """Return a cached model, loading it via `loader()` on miss."""
    cached = _peek(key)
    if cached is not None:
        return cached
    if _family_of(key) not in PINNED_FAMILIES:
        pinned_ready.wait(PINNED_READY_WAIT_SECONDS)

    # Load outside the registry lock: a load can take minutes and must not block
    # `health` or another family's lookup. Under `_load_lock` though — model construction
    # across families is not thread-safe, see there. Re-check the cache once it is held: a
    # load we queued behind may have been for this very key, and loading it a second time
    # would build a whole second copy just to throw it away below.
    with _load_lock:
        cached = _peek(key)
        if cached is not None:
            return cached
        with _lock:
            _loading[key] = time.time()
        before = _memory_now()
        try:
            model = loader()
        finally:
            with _lock:
                _loading.pop(key, None)
        size = _memory_cost(before, _memory_now())

        with _lock:
            _sizes[key] = size
            _cache[key] = model
            _last_used[key] = _loaded_at[key] = time.time()
        _forget_own_card_reading()
    return model


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/statm") as statm:
            return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576
    except (OSError, ValueError, IndexError):
        return None


def _memory_now() -> dict:
    """Card and process memory right now, for `_memory_cost`. Never imports torch."""
    torch = sys.modules.get("torch")
    allocated = None
    if torch is not None:
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1048576
        except Exception:  # noqa: BLE001 - diagnostics must never break a load
            allocated = None
    return {"at": time.monotonic(), "free_vram": free_vram_mb(), "allocated": allocated, "rss": _rss_mb()}


def _memory_cost(before: dict, after: dict) -> dict:
    """What a load cost between two `_memory_now` readings. See `_sizes`.

    The card's free memory is the measure when both readings have it; a load that imported torch
    itself has no before-reading, and then torch's own allocation (zero before it was imported)
    stands in — it misses onnxruntime's share, which only the families on onnxruntime have.
    """
    size: dict = {"load_seconds": round(after["at"] - before["at"], 1)}
    if before["free_vram"] is not None and after["free_vram"] is not None:
        size["vram_mb"] = max(0, round(before["free_vram"] - after["free_vram"]))
    elif after["allocated"] is not None:
        size["vram_mb"] = max(0, round(after["allocated"] - (before["allocated"] or 0)))
    if before["rss"] is not None and after["rss"] is not None:
        size["ram_mb"] = max(0, round(after["rss"] - before["rss"]))
    return size


def _forget(keys: list[str]) -> None:
    """Drop `keys` from the cache, reporting them as unloading until the reclaim is done.

    The cache entry goes first, under `_lock`, so no new request picks up a model being freed;
    the memory itself only returns with `cleanup_torch`, and `health` says so meanwhile.
    """
    now = time.time()
    with _lock:
        for key in keys:
            if _cache.pop(key, None) is not None:
                _unloading[key] = now
            _last_used.pop(key, None)
            _loaded_at.pop(key, None)
    try:
        cleanup_torch()
    finally:
        with _lock:
            for key in keys:
                _unloading.pop(key, None)
                _devices.pop(key, None)
        _forget_own_card_reading()


def forget_copies(family: str, device: str) -> None:
    """Unload `family`'s copies on `device`. For a caller on the family's model thread, inside its
    lock, so no request is using them."""
    with _lock:
        keys = [key for key in _cache if _family_of(key) == family and _device_of(key) == device]
    if keys:
        _forget(keys)


def evict(key: str) -> None:
    """Drop one family from the cache (used on CUDA OOM retry)."""
    _forget([key])


def unload(families: list[str] | None = None, device: str | None = None, wait: bool = False) -> list[str]:
    """Unload resident copies now. Returns their keys.

    Without `families`, every copy that is not pinned or in use (`evict_idle` at zero). With them,
    those families' copies, only those on `device` if given; `wait` unloads a family in use once
    its work is done instead of leaving it.
    """
    if families is None:
        return evict_idle(0)
    unloaded: list[str] = []
    for family in families:
        keys = [key for key in loaded() if _family_of(key) == family and (device is None or key.split(':')[1:2] == [device])]
        if not keys:
            continue
        lock = family_lock(family)
        if wait:
            with lock:
                _forget(keys)
        elif lock.busy():
            continue
        else:
            _forget(keys)
        unloaded += keys
    return unloaded


def evict_idle(max_idle: float = IDLE_EVICT_SECONDS, keep: tuple[str, ...] = ()) -> list[str]:
    """Unload resident families untouched for `max_idle` seconds. Returns their keys.

    Families in PINNED_FAMILIES are never evicted — see the note there — nor those in `keep`, nor
    one whose family lock is held or waited on right now.
    """
    now = time.time()
    with _lock:
        stale = [
            k for k, seen in _last_used.items()
            if now - seen > max_idle and _family_of(k) not in PINNED_FAMILIES and _family_of(k) not in keep
            and not (_family_of(k) in _family_locks and _family_locks[_family_of(k)].busy())
        ]
    if stale:
        _forget(stale)
    return stale


def loaded() -> list[str]:
    with _lock:
        return sorted(_cache)


def catalog() -> dict[str, dict]:
    """Every family this worker can load: its model, configured device, purpose and the ops using it.

    So `health` can list a model that is not resident, not only the ones that are, and say what
    each is for to someone who did not choose it. The device is the setting (`auto`, `cpu`,
    `cuda`), not a probe: answering it must not import torch.
    """
    return {
        "insightface": {
            "model": "insightface/buffalo_l", "device": "auto",
            "purpose": "Finds faces in photos and video frames and turns each into a fingerprint, so the same person "
                       "can be recognised across the archive.",
            "ops": ["faces.detect", "faces.detect_video", "faces.embed", "speech.active_speakers"],
        },
        "whisper": {
            "model": WHISPER_MODEL, "device": "auto",
            "purpose": "Speech recognition: transcribes what is said in videos and recordings, detecting the language.",
            "ops": ["speech.transcribe"],
        },
        "sortformer": {
            "model": SORTFORMER_MODEL, "device": "auto",
            "purpose": "Speaker diarization: splits a recording into turns by who is speaking, overlapping speech included.",
            "ops": ["speech.diarize", "speech.voiceprint"],
        },
        "titanet": {
            "model": TITANET_MODEL, "device": TITANET_DEVICE,
            "purpose": "Voiceprints: a fingerprint per speech turn, so a voice can be matched to a person.",
            "ops": ["speech.voiceprint"],
        },
        "asd": {
            "model": f"LR-ASD/{LR_ASD_WEIGHTS}", "device": LR_ASD_DEVICE,
            "purpose": "Active speaker detection: scores whether a face on screen is the one talking, to tie a voice to a face.",
            "ops": ["speech.active_speaker", "speech.active_speakers"],
        },
        "siglip-visual": {
            "model": IMAGE_EMBED_MODEL, "device": IMAGE_EMBED_DEVICE,
            "purpose": "Picture embeddings: places each photo where similar pictures, and pictures matching a text, "
                       "lie close — picture search and \"more like this\".",
            "ops": ["vision.embed"],
        },
        "siglip-text": {
            "model": IMAGE_EMBED_MODEL, "device": IMAGE_EMBED_TEXT_DEVICE,
            "purpose": "The text side of picture search: turns a search query into the same space as the pictures.",
            "ops": ["vision.embed_text"],
        },
        "sscd": {
            "model": COPY_DETECT_MODEL, "device": COPY_DETECT_DEVICE,
            "purpose": "Copy detection: a fingerprint per picture or video keyframe that stays close for a resized, cropped, "
                       "edited or rescanned copy and apart for another photo of the same scene — duplicates and photo stacks.",
            "ops": ["fingerprint.compute"],
        },
        "transnet": {
            "model": "transnetv2", "device": TRANSNET_DEVICE,
            "purpose": "Shot boundaries: where a film or a tape cuts from one shot to the next, for a video's chapters.",
            "ops": ["video.shots"],
        },
        "htr": {
            "model": os.path.basename(os.path.normpath(HTR_ROOT)), "device": HTR_DEVICE,
            "purpose": "Writes the text of handwritten lines in pictures — letters, captions on photo cards — which "
                       "the text reader found and judged to be writing.",
            "ops": ["ocr.read", "document.extract"],
        },
        "htr-en": {
            "model": os.path.basename(os.path.normpath(HTR_EN_ROOT)), "device": HTR_EN_DEVICE,
            "purpose": "Writes the text of handwritten lines in English pictures, which the text reader found and "
                       "judged to be writing.",
            "ops": ["ocr.read", "document.extract"],
        },
        "text-language": {
            "model": "lingua", "device": "cpu",
            "purpose": "Tells which language the text read in a picture is in, so its handwriting goes to a "
                       "transcriber for that language.",
            "ops": ["ocr.read", "document.extract"],
        },
        "ocr": {
            "model": ocr_model(),
            "device": OCR_DEVICE,
            "purpose": "Reads text in pictures — signs, shop fronts, photographed pages — line by line with where each "
                       "stands, so a picture is found by the words in it.",
            "ops": ["ocr.read"],
        },
        "text-embed": {
            "model": TEXT_EMBED_MODEL, "device": TEXT_EMBED_DEVICE,
            "purpose": "Text embeddings for documents, text in pictures and transcripts, so text search finds meaning rather "
                       "than exact words.",
            "ops": ["text.embed"],
        },
        "text-query": {
            "model": TEXT_EMBED_MODEL, "device": TEXT_EMBED_QUERY_DEVICE,
            "purpose": "The same text embedder for search queries, kept loaded on its own so a search never waits "
                       "for a load or behind a batch.",
            "ops": ["text.embed"],
        },
    }


def _device_of(key: str) -> str | None:
    """`cuda` or `cpu` from a `family:device[:compute]` key, or what its loader recorded."""
    parts = key.split(':')
    return parts[1] if len(parts) > 1 else _devices.get(key)


def family_states() -> dict[str, dict]:
    """Every family in `catalog()`, and any other this worker holds, with each resident copy.

    A family can have more than one copy (`insightface` beside `insightface:cpu` after a CUDA OOM), so
    the state is per copy: `loading`, `loaded` or `unloading`, since when, and when it was last
    used. No copies means not loaded. `running` is a request holding the family's lock and
    `waiting` how many queue behind it — the difference between a model that is resident and
    one that is working.
    """
    with _lock:
        copies: dict[str, dict] = {}
        for key, since in _loading.items():
            copies[key] = {"state": "loading", "since": round(since, 1)}
        for key in _cache:
            copies[key] = {"state": "loaded", "since": round(_loaded_at[key], 1) if key in _loaded_at else None}
            if key in _last_used:
                copies[key]["last_used"] = round(_last_used[key], 1)
        for key, since in _unloading.items():
            copies[key] = {"state": "unloading", "since": round(since, 1)}
        devices = {key: _device_of(key) for key in copies}
        locks = dict(_family_locks)
        sizes = {key: dict(size) for key, size in _sizes.items()}

    families = {name: {**info, "pinned": name in PINNED_FAMILIES, "copies": []} for name, info in catalog().items()}
    for key in sorted(copies):
        family = _family_of(key)
        entry = families.setdefault(family, {"model": None, "device": None, "pinned": family in PINNED_FAMILIES, "copies": []})
        copy = {"key": key, **copies[key], **sizes.get(key, {})}
        if devices[key]:
            copy["device"] = devices[key]
        entry["copies"].append(copy)
    # What each key cost when it last loaded, resident or not — the size of a model not loaded now.
    for key, size in sorted(sizes.items()):
        family = _family_of(key)
        if family in families:
            families[family].setdefault("measured", {})[key] = size
    for family, lock in locks.items():
        if family in families:
            held, waiting = lock.snapshot()
            families[family]["running"] = held
            families[family]["waiting"] = waiting
    return families


# --- Model families -------------------------------------------------------------

def evict_insightface() -> None:
    """Drop the InsightFace session so the next call re-creates it (CUDA OOM retry)."""
    evict("insightface")
    evict("insightface:cpu")


# The most of the card this worker treats as there, in MB; 0 means the whole card. A smaller
# number makes it behave as it would on a smaller card — how an 8 or 12 GB card is rehearsed on a
# 16 GB one — and leaves the rest to whatever else shares the card.
CARD_BUDGET_MB = float(os.environ.get('CFG_CARD_BUDGET_MB', '0'))

# The most of the card EMS's own processes hold, in MB; 0 means all of it (EMS_CARD_SHARE_MB). A
# host whose card another service shares sets it to what that service leaves EMS, so a family
# never loads into the other's part while it happens to be free — the llm restarting, say.
CARD_SHARE_MB = float(os.environ.get('CFG_CARD_SHARE_MB', '0'))
_OWN_CACHE_SECONDS = 5.0
_own_cache: dict = {"at": float("-inf"), "mb": None}


def _forget_own_card_reading() -> None:
    """Have the next `_own_card_mb` ask `nvidia-smi` again: a family has just loaded or unloaded.
    A reading from before an unload still counts what was unloaded, and the family that unloaded
    it to make room would go to the CPU as if it had not."""
    _own_cache["at"] = float("-inf")


def _own_card_mb() -> float | None:
    """What this container's processes hold of the card, in MB, or None if `nvidia-smi` cannot say.

    `nvidia-smi` in a container lists only that container's processes, which is what makes this
    EMS's own use and not the card's. One reading serves for a few seconds.
    """
    now = time.monotonic()
    if now - _own_cache["at"] < _OWN_CACHE_SECONDS:
        return _own_cache["mb"]
    mb = None
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            done = subprocess.run([smi, "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=5)
            if done.returncode == 0:
                mb = sum(float(line) for line in done.stdout.split() if line.replace(".", "", 1).isdigit())
        except (OSError, subprocess.TimeoutExpired):
            mb = None
    _own_cache.update(at=now, mb=mb)
    return mb


def _card_mem_mb() -> tuple[float, float] | None:
    """(free, total) on the device in MB, as CARD_BUDGET_MB and CARD_SHARE_MB let this worker see it.

    Device-level rather than torch-level: the whole point is to see memory held by onnxruntime
    and CTranslate2, which torch's own accounting cannot see. None if torch is not loaded or
    there is no GPU.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, total = (value / 1048576 for value in torch.cuda.mem_get_info())
    except Exception:  # noqa: BLE001 - diagnostics must never break the caller
        return None
    if 0 < CARD_BUDGET_MB < total:
        used = total - free
        free, total = max(0.0, CARD_BUDGET_MB - used), CARD_BUDGET_MB
    if 0 < CARD_SHARE_MB < total:
        own = _own_card_mb()
        if own is not None:
            free, total = max(0.0, min(free, CARD_SHARE_MB - own)), CARD_SHARE_MB
    return free, total


def free_vram_mb() -> float | None:
    """Free VRAM on the device (within CARD_BUDGET_MB), or None if torch is not loaded or there
    is no GPU."""
    mem = _card_mem_mb()
    return None if mem is None else mem[0]


def card_total_mb() -> float | None:
    """The card's size (within CARD_BUDGET_MB), or None without one."""
    mem = _card_mem_mb()
    return None if mem is None else mem[1]


# What a family needs free on the card to load and work there, in MB: what it holds plus its
# working memory, measured on fry (wow notes/host-and-runtime-limits.md).
CARD_NEED_MB = {
    "insightface": INSIGHTFACE_RESET_FREE_MB,
    "siglip-visual": SIGLIP_VISUAL_MIN_FREE_MB,
    "sscd": SSCD_MIN_FREE_MB,
    "transnet": TRANSNET_MIN_FREE_MB,
    "ocr": OCR_MIN_FREE_MB,
    "htr": HTR_MIN_FREE_MB,
    "htr-en": HTR_EN_MIN_FREE_MB,
    "text-embed": TEXT_EMBED_MIN_FREE_MB,
    "sortformer": float(os.environ.get('CFG_SORTFORMER_MIN_FREE_MB', '1500')),
    "titanet": float(os.environ.get('CFG_TITANET_MIN_FREE_MB', '500')),
    "asd": float(os.environ.get('CFG_LR_ASD_MIN_FREE_MB', '300')),
    # large-v3 in float16, beam 5: ~4.0 GB on fry while transcribing (2026-10-10). int8 holds
    # about half the weights.
    "whisper": float(os.environ.get('CFG_WHISPER_MIN_FREE_MB', '4000' if WHISPER_COMPUTE["cuda"] == 'float16' else '2400')),
}


# A worker process's CUDA context, which no family's figure counts.
CUDA_CONTEXT_MB = 400.0


def _committed_mb() -> float:
    """What the families on the card have claimed there, in MB.

    Each resident family claims its need — what it holds and what it works with — not what it
    holds when measured between requests: two families each within the card's free memory at
    load time still ran it out of memory at the same moment, mid-request.
    """
    with _lock:
        families = {_family_of(key) for key in _cache if _device_of(key) == "cuda"}
    return sum(CARD_NEED_MB.get(name, 0.0) for name in families)


def _room_mb() -> float | None:
    """What is left of EMS's part of the card once every claim is counted, in MB — what it holds
    and what is free to it (within EMS_CARD_SHARE_MB), less the context and the claims — or None
    when `nvidia-smi` cannot say what EMS holds."""
    free = free_vram_mb()
    own = _own_card_mb()
    if free is None or own is None:
        return None
    return own + free - CUDA_CONTEXT_MB - _committed_mb()


def place(family: str, setting: str = "auto") -> str:
    """Where `family` runs this time: `cuda` or `cpu`.

    A setting that names a device is obeyed. With `auto` the card is used when it has the family's
    need free now and left after every claim on it (`_room_mb`) — after unloading families idle
    past the grace — and the CPU otherwise. Every family placed here runs the same weights on either
    device, so where it lands changes only how fast it is, and a family never loads onto a card
    that cannot hold it: no out-of-memory, on a 16 GB card or an 8 GB one (ein/core#515).
    A family already resident stays where it is rather than load a second copy — except one on the
    CPU, which goes back to the card once the card has room for it again (`_back_to_card`).

    Runs on the family's own model thread: `select_device` is its first CUDA call.
    """
    if setting in ("cpu", "cuda"):
        return setting
    if select_device() != "cuda":
        return "cpu"
    with _lock:
        resident = [key for key in _cache if _family_of(key) == family]
    if resident:
        device = _device_of(resident[0]) or "cuda"
        if device == "cpu" and _back_to_card(family, resident):
            return "cuda"
        return device
    why = _short_of_room(family)
    if why is not None:
        print(f"[model-server] {family}: {why} — on the CPU", file=sys.stderr, flush=True)
        return "cpu"
    return "cuda"


def _back_to_card(family: str, resident: list[str]) -> bool:
    """Whether `family`, resident on the CPU because the card had no room when it loaded, moves to the
    card now: once a `CARD_RETURN_SECONDS`, it looks whether the card has room for it — after
    unloading families idle past the grace — and if so its CPU copies are unloaded for the card's
    to load. A family in use the whole of a crawl is never idle, so without this it never left the
    CPU: on fry, faces ran there for ten hours beside an idle card (ein/core#736).

    Runs on the family's own model thread, inside its lock, so no request is using the copies it
    unloads."""
    now = time.monotonic()
    if now - _card_return_at.get(family, float("-inf")) < CARD_RETURN_SECONDS:
        return False
    _card_return_at[family] = now
    if _short_of_room(family) is not None:
        return False
    forget_copies(family, "cpu")
    print(f"[model-server] {family}: the card has room again — back from the CPU", file=sys.stderr, flush=True)
    return True


def _short_of_room(family: str) -> str | None:
    """Why the card cannot take `family` now, or None when it can — after letting torch give back
    what it holds unused and unloading families idle past the grace (`ROOM_IDLE_SECONDS`)."""
    need = CARD_NEED_MB.get(family)
    if need is None:
        return None

    def short() -> str | None:
        free, room = free_vram_mb(), _room_mb()
        if free is not None and free < need:
            return f"{free:.0f}MB free on the card, needs {need:.0f}MB"
        if room is not None and room < need:
            return f"{room:.0f}MB of the card left after what the families on it claim, needs {need:.0f}MB"
        return None

    if short() is None:
        return None
    cleanup_torch()
    if short() is not None:
        evicted = evict_idle(ROOM_IDLE_SECONDS, keep=(family,))
        if evicted:
            print(f"[model-server] {family}: unloaded idle {', '.join(evicted)} to load on the card", file=sys.stderr, flush=True)
    return short()


def whisper_device() -> str:
    """Where the next transcription runs: the card on a host that has one, the CPU on one that has
    not (WHISPER_COMPUTE). A card without room now raises `CardBusy`: the caller waits rather than
    have this transcript decoded somewhere else than every other on this host. Runs on whisper's
    own model thread, like `place`."""
    if select_device() != "cuda":
        return "cpu"
    with _lock:
        resident = any(_family_of(key) == "whisper" and _device_of(key) == "cuda" for key in _cache)
    why = None if resident else _short_of_room("whisper")
    if why is not None:
        raise CardBusy(f"whisper waits for the card: {why}")
    return "cuda"


def reset_insightface_if_low() -> float | None:
    """Recreate InsightFace's sessions when the device is short on VRAM.

    onnxruntime's CUDA arena grows and never returns memory, and `gpu_mem_limit` cannot
    bound it usefully here: the limit is per-session and buffalo_l loads five of them, so a
    cap large enough for each is five times too large in aggregate, while a cap small enough
    in aggregate starves one session mid-inference. The arenas grow by tens of MB per image
    until the card is full, at which point a CUDA OOM evicts the other models *and* drops
    InsightFace onto the CPU provider, and indexing slows to a fraction with the GPU idle.

    Dropping the sessions frees every arena at once; the next call rebuilds them in ~1-2s,
    which amortises to nothing over the hundreds of images between resets. Returns the free
    VRAM that triggered a reset, or None when nothing was done.
    """
    free_mb = free_vram_mb()
    if free_mb is None or free_mb >= INSIGHTFACE_RESET_FREE_MB:
        return None
    # Hold the family lock so no request is mid-inference on a session being dropped.
    with family_lock("insightface"):
        if "insightface" not in loaded() and "insightface:cpu" not in loaded():
            return None
        evict_insightface()
    cleanup_torch()
    return free_mb


def insightface(force_cpu: bool = False):
    """InsightFace buffalo_l analyser (512-dim embeddings).

    `force_cpu` pins it to the CPU provider — used after a CUDA OOM.
    """
    key = "insightface:cpu" if force_cpu else "insightface"

    def load():
        import sys
        import onnxruntime
        from insightface.app import FaceAnalysis

        # Prefer the GPU execution provider when onnxruntime-gpu finds CUDA, always keep
        # CPU as fallback. ctx_id selects the GPU device (>= 0) or CPU (< 0).
        use_cuda = not force_cpu and 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
        # Bound the CUDA arena. onnxruntime defaults to kNextPowerOfTwo with no limit and
        # never returns the memory, so sustained face detection walks the card, tens of MB per
        # image. At the top it OOMs, which evicts the other models *and* reloads InsightFace on
        # the CPU provider — one arena taking down both.
        # kSameAsRequested stops the doubling; the cap is a backstop well above the ~1-2GB
        # this model actually needs.
        cuda_opts = {
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': str(INSIGHTFACE_GPU_MEM_LIMIT_BYTES),
        }
        providers = ([('CUDAExecutionProvider', cuda_opts)] if use_cuda else []) + ['CPUExecutionProvider']
        ctx_id = 0 if use_cuda else -1
        root = os.environ.get('INSIGHTFACE_HOME', os.path.expanduser('~/.insightface'))

        # FaceAnalysis prints "Applied providers..." to stdout; keep it out of the worker's output.
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            app = FaceAnalysis(name='buffalo_l', root=root, providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
        with _lock:
            _devices[key] = "cuda" if use_cuda else "cpu"
        return app

    return _get(key, load)


def text_embed_device() -> str:
    return place("text-embed", TEXT_EMBED_DEVICE)


def text_embed_query_device() -> str:
    return select_device() if TEXT_EMBED_QUERY_DEVICE == 'auto' else TEXT_EMBED_QUERY_DEVICE


def text_embedder(family: str, device: str):
    """TEXT_EMBED_MODEL through sentence-transformers, keyed by family so queries load their own."""
    key = f"{family}:{device}"

    def load():
        import torch
        from sentence_transformers import SentenceTransformer
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = SentenceTransformer(
            TEXT_EMBED_MODEL, revision=TEXT_EMBED_REVISION, device=device, trust_remote_code=True,
            model_kwargs={"dtype": dtype, "default_task": "retrieval"},
        )
        model.max_seq_length = TEXT_EMBED_MAX_TOKENS
        return model

    return _get(key, load)


def image_embed_device() -> str:
    return place("siglip-visual", IMAGE_EMBED_DEVICE)


def image_embed_text_device() -> str:
    """Where search queries are embedded: IMAGE_EMBED_TEXT_DEVICE, `auto` following the pictures."""
    return select_device() if IMAGE_EMBED_TEXT_DEVICE == "auto" else IMAGE_EMBED_TEXT_DEVICE


def image_embed_root() -> str:
    return os.path.join(IMAGE_EMBED_ROOT, IMAGE_EMBED_MODEL)


def siglip_visual(device: str):
    """The picture tower of IMAGE_EMBED_MODEL (image_embed.Visual)."""
    key = f"siglip-visual:{device}"

    def load():
        from image_embed import Visual
        return Visual(image_embed_root(), device, IMAGE_EMBED_GPU_MEM_LIMIT_BYTES)

    return _get(key, load)


def copy_detect_device() -> str:
    return place("sscd", COPY_DETECT_DEVICE)


def sscd(device: str):
    """COPY_DETECT_MODEL's descriptor (copy_detect.Descriptor)."""
    key = f"sscd:{device}"

    def load():
        from copy_detect import Descriptor
        return Descriptor(os.path.join(COPY_DETECT_ROOT, f"{COPY_DETECT_MODEL}.torchscript.pt"), device)

    return _get(key, load)


def transnet_device() -> str:
    return place("transnet", TRANSNET_DEVICE)


def transnet(device: str):
    """TransNetV2's cut scorer (shots.ShotScorer)."""
    key = f"transnet:{device}"

    def load():
        from shots import ShotScorer
        return ShotScorer(os.path.join(TRANSNET_ROOT, TRANSNET_WEIGHTS), device)

    return _get(key, load)


def ocr_model() -> str:
    """The text reader's producer: the photo engine's tier, then each later reader the image has."""
    readers = [os.path.basename(os.path.normpath(root)) for root in (OCR_HAND_ROOT, HTR_ROOT, HTR_EN_ROOT) if os.path.isdir(root)]
    return "+".join([f"PP-OCRv6_{OCR_PHOTO_TIER}", *readers])


# Each transcriber family: its directory, its device setting, and the line size it reads at.
_HTR = {"htr": (lambda: HTR_ROOT, lambda: HTR_DEVICE, HTR_LINE_SIZE), "htr-en": (lambda: HTR_EN_ROOT, lambda: HTR_EN_DEVICE, None)}


def htr_family(language: str | None) -> str | None:
    """The transcriber family in the image that writes `language` (None: unknown), or None."""
    family = next((name for name, languages in HTR_LANGUAGES.items() if language in languages), None if language else "htr")
    return family if family and os.path.isdir(_HTR[family][0]()) else None


def htr_device(family: str = "htr") -> str:
    return place(family, _HTR[family][1]())


def htr(device: str, family: str = "htr"):
    """A handwriting transcriber (htr.LineTranscriber)."""
    key = f"{family}:{device}"
    root, _, size = _HTR[family]

    def load():
        from htr import LineTranscriber
        return LineTranscriber(root(), device, size)

    return _get(key, load)


def text_language():
    """The language identifier for read text (text_language.LanguageIdentifier), on the CPU."""
    def load():
        from text_language import LanguageIdentifier
        return LanguageIdentifier()

    return _get("text-language:cpu", load)


def ocr_device() -> str:
    return place("ocr", OCR_DEVICE)


def ocr_photo(device: str):
    """The photo engine at OCR_PHOTO_TIER (text_reader.PhotoReader)."""
    key = f"ocr:{device}"

    def load():
        from text_reader import PhotoReader
        hand = OCR_HAND_ROOT if os.path.isdir(OCR_HAND_ROOT) else None
        return PhotoReader(OCR_ROOT, OCR_PHOTO_TIER, device, OCR_GPU_MEM_LIMIT_BYTES, OCR_MAX_SIDE, hand)

    return _get(key, load)


def siglip_text(device: str):
    """The text tower of IMAGE_EMBED_MODEL (image_embed.Textual), for search queries."""
    key = f"siglip-text:{device}"

    def load():
        from image_embed import Textual
        return Textual(image_embed_root(), device, IMAGE_EMBED_GPU_MEM_LIMIT_BYTES)

    return _get(key, load)


def whisper(device: str, compute_type: str):
    """faster-whisper model named by WHISPER_MODEL."""
    key = f"whisper:{device}:{compute_type}"

    def load():
        from faster_whisper import WhisperModel
        return WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)

    return _get(key, load)


def sortformer(device: str):
    """Sortformer speaker diarizer named by SORTFORMER_MODEL.

    Same baked-model rule as WHISPER_MODEL: this MUST match the Containerfile or the first
    diarization pulls the checkpoint at runtime, with no network guarantee. End-to-end, and it models overlapping speech rather than assuming one voice
    at a time — which is why the transcript's own whisper segmentation is not reused as a
    speaker boundary.
    """
    key = f"sortformer:{device}"

    def load():
        from nemo.collections.asr.models import SortformerEncLabelModel

        model = SortformerEncLabelModel.from_pretrained(SORTFORMER_MODEL)
        model.eval()
        return model.to(device)

    return _get(key, load)


def titanet(device: str):
    """TitaNet speaker-verification model named by TITANET_MODEL. 192-dim embeddings."""
    key = f"titanet:{device}"

    def load():
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        model = EncDecSpeakerLabelModel.from_pretrained(TITANET_MODEL)
        model.eval()
        return model.to(device)

    return _get(key, load)


def titanet_device() -> str:
    """Device for the voiceprint family, honouring TITANET_DEVICE. See the note there."""
    return place("titanet", TITANET_DEVICE)


def lr_asd_device() -> str:
    """Device for the active-speaker family, honouring LR_ASD_DEVICE. See the note there."""
    return place("asd", LR_ASD_DEVICE)


def sortformer_device() -> str:
    return place("sortformer")


def insightface_device() -> str:
    return place("insightface")


def lr_asd(device: str):
    """LR-ASD active-speaker model and its scoring head. Returns (model, head).

    The head comes back separately because the checkpoint is a *training* state dict: it holds
    the backbone under `model.` and the two-class classifier under `lossAV.FC`, and the
    repo's own `ASD` wrapper that owns both also pulls in an optimizer, a scheduler and pandas
    on import. Loading the two pieces directly keeps inference free of all of that, and the
    score is then `softmax(head(outsAV))[:, 1]` — the same quantity `ASD.evaluate_network`
    reads, without the training machinery around it.
    """
    key = f"asd:{device}"

    def load():
        import sys
        import torch

        if LR_ASD_HOME not in sys.path:
            sys.path.insert(0, LR_ASD_HOME)
        from model.Model import ASD_Model

        weights = os.path.join(LR_ASD_HOME, 'weight', LR_ASD_WEIGHTS)
        state = torch.load(weights, map_location='cpu')

        model = ASD_Model()
        backbone = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
        # strict=False: the checkpoint carries the loss heads too, which are not this module's.
        model.load_state_dict(backbone or state, strict=False)

        head = torch.nn.Linear(128, 2)
        head.load_state_dict({'weight': state['lossAV.FC.weight'], 'bias': state['lossAV.FC.bias']})

        return model.eval().to(device), head.eval().to(device)

    return _get(key, load)
