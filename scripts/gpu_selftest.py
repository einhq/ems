#!/usr/bin/env python3
"""gpu-selftest — prove the inference stack really runs on the GPU inside the EMS image.

The Containerfile's build-time check only asserts that CUDAExecutionProvider is *listed*.
That says nothing about whether kernels exist for the host's compute capability: on a
Blackwell card (sm_120) a cu124 torch imports fine, reports cuda available, and then dies
on the first real kernel launch. InsightFace is worse — it silently falls back to CPU and
you only notice via throughput.

So every check here runs an actual computation and inspects what the loaded model/session
ended up on. Every check runs, and it exits non-zero if any of them failed.

Run inside the running container with the GPU passed through (`make gpu-check`):

    podman exec ems python3 scripts/gpu_selftest.py

--allow-cpu-faces downgrades a CPU-bound InsightFace from failure to warning (some
onnxruntime-gpu builds ship no sm_120 kernels; faces still work, just slower).
"""

import argparse
import sys
import traceback

import numpy as np

FAILURES: list[str] = []
WARNINGS: list[str] = []
SKIPPED: list[str] = []
ONLY: list[str] = []


def check(name):
    """Run a check, turning any exception into a recorded failure.

    Honours --only, which exists because the checks are not equal in cost: on a card that is
    already serving a crawl the whole suite may not fit, while verifying one model against the
    same image can.
    """

    def decorator(fn):
        if ONLY and not any(token.lower() in name.lower() for token in ONLY):
            SKIPPED.append(name)
            return fn
        print(f'\n=== {name} ===')
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a failed check must not stop the rest
            FAILURES.append(f'{name}: {exc}')
            print(f'FAIL {name}: {exc}')
            traceback.print_exc(limit=3)
        return fn

    return decorator


def synthetic_face_image() -> np.ndarray:
    """A crude frontal face on a plain background — enough to exercise the detector.

    Detection is not asserted (a synthetic blob may find zero faces); what matters is that
    the session runs on CUDA without raising.
    """
    import cv2

    img = np.full((480, 480, 3), 200, dtype=np.uint8)
    cv2.ellipse(img, (240, 250), (110, 145), 0, 0, 360, (170, 150, 135), -1)
    cv2.circle(img, (200, 215), 14, (60, 60, 60), -1)
    cv2.circle(img, (280, 215), 14, (60, 60, 60), -1)
    cv2.ellipse(img, (240, 320), (45, 22), 0, 0, 180, (90, 70, 70), -1)
    return img


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-cpu-faces', action='store_true',
                        help='treat an InsightFace CPU fallback as a warning, not a failure')
    parser.add_argument('--only', default='',
                        help='comma-separated substrings of check names to run (e.g. "nemo,lr-asd"); '
                             'this is how one model is verified on a card that is already busy')
    args = parser.parse_args()
    ONLY.extend(token.strip() for token in args.only.split(',') if token.strip())

    @check('torch / CUDA')
    def _torch():
        import torch

        print('torch:', torch.__version__, '| built for CUDA:', torch.version.cuda)
        if not torch.cuda.is_available():
            raise RuntimeError('torch.cuda.is_available() is False — no device passed through?')

        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        arch_list = torch.cuda.get_arch_list()
        print(f'device: {name} (sm_{major}{minor})')
        print('kernels built for:', ' '.join(arch_list))

        if f'sm_{major}{minor}' not in arch_list:
            # PTX for a lower arch can JIT forward, so this is not fatal on its own —
            # the matmul below is the real verdict.
            print(f'WARN: no native sm_{major}{minor} cubin; relying on PTX JIT')

        # The actual test: a kernel launch. A wheel without kernels for this arch raises
        # "CUDA error: no kernel image is available for execution on the device" here.
        a = torch.randn(512, 512, device='cuda')
        result = (a @ a.T).sum().item()
        torch.cuda.synchronize()
        if not np.isfinite(result):
            raise RuntimeError(f'cuda matmul produced a non-finite result: {result}')
        print(f'cuda matmul OK (checksum {result:.3f})')
        print(f'VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    @check('onnxruntime / InsightFace')
    def _insightface():
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        providers = ort.get_available_providers()
        print('onnxruntime:', ort.__version__, '| providers:', providers)
        if 'CUDAExecutionProvider' not in providers:
            raise RuntimeError(f'CUDAExecutionProvider missing — CPU wheel shadowing? {providers}')

        app = FaceAnalysis(name='buffalo_l', root='/opt/insightface',
                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Ask the loaded sessions what they actually bound to, rather than trusting the
        # requested provider list — this is where a silent CPU fallback shows up.
        bound = {name: model.session.get_providers()[0] for name, model in app.models.items()}
        for name, provider in sorted(bound.items()):
            print(f'  {name}: {provider}')

        faces = app.get(synthetic_face_image())
        print(f'detection ran, {len(faces)} face(s) on the synthetic image')

        cpu_only = [n for n, p in bound.items() if p != 'CUDAExecutionProvider']
        if cpu_only:
            msg = f'InsightFace models on CPU: {cpu_only} (no sm_120 kernels in this onnxruntime-gpu build?)'
            if args.allow_cpu_faces:
                WARNINGS.append(msg)
                print('WARN:', msg)
            else:
                raise RuntimeError(msg)

    @check('jina-v5-text embed (card float16 against CPU float32)')
    def _text_embed():
        import os

        import torch
        from sentence_transformers import SentenceTransformer

        repo = os.environ.get('CFG_TEXT_EMBED_MODEL', 'jinaai/jina-embeddings-v5-text-small')
        revision = os.environ.get('CFG_TEXT_EMBED_REVISION')
        texts = ['en segelbåt vid bryggan', 'Document text about a birthday cake']
        vectors = {}
        for device, dtype in (('cuda', torch.float16), ('cpu', torch.float32)):
            model = SentenceTransformer(repo, revision=revision, device=device, trust_remote_code=True,
                                        model_kwargs={'dtype': dtype, 'default_task': 'retrieval'})
            vectors[device] = model.encode(texts, prompt_name='document', normalize_embeddings=True)
            del model
            torch.cuda.empty_cache()
        agreement = (vectors['cuda'] * vectors['cpu']).sum(axis=1).min()
        print('embedding dim:', vectors['cuda'].shape[-1], f'| card/CPU cosine {agreement:.6f}')
        # A query embedded on the CPU searches documents embedded on the card (model_registry).
        if agreement < 0.999:
            raise RuntimeError(f'card and CPU vectors disagree (cosine {agreement:.6f})')

    @check('SigLIP2 image-text embed')
    def _image_embed():
        import os
        import sys

        from PIL import Image

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ems', 'modelworker'))
        from image_embed import Textual, Visual

        root = os.path.join(os.environ.get('CFG_IMAGE_EMBED_ROOT', '/opt/image-embed'),
                            os.environ.get('CFG_IMAGE_EMBED_MODEL', 'ViT-SO400M-16-SigLIP2-384__webli'))
        visual, textual = Visual(root, 'cuda', 3 << 30), Textual(root, 'cuda', 3 << 30)
        if 'CUDAExecutionProvider' not in visual.session.get_providers():
            raise RuntimeError('the visual tower stayed on the CPU provider')
        # A real, if tiny, retrieval: each flat colour must be closest to its own name, in
        # Swedish, so a broken tokenizer or preprocessing fails here rather than in search.
        pictures = visual.embed([Image.new('RGB', (384, 384), c) for c in ('red', 'blue')])
        words = textual.embed(['en röd yta', 'en blå yta'])
        score = [[sum(a * b for a, b in zip(p, w)) for w in words] for p in pictures]
        print('dimension:', len(pictures[0]), '| red·(röd, blå):', [round(x, 3) for x in score[0]], '| blue·(röd, blå):', [round(x, 3) for x in score[1]])
        if not (score[0][0] > score[0][1] and score[1][1] > score[1][0]):
            raise RuntimeError('the colours did not match their names')

    @check('SSCD copy descriptor (card against CPU)')
    def _copy_detect():
        import os
        import sys

        from PIL import Image, ImageOps

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ems', 'modelworker'))
        from copy_detect import Descriptor

        path = os.path.join(os.environ.get('CFG_COPY_DETECT_ROOT', '/opt/copy-detect'),
                            os.environ.get('CFG_COPY_DETECT_MODEL', 'sscd_disc_mixup') + '.torchscript.pt')
        # A picture, its mirror image and a crop of it: copies of one another.
        picture = Image.radial_gradient('L').convert('RGB').resize((640, 480))
        picture.paste((200, 40, 40), (60, 60, 260, 200))
        copies = [picture, ImageOps.mirror(picture), picture.crop((40, 30, 600, 450))]
        card, cpu = Descriptor(path, 'cuda').describe(copies), Descriptor(path, 'cpu').describe(copies)
        agreement = min(sum(a * b for a, b in zip(x, y)) for x, y in zip(card, cpu))
        alike = min(sum(a * b for a, b in zip(card[0], other)) for other in card[1:])
        print('dimension:', len(card[0]), f'| card/CPU cosine {agreement:.6f} | copies at least {alike:.3f}')
        if agreement < 0.999:
            raise RuntimeError(f'card and CPU descriptors disagree (cosine {agreement:.6f})')
        if alike < 0.75:
            raise RuntimeError(f'a mirror image or a crop is no copy to it (cosine {alike:.3f})')

    @check('TransNetV2 shot scores (card against CPU)')
    def _shots():
        import os
        import subprocess
        import sys
        import tempfile

        import torch

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ems', 'modelworker'))
        import model_registry
        import shots
        from shots import ShotScorer

        weights = os.path.join(model_registry.TRANSNET_ROOT, model_registry.TRANSNET_WEIGHTS)
        # Three scenes of 80 frames each with hard cuts between them: the new shots start at frames
        # 80 and 160. Five windows, so the card runs full batches and a short last one.
        graph = ('testsrc2=size=480x270:rate=25,trim=end_frame=80,setpts=PTS-STARTPTS[a];'
                 'mandelbrot=size=480x270:rate=25,trim=end_frame=80,setpts=PTS-STARTPTS[b];'
                 'testsrc=size=480x270:rate=25,trim=end_frame=80,setpts=PTS-STARTPTS[c];'
                 '[a][b][c]concat=n=3:v=1:a=0,scale=48:27,format=rgb24')
        with tempfile.TemporaryDirectory() as tmp:
            frames = os.path.join(tmp, 'frames.rgb')
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-filter_complex', graph, '-f', 'rawvideo', frames], check=True, capture_output=True)
            count = shots.frame_count(frames)
            torch.cuda.reset_peak_memory_stats()
            card = ShotScorer(weights, 'cuda').score_file(frames, count)[0]
            held = torch.cuda.max_memory_reserved() / 2**20
            cpu = ShotScorer(weights, 'cpu').score_file(frames, count)[0]
        apart = max(abs(a - b) for a, b in zip(card, cpu))
        # The network marks the old shot's last frame, which shot_rules.peaks moves on by one.
        top = sorted(sorted(range(count), key=lambda i: -card[i])[:2])
        rest = max(s for i, s in enumerate(card) if i not in top)
        need = model_registry.CARD_NEED_MB['transnet']
        print(f'peaks at frames {top} scoring {[round(card[i], 3) for i in top]} | the rest at most {rest:.3f} | card/CPU apart {apart:.4f} | {held:.0f} MB of the card, {need:.0f} asked')
        if top != [79, 159] or min(card[i] for i in top) < 0.5 or rest > 0.5:
            raise RuntimeError(f'cuts into frames 80 and 160 peak at {top}, the rest up to {rest:.3f}')
        if apart > 0.02:
            raise RuntimeError(f'card and CPU scores disagree by {apart:.4f}')
        if held > need:
            raise RuntimeError(f'scoring held {held:.0f} MB of the card, over the {need:.0f} MB it asks for (CARD_NEED_MB)')

    @check('PP-OCRv6 text reader (card against CPU)')
    def _text_reader():
        import os
        import sys

        from PIL import Image, ImageDraw, ImageFont

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ems', 'modelworker'))
        from text_reader import PhotoReader

        root = os.environ.get('CFG_OCR_ROOT', '/opt/ocr')
        tier = os.environ.get('CFG_OCR_PHOTO_TIER', 'medium')
        picture = Image.new('RGB', (900, 260), 'white')
        draw = ImageDraw.Draw(picture)
        # PIL's own font has no å/ä/ö, which would read as missing letters.
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 64)
        draw.text((40, 30), 'Hamngatan 4', fill='black', font=font)
        draw.text((40, 140), 'Älvsjö 1932', fill='black', font=font)
        card = PhotoReader(root, tier, 'cuda', 2 << 30, 1600)
        cpu = PhotoReader(root, tier, 'cpu', 2 << 30, 1600)
        on_card, on_cpu = card.read(picture), cpu.read(picture)
        providers = card.engine.text_det.session.session.get_providers()
        print('card:', on_card['text'].replace('\n', ' | '), '| cpu:', on_cpu['text'].replace('\n', ' | '), '| providers:', providers)
        if providers[0] != 'CUDAExecutionProvider':
            raise RuntimeError(f'the detector runs on {providers[0]}')
        if on_card['text'] != on_cpu['text']:
            raise RuntimeError('card and CPU read different text')
        if 'Hamngatan 4' not in on_card['text'] or 'Älvsjö' not in on_card['text']:
            raise RuntimeError(f'misread: {on_card["text"]!r}')

    @check('handwriting recogniser (card against CPU)')
    def _hand_reader():
        import os
        import sys

        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ems', 'modelworker'))
        from text_reader import HandReader

        root = os.environ.get('CFG_OCR_HAND_ROOT', '/opt/ocr/multicentury-ppocrv6-medium')
        line = Image.new('RGB', (520, 70), 'white')
        ImageDraw.Draw(line).text((10, 8), 'Elin Andersson', fill='black',
                                  font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Oblique.ttf', 48))
        pixels = np.asarray(line)[:, :, ::-1].copy()
        card = HandReader(root, 'cuda', 1 << 30, -1).read(pixels)
        cpu = HandReader(root, 'cpu', 1 << 30, -1).read(pixels)
        print('card:', card, '| cpu:', cpu)
        if card[0] != cpu[0]:
            raise RuntimeError('card and CPU read the line differently')
        if 'Andersson' not in card[0]:
            raise RuntimeError(f'misread: {card[0]!r}')

    @check('handwriting transcriber (card against CPU)')
    def _transcriber():
        import os
        import sys

        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ems', 'modelworker'))
        from htr import LineTranscriber

        root = os.environ.get('CFG_HTR_ROOT', '/opt/htr/multicentury-htr-model')
        line = Image.new('RGB', (900, 120), 'white')
        ImageDraw.Draw(line).text((14, 20), 'Kära lilla Mormor', fill='black',
                                  font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Oblique.ttf', 72))
        pixels = np.asarray(line)[:, :, ::-1].copy()
        card = LineTranscriber(root, 'cuda', (192, 1024)).read([pixels])[0]
        cpu = LineTranscriber(root, 'cpu', (192, 1024)).read([pixels])[0]
        print('card:', card, '| cpu:', cpu)
        if card != cpu:
            raise RuntimeError('card and CPU write the line differently')
        if 'lilla' not in card:
            raise RuntimeError(f'misread: {card!r}')

    @check('English handwriting transcriber (card against CPU)')
    def _english_transcriber():
        import os
        import sys

        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ems', 'modelworker'))
        from htr import LineTranscriber

        root = os.environ.get('CFG_HTR_EN_ROOT', '/opt/htr/trocr-large-handwritten')
        line = Image.new('RGB', (900, 120), 'white')
        ImageDraw.Draw(line).text((14, 20), 'Dear old sister', fill='black',
                                  font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Oblique.ttf', 72))
        pixels = np.asarray(line)[:, :, ::-1].copy()
        card = LineTranscriber(root, 'cuda').read([pixels])[0]
        cpu = LineTranscriber(root, 'cpu').read([pixels])[0]
        print('card:', card, '| cpu:', cpu)
        if card != cpu:
            raise RuntimeError('card and CPU write the line differently')
        if 'sister' not in card.lower():
            raise RuntimeError(f'misread: {card!r}')

    @check('language of read text')
    def _text_language():
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ems', 'modelworker'))
        from text_language import LanguageIdentifier

        identifier = LanguageIdentifier()
        swedish = identifier.detect('Kära lilla Mormor, tack för brevet. Vi har det bra här och hoppas att ni mår bra.')
        english = identifier.detect('Dear old sister, I am writing tonight in case this may catch some extra mail out to you.')
        name = identifier.detect('Anna Andersson 1932')
        print('sv:', swedish, '| en:', english, '| a name:', name)
        if (swedish or {}).get('language') != 'sv' or (english or {}).get('language') != 'en' or name is not None:
            raise RuntimeError('languages told wrong')

    @check('faster-whisper transcribe')
    def _whisper():
        import subprocess
        import tempfile

        from faster_whisper import WhisperModel

        model = WhisperModel('base', device='cuda', compute_type='float16')
        print('whisper base loaded on cuda')

        # Constructing the model is NOT enough: CTranslate2 resolves cuBLAS/cuDNN through the
        # system linker lazily, on the first inference. A CUDA-13 image (where CTranslate2's
        # libcublas.so.12 is missing) loads the model happily and only then fails with
        # "Library libcublas.so.12 is not found or cannot be loaded". So actually decode.
        with tempfile.TemporaryDirectory() as tmp:
            wav = f'{tmp}/tone.wav'
            subprocess.run(
                ['ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
                 '-ar', '16000', '-ac', '1', '-y', wav],
                check=True, capture_output=True,
            )
            # The same decode arguments inference_ops.transcribe uses, because they are
            # what can fail on their own. vad_filter in particular loads a *second* model
            # — the Silero VAD ONNX — which the image prewarms separately; if that asset is
            # missing this is where it shows, rather than inside a request with no network. A sine tone has no speech, so the VAD is expected to return nothing.
            segments, info = model.transcribe(
                wav,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            list(segments)  # generator — nothing runs until it is consumed

            # And once over speech, to prove segments come back timed. Espeak is not in the
            # image, so the "speech" is a frequency sweep: whisper will hear something and
            # hallucinate words for it, which is fine — what is under test is the shape of
            # what comes back, not the words.
            speech = f'{tmp}/sweep.wav'
            subprocess.run(
                ['ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=200:duration=3',
                 '-af', 'vibrato=f=8:d=0.9,volume=3', '-ar', '16000', '-ac', '1', '-y', speech],
                check=True, capture_output=True,
            )
            # word_timestamps on, whatever the deployment's default is: the shape of what it
            # returns is what inference_ops.transcribe passes on as each segment's `words`,
            # and a silent change to it would only surface far downstream.
            timed, _timed_info = model.transcribe(
                speech, beam_size=1, condition_on_previous_text=False, word_timestamps=True,
            )
            timed = list(timed)
            for segment in timed:
                if segment.start is None or segment.end is None:
                    raise RuntimeError('whisper returned a segment with no timings')
                if segment.end < segment.start:
                    raise RuntimeError(f'whisper segment ends before it starts: {segment.start} -> {segment.end}')
            print(f'timed segments: {len(timed)}')

            worded = [s for s in timed if getattr(s, 'words', None)]
            if timed and not worded:
                raise RuntimeError('word_timestamps=True returned segments carrying no words')
            for segment in worded:
                for word in segment.words:
                    if word.start is None or word.end is None or not str(word.word):
                        raise RuntimeError(f'whisper returned a malformed word: {word!r}')
            if worded:
                sample = worded[0].words[0]
                print(f'word timings present, e.g. {sample.word!r} at {sample.start:.2f}-{sample.end:.2f}s')

        if not info.language:
            raise RuntimeError('whisper reported no detected language')
        print(f'transcribe ran on cuda (detected language: {info.language})')

    @check('NeMo diarization / voiceprint')
    def _voice():
        import subprocess
        import tempfile

        import torch
        from nemo.collections.asr.models import EncDecSpeakerLabelModel, SortformerEncLabelModel

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ems', 'modelworker'))
        import model_registry

        # Loading is not the test — NeMo sits on lightning/hydra and on the same torch the
        # rest of the image uses, so what can break is an import or a kernel, not a download.
        diarizer = SortformerEncLabelModel.restore_from(
            model_registry._nemo_checkpoint(model_registry.SORTFORMER_MODEL, model_registry.SORTFORMER_REVISION),
        )
        model_registry.set_streaming(diarizer)
        diarizer.eval().to('cuda')
        device = next(diarizer.parameters()).device
        print('sortformer on:', device)
        if device.type != 'cuda':
            raise RuntimeError(f'sortformer landed on {device}, not cuda')

        with tempfile.TemporaryDirectory() as tmp:
            # A frequency sweep, not silence: the VAD in front of the diarizer would drop
            # pure silence and the model would never run a kernel at all, which is exactly
            # the load-only pass this script exists to avoid.
            wav = f'{tmp}/sweep.wav'
            subprocess.run(
                ['ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=200:duration=8',
                 '-af', 'vibrato=f=8:d=0.9,volume=3', '-ar', '16000', '-ac', '1', '-y', wav],
                check=True, capture_output=True,
            )
            # Shape, not content: a sweep is not speech, so zero turns is a fine answer.
            turns = diarizer.diarize(audio=[wav], batch_size=1)
            if not isinstance(turns, list) or not turns:
                raise RuntimeError(f'diarize returned {turns!r}, expected one entry per input')
            print(f'diarize ran, {len(turns[0])} turn(s) on a synthetic sweep')

            # On the card, where `auto` runs it (ein/core#472), and on the CPU, where
            # EMS_TITANET_DEVICE=cpu and a host without a card run it. The two must agree:
            # voiceprints from either are matched against each other.
            embedder = EncDecSpeakerLabelModel.restore_from(
                model_registry._nemo_checkpoint(model_registry.TITANET_MODEL, model_registry.TITANET_REVISION),
            )
            vectors = {}
            for device in ('cuda', 'cpu'):
                embedder.eval().to(device)
                vector = embedder.get_embedding(wav).reshape(-1).float().cpu()
                if vector.shape[0] != 192:
                    raise RuntimeError(f'expected a 192-dim voiceprint on {device}, got {vector.shape[0]}')
                if not torch.isfinite(vector).all():
                    raise RuntimeError(f'voiceprint on {device} contains non-finite values')
                vectors[device] = vector / vector.norm()
            agreement = float((vectors['cuda'] * vectors['cpu']).sum())
            print(f'voiceprint dim: 192 | card/CPU cosine {agreement:.6f}')
            if agreement < 0.999:
                raise RuntimeError(f'card and CPU voiceprints disagree (cosine {agreement:.6f})')

    @check('LR-ASD active speaker')
    def _asd():
        import os

        import torch

        lr_home = os.environ.get('LR_ASD_HOME', '/opt/lr-asd')
        weight = os.path.join(lr_home, 'weight', 'finetuning_TalkSet.model')
        if not os.path.exists(weight):
            raise RuntimeError(f'LR-ASD weights missing at {weight}')
        sys.path.insert(0, lr_home)
        from model.Model import ASD_Model

        model = ASD_Model()
        state = torch.load(weight, map_location='cpu')
        backbone = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')} or state
        model.load_state_dict(backbone, strict=False)
        model.eval().to('cuda')

        # 3s of video at 25fps: 75 grayscale 112x112 face crops, with MFCC at 4x that rate.
        frames = 75
        visual = torch.rand(1, frames, 112, 112, device='cuda') * 255
        audio = torch.rand(1, frames * 4, 13, device='cuda')
        with torch.no_grad():
            out_av, _out_v = model(audio, visual)
        torch.cuda.synchronize()
        if not torch.isfinite(out_av).all():
            raise RuntimeError('LR-ASD produced non-finite scores')
        print(f'LR-ASD forward OK on cuda, output {tuple(out_av.shape)}')

    print('\n' + '=' * 60)
    if SKIPPED:
        print(f'skipped {len(SKIPPED)} check(s) not matching --only: {", ".join(SKIPPED)}')
    for warning in WARNINGS:
        print('WARN:', warning)
    if FAILURES:
        print(f'FAILED ({len(FAILURES)}):')
        for failure in FAILURES:
            print('  -', failure)
        return 1
    print('all GPU checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
