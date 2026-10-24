# EMS API

EMS (EiN Media Service) is a stateless service that answers questions about media bytes and
produces media bytes: probing, renditions, transcodes, fingerprints, face detection, speech,
image-text embeddings, text embeddings, document text extraction. It owns the tools (ffmpeg,
ImageMagick, exiftool, chromaprint) and the models (SigLIP2, InsightFace, a text
embedder (jina-v5-text), PP-OCRv6, whisper, NeMo diarization and voiceprints, LR-ASD) so that a caller never
depends on which tool or model answers.

This document is the contract. It is written to stand on its own: a caller with a directory of
files and an HTTP client can use every operation here without knowing anything about EiN.

Status: everything in §4 marked _(implemented)_ answers today; `image.decode` and `audio.extract` are specified here and not yet needed by any caller. Open questions are at the end.

---

## 1. Principles

- **Stateless.** EMS holds no catalog, no ids, no cache of results. Every request carries
  everything the answer depends on; the same request gives the same answer for the same model
  and tool versions, and every answer says which versions produced it.
- **Files by path.** Inputs and outputs are absolute paths on a filesystem EMS can see. EMS reads
  the input where it is and writes the output where it is told, byte-exactly at that path. It
  never chooses names, never writes anywhere it was not asked to, and never cleans up after the
  caller.
- **Explicit parameters.** Anything that changes the answer is a request parameter: the display
  frame of an image, target sizes, language hints. EMS reads no sidecar, no database, no
  convention.
- **Normalized results with the raw answer attached.** Where a tool's output is the result
  (`media.probe`), EMS returns a normalized structure it will keep stable across tool changes, and
  the tool's raw output beside it under `raw.<tool>` for callers that need more. A caller reading
  `raw.*` accepts that it is bound to that tool.
- **Fine-grained ops, and composites where they make a better API.** `speech.diarize` is the
  model; `speech.voiceprint` is the useful thing. Both exist.
- **Consistent shapes.** Every file input looks the same, every long op streams the same events,
  every error has the same envelope.

## 2. Transport

HTTP/1.1, JSON request bodies, JSON responses. All operations are `POST /v0/<namespace>/<op>`
except `GET /v0/health` and `GET /v0/capabilities`.

```
POST /v0/faces/detect
Authorization: Bearer <key>
Content-Type: application/json

{ "file": { "path": "/files/S1/01K…jpg", "angle": 90 } }
```

Every request body may carry an `options` object:

| Field      | Type                         | Meaning                                                                                                                |
| ---------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `priority` | `"interactive"` \| `"batch"` | `interactive` enters a model family's queue ahead of waiting batch work. Default `batch`.                              |
| `deadline` | number, epoch seconds        | Work still queued past this instant is discarded and answered with `deadline_exceeded`. Work already running finishes. |

### 2.1 Responses

Success:

```json
{ "ok": true, "result": { … }, "meta": { "took_ms": 412, "producer": "qwen3-vl-8b/nf4", "api": "3.3.0" } }
```

`meta.producer` names the model or tool that answered, with enough detail to detect a change —
a caller that stores results beside it can tell stale from current. `meta` may carry more
(`device`, `queued_ms`).

Failure (HTTP status 4xx/5xx and the same envelope):

```json
{ "ok": false, "error": { "code": "undecodable", "message": "moov atom not found", "permanent": true } }
```

`permanent: true` means the same request will fail the same way for these bytes — a truncated
container, an unsupported format, a path outside the roots. A caller should not retry it.
`permanent: false` (or absent) is transient: a queue full, a model still loading, a deadline.

### 2.2 Streaming long operations

Any op may be asked to stream by sending `Accept: text/event-stream`. The response is then a
server-sent event stream that ends with exactly one terminal event:

```
event: progress
data: {"phase":"decode","fraction":0.31,"message":"frame 4812/15500"}

event: log
data: {"level":"warn","message":"NVDEC has no decoder for dvvideo, decoding on the CPU"}

event: result
data: {"ok":true,"result":{…},"meta":{…}}
```

or `event: error` with the failure envelope. `fraction` is optional and monotonic when present;
`phase` names what is happening in the op's own vocabulary. Without `text/event-stream` the same
op blocks and returns the terminal envelope as its body — the two forms carry the same result.

**Closing the connection cancels the operation.** A running tool is killed, a queued model request
is discarded, and any output file the op had started writing is removed. A model request already
running finishes its forward pass, which cannot be stopped partway, and its answer is dropped.
That is the whole cancellation API; there are no job ids.

Ops marked _long_ below are the ones where streaming is worth asking for. Nothing stops a caller
streaming a short op.

### 2.3 Authentication

`Authorization: Bearer <key>`. Keys are lines in a keys file the operator points EMS at;
there is one role. A missing or unknown key is `401 unauthorized`.

### 2.4 Versioning

The path prefix is the major version. `GET /v0/capabilities` reports `api` as a semver string.
Below 1.0.0 the minor is the breaking part, as in a caret range: patch versions add fields and ops,
never remove or reinterpret, and a breaking change is a new minor. From 1.0.0 the same holds for
minor and major, and a new major is a new path prefix. A
caller should refuse to run against a major it does not know and may warn on a minor below what
it was written for.

### 2.5 Admission

EMS bounds concurrent work in three lanes — `models` (requests forwarded to the model worker,
which serialises per model family behind that), `tools` (CPU and subprocess work) and `encodes`
(whole-file decodes: transcodes, storyboards and shot decodes, so a long one never sits ahead of the probes and
renditions) — and reports the bounds in `capabilities.limits`. Past `queue_depth` waiters a new request is
`503 busy` with `Retry-After`. Interactive requests may use `interactive_reserve` extra slots,
so a crawl never starves a person waiting.

### 2.6 The card

Where a model family runs is EMS's decision, made each time it loads one, and changes only how
fast it answers, never what (`model_registry.place`):

- **Claims, not moments.** Each family on the card claims its whole need, its weights and what it
  works with mid-request (`CARD_NEED_MB`). A family goes on the card when its need is free now and
  still left after every claim; otherwise the same weights run on the CPU. Free memory measured
  between requests overcommitted the card: two families each placed within it ran out together.
  Before a family goes to the CPU, families idle past `EMS_ROOM_IDLE` (30 s) are unloaded to make
  room.
- **whisper in one place per host.** On the card in float16 on a host with one
  (`EMS_WHISPER_CARD_COMPUTE`), waiting for room there as `model_unavailable`; on the CPU in int8
  on a host without one (`EMS_WHISPER_CPU_COMPUTE`). The place is in the transcript signature.
- **A shared card.** `EMS_CARD_SHARE_MB` caps what EMS's own processes hold (counted with
  `nvidia-smi`, which in the container lists only them), for a card another service shares; the
  other side is set to leave EMS the same number (ein/llm's `LLM_CARD_RESERVE_MB`).
  `EMS_CARD_BUDGET_MB` makes the card look smaller, to rehearse one.

`system.health` lists each resident copy with its device and measured size, and `degraded` names
a card the host should have and does not see, or one whose context is lost. EiN's `docs/architecture/media-hardware.md` has the
whole picture, the pipelines' side included.

## 3. Files, roots and frames

### 3.1 Paths and roots

EMS is configured with **roots**: directories it may read (`ro`) or read and write (`rw`).
A request naming a path outside every root, or an output path under a read-only root, is
`403 path_outside_roots` (permanent). Paths are resolved after following symlinks, and the
resolved path must also lie under a root — a symlink that leaves the roots is refused. Roots are
listed in `capabilities.roots`.

An **output path** is written exactly as given. EMS writes through a private temporary file in
the same directory and renames it into place, so a reader never sees a partial file at the
output path; a failed op leaves nothing there. The directory must exist and be writable. The
extension does not choose the format — the request does — but EMS refuses a mismatch it can see
(`.avif` asked to hold `mp4`) as `invalid_request`.

### 3.2 File input

Every operation that reads media takes a `file` object:

```json
{ "path": "/files/S1/01K…", "angle": 90, "mirror": false, "mimetype": "image/heic" }
```

| Field      | Type                  | Meaning                                                                          |
| ---------- | --------------------- | -------------------------------------------------------------------------------- |
| `path`     | string                | Absolute path under a root. Required.                                            |
| `angle`    | 0 \| 90 \| 180 \| 270 | The turn still owed on these bytes to show them upright — see 3.3. Default 0.    |
| `mirror`   | boolean               | Flip after rotating. Default false.                                              |
| `mimetype` | string                | A hint. EMS sniffs when absent, and trusts the sniff over a hint it contradicts. |

### 3.3 The display frame

Images and video frames are handled in the **display frame**: the stored raster rotated
counter-clockwise by `angle`, then flipped horizontally when `mirror` is set. EMS never applies
the file's own EXIF `Orientation` or a container's rotation matrix on its own; the caller states
the turn owed, and that is the only rotation applied. This is what lets a caller override a wrong
tag and be sure every op agrees.

Every coordinate EMS returns or accepts — face boxes, landmarks, crop boxes, tracks — is a
**fraction of the display frame** (`0..1`, origin top-left), so the same numbers apply to the
original and to every rendition of it. Where a result reports the frame it measured
(`frame: {width, height}`), those are the pixel dimensions of the display frame.

For formats whose decoder already applies a stored rotation (HEIF `irot`), EMS accounts for it
internally: the caller still states the _total_ turn owed relative to what a naive viewer would
show, and EMS applies only the residual. That rule is what makes `angle` the same number for a
JPEG and a HEIC.

### 3.4 Output specifications

Ops that write files take `output` objects:

```json
{ "path": "/files-volatile/…/512x512.avif", "format": "avif", "quality": 58 }
```

The set of formats and their parameters is per op; `capabilities` lists what this EMS can encode.

## 4. Operations

Namespaces: `media`, `image`, `video`, `audio`, `fingerprint`, `faces`, `speech`, `vision`,
`text`, `document`, `system`.

### 4.1 `media.probe` _(implemented)_

What is this file? One call, all tools.

Request: `{ "file": {…}, "raw": ["exiftool", "ffprobe"] }` — `raw` lists which tool outputs to
attach (default: all that ran).

Result:

```json
{
  "mimetype": "video/quicktime",
  "kind": "video",
  "size": 242120443,
  "sha256": "…",
  "container": "mov,mp4,m4a,3gp,3g2,mj2",
  "duration": 61.4,
  "bitrate": 31500000,
  "picture": { "width": 3840, "height": 2160, "codec": "hevc", "rotation": 90, "sar": 1, "fps": 29.97, "interlaced": false },
  "sound": { "codec": "aac", "channels": 2, "sample_rate": 48000 },
  "captured_at": [ { "value": "2025-09-27T15:26:32", "source": "DateTimeOriginal", "zone": "+02:00" } ],
  "gps": { "lat": 59.33, "lon": 18.07, "alt": 12, "dop": 1.2 },
  "device": { "make": "DJI", "model": "FC3582", "serial": "…" },
  "decodable": true,
  "raw": { "exiftool": { … }, "ffprobe": { … } }
}
```

- `kind` is `image | video | audio | document | other`.
- `picture.rotation` is what the _file_ says: the degrees clockwise a naive viewer must turn the
  stored raster to show it upright — exiftool's `Rotation` for a video container, or EXIF
  `Orientation` mapped to degrees for an image (also reported raw as `picture.orientation`, with
  `picture.mirrored: true` for the flipped variants). A video's display-matrix value is attached
  as `picture.matrix_rotation` in ffmpeg's own sign convention. All of it is reported, never
  applied: the caller decides what turn it owes (§3.3). `picture.width/height` are the stored
  raster.
- `captured_at` is a list of candidates in the tool's own tag names, ordered by how much the
  tool trusts them; the caller decides. `zone` is present only when the file carries one.
- `decodable: false` comes with `reason` and means no decoder will ever get a frame out of these
  bytes (the container index is missing). It is a property of the file, so an op that has to
  decode it (frames, poster, audio, transcode) fails `undecodable` with `permanent: true`: when
  its decode fails, the op asks ffprobe, and a refusal there makes the failure the file's. A
  decode that fails on a file ffprobe reads is `tool_failed`, which a caller may retry.
- `sha256` is computed only when asked (`"hash": true`); it is a full read.
- `raw.exiftool` is exiftool's numeric, ungrouped JSON (`-n -j`); `raw.ffprobe` is
  `-show_format -show_streams` JSON. Their tag names belong to those tools.
- ffprobe runs for video and audio (and for anything exiftool could not read). An EMS without
  ffprobe still answers from exiftool and says so in `meta.warnings`; `decodable` is then
  unmeasured and reported `true`.

### 4.2 `image.renditions` _(implemented)_

Several sized renditions of one image, from one decode. _long_ for large sources.

```json
{
  "file": { "path": "…/IMG_1.CR2", "angle": 270, "mimetype": "image/x-canon-cr2" },
  "targets": [
    { "output": { "path": "…/320x320.avif", "format": "avif", "quality": 55 }, "box": { "width": 320, "height": 320 }, "fit": "cover" },
    { "output": { "path": "…/512x512.avif", "format": "avif", "quality": 58 }, "box": { "width": 512, "height": 512 }, "fit": "contain" },
    {
      "output": { "path": "…/face.avif", "format": "avif", "quality": 70 },
      "crop": { "x": 0.41, "y": 0.22, "width": 0.09, "height": 0.13 },
      "pad": 0.3,
      "box": { "width": 384, "height": 384 },
      "fit": "contain"
    }
  ]
}
```

- `box` is a bounding box in pixels of the display frame; either side may be omitted. `fit`:
  `contain` (default) fits inside and **never upscales** — a small source lands at its own
  size; `cover` fills the box exactly, cropping the centre, and does upscale; `fill` stretches
  to the box. No `box` keeps the source size, which is how a crop is taken at full resolution.
- `crop` (fractions of the display frame) cuts first; `pad` grows the crop by that fraction of
  its own size on each side, clamped to the frame. The cut is then fitted like any target. The
  fraction is resolved against the raster EMS decoded — so for a RAW, whose readers disagree on
  which raster the file _is_ (embedded preview against sensor image), a box measured by
  `faces.detect` cuts the same face here.
- Output formats: `avif` (default), `webp`, `jpeg`, `png`. `quality` is the encoder's own scale
  (default 60). A path whose extension names another image format is `invalid_request`.
- Renditions carry **no metadata** (EXIF stripped) and **no orientation tag** — the pixels are
  already upright, and a viewer applying a leftover tag would turn them back.
- All targets are written or none: every output is validated before any is put in place.
- A RAW whose sensor image cannot be decoded (a truncated raw section) is rendered from its
  embedded JPEG when it has one, and the result says so (`"source": "preview"`).
- At most 32 targets.

Result: `{ "frame": {"width", "height"}, "targets": [ { "path", "width", "height", "bytes" } ] }` —
the display frame's pixel size and the measured size of each written file, in order.

### 4.3 `image.decode`

The display frame as a plain image, for formats other tools cannot open (RAW, HEIF).

`{ "file": {…}, "output": { "path": "…/decoded.png", "format": "png" | "jpeg", "max": 4096 } }`
→ `{ "frame": {…}, "path" }`. `max` bounds the longer side.

### 4.4 `video.frames` _(implemented)_

Frames as images. _long_.

```json
{ "file": {…}, "at": [1.0, 10.5, 40.0], "count": 5, "strategy": "representative",
  "output": { "dir": "/scratch/…", "format": "jpeg", "quality": 90, "max": 1280 } }
```

Either `at` (seconds) or `count` + `strategy`. A `count` divides the picture's length (the video
stream's duration, which can be shorter than the file's) into that many equal spans and takes a
frame from each: `representative` the one of the span's first 300 frames nearest their average
colour (ffmpeg's `thumbnail`), which passes over a black leader or a flash; `even` the frame at the
span's middle. A span past the last frame gives none, so fewer may come back; a picture of unknown
length is one span. Frames come out in the display frame `file.angle`/`file.mirror` name, with
square pixels (_5.5.3_); `max` bounds the longer side.
Result: `{ "frames": [ { "path", "t" } ] }`, `t` the frame's second in the file.

### 4.5 `video.poster` _(implemented)_

One frame, fitted like a rendition. `{ "file", "at": 1.0, "deinterlace": false, "targets": [ … as 4.2 … ] }`
→ as 4.2 plus `"t"`. A convenience over `video.frames` + `image.renditions` that decodes once:
the frame is taken at `at` seconds (default 1, past the black leader most cameras record) in the
display frame with square pixels; `deinterlace` runs yadif first. When the picture ends before
`at`, the poster is the first frame and `t` says so (0): a container's duration counts its longest
stream, so a caller cannot know where the picture stops. A file with no frame at all is
`undecodable`.

### 4.5a `video.storyboard` _(implemented, 5.4)_

Small frames at an even interval, tiled into one sheet: what a player's timeline shows under the
pointer.

```json
{
  "file": { "path": "…/1920x1080.mp4" },
  "tile": 160,
  "columns": 12,
  "max_tiles": 240,
  "min_interval": 1.0,
  "output": { "path": "…/storyboard-160.avif", "format": "avif", "quality": 55 }
}
```

- The file is upright — a rendition — so `angle` and `mirror` are refused rather than applied.
- A tile is the picture's display shape (its stored pixels widened by the sample aspect ratio),
  its longer side `tile`, both sides even. Tile `i` is the frame nearest `i × interval` seconds,
  laid out row by row; the interval is the longer of `min_interval` and the picture's length over
  `max_tiles`, so a video of any length is one sheet. A sheet that could exceed 4096 px a side is
  refused.
- The whole picture is decoded on the card; without one, or when that fails, only keyframes are,
  so a tile repeats until the next keyframe. `meta.producer` names which ran
  (`ffmpeg/9.0.1/cuda+magick/…`, `…/keyframes+…`).
- The sheet is encoded as `image.renditions` encodes, through a temporary file beside it. It runs
  in the `encodes` lane.

Result:
`{ "path", "width", "height", "bytes", "tile": { "width", "height" }, "columns", "rows", "count", "interval", "duration" }`
— `count` the tiles the sheet holds, which can be fewer than `columns × rows` when the picture
ends before the last tick.

### 4.5b `video.shots` _long_ _(implemented, 5.5)_

Where a video cuts from one shot to the next: the starts of its chapters.

```json
{ "file": { "path": "…/1920x1080.mp4" }, "threshold": 0.6, "nms": 0.5, "merge": 1.0, "min_shot": 3.0, "min_change": 0.1, "debug": false }
```

- The file is upright — a rendition — so `angle` and `mirror` are refused rather than applied.
- Every frame is decoded once, on the card when there is one, to 48×27, and scored by TransNetV2
  (model family `transnet`) for being the frame of a cut, in 100-frame windows 50 apart.
- The scores become cuts by these rules, every one a parameter:
  - a cut is a frame scoring at least `threshold`, with no higher one within `nms` seconds; the
    network marks the old shot's last frame, and the cut's `t` is the next one, the new shot's first;
  - black or white leader (blank frames — luma mean under 18 or over 235, spread under 6 at
    48×27 — for at least 0.2 s) is no cut, and its end is one (`kind: "leader"`, `score: null`)
    unless it opens or closes the file;
  - the colours a second before and after a cut must differ by `min_change` (a Bhattacharyya
    distance of median 8×8×8 RGB histograms), which a flash or a flicker does not;
  - cuts under `merge` seconds apart are one, the stronger;
  - no shot is shorter than `min_shot` seconds: the weaker cut of a short one goes.
- The defaults are the ones measured on the archive's own film and tape transfers and phone clips
  (`scripts/bench/shots-bench.py`): every cut of an edited 8 mm transfer, frame-exact, and none
  on 26 of 28 phone clips.
- Without a card the CPU decodes the file only when the whole decode fits the CPU budget a
  transcode is held to (§4.6); a file over it is **`refused`**, permanent at that budget.

Result:
`{ "duration", "frames", "cuts": [ { "t", "score", "kind" } ], "gaps": [ { "start", "end" } ], "params": { … } }`
— `t` in seconds on the file's own clock, at the new shot's first frame, `params` the rules the
cuts were found with. With
`debug`, also every `candidates` entry over the threshold and the per-frame `scores`.
`meta.producer` names the model and the decode (`transnetv2+ffmpeg/9.0.1/cuda`).

### 4.6 `video.transcode` _long_ _(implemented)_

```json
{
  "file": { "path": "…/DJI_0114.MP4", "angle": 0 },
  "output": { "path": "…/1920x1080.mp4", "format": "mp4" },
  "video": { "codec": "av1", "box": { "width": 1920, "height": 1080 }, "quality": 32, "speed": 8, "deinterlace": false },
  "audio": { "codec": "aac", "bitrate": "128k" },
  "clip": { "start": 1.0, "duration": 20.0 },
  "hints": { "source_codec": "hevc" }
}
```

- `video.box` is a bounding box; the picture is fitted inside and never upscaled, and comes
  out in the display frame (the caller's `angle`/`mirror`; the container's own rotation matrix
  is neither applied nor passed on).
- `video.quality` is on the codec's own scale (AV1: 0–63, lower is better); `speed` likewise
  (SVT-AV1's 0–13). EMS maps them onto whichever encoder it uses — on NVENC, `quality + 6` as
  `cq`, measured to give the same bytes — so the same request produces comparable output on
  either.
- `audio: null` drops the sound; `clip` cuts before encoding (`start` is an input-side seek,
  `duration` an output-side length).
- `hints.source_codec` (ffmpeg's name) lets EMS skip the GPU decode it knows will fail for a
  codec NVDEC cannot read (DV, MJPEG, ProRes, H.263, MS-MPEG4v3); without it the failure is
  learnt from one ffmpeg run. Unknown codecs are tried.

EMS chooses the chain and falls back on its own — everything in VRAM, then a CPU decode into
the GPU encoder, then the CPU alone — and `meta.producer` names what ran
(`ffmpeg/9.0.1/av1_nvenc/cuda`, `…/av1_nvenc/upload`, `…/libsvtav1`). A CPU decode's memory
footprint is projected from the frame count before it starts and a source over the budget is
encoded as several windows, one process each, joined by a stream copy with the audio encoded
once over the whole clip; only a source whose windows cannot be placed is **`refused`** —
permanent for that source at that budget. `capabilities.encode.cpu_budget` is the budget's
signature, for a caller that records refusals and wants to know when a raised budget makes
them stale. Progress events carry `fraction` by time.

Result: `{ "path", "width", "height", "duration", "bytes", "video_codec", "audio_codec" }`.
Transcodes run in the `encodes` lane (`EMS_LIMIT_ENCODES`), so a long encode never sits ahead
of the probes and renditions queued in the tool lane.

### 4.7 `audio.transcode` _(implemented)_

`{ "file", "output": { "path": "…/128k-44100.m4a", "format": "m4a" }, "audio": { "codec": "aac", "bitrate": "128k", "sample_rate": 44100 } }`.
`bitrate` and `sample_rate` are **ceilings**: EMS never exceeds the source's own, and keeps
mono for a mono source. A source that is already AAC in an MP4-family container is remuxed
rather than re-encoded. The output is the `ipod` flavour of MP4 (ftyp brand `M4A`), so a
sniffer reads it as audio. A file with no audio stream is `refused`.
Result: `{ "path", "duration", "bytes", "bitrate", "sample_rate", "channels", "remuxed" }`.

### 4.8 `audio.extract`

The sound of any file as PCM, for callers with their own audio models.
`{ "file", "output": { "path": "…/audio.wav", "format": "wav", "sample_rate": 16000, "channels": 1 } }`
→ `{ "path", "duration" }`, or `{ "path": null, "duration": 0 }` when the file has no audio
stream — a normal answer, not an error.

### 4.9 `audio.waveform` _(implemented)_

A rendered waveform picture. `{ "file", "targets": [ … as 4.2 … ], "style": { "foreground": "#4a90f0", "background": "#0a0b0e", "baseline": "#2b70d6", "aspect": 3 } }`.
Every target needs a `box` with both sides: a `cover` target is drawn square, any other as a
banner `aspect` times wider than tall that fits inside the box. Each picture is drawn at twice
its size and scaled down once (the drawing has no antialiasing of its own), from a single
decode of the recording. `baseline: null` draws no centre line. Result: `{ "targets": [ … as 4.2 … ] }`.

### 4.10 `fingerprint.compute` _(implemented)_

`{ "file", "kinds": ["sscd-image", "sscd-video", "chromaprint"] }` — default: every kind that
applies to the file's `kind`. Result:

```json
{ "fingerprints": [ { "kind": "sscd-image", "dimension": 512, "vector": [0.031, -0.012, …], "label": "primary" },
                    { "kind": "chromaprint", "label": "primary", "fingerprint": [514601615, 250901166, …] } ] }
```

`sscd-image` and `sscd-video` (_5.0_) are SSCD copy-detection descriptors (`sscd_disc_mixup`,
model family `sscd`): unit vectors whose cosine is high for copies of one picture — re-encoded,
resized, cropped, recoloured, overlaid, a scan of the print, a screenshot — and low for another
photo of the same scene. A picture's is taken in the display frame `file.angle`/`file.mirror`
name; a video gets one per keyframe (eight, `keyframe-0`…, as `video.frames` `representative`
picks them over the whole picture), labelled in order. The producer is the
model's name, with the tools that ran. `chromaprint` (_3.6_) is
the whole recording's raw chromaprint (`fpcalc -raw -length 0`, algorithm 2): one unsigned 32-bit
value per 1365/11025 s (about 0.124 s). It is not a vector: two copies of a recording match once
aligned, by the bit error rate over their overlap at the best offset. An audio too short to
fingerprint, under about two and a half seconds, yields no `chromaprint` entry rather than an error.

### 4.11 `faces.detect` _(implemented)_

`{ "file": {…}, "min_size": 0.02 }` → boxes, landmarks and embeddings in one call:

```json
{ "frame": { "width": 4032, "height": 3024 },
  "faces": [ { "box": { "x": 0.41, "y": 0.22, "width": 0.09, "height": 0.13 }, "confidence": 0.93,
               "landmarks": [[0.44, 0.27], …], "pose": { "pitch": -4.5, "yaw": 12.0, "roll": 1.3 },
               "embedding": [ … 512 floats … ] } ] }
```

`min_size` is the smallest face to report, as a fraction of the frame's shorter side.

`pose` is the head's orientation in degrees, from buffalo_l's 3D landmark model, in the display
frame: `pitch` up/down, `yaw` left/right turn, `roll` tilt; all three 0 is a face looking straight
into the camera. Omitted when the model pack provides none. It says nothing about occlusion —
`confidence` is the nearest signal for a face partly hidden.

### 4.11a `faces.detect_video` _long_ _(implemented)_

The faces a video shows, each with when it is seen, in one request:
`{ "file": { "path", "angle", "mirror" }, "interval": 1.0, "min_size": 0.02, "min_confidence": 0.7 }` →

```json
{ "frame": { "width": 1920, "height": 1080 }, "duration": 63.2, "interval": 1.0, "samples": 63,
  "faces": [ { "appearances": [[0.5, 9.5], [41.5, 44.5]], "detections": 12, "t": 5.0,
               "box": { "x": 0.41, "y": 0.22, "width": 0.09, "height": 0.13 }, "confidence": 0.93,
               "landmarks": [[0.44, 0.27], …], "pose": { "pitch": -4.5, "yaw": 12.0, "roll": 1.3 },
               "embedding": [ … 512 floats … ],
               "track": [[0.0, 0.40, 0.21, 0.09, 0.13], [1.0, 0.41, 0.22, 0.09, 0.13], …] } ] }
```

A frame every `interval` seconds (0.2–60, default 1), in the display frame the caller's
`angle`/`mirror` name, read in one pass over the video. A video longer than 1800 intervals is
sampled more sparsely so it gives no more frames than that; `interval` in the answer is the spacing
used. Detections smaller than `min_size` of the frame's shorter side, or below `min_confidence`,
are left out.

The detections are followed from frame to frame and the tracks of the file grouped into faces:
one per person seen, as far as the vectors tell. Two tracks seen in the same frame are never one
face, and a person may still come back as two when the vectors are not alike enough. Per face:

- `appearances` — when it is on screen, `[start, end]` in seconds, padded by half an interval.
- `t`, `box`, `confidence`, `landmarks`, `pose` — its clearest detection (confidence, size and a
  head turned towards the camera), as in `faces.detect`: fractions of the display frame at `t`.
  A crop of the face is `video.poster` at `t` with this box as `crop`.
- `embedding` — the normalized mean of its best detections' vectors, comparable by cosine with
  `faces.detect`'s.
- `track` — where it was in every sampled frame it was seen in, `[t, x, y, width, height]` in
  time order, in fractions of the display frame: at most one a frame, so a box can follow the
  face through the video.

Faces come first seen first. A file with no picture shows none; one no decoder can read is
`undecodable`. `duration` is the picture's length, the stream's own before the container's.

### 4.12 `faces.embed` _(implemented)_

An embedding for a face the caller already located: `{ "file", "box": {…} }` →
`{ "embedding", "confidence", "pose"? }` (`pose` as in `faces.detect`), or `result: null` when no
face is found inside the box.

### 4.13 `speech.transcribe` _long_ _(implemented)_

```json
{ "file": {…}, "language": "sv", "vad": { "min_silence_ms": 500 }, "word_timestamps": false }
```

Any container with sound; EMS extracts the audio itself. `language` is an ISO-639-1 code that
pins the decode and skips detection. Without it EMS detects the language over several windows
spread across the speech the VAD found (not whisper's own first-30-seconds guess, which
mislabels a Swedish clip that opens on noise as Norwegian and decodes all of it that way), and
lets the operator's expected languages (`EMS_WHISPER_LANGUAGES`, with `EMS_WHISPER_LANGUAGE_FLOOR`
and `EMS_WHISPER_LANGUAGE_WINDOWS`, §6) win when the averaged probability reaches the floor —
so a genuinely foreign clip is still decoded as what it is. The reported `language_probability`
is that averaged probability for the language chosen.

```json
{ "speech": true, "language": "sv", "language_probability": 0.98, "duration": 61.4,
  "text": "…", "segments": [ { "start": 0.8, "end": 4.1, "text": "…", "words": [ { "start", "end", "word" } ] } ] }
```

`speech: false` comes with an empty `segments` and is a normal answer for a silent clip; a file
with no sound track at all answers `audio: false` as well, so the two can be told apart.

### 4.14 `speech.diarize` _long_ _(implemented)_

`{ "file" }` → speaker turns:

```json
{ "speech": true, "duration": 61.4, "window": 120, "turns": [{ "start": 0.8, "end": 4.1, "speaker": "w0/speaker_0" }] }
```

The primitive under `speech.voiceprint`, exposed for inspection. The model runs in fixed
windows sized for the card, so `speaker` labels are **local to their window** — the same person
in two windows gets two labels, and a turn crossing a boundary arrives as two turns that meet
there. A caller that wants one speaker per file asks `speech.voiceprint`, which undoes the
windowing itself; the window is a fact about EMS's hardware and not the caller's to reason
about.

### 4.15 `speech.voiceprint` _long_ _(implemented)_

Who speaks when in a recording, with a voiceprint for each turn and for each speaker:
`{ "file" }` →

```json
{ "speech": true, "audio": true, "duration": 61.4,
  "turns": [ { "start": 0.8, "end": 4.1, "speaker": "speaker_0", "embedding": [ … ], "dimension": 192 } ],
  "speakers": [ { "id": "speaker_0", "turns": [ [0.8, 4.1], [9.2, 15.0] ], "talkSeconds": 9.1,
                  "vector": [ … ], "dimension": 192 } ] }
```

`speakers` is the same answer per voice, in order of first appearance (API 3.5): its turns, its
talk time, and one voiceprint for all of it — the mean of its turns' unit vectors weighted by
each turn's length, made unit again. A turn with `speaker: null` belongs to no speaker.

`speaker` is **file-wide** (`speaker_0`, `speaker_1`, … in order of first appearance): EMS
diarizes in windows, embeds every turn, pools each window-local speaker's voiceprints and merges
the ones that match — refusing to merge two labels from the _same_ window, which the diarizer
already said are different people — and rejoins the turns a window boundary cut in half. Only a
window-local label with enough embedded voice (10 s) can found a speaker; a label with less
joins the speaker it most resembles at a lower bar, and if it resembles none, or had nothing
long enough to embed, its turns come back with `speaker: null` — somebody spoke there whom
nothing could attribute. A turn too short to characterise a voice (under 1.5 s) is still listed,
with `embedding` and `dimension` `null`, so the timeline stays complete. `speech: false` comes
with empty `turns`; a file with no sound track answers `audio: false` as well.

One request per recording, whatever its length. The sound is extracted once and every turn is
sliced from the samples in memory, so the cost is the diarization plus one model call per
turn, not one decode per turn (the shape the per-span form of this op had until 2026-09-22,
which put a 45-minute tape side past a 15-minute step budget). The producer names both
models, `<diarizer>+<embedder>`.

The extracted sound is kept in scratch for a while (`EMS_AUDIO_CACHE_SECONDS`, default 30 min,
within `EMS_AUDIO_CACHE_MB`), so a `speech.active_speaker` score per turn afterwards decodes the
recording once, not hundreds of times.

### 4.16 `speech.active_speakers` _long_ _(implemented)_

Per speaker of a recording, the visible face whose lips move with its voice, in one request:
`{ "file": { "path", "angle", "mirror" }, "speakers": [ { "id": "speaker_0", "turns": [ [3.2, 9.8], … ] } ] }` →

```json
{ "speakers": [ { "id": "speaker_0", "score": 0.83, "t": 5.48,
                  "box": { "x": 0.41, "y": 0.18, "width": 0.12, "height": 0.21 },
                  "confidence": 0.86, "embedding": [ … ] } ],
  "windows": 4, "frames": 200 }
```

LR-ASD reads lip movement against the sound at 25 fps, so it is given consecutive frames: up to
4 s from the middle of each of a speaker's three longest turns (2 s or more), decoded in file
order in one pass over the video, in the display frame the caller's `angle`/`mirror` name. Faces
are detected on every fifth frame and followed by overlap; each face seen long enough is scored
against the window's sound, prepared as LR-ASD's own pipeline prepares it — the lower face cropped
square from the smoothed track, and MFCCs of the 16-bit samples. Per speaker the best-scoring face is answered, with its clearest
detection: `box` in fractions of the display frame at `t` seconds and `embedding`, InsightFace's
512-dim vector, for the caller to recognise. `score` is the mean probability (0..1) that the face
is the one speaking; what counts as enough is the caller's call. A speaker with no face to score
is left out. A file with no sound answers `speakers: []`.

### 4.16a `speech.active_speaker` _(implemented)_

Is the face in `track` the one talking between `start` and `end`?
`{ "file", "start", "end", "track": [ { "t": 12.5, "box": {…} }, … ] }` →
`{ "score": 0.81, "frames": 40 }`, or `result: null` when there is too little to judge (a track of
a few frames, a span with no audio). Silence is not evidence of not speaking, so it is no answer
rather than a low score.

### 4.18a `vision.embed` _(implemented)_

A picture's vector in the image-text space that `vision.embed_text` shares, for search by what
is in the picture: `{ "file": {…} }` →

```json
{ "embedding": [ … 1152 floats … ], "dimension": 1152 }
```

for an image (RAW/HEIF decoded out of process when the model's readers cannot open it), and for
a video one vector per keyframe — eight, as `video.frames` `representative` picks them over the
whole picture, each with its second in the file (_5.0.1_):

```json
{ "frames": [ { "at": 3.2, "embedding": [ … ] }, … ], "dimension": 1152 }
```

Normalized; compare by cosine. Audio is `unsupported`. `meta.producer` is the model
(`ViT-SO400M-16-SigLIP2-384__webli`, SigLIP2 So400m/16-384 from Immich's ONNX export), float32
on every device so a CPU host writes the same vectors as a GPU one. A producer change changes
every stored vector's meaning: the caller re-embeds.

### 4.18b `vision.embed_text` _(implemented)_

Search text as a vector in the same space: `{ "text": "segelbåt" }` →
`{ "embedding": [ … ], "dimension": 1152 }`. Multilingual (SigLIP2's Gemma tokenizer); text is
lower-cased and stripped of punctuation as the model was trained. Meant for the interactive
path: the text tower is small and pinned where `EMS_MODEL_PINNED` names `siglip-text`.

### 4.19 `text.embed` _(implemented)_

`{ "text": "…" }` → `{ "embedding": [ … ], "dimension": 1024 }`, or `{ "texts": [ "…", … ] }` (at
most 256) → `{ "embeddings": [ [ … ], … ], "dimension": 1024 }` in the same order. Normalized.

The embedder is the configured one, named in capabilities `models.text_embed` and in
`meta.producer` (jina-embeddings-v5-text-small, 1024 dimensions). `role` is `query` for a search
and `document` (the default) for the text it should find; the model embeds them differently. A
text over `EMS_TEXT_EMBED_MAX_TOKENS` (8192) is `invalid_request`, never cut: the caller chunks.
Queries run on their own pinned family (`text-query`, on the CPU by default), so a search does not
wait behind a batch of documents.

`model: "current"` is still accepted, as 2.1 callers send it. _3.0_ removed `model: "legacy"`
(paraphrase-multilingual-MiniLM-L12-v2, 384 dimensions, a 128-token window that cut longer text
silently), which was the default in 2.x; it is now `invalid_request`.

A change of the configured embedder changes every stored vector's meaning: the caller compares
`meta.producer` with what it stored and re-embeds.

### 4.21 `document.extract` _(implemented)_

`{ "file", "ocr": "auto" | "always" | "never" }` →
`{ "text": "…", "pages": 5, "method": "pdf-text" | "ocr" | "docx" | "odf" | "text" }`. `auto`
reads a PDF page as a picture only when it carries no usable text layer (fewer than
`EMS_OCR_MIN_PAGE_CHARS` characters), rasterized at `EMS_OCR_DPI` and at most `EMS_OCR_MAX_PAGES`
pages; a picture, a camera RAW through its preview included, is read whole in the display frame
`file.angle`/`file.mirror` name. Both go to the text reader `ocr.read` runs (PP-OCRv6, model family
`ocr`), and `meta.producer` names it when it read the text (_5.2_; Tesseract before).
`languages` is accepted and changes nothing: the reader tells each picture's language itself (§4.21a). A document that
cannot be opened is `unsupported` (permanent); a reader that is busy or down fails the request
rather than answering empty text.

### 4.21a `ocr.read` _(implemented, 5.1)_

The lines of text in a picture, for search by the words in it: `{ "file": {…}, "engine": "photo",
"language": "en" }` (`language` optional) →

```json
{ "text": "Hamngatan 4\nÖppet 10–18",
  "lines": [ { "text": "Hamngatan 4", "box": [[0.12, 0.30], [0.48, 0.29], [0.48, 0.35], [0.12, 0.36]], "score": 0.97 }, … ],
  "width": 4032, "height": 3024 }
```

`engine: photo` (the default and, for now, the only one) is PP-OCRv6 through RapidOCR, model
family `ocr`: one multilingual model that reads Swedish without a language setting, made for scene
text as much as for pages. The picture is read in the display frame `file.angle`/`file.mirror`
name (RAW/HEIF decoded out of process when the readers cannot open it); a 180° line is turned
before it is read. `box` is a line's four corners, clockwise from its top left as the text runs,
as fractions of `width` and `height` (the display frame's pixels). Lines come top to bottom; a
line scoring below `EMS_OCR_MIN_LINE_SCORE` (0.8), or holding fewer than two letters or digits, is left out.
A line PP-OCRv6 reads below the floor is cut out along its corners and read again by a
handwriting recogniser, the National Archives of Finland's `multicentury-ppocrv6-medium` (_5.2_),
and kept if that reading reaches the floor: handwritten captions and worn print, without changing
any line PP-OCRv6 reads well. The text of such a line is then written by the same archive's TrOCR-large
handwriting model (`multicentury-htr-model`, model family `htr`, loaded when a picture has such lines):
the recogniser judges that a line is writing, the transcriber reads it, at about half the word errors
on the archive's handwritten letters. On a picture that is writing — at least four of its kept lines,
and most of them, came through the recogniser — the transcriber writes every kept line, as PP-OCRv6 is
at times sure and wrong on a hand; a photo card's printed imprint beside a name or two keeps its reading.

The language of the text decides which transcriber writes it (_5.3_). It is told from the lines kept
so far by lingua, every language it knows, and comes back as `meta.language`:
`{ "code": "en", "confidence": 0.98, "given": false }` (ISO 639-1), or `null` when the text has fewer
than `EMS_OCR_LANGUAGE_MIN_LETTERS` (40) letters or no language reaches `EMS_OCR_LANGUAGE_MIN_CONFIDENCE`
(0.5) — a name or a date on a card. A request's `language` (two lowercase letters, as a person set
it) is used instead and comes back with `given: true` and no confidence. Swedish, Finnish and no
language go to the transcriber above, as described. English goes to Microsoft's TrOCR-large fine-tuned
on English handwriting (`trocr-large-handwritten`, model family `htr-en`); since the recogniser scores
English writing under its floor, an English picture is writing when fewer than half of at least three
lines PP-OCRv6 found read at 0.95 or above, and then every line it reads at 0.6 or above is written by
that transcriber, a line under the floor kept only if the transcriber writes two letters or digits there.
Any other language has no transcriber: a line the recogniser kept keeps its reading.

A picture without text answers `text: ""` and no lines. Anything but an image is `unsupported`.
`meta.producer` names the engines the image has, float32 on every device
(`PP-OCRv6_medium+multicentury-ppocrv6-medium+multicentury-htr-model+trocr-large-handwritten`;
`PP-OCRv6_medium` in 5.1); a producer change is the caller's signal to read again.

### 4.22 `system.health` — `GET /v0/health` _(implemented)_

Liveness and what is loaded:

```json
{ "ok": true, "api": "3.4.0", "uptime": 86400,
  "gpu": { "present": true, "vram_allocated_mb": 6100, "vram_reserved_mb": 7900,
           "util_pct": 37, "mem_used_mb": 8123, "mem_total_mb": 16311, "temp_c": 51,
           "power_w": 21.6, "power_limit_w": 180.0, "driver": "610.43.02", "name": "NVIDIA GeForce RTX 5060 Ti",
           "processes": [ { "pid": 49, "used_mb": 1810, "worker": "batch" } ] },
  "models": { "whisper:cuda:float16": { "loaded": true, "device": "cuda" }, "text-query": { "loaded": true, "device": "cpu" } },
  "worker": { "running": true, "restarts": 0 },
  "process": { "rss_mb": 6200, "max_rss_mb": 12000, "recycles": 0, "uptime": 86000,
               "query": { "rss_mb": 1500, "max_rss_mb": 6144, "recycles": 0, "uptime": 86000 } },
  "degraded": [],
  "queues": { "models": { "running": 1, "waiting": 3, "limit": 8, "reserve": 1 }, "tools": { … } },
  "families": {
    "whisper": { "model": "large-v3", "device": "auto", "pinned": false, "running": true, "waiting": 2,
                 "purpose": "Speech recognition: transcribes what is said …", "ops": [ "speech.transcribe" ],
                 "copies": [ { "key": "whisper:cuda:float16", "state": "loaded", "since": 1780000000.0, "last_used": 1780003600.0,
                               "device": "cuda", "worker": "batch", "vram_mb": 3616, "ram_mb": 17, "load_seconds": 119.6 } ],
                 "measured": { "whisper:cuda:float16": { "vram_mb": 3616, "ram_mb": 17, "load_seconds": 119.6 } } },
    "insightface": { "model": "insightface/buffalo_l", "device": "auto", "pinned": false, "copies": [] } },
  "requests": [ { "op": "speech.transcribe", "lane": "models", "interactive": false,
                  "queued": 1780003590.0, "started": 1780003595.0, "family": "whisper" } ],
  "build": { "version": "0.1.0", "commit": "e737f1d", "date": "2026-10-04T06:51:28+02:00" },
  "warnings": [ "no keys configured: authentication is off" ] }
```

`models` lists what is resident right now (the worker loads lazily and evicts idle families);
`worker.running: false` means every model op answers `model_unavailable` until it is back.

`families` (_3.2_) lists every model family the workers can load, resident or not: the model it
holds, the device it is configured for (`auto`, `cpu` or `cuda` — a setting, not a probe), whether
it is pinned, and each copy in memory with its `state` (`loading`, `loaded` or `unloading`), the
wall-clock `since`, `last_used`, the `device` it is on and the `worker` (`batch`, `query`) holding
it. No copies means not loaded. A family can have two copies, `insightface` beside `insightface:cpu`
after a CUDA OOM. `running` is a request holding the family right now and `waiting` how many queue
behind it — resident and working are different things. Absent while the worker is down.

`requests` (_3.2_) lists every admission slot held or waited for, oldest first: the `op`, its
`lane`, whether it is `interactive`, when it `queued` and, once admitted, when it `started`.
`build` (_3.2_) is the package version and, when the image was built from a git checkout, its
`commit` and commit `date`.

_3.3_ adds, for a status page: each family's `purpose` (what it is for, in a sentence) and `ops`
(the ops that use it); each copy's `vram_mb`, `ram_mb` and `load_seconds`, measured when it loaded
— the card's free memory and the worker's RSS before and after, so approximate when other work
runs beside the load — and `measured`, the same per key kept after an unload, so a family that is
not resident still says what it costs; a request's `family`, the model it is using right now
(a recording's `speech.voiceprint` is `sortformer`, then `titanet`); `process.query`, the query worker's own
memory; and in `gpu` the card's `name`, `driver`, `power_w` and `power_limit_w`, and `processes`,
what each of this container's processes holds of the card, a worker's named by `worker`.
`nvidia-smi` in a container sees only that container's processes, so `mem_used_mb` less their sum
is held by other containers on the same card.

`card` (_3.4_) says who has the card (`ems/card.py`): `mode` is `busy` while batch model work
arrives, `quiet` after `quiet_after` seconds without any, `switching` between them, and `off` with
no card or no query worker; `since` and `last_batch` are wall-clock; `search_on_card` lists the
query families answered by the card worker, and `skipped` why one is not (the free memory it would
need). The card worker's copies appear in `families` with `worker: "card"`. `plan` says, for every
family, where it runs in each mode — `{"busy": "card" | "cpu", "quiet": "card" | "cpu" | "unloaded",
"why"?}`: batch families where their device setting says while busy and unloaded while quiet, the
document embedder its setting while busy and the card while quiet, the query families the CPU while
busy and the card while quiet if they fit. While busy the quiet column is a prediction from the
card's free memory less what the batch and card workers hold; `why` names the shortfall.

`degraded` lists what is wrong with the card — the things that are otherwise indistinguishable
from "slow": a card the host should have and does not see (`EMS_EXPECT_GPU`), and `what: "gpu"`,
which means the model
worker's CUDA context is gone after a sticky error (a kernel killed by the driver's watchdog, an
illegal memory access): every op on the card fails until the worker has drained and replaced
itself, which it does on its own, at most once every ten minutes
(`CFG_MODEL_GPU_RECYCLE_GRACE`).

`gpu` has two views of the card. `vram_allocated_mb` / `vram_reserved_mb` are torch's own
accounting for this EMS's models, present while the model worker has torch loaded. `util_pct`,
`mem_used_mb`, `mem_total_mb` and `temp_c` are the whole card as `nvidia-smi` reports it — every
process on it, the transcodes included — present when the container can run `nvidia-smi`, and
read at most once every few seconds however often health is asked. A caller charting the card
(EiN's metrics sampler) reads the second set and treats its absence as "no card to chart",
not as an idle one.

### 4.23 `system.capabilities` — `GET /v0/capabilities` _(implemented)_

What this EMS can do, for a caller to check before it relies on it:

```json
{ "api": "3.3.0", "ops": [ "media.probe", "faces.detect", … ],
  "roots": [ { "path": "/files", "mode": "ro" }, { "path": "/files-volatile", "mode": "rw" } ],
  "tools": { "exiftool": "13.10", "ffprobe": "9.0.1", "ffmpeg": "9.0.1", "fpcalc": "1.5.1", "magick": "7.1.1" },
  "models": { "text_embed": "jina-embeddings-v5-text-small", "whisper": "large-v3", "faces": "insightface/buffalo_l",
              "image_embed": "ViT-SO400M-16-SigLIP2-384__webli", "ocr": "PP-OCRv6_medium+multicentury-ppocrv6-medium+multicentury-htr-model+trocr-large-handwritten" },
  "limits": { "models": 8, "tools": 6, "encodes": 2, "queue_depth": 64, "interactive_reserve": 1 },
  "encode": { "video_encoder": "av1_nvenc", "cpu_budget": "4000/170/64" },
  "signatures": { "transcribe": "large-v3/…" } }
```

`encode.video_encoder` is what `video.transcode` will run first, and `encode.cpu_budget` the
signature of the CPU-decode budget a refusal is measured against — a caller that records
refusals compares it to know when a raised budget has made them stale. `signatures.transcribe` is what a transcript
decoded right now would be stamped with — every setting that decides `speech.transcribe`'s
output, as one string, among them where this host decodes and in what precision (`on=cuda:float16`
on a host with a card, `on=cpu:int8` on one without; `EMS_WHISPER_CARD_COMPUTE`,
`EMS_WHISPER_CPU_COMPUTE`). A host decodes in that one place only — a card short of room makes the
request wait (`model_unavailable`) rather than move it to the CPU — so a transcript never depends
on what else was on the card. A caller holding old transcripts can tell which were made with other
settings; it is absent while the model worker is down.

## 5. Error codes

| Code                 | HTTP | Permanent | Meaning                                                                                                        |
| -------------------- | ---- | --------- | -------------------------------------------------------------------------------------------------------------- |
| `unauthorized`       | 401  | —         | Missing or unknown key                                                                                         |
| `invalid_request`    | 400  | yes       | Malformed body, unknown op, contradictory parameters                                                           |
| `path_outside_roots` | 403  | yes       | Input or output path not under an allowed root                                                                 |
| `not_found`          | 404  | yes       | Input path does not exist                                                                                      |
| `unsupported`        | 415  | yes       | EMS has no tool or model for this format or kind                                                               |
| `undecodable`        | 422  | yes       | The bytes cannot be opened by any decoder (see `media.probe`)                                                  |
| `refused`            | 422  | yes       | EMS will not do this work within its own budget (a CPU encode too large)                                       |
| `busy`               | 503  | no        | Admission queue full; `Retry-After` set                                                                        |
| `deadline_exceeded`  | 504  | no        | Discarded from the queue past `options.deadline`                                                               |
| `model_unavailable`  | 503  | no        | The model needed is loading, its device is unusable right now, or the card is out of memory; `Retry-After` set |
| `tool_failed`        | 500  | no        | A tool or model failed for a reason EMS could not classify; `message` carries what it said                     |
| `internal`           | 500  | no        | EMS's own fault                                                                                                |

## 6. Configuration (operator)

Environment only. `EMS_KEYS_FILE` (one key per line) or `EMS_KEY`, `EMS_ROOTS`
(`/files:ro,/old:ro,/files-volatile:rw,/files-tmp:rw`), `EMS_PORT`, `EMS_HOST`, `EMS_SCRATCH`,
the admission limits (`EMS_LIMIT_MODELS`, `EMS_LIMIT_TOOLS`, `EMS_LIMIT_ENCODES`,
`EMS_QUEUE_DEPTH`, `EMS_INTERACTIVE_RESERVE`), `EMS_EXIFTOOL_WORKERS`, `EMS_EXPECT_GPU` (`1`: a card
is expected, and health warns when none is visible), `EMS_CARD_BUDGET_MB` (the most of the card the models
treat as there — how a smaller card is rehearsed; 0 is the whole card), `EMS_CARD_SHARE_MB` (the most
of the card EMS's own processes hold, for a card shared with another service such as the llm: a
model that would take EMS past it runs on the CPU, even while the rest of the card is free; 0 is
all of it), `EMS_TOOL_STALL` (how long a
tool may go without using CPU or moving a byte before it is stopped as wedged; not a limit on how
long it may take), the transcode budget (`EMS_CPU_ENCODE_MAX_MB`,
`EMS_CPU_ENCODE_MB_PER_1K_FRAMES`, `EMS_CPU_ENCODE_MAX_SEGMENTS`), the encoder
(`EMS_VIDEO_ENCODER`, empty for automatic; `EMS_NVENC_CQ_OFFSET`, `EMS_NVENC_PRESET`), a scanned
PDF's pages (`EMS_OCR_DPI`, `EMS_OCR_MIN_PAGE_CHARS`, `EMS_OCR_MAX_PAGES`), and the model worker's own settings — every `EMS_<NAME>` reaches it as
`CFG_<NAME>`: `EMS_WHISPER_MODEL`, `EMS_WHISPER_LANGUAGE` (a pin),
`EMS_WHISPER_LANGUAGES` / `EMS_WHISPER_LANGUAGE_FLOOR` / `EMS_WHISPER_LANGUAGE_WINDOWS` (detection
biased towards the languages the archive holds; `ems/modelworker/model_registry.py` has the
reasoning), `EMS_MODEL_PINNED`, `EMS_MODEL_IDLE_EVICT`, `EMS_ROOM_IDLE` (how long a family may sit
idle before it is unloaded to make room for another on the card; 30), `EMS_MODEL_MAX_RSS`, `EMS_TITANET_DEVICE`,
`EMS_IMAGE_EMBED_MODEL` / `EMS_IMAGE_EMBED_DEVICE` / `EMS_IMAGE_EMBED_GPU_MEM_MB` (`vision.embed`), `EMS_IMAGE_EMBED_TEXT_DEVICE` (`vision.embed_text`; `cpu` by default, ~0.3 s a query, so the card keeps room for the batch families), `EMS_COPY_DETECT_DEVICE` (`fingerprint.compute`'s `sscd-*` kinds; `auto`), `EMS_TRANSNET_DEVICE` (`video.shots`; `auto`), `EMS_OCR_DEVICE` / `EMS_OCR_PHOTO_TIER` (`medium`; `small` is baked too) / `EMS_OCR_HAND_ROOT` (the handwriting recogniser's directory) / `EMS_HTR_ROOT` / `EMS_HTR_DEVICE` (the transcriber's; `auto`) / `EMS_HTR_EN_ROOT` / `EMS_HTR_EN_DEVICE` (the English transcriber's; `auto`) / `EMS_OCR_LANGUAGE_MIN_LETTERS` / `EMS_OCR_LANGUAGE_MIN_CONFIDENCE` / `EMS_OCR_MIN_LINE_SCORE` / `EMS_OCR_MAX_SIDE` (1600: the longest side the detector sees, whose arena grows with the area) / `EMS_OCR_GPU_MEM_MB` (2048 per session) (`ocr.read`; `auto`), `EMS_TEXT_EMBED_DEVICE` / `EMS_TEXT_EMBED_QUERY_DEVICE` / `EMS_TEXT_EMBED_MAX_TOKENS` (`text.embed`; documents on the card, queries on the CPU by default), …
Changing any of them is a restart.

Search queries have a model worker of their own, `EMS_QUERY_WORKER` (on by default; `0` keeps
everything in one worker). It holds `text.embed` with `role: query` and `vision.embed_text`, pinned
on the CPU, at normal priority, with a few threads (`EMS_QUERY_WORKER_THREADS`, the
cores less two and at most four: a short query split across every core waits on whichever thread
other work has descheduled) and under its own ceiling (`EMS_QUERY_WORKER_MAX_RSS`, `6g`; socket
`EMS_QUERY_WORKER_SOCKET`), while the batch worker leaves those families out of its
pins and runs at `EMS_WORKER_NICE` (10). In one worker a query shared the CPU and the interpreter
lock with diarization and took 1.5-9 s instead of 0.3. If the query worker is down, queries go to
the batch worker.

The card goes to batch work while there is some and to search while there is none (_3.4_). After
`EMS_CARD_QUIET_AFTER` seconds (600; `0` turns this off) without a batch model request, the batch
worker unloads what it holds and a third model worker, the card worker, starts with the query
families whose card copies fit what `nvidia-smi` then reports free — `EMS_CARD_TEXT_QUERY_MB` (1800)
first, then `EMS_CARD_SIGLIP_TEXT_MB` (3100), plus `EMS_CARD_CONTEXT_MB` (400) for its CUDA context
and `EMS_CARD_MARGIN_MB` (1024) to spare — so a small card takes text-query alone, or nothing, and a
host without a card changes nothing. Queries go to it once it has loaded them, and documents embed
on the card too when it has room. The next batch request waits until the card worker has stopped
and the document embedder has left the card, so a batch family never loads beside them; searches
never wait, because the query worker answers on the CPU throughout. The copies live in a process of
their own because libraries in one that can see the card touch it (peft loads adapters onto it),
and stopping the process gives back everything, its CUDA context included. Its socket is `EMS_CARD_WORKER_SOCKET` and its ceiling `EMS_CARD_WORKER_MAX_RSS` (`6g`).
Measured on fry: a query 107 ms (text) and 7 ms (pictures) on the card against ~0.3 s each on an
idle CPU and several seconds on a busy one. `health` lists both workers' models and reports the query worker under
`worker.query` (_3.1_). `capabilities` and `health` are how a caller learns what a
given EMS is running; there is no configuration API.

Sizing the tool lane: every tool runs single-threaded (`MAGICK_THREAD_LIMIT=1` in the image), so
`EMS_LIMIT_TOOLS` is a number of cores. Requests queue
FIFO within a lane; a caller that runs several kinds of tool work at once (renditions beside text extraction)
bounds each kind on its own side, or the cheap one waits behind the slow one.

## 7. Open questions

- `media.probe`: should `captured_at` interpretation (which tag wins, zone handling) move into
  EMS as a normalized `captured_at_best`, or stay a caller rule? Proposed: stay a caller rule for
  now; EMS reports candidates.
- `image.renditions`: `targets` is capped at 32. `cover` upscales on purpose (a square thumb
  owes its callers a square); should there be a `fit` that refuses instead?
- Streaming: SSE (`text/event-stream`) or NDJSON? SSE is proposed; both are trivial on both sides.
