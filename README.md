# EMS — EiN Media Service

A stateless service that answers questions about media bytes and produces media
bytes: probing, renditions, transcodes, fingerprints, face detection, speech,
image-text embeddings, text embeddings, document text. It owns the tools
(ffmpeg, ImageMagick, exiftool, chromaprint) and the models (SigLIP2,
InsightFace, a text embedder (jina-v5-text), PP-OCRv6, whisper, NeMo diarization and
voiceprints, LR-ASD) behind one HTTP API, so a caller never depends on which
tool or model answers. **The contract is [docs/api.md](docs/api.md).**

It is a keyed service on identically mounted volumes, and it keeps no cache: the
caller decides where results live.

## Shape

Two processes in one container. The **front** (`python -m ems`, FastAPI)
validates requests, spawns the tools, and holds the composition. The **model
worker** (`ems/modelworker/`, the resident model server) holds the models behind
a Unix socket and recycles itself on its RSS ceiling without taking the listener
down; the front supervises and restarts it.

```
POST /v0/<namespace>/<op>     JSON in, JSON out; Accept: text/event-stream for progress
GET  /v0/health               liveness, loaded models, VRAM, degraded fallbacks
GET  /v0/capabilities         ops, roots, tools, models, limits
```

Files are absolute paths under configured **roots**
(`EMS_ROOTS=/files:ro,/old:ro,/files-volatile:rw,/files-tmp:rw`); anything
outside is refused. Mount the caller's directories at the same container paths
on both sides and nothing is ever copied.

## Run

```bash
make build                         # the image (tens of GB: models are baked in)
make run GPU=1 EMS_KEY=… FILES_PATH=… OLD_PATH=… VOLATILE_PATH=… TMP_PATH=…
make health                        # GET /v0/health
make gpu-check                     # real inference per model, reports the device each bound to
```

On another host, sync this checkout there and run the same targets with that host's variables.

### A GPU host

Before the first `run GPU=1` on a new host, and again after reinstalling its OS:

1. **NVIDIA driver >= 610** (the nvenc API the bundled ffmpeg wants; see
   `Containerfile`).
2. **nvidia-container-toolkit >= 1.18**, the first to ship
   `nvidia-cdi-refresh.service`. That unit writes the CDI spec
   (`/var/run/cdi/nvidia.yaml`) from the devices the host actually has, at every
   boot and whenever the driver or toolkit changes. Do not run
   `nvidia-ctk cdi generate
   --output=/etc/cdi/…` by hand: that spec names the
   card numbers of that moment and podman falls back to it.
3. **Hook the podman fix into that refresh**, once per host, with sudo there:

   ```bash
   sudo bash deploy/install-cdi-refresh.sh     # on the host itself
   ```

   Podman 4.9.3 (Ubuntu 24.04 / Mint 22) rejects the spec as nvidia-ctk >= 1.17
   writes it (`unknown field "additionalGids"`), and
   `--device nvidia.com/gpu=all` then finds no GPU. The installer makes the
   refresh unit run `deploy/fix-cdi.sh` after each regeneration and removes any
   `/etc/cdi/nvidia.yaml`. It ends with the unit's status and a count of the
   spec's `cdiVersion`, `additionalGids` and `/dev/dri/card` lines. It survives
   reboots and toolkit upgrades.
4. `make gpu-check` once EMS runs: every model reports
   the device it bound to.

A container keeps the device paths it was created with. If a reboot renumbers
`/dev/dri/cardN`, `podman start ems` fails. `make run` (`--replace`) recreates
the container from the current spec.

Configuration is environment only — see `docs/api.md` §6 and `ems/config.py`.
The model worker's own settings are `EMS_*` too (`EMS_WHISPER_MODEL`,
`EMS_MODEL_PINNED`, …); the front maps them onto the worker's `CFG_*` namespace
when it spawns it.

## Develop

```bash
make venv && make test             # the front's tests: no models, no GPU
```

The tests script the model worker's socket protocol (`tests/conftest.py`), so
every op is exercised end to end without torch. `tests/test_probe_tools.py` runs
against a real exiftool when one is installed.

## Status

Implemented: the model primitives and composites, `media.probe`,
`fingerprint.compute`, `document.extract`, the renditions (`image.renditions`,
`video.poster`, `video.storyboard`, `audio.waveform`), the shots (`video.shots`) and the transcodes (`video.transcode`,
`audio.transcode`). `docs/api.md` is the contract; `image.decode` and
`audio.extract` are specified there and not yet needed by any caller.

## Known issues

- **The ffmpeg pin is written twice.** `Containerfile` declares
  `ARG FFMPEG_BUILD`/`FFMPEG_ASSET` with defaults and the Makefile passes the
  same two names as `--build-arg`, so the Makefile's values win and a repin made
  only in the Containerfile builds the old ffmpeg without a word (podman reuses
  the cached layer). Bump both together, and check `ffmpeg -version` inside the
  container afterwards.
