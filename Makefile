# EMS. `make help` lists the targets.
#
# Local: the venv runs the front's tests (no models, no tools needed beyond what the tests
# skip without). Container: the real thing, built and run here, or on a host by whatever deploys
# it, with the host's settings as these variables.

.DEFAULT_GOAL := help
SHELL := /bin/bash

VENV        ?= .venv
PY          := $(VENV)/bin/python
IMAGE_NAME  ?= ems
CONTAINER_NAME ?= ems
HOST        ?= fry

# Build arguments, passed as --build-arg: these win over the Containerfile's defaults.
TORCH_CUDA      ?= cu130
FFMPEG_BUILD    ?= autobuild-2026-08-31-13-27
FFMPEG_ASSET    ?= ffmpeg-n9.0.1-11-ge47273f4d9-linux64-gpl-9.0

# Run arguments. The four host directories are mounted at the SAME container paths the caller
# mounts them at (/files, /old, /files-volatile, /files-tmp): the paths the caller resolves
# inside its container are the paths EMS opens inside its own, and nothing is ever copied.
# EMS_ROOTS names exactly those four, read-only for the originals.
EMS_PORT        ?= 8181
EMS_KEY         ?= let-me-in
FILES_PATH      ?= /srv/files
OLD_PATH        ?= /srv/old
VOLATILE_PATH   ?= /srv/files-volatile
TMP_PATH        ?= /srv/files-tmp
EMS_ROOTS       ?= /files:ro,/old:ro,/files-volatile:rw,/files-tmp:rw
EMS_MEMORY      ?= 16g
GPU             ?=
GPU_FLAG        := $(if $(GPU),--device nvidia.com/gpu=all,)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- local ---

venv: ## Create the dev venv (python3-venv without ensurepip is fine; pip bootstraps it)
	python3 -m venv --without-pip $(VENV)
	pip3 --python $(PY) install --disable-pip-version-check -e '.[test]' 2>/dev/null || pip3 --python $(PY) install --disable-pip-version-check fastapi 'uvicorn[standard]' httpx pytest pytest-asyncio

test: ## Run the front's tests
	$(PY) -m pytest -q

serve: ## Serve locally from the venv (no models; EMS_ROOTS and EMS_KEY from the environment)
	$(PY) -m ems

# --- container ---

build: ## Build the image (TMPDIR on the graph root: the model layers are tens of GB)
	@# What `health` reports as `build`. On a host the checkout may have no .git, and whatever
	@# synced it has written the file, so a failed `git` must leave the file alone.
	@if commit=$$(git describe --always --dirty 2>/dev/null) && date=$$(git log -1 --format=%cI 2>/dev/null); then \
		echo "$$commit $$date" > ems/BUILD; fi
	TMPDIR="$$(podman info --format '{{.Store.GraphRoot}}')/tmp"; mkdir -p "$$TMPDIR"; \
	TMPDIR="$$TMPDIR" podman build -t $(IMAGE_NAME) -f Containerfile \
		--build-arg TORCH_CUDA=$(TORCH_CUDA) \
		--build-arg FFMPEG_BUILD=$(FFMPEG_BUILD) \
		--build-arg FFMPEG_ASSET=$(FFMPEG_ASSET) \
		.

run: ## Run the container (GPU=1 for NVIDIA; FILES_PATH/OLD_PATH/VOLATILE_PATH/TMP_PATH mount at /files, /old, /files-volatile, /files-tmp)
	@mkdir -p $(VOLATILE_PATH) $(TMP_PATH)
	podman run --replace -d --name $(CONTAINER_NAME) --restart always --stop-timeout 60 \
		--health-cmd "python3 -m ems.healthcheck" --health-interval 30s --health-timeout 15s --health-start-period 120s \
		--memory $(EMS_MEMORY) --memory-swap $(EMS_MEMORY) \
		$(GPU_FLAG) --network=host \
		-e EMS_PORT=$(EMS_PORT) -e EMS_KEY=$(EMS_KEY) \
		-e EMS_ROOTS=$(EMS_ROOTS) \
		$(if $(EMS_LIMIT_TOOLS),-e EMS_LIMIT_TOOLS=$(EMS_LIMIT_TOOLS),) \
		$(if $(EMS_LIMIT_MODELS),-e EMS_LIMIT_MODELS=$(EMS_LIMIT_MODELS),) \
		$(if $(EMS_LIMIT_ENCODES),-e EMS_LIMIT_ENCODES=$(EMS_LIMIT_ENCODES),) \
		$(if $(EMS_CPU_ENCODE_MAX_MB),-e EMS_CPU_ENCODE_MAX_MB=$(EMS_CPU_ENCODE_MAX_MB),) \
		$(if $(EMS_CPU_ENCODE_MB_PER_1K_FRAMES),-e EMS_CPU_ENCODE_MB_PER_1K_FRAMES=$(EMS_CPU_ENCODE_MB_PER_1K_FRAMES),) \
		$(if $(EMS_CPU_ENCODE_MAX_SEGMENTS),-e EMS_CPU_ENCODE_MAX_SEGMENTS=$(EMS_CPU_ENCODE_MAX_SEGMENTS),) \
		$(if $(EMS_VIDEO_ENCODER),-e EMS_VIDEO_ENCODER=$(EMS_VIDEO_ENCODER),) \
		$(if $(EMS_TOOL_STALL),-e EMS_TOOL_STALL=$(EMS_TOOL_STALL),) \
		$(if $(EMS_CARD_BUDGET_MB),-e EMS_CARD_BUDGET_MB=$(EMS_CARD_BUDGET_MB),) \
		$(if $(EMS_CARD_SHARE_MB),-e EMS_CARD_SHARE_MB=$(EMS_CARD_SHARE_MB),) \
		$(if $(EMS_EXPECT_GPU),-e EMS_EXPECT_GPU=$(EMS_EXPECT_GPU),) \
		$(if $(EMS_MODEL_PINNED),-e EMS_MODEL_PINNED=$(EMS_MODEL_PINNED),) \
		$(if $(EMS_MODEL_IDLE_EVICT),-e EMS_MODEL_IDLE_EVICT=$(EMS_MODEL_IDLE_EVICT),) \
		$(if $(EMS_MODEL_MAX_RSS),-e EMS_MODEL_MAX_RSS=$(EMS_MODEL_MAX_RSS),) \
		$(if $(EMS_TITANET_DEVICE),-e EMS_TITANET_DEVICE=$(EMS_TITANET_DEVICE),) \
		$(if $(EMS_IMAGE_EMBED_DEVICE),-e EMS_IMAGE_EMBED_DEVICE=$(EMS_IMAGE_EMBED_DEVICE),) \
		$(if $(EMS_COPY_DETECT_DEVICE),-e EMS_COPY_DETECT_DEVICE=$(EMS_COPY_DETECT_DEVICE),) \
		$(if $(EMS_TRANSNET_DEVICE),-e EMS_TRANSNET_DEVICE=$(EMS_TRANSNET_DEVICE),) \
		$(if $(EMS_OCR_DEVICE),-e EMS_OCR_DEVICE=$(EMS_OCR_DEVICE),) \
		$(if $(EMS_IMAGE_EMBED_TEXT_DEVICE),-e EMS_IMAGE_EMBED_TEXT_DEVICE=$(EMS_IMAGE_EMBED_TEXT_DEVICE),) \
		$(if $(EMS_TEXT_EMBED_DEVICE),-e EMS_TEXT_EMBED_DEVICE=$(EMS_TEXT_EMBED_DEVICE),) \
		$(if $(EMS_TEXT_EMBED_QUERY_DEVICE),-e EMS_TEXT_EMBED_QUERY_DEVICE=$(EMS_TEXT_EMBED_QUERY_DEVICE),) \
		$(if $(EMS_LR_ASD_DEVICE),-e EMS_LR_ASD_DEVICE=$(EMS_LR_ASD_DEVICE),) \
		$(if $(EMS_WHISPER_LANGUAGE),-e EMS_WHISPER_LANGUAGE=$(EMS_WHISPER_LANGUAGE),) \
		$(if $(EMS_WHISPER_LANGUAGES),-e EMS_WHISPER_LANGUAGES=$(EMS_WHISPER_LANGUAGES),) \
		$(if $(EMS_WHISPER_LANGUAGE_FLOOR),-e EMS_WHISPER_LANGUAGE_FLOOR=$(EMS_WHISPER_LANGUAGE_FLOOR),) \
		$(if $(EMS_WHISPER_LANGUAGE_WINDOWS),-e EMS_WHISPER_LANGUAGE_WINDOWS=$(EMS_WHISPER_LANGUAGE_WINDOWS),) \
		$(if $(TZ),-e TZ=$(TZ),) \
		-v $(FILES_PATH):/files:ro \
		-v $(OLD_PATH):/old:ro \
		-v $(VOLATILE_PATH):/files-volatile \
		-v $(TMP_PATH):/files-tmp \
		$(IMAGE_NAME)

stop: ## Stop the container
	podman stop -t 60 $(CONTAINER_NAME)

logs: ## Tail the container's log
	podman logs -f $(CONTAINER_NAME)

shell: ## A bash shell inside the running container
	podman exec -it $(CONTAINER_NAME) bash

gpu-check: ## Real inference on every model inside the running container (what device each bound to)
	podman exec $(CONTAINER_NAME) python3 scripts/gpu_selftest.py $(GPU_CHECK_ARGS)

# The whisper diagnostic, run against the image with the archive mounted read-only at the same
# paths the service sees (/files with its symlinks into /old).
whisper-coverage: ## Compare whisper decodes (WHISPER_COVERAGE_ARGS="--file /files/..." or "--sweep --file ... --silent ...")
	podman run --rm $(GPU_FLAG) -v $(PWD):/work:ro -v $(FILES_PATH):/files:ro -v $(OLD_PATH):/old:ro -w /work --entrypoint python3 $(IMAGE_NAME) \
		scripts/bench/whisper-coverage.py $(WHISPER_COVERAGE_ARGS)

OCR_SPOTCHECK_DIR ?= /var/tmp/ocr-spotcheck
# Weights a bench engine fetches (PaddleOCR-VL), kept between runs and out of the image.
OCR_SPOTCHECK_CACHE ?= /var/tmp/ocr-spotcheck-cache
ocr-spotcheck: ## Read pictures with OCR engines side by side (list.tsv and report.html in OCR_SPOTCHECK_DIR)
	mkdir -p $(OCR_SPOTCHECK_DIR) $(OCR_SPOTCHECK_CACHE)
	podman run --rm $(GPU_FLAG) -v $(PWD):/work:ro -v $(FILES_PATH):/files:ro -v $(OLD_PATH):/old:ro -v $(OCR_SPOTCHECK_DIR):$(OCR_SPOTCHECK_DIR) \
		-v $(OCR_SPOTCHECK_CACHE):/bench-cache -e HF_HUB_CACHE=/bench-cache -w /work --entrypoint python3 $(IMAGE_NAME) \
		scripts/bench/ocr-spotcheck.py --list $(OCR_SPOTCHECK_DIR)/list.tsv --out $(OCR_SPOTCHECK_DIR) $(OCR_SPOTCHECK_ARGS)

SHOTS_BENCH_DIR ?= /var/tmp/shots-bench
# Asks the running EMS (video.shots) on the host's network; ffmpeg in the image cuts the strips.
shots-bench: ## video.shots against listed cuts and clips without (report.tsv in SHOTS_BENCH_DIR)
	mkdir -p $(SHOTS_BENCH_DIR)
	podman run --rm --network=host -e EMS_KEY=$(EMS_KEY) -e EMS_PORT=$(EMS_PORT) -v $(PWD):/work:ro -v $(FILES_PATH):/files:ro -v $(OLD_PATH):/old:ro \
		-v $(VOLATILE_PATH):/files-volatile:ro -v $(SHOTS_BENCH_DIR):$(SHOTS_BENCH_DIR) -w /work --entrypoint python3 $(IMAGE_NAME) \
		scripts/bench/shots-bench.py --out $(SHOTS_BENCH_DIR) $(SHOTS_BENCH_ARGS)

health: ## Ask the running EMS how it is
	@curl -s -H "Authorization: Bearer $(EMS_KEY)" http://localhost:$(EMS_PORT)/v0/health | python3 -m json.tool

.PHONY: help venv test serve build run stop logs shell gpu-check health whisper-coverage ocr-spotcheck shots-bench
