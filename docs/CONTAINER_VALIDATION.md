# Scope / Docker integration validation

Integration date: 2026-11-05. Branch: `feature/scope-docker-local`.
Base: local Scope tip `876099d` (one commit ahead of fetched Scope `b39676c`,
without divergence). Merged Docker tip: `551e9ad`. Existing branches were not moved.

## Passed in the integration workspace

- `QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v`:
  **160 tests passed**. This includes Scope UI/provenance, both model profiles,
  offload choices, downloads, exports, setup, and container integration checks.
- New integration checks run startup configuration in fresh processes for CPU
  and CUDA preferences, preserving the model/offload/device selection and fixture
  files in both Qwen/SAE pairs. They reject changed preferences and corrupt files.
  These are filesystem/process checks, not Docker volume-recreation tests.
- Real spawned worker startup rejection and cancellation leave no child worker.
  These checks require no model weights or GPU.
- `ruff check src/backend scripts tests docker`, formatting checks for container
  Python/new integration files, shell syntax checks, and `git diff --check` passed.
- Host `pip check` and runtime imports passed. A real bitsandbytes 0.50.2 CPU NF4
  linear operation passed. The host uses Python 3.15 / PyTorch 2.15.0+cu132;
  this does **not** verify the Python 3.11 / PyTorch 2.9.1 container environment.
- Official standalone Compose **2.24.7**, verified against its release checksum,
  validated CPU and CUDA configurations both offline and online. The checks assert
  distinct local tags, `pull_policy: never`, correct wheel indexes/GPU reservations,
  network isolation, localhost binding, read-only secrets, and persistent home.
  Reproduce with `python3 scripts/check_container_config.py` on a Docker host or
  pass `--compose /path/to/docker-compose` without a daemon.
- CI shell blocks passed syntax checks; both matrix selection scripts were run
  with temporary environment files. Compose confirmed the CPU runner removes GPU
  reservations while retaining the intended CPU or CUDA local image.
- A temporary checker using Docker's `github.com/moby/patternmatcher` v0.6.0
  enumerated **51 allowed context files** and passed **22 inclusion/exclusion
  probes**, including nested secrets, local Scope weights/checkpoint metadata,
  caches, and build outputs. Required nested application assets remain included.
  Only the exact shipped GraphSAGE `.pt`/`.json` pair is allowed from model weights.

## Blocked / not established

Both `docker compose build` and the CPU override build returned `docker: command
not found`. No image was built or container started in this workspace. Therefore
container dependency resolution/imports, noVNC/desktop operation, authenticated
HTTP exports, container worker cleanup, and volume recreation remain unverified
here. The retained Container desktop CI workflow now builds both variants and
runs those checks; it has not been dispatched or claimed to have passed.

Host PyTorch reports CUDA unavailable (NVML initialization fails). No real 9B or
2B model pair was downloaded or loaded. CUDA inference, Scope capture under a
real model, and peak RAM/VRAM acceptance remain unverified. A successful desktop
startup, tiny CPU NF4 check, or CUDA tensor check cannot establish that 9B fits
16 GiB RAM / 6 GiB VRAM. See [Scope hardware acceptance](SCOPE_HARDWARE.md).

## Docker-host follow-up

Use the [container guide](container.md) to build and start each variant, then run
`pip check`, `check_runtime.py --variant cpu|cuda`, and `smoke_test.py` inside it.
The CI workflow uses disposable volumes to seed settings and model-storage marker
files, recreates the service, and checks persistence. Never run its persistence
seed helper against a volume containing user preferences.

On an NVIDIA host, additionally run `check_cuda.py` and the full real-model
acceptance checks documented in the Scope hardware notes. Keep the configurable
16 GiB container default until workload measurements justify another limit;
changing the container limit does not change Scope's application memory budgets.
