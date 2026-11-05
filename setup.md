# Scope branch setup

This branch is under validation. See [Scope configuration](docs/SCOPE.md) for
both model pairs, CPU offloading, memory limits and outstanding checks.

Install dependencies with `python run.py setup --dev` (which invokes
`scripts/setup_environment.py`), then
launch with `python run.py`. Both entry points and dependency installation have
been exercised in this checkout. Models provides paired download buttons for
9B and 2B; both can remain installed. Select the desired pair separately.

Rerunning setup preserves the selected model and CPU-offload policy unless you
explicitly pass their flags. `python run.py setup --download-both` downloads both
model/SAE pairs and embeddings without changing the selected model.

Setup also accepts `--model 9b` or `--model 2b`, `--download-models`,
`--device cpu` or `--device cuda`, and `--cpu-offload none`, `sae`, or `modules`.
For example:

```sh
python scripts/setup_environment.py --model 2b --device cpu --download-models
```

Argument handling and download orchestration are tested offline; the full model
download/loading workflow has not yet passed a real-model smoke test.
The old `--runtime` option and GGUF import are removed.

`--skip-install` checks the environment and installs local GraphSAGE checkpoint
pairs without reinstalling dependencies. Keep each original `.pt`/`.json` pair
in `model weights/`. The separate trainer remains available through
`python run.py trainer`. FEVER download and resumable indexing remain in the app.

`--reset-settings` clears model, offload, device and other preferences while
preserving conversations and files. `--factory-reset` deletes application
storage, including both downloaded model pairs. `--reset-venv` recreates the
virtual environment. Close the application before setup or reset operations.
Use `--data-dir PATH` consistently for custom storage. Legacy GGUF/runtime files
are unused and are not automatically deleted on upgrade.

Verified offline checks:

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
.venv/bin/ruff check src/backend scripts tests
.venv/bin/python -m pip check
```

The requested limits are 16 GiB CPU RAM or 6 GiB CUDA VRAM. These are validation
targets, not measured guarantees. CPU-only is explicit and slower; CUDA requires
a functioning compatible NVIDIA driver. Offloading can reduce VRAM while raising
RAM consumption. The application rejects configurations failing its memory
checks; full loading peaks and latency remain to be measured.
