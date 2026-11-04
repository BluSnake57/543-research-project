# Sentinel-X setup

Use `scripts/setup_environment.py` from this checkout. The same Python script runs
on Windows, macOS, and Linux. It creates `.venv`, installs the repository's Python
dependencies, checks their consistency, and installs valid local GraphSAGE checkpoints.
Run it as your normal user, with Sentinel-X closed.

## Prerequisites

- A 64-bit Python installation in the project's declared range, 3.11–3.15, with
  `venv` and `pip`. Python 3.13 is the version documented for the original research.
  Available dependency wheels still depend on Python version, OS, and architecture.
- Internet for initial package/model downloads and enough disk space for PyTorch,
  runtime libraries, and the approximately 5.7 GB Qwen model. FEVER is optional and
  requires substantially more storage after extraction.
- A graphical desktop for the application. GPU acceleration is optional.

## Model files

Place original trained checkpoint pairs in the repository's `model weights` folder:

```text
model weights/
  fe44637dfc4d49fe9cd402c4ce6cc21d.pt
  fe44637dfc4d49fe9cd402c4ce6cc21d.json
  Qwen3.5-9B-Q4_K_M.gguf                       (optional)
```

One pair represents one trained model variant, not one tweet. Retain matching
filenames and the original metadata. The JSON includes normalization statistics,
architecture, cutoff, feature names, and evaluation results. Neither
`checkpoint_map.json` nor `model_configuration.json` replaces it.

The demonstration `.pt` and its matching checkpoint JSON are included in this
checkout and are installed automatically. For additional checkpoints, obtain both
files from the original training output. Setup reports incomplete pairs and exits
with status 2 after configuring the environment. Do not invent normalization
values to fill in missing metadata.

Small GraphSAGE `.pt` checkpoints and their `.json` metadata in this folder are
versioned in Git. Large downloaded models remain ignored. The default checkpoint
map lives in `results/final/checkpoint_map.json`. Feature tables under `results/` are versioned;
they are used in training/evaluation and are not imported as weights.

## Windows (PowerShell)

From the repository root, using an installed Python 3.13:

```powershell
py -3.13 scripts/setup_environment.py
py -3.13 run.py
```

Substitute your installed supported version for `-3.13`. Activation is unnecessary,
so PowerShell's activation-script execution policy does not affect these commands.

## macOS

Use a supported Python installation matching your Mac's architecture:

```bash
python3 scripts/setup_environment.py
python3 run.py
```

The `cpu` runtime option below also covers the project's Apple Metal runtime.
CUDA is not an option for macOS. Prebuilt packages must exist for your architecture.

## Linux

Use a supported Python installation:

```bash
python3 scripts/setup_environment.py
python3 run.py
```

If Python reports that `venv` or `ensurepip` is unavailable, install your
distribution's venv support for that Python version. Linux Qt also needs system
graphics libraries. This repository's Ubuntu CI installs `libegl1`, `libopengl0`,
and `libxkbcommon0`; other distributions use different package names. Missing Qt
platform libraries must be installed with your OS package manager.

## Download models and runtime

To also download Qwen, the embedding model, and a precompiled CPU runtime during
initial setup, append:

```bash
python3 scripts/setup_environment.py --download-models --runtime cpu
```

On Windows, use `py -3.13` in place of `python3`. For subsequent runs, add
`--skip-install` to reuse the installed environment. Downloads use the project's
existing verification and runtime selection code. Supported runtime options are
`cpu`, `cuda`, `vulkan`, and `rocm`, subject to platform and pinned release availability.

For NVIDIA on supported Windows/Linux machines, choose `--runtime cuda` and install
a compatible NVIDIA driver through your OS. PyTorch GPU support and llama.cpp GPU
support are separate; selecting the runtime does not change the installed PyTorch
build. GPU packages may require additional system libraries. The script does not
install drivers or change system packages.

Without those flags, use the application's setup screen / Settings > Models to
download weights and choose a runtime. A local GGUF with the exact filename above
is configured in place; keep that file in place. FEVER is optional and is configured
through the app; use Evidence > Build / resume FEVER index after downloading it.

## Rerun after adding checkpoint files

```bash
python3 scripts/setup_environment.py --skip-install
```

Use `--venv venv` if reusing an existing environment named `venv` instead of `.venv`.
The script reuses identical checkpoint files and refuses to overwrite different
files with the same name. It preserves an existing checkpoint selection. With no
selection, it selects the installed `twitter15_emotional_language` entry from the
optional map, or the sole installed checkpoint. For multiple other checkpoints,
choose one in Settings > Models.

Launch with `python3 run.py` (`py -3.13 run.py` on Windows). The launcher uses
`.venv` automatically; pass `--venv venv` for a different environment. It forwards
application arguments, so `python3 run.py trainer` opens the trainer and
`python3 run.py --help` lists application commands.

Default application storage is:

| Platform | Location |
| --- | --- |
| Windows | `%LOCALAPPDATA%\Sentinel-X` |
| macOS | `~/Library/Application Support/Sentinel-X` |
| Linux | `~/.local/share/sentinel-x` (or `$XDG_DATA_HOME/sentinel-x`) |

Checkpoints go in `checkpoints/` below that location. The existing
`SENTINEL_X_DATA_DIR` override is respected. For a custom `--data-dir`, pass the same
directory when launching, before `gui`:

```bash
python3 scripts/setup_environment.py --skip-install --data-dir ./user-data
python3 run.py --data-dir ./user-data gui
```

## Verify and use

On Linux/macOS (use `.\.venv\Scripts\python.exe` on Windows):

```bash
.venv/bin/python -m pip check
.venv/bin/python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

A false CUDA result is expected for CPU-only setups and Macs. In the app, verify
Qwen generation, select the installed GraphSAGE checkpoint, and import/attach a
propagation graph before requesting a virality prediction. The hybrid checkpoint
requires source text for Qwen feature extraction. Training datasets are only needed
for retraining or reproducing experiments; see the README for training examples.
The supplied metadata preserves its original Windows training-file path. When
using CLI prediction with saved semantic features on another machine, pass
`--llm-features results/twitter15_llm_features.json` explicitly. Interactive hybrid
prediction generates the required scores through Qwen automatically.

## Development and checks

Use `--dev` during setup to include the development requirements. From the root:

```bash
.venv/bin/python -m unittest discover -s tests -p test_environment_setup.py -v
.venv/bin/python -m ruff check run.py scripts/setup_environment.py tests/test_environment_setup.py
.venv/bin/python -m ruff format --check run.py scripts/setup_environment.py tests/test_environment_setup.py
```

The installer tests exercise real PyTorch checkpoint loading in temporary folders,
missing metadata, corrupt weights, conflict handling, selection, and path conventions.
They do not download models. Linux checks were run locally; Windows/macOS command
paths are covered by tests, but native installation on those OSes still requires
verification. Dependency constraints are ranges, so setup is not an exact recreation
of the original research environment.
