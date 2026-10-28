# Veritas

A local desktop research app for **evidence-based fact checking** and **GraphSAGE virality prediction**. Electron/React provides the desktop interface; a packaged Python worker manages llama.cpp, evidence retrieval, and training. Existing `AGENTS.md` is preserved.

- Multiple persistent conversations with post-dataset attachments and streamed responses.
- Qwen3.5-9B GGUF through an application-managed `llama-server`.
- Local RAG from a directory named `resources`, including optional FEVER Wikipedia evidence.
- Native Twitter15/16 imports and reproducible binary GraphSAGE training in the UI or CLI.
- Local skill folders; the bundled GraphSAGE skill lets Qwen request graph predictions.
- Explicit review of inferred propagation links/times, with experimental-result warnings.

This repository contains the implementation and packaging workflows. **It does not include prebuilt installers, downloaded model weights, or a bundled llama.cpp binary.** See [validation status](docs/VALIDATION.md) for checks actually run.

## Development setup

Use **Python 3.11** (recommended), Node.js 22, npm, Git, and a C/C++ toolchain with CMake. `hnswlib` may require compilation. Use a virtual environment so dependencies do not modify system Python.

The desktop targets Electron 45, listed on the [official stable release page](https://releases.electronjs.org/). Install dependencies online to generate the npm lockfile; one could not be resolved in this offline workspace.

```bash
python3.11 -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm install
```

`requirements.txt` lists runtime dependencies and installs this repository in editable mode. `pyproject.toml` also supports the standard `python -m pip install -e .` workflow. For tests, linting, and packaging:

```bash
python -m pip install -r requirements-dev.txt
```

Build the pinned llama.cpp runtime (network access required):

```bash
python scripts/build_llama.py --backend cpu
```

Use `--backend metal` on Apple Silicon, `--backend cuda` with a compatible NVIDIA CUDA toolkit, or `--backend vulkan` with a Vulkan SDK. CPU is always available in the CPU build; automatic GPU offload requires an accelerated build. In Models, choose the resulting `build/llama/bin/llama-server` executable (`.exe` on Windows), or assemble the worker as described below.

```bash
npm run dev
```

The launcher uses `.venv` automatically; set `VERITAS_PYTHON` to override the interpreter. First launch queues automatic Qwen and embedding downloads. Downloads need internet access; inference and retrieval operate offline afterward. Use Models to retry/resume downloads, import a local GGUF, choose context/acceleration, and run **Verify generation & tools**. FEVER is downloaded only when requested.

For a production frontend build without an installer: `npm run build`, then `npm start` with the Python environment and runtime configured. This is a proposed development workflow; dependency installation and the frontend build could not be completed in the initial restricted environment.

## Use the app

1. Complete model setup in **Models** and run the runtime smoke check.
2. In **Resources**, select the resources directory and choose **Index directory**, or download/index FEVER. TXT, Markdown, JSON, JSONL, and CSV are supported. Reindex to apply changes or deletions. Missing embeddings produce an explicitly reported keyword-only index; download embeddings and reindex for hybrid retrieval.
3. In **Training**, import `twitter15`, `twitter16`, or their parent directory. Train and select a checkpoint. Defaults: viral at **100 recorded posts**, observation window **60 minutes**, seed **42**.
4. In **Conversations**, import JSON/JSONL posts, review field mappings, and select graph attachments. For text-only bundles, select grouping as one cascade when appropriate. Inspect incomplete graphs, request estimation, and **review/accept** the proposal before prediction.
5. Ask about factual claims, virality, or both. Verdicts and graph results appear separately, with evidence excerpts and prediction metadata.

Conversations and resources are shared only within this local installation; graph attachments scope Qwen's prediction tools to the conversation. Large bundles that exceed the configured context are rejected rather than silently dropping posts. Split these into smaller batches or increase context if memory permits.

### Example post input

```json
[
  {"id":"p1","text":"Mars has two moons.","author_id":"u1","timestamp":"2026-01-01T00:00:00Z"},
  {"id":"p2","text":"Sharing this claim","author_id":"u2","timestamp":"2026-01-01T00:03:00Z","parent_id":"p1"}
]
```

Numeric timestamps are Unix **seconds**; string timestamps must include a timezone. Optional `cascade_id` groups posts explicitly. JSON field aliases and a Qwen-assisted mapping workflow are supported. IDs and observed relationships are validated; Qwen cannot silently overwrite an observed parent or timestamp. See [data and model behavior](docs/DATA_AND_MODELS.md).

## CLI

The CLI and UI share training, preprocessing, and inference implementations:

```bash
veritas train /path/to/datasets --threshold 100 --window 60 --epochs 80 --seed 42
veritas index /path/to/resources
veritas predict /path/to/canonical-graphs.json /path/to/checkpoints/MODEL_ID.json
```

The `predict` input is the canonical graph representation shown by **Inspect / review**, not raw posts. Multiple graphs may be passed as an array. Estimated graphs must include explicit acceptance. `veritas --data-dir /custom/path train ...` overrides the application-data location.

## Build installers

Build on each target operating system; Python freezing and native libraries are platform-specific:

```bash
python scripts/lock_manifest.py
python scripts/build_llama.py --backend cpu
python scripts/package_worker.py
npm run make
```

Use `metal` on Apple Silicon. Artifacts appear in `out/make`: Windows Squirrel installer, Linux DEB/RPM/ZIP, and macOS ZIP containing the `.app`. Python, Docker, and a browser are not required by end users. The packaged runtime and Python libraries increase installer size; model weights download separately.

`lock_manifest.py` resolves immutable model revisions and upstream artifact hashes without downloading weights. Packaging includes this release manifest. Development installs pin resolved metadata in the application-data model catalog on first download. Downloads verify LFS SHA-256 or Git blob SHA-1, support resume, and write a ready marker only when complete. FEVER lacks an upstream checksum in this integration; it uses HTTPS, ZIP CRC validation, and a recorded local digest.

`.github/workflows/package.yml` builds on Windows x64, Linux x64, and macOS Apple Silicon. Default workflow variants use CPU on Windows/Linux and Metal on macOS. CUDA/Vulkan variants can be built with their SDKs. Apple signing/notarization reads `APPLE_IDENTITY`, `APPLE_ID`, `APPLE_PASSWORD`, and `APPLE_TEAM_ID`; unsigned local packages are development artifacts. Windows public distribution also requires configuring a signing identity. No installers have been built or signed in this workspace.

## Checks

```bash
python -m unittest discover -s tests -v
ruff check src/backend scripts tests
npm test
npm run build
```

Python tests cover graph parsing, temporal/split leakage, retrieval provenance, label exclusion, estimate review, downloads, conversation tool calls, persistence, API authentication, and real PyTorch training/checkpoint inference. Dependency-dependent tests skip explicitly when their libraries are absent. CI installs dependencies before running these checks. Real Qwen inference additionally requires the multi-gigabyte model and compiled runtime; use Models → Verify generation & tools.

## Layout and local data

- `src/desktop`: Electron main process, restricted preload bridge, and IPC checks.
- `src/renderer`: React interface.
- `src/backend/veritas`: local API, inference, RAG, skills, graph import, training, and CLI.
- `skills/graphsage`: bundled skill instructions and tool manifest.
- `tests`, `scripts`, `docs`: verification, build tools, and model/data notes.

Open the application-data directory from the sidebar. It contains `resources`, `models`, `graphs`, `checkpoints`, `skills`, `index`, logs, and SQLite records. Set `VERITAS_DATA_DIR` to relocate it. Imported resource directories may remain external and are read in place. Do not commit datasets, model weights, credentials, or generated indexes.

To add a skill, follow [the local skill format](docs/SKILLS.md). Contributions should include focused behavior tests and an explanation of verification limits.
