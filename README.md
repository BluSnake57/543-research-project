# Sentinel-X

A **native Python desktop app** for local evidence-based fact checking and GraphSAGE virality prediction. The interface uses Qt Widgets through PySide6 Essentials. There is no JavaScript, Electron, React, npm, browser engine, or application HTTP server.

Qwen3.5-9B runs through a managed llama.cpp executable. Multiple saved conversations can attach post datasets, retrieve evidence from `resources`, and invoke the GraphSAGE skill. JSON/JSONL data imports remain supported; those are dataset formats, not JavaScript.

## Install and run

Use a 64-bit Python 3.11–3.14 installation. Python 3.11 is the conservative choice for dependency availability; the native application and tests have also run here on Python 3.14.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

Install and launch:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m sentinel_x
```

`sentinel-x` and `sentinel-x-desktop` are also installed as launch commands. No Node.js or compiler is required for normal setup.

**Dependencies use precompiled wheels.** `requirements.txt` enforces `--only-binary=:all:` so pip does not silently compile a native dependency. This repository itself is pure Python and installs in editable mode. If pip reports that no matching binary exists, use a supported 64-bit Python/platform combination rather than installing a compiler. The previous `hnswlib` dependency has been removed; the semantic index now uses precompiled NumPy and memory-mapped vectors. An old environment may retain unused packages, but they are no longer imported or required.

First launch queues a download of an official **precompiled llama-server**, Qwen3.5-9B Q4_K_M GGUF, and BGE-small-en-v1.5 embeddings. Runtime selection matches the OS and architecture, verifies GitHub's published SHA-256 digest, and persists the chosen artifact. Models also verify pinned source metadata and checksums. Download progress, errors, retry, and cancellation are available in Activity and Models. CPU runtime downloads are preferred on Windows/Linux; macOS uses the matching CPU/Metal release. You can choose an existing executable instead.

Internet access is required for downloads; inference and imported data remain local afterward. The runtime installer does not fall back to compiling code. If no compatible artifact is published or its checksum is unavailable, Models explains the error and lets you select a verified local executable. Actual online downloads and Qwen generation have not been exercised in this workspace.

## Use the app

- **Conversations:** create/search/rename/delete chats, import JSON/JSONL posts, map their fields, and select attached cascades. Ask about factual claims, virality, or both. Responses show verdicts, evidence excerpts, and separate graph predictions.
- **Resources:** select a directory named `resources` or another local directory, then index TXT, Markdown, JSON, JSONL, or CSV. Reindex after edits/deletions. Semantic search combines NumPy cosine similarity with SQLite full-text results. Until embeddings are downloaded and indexed, the reported fallback is keyword search.
- **Skills:** enable/disable local skills or import folders containing instructions and a tool manifest. GraphSAGE is included. Skill files cannot execute arbitrary code.
- **Training:** import native Twitter15/16 directories or their parent, train, inspect metrics, and select a checkpoint. Defaults: viral at **100 recorded posts**, observed during the first **60 minutes**, seed **42**.
- **Models:** download/retry models and the precompiled runtime, import a GGUF, select an executable, configure context/acceleration, and run **Verify generation & tools**.
- **Activity:** inspect jobs, progress, results, errors, and cancellation.

For missing graph metadata, inspect the cascade, ask Qwen to propose links/timing, and review the proposal's tables and assumptions before accepting. Estimates remain prominently labeled experimental. No virality prediction is permitted for an unaccepted estimated graph.

Example post dataset:

```json
[
  {"id":"p1","text":"Mars has two moons.","author_id":"u1","timestamp":"2026-01-01T00:00:00Z"},
  {"id":"p2","text":"Sharing this claim","author_id":"u2","timestamp":"2026-01-01T00:03:00Z","parent_id":"p1"}
]
```

Numeric timestamps are Unix seconds; string timestamps require a timezone. Optional `cascade_id` groups posts. For a text-only bundle that should be estimated as one cascade, choose that grouping during import. Large bundles that exceed the model context are rejected rather than silently dropping posts.

## FEVER and prediction behavior

Optional FEVER setup downloads the historical Wikipedia evidence corpus, then builds an index. Claim labels and evaluation annotations stay outside the evidence directory. Citations retain page/sentence provenance. Factual verdicts are **supported**, **refuted**, or **insufficient evidence**; a missing valid citation cannot support a definitive verdict.

GraphSAGE predicts recorded cascade growth, independently of factual support. Twitter15/16 rumor labels are not virality labels. Training derives its target from recorded cascade size and excludes future nodes/edges from observed features. Real-world reach and estimated-graph quality are not established by dataset test metrics. See [data and model details](docs/DATA_AND_MODELS.md).

Full FEVER indexing needs substantial disk space and time. Memory-mapped vector shards bound search memory, but exact semantic search over a large corpus trades speed for simpler installation. Qwen's 9B weights and context also need substantial RAM; CPU responses may be slow.

## CLI and tests

```bash
sentinel-x train /path/to/datasets --threshold 100 --window 60 --epochs 80 --seed 42
sentinel-x index /path/to/resources
sentinel-x predict /path/to/canonical-graphs.json /path/to/checkpoints/MODEL_ID.json
```

The prediction input is the canonical graph shown in graph details, not raw post JSON. Use `sentinel-x --data-dir /custom/path` before the subcommand to relocate local state.

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
ruff check src/backend scripts tests
```

For headless CI, set `QT_QPA_PLATFORM=offscreen` (Linux: prefix the test command with that assignment). The tests include a real Qt window, all six pages, conversation/attachment interactions, NumPy semantic retrieval, precompiled-runtime archive verification, and real PyTorch GraphSAGE training/checkpoint/skill inference. See [validation status](docs/VALIDATION.md) for limitations.

## Package standalone applications

On each target operating system:

```bash
python -m pip install -r requirements-dev.txt
python scripts/lock_manifest.py
python scripts/package_app.py
```

PyInstaller freezes the Python application and installed binary libraries. This packages existing code; it does not compile PyTorch, Qt, or a C++ runtime from source. Output is a Windows ZIP containing `Sentinel-X.exe` and supporting files, a macOS ZIP containing `Sentinel-X.app`, or a Linux tarball containing the executable and libraries. Keep the entire extracted application directory together. End users do not need Python.

The packaged app downloads the precompiled runtime and models on first launch. To bundle an existing runtime instead, use `python scripts/package_app.py --llama-dir /path/to/precompiled/binaries`. No CMake step is needed for normal packaging. `scripts/build_llama.py` remains an optional developer tool for custom accelerated source builds.

`.github/workflows/package.yml` defines Windows x64, Linux x64, and macOS Apple Silicon builds. Signing/notarization and public distribution are separate release steps; installers have not been built here.

## Source and local data

Application code lives in `src/backend/sentinel_x`: `desktop.py` is the Qt UI and `service.py` directly connects it to inference, retrieval, skills, and training. Bundled configuration and the GraphSAGE skill are package assets under `src/backend/sentinel_x/assets`. Tests, build scripts, and documentation are in `tests`, `scripts`, and `docs`.

Local conversations, resources, model catalogs, checkpoints, indexes, and logs use the platform application-data directory. Open it from the sidebar or set `SENTINEL_X_DATA_DIR` to override it. Existing conversations/checkpoints remain compatible. Reindex an older HNSW evidence index to migrate it to NumPy shards.

When upgrading from the former Veritas name, Sentinel-X automatically reuses an existing legacy data directory/database. The former `VERITAS_DATA_DIR` override is also recognized; `SENTINEL_X_DATA_DIR` takes precedence.

Keep weights, datasets, credentials, and generated indexes out of Git. `AGENTS.md` remains unchanged. [Skill authoring](docs/SKILLS.md) describes the supported extension format.
