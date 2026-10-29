# Sentinel-X

A **native Python desktop app** for local evidence-based fact checking and GraphSAGE virality prediction. The interface uses Qt Widgets through PySide6 Essentials. There is no JavaScript, Electron, React, npm, browser engine, or application HTTP server.

Qwen3.5-9B runs through the llama.cpp runtime included with the application. Multiple saved conversations can attach post datasets, retrieve evidence from `resources`, and invoke the GraphSAGE skill. JSON/JSONL data imports remain supported; those are dataset formats, not JavaScript.

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

Packaged releases **include llama.cpp**, its shared libraries, and runtime licenses. Users never download or compile the runtime during setup. Extract the release archive and keep the application directories intact. The companion **Sentinel-X Trainer** is a separate executable/application beside Sentinel-X.

On opening Sentinel-X, a setup dialog offers Qwen3.5-9B Q4_K_M weights, BGE-small embeddings, and optional FEVER evidence. Nothing downloads before you choose **Download selected**. **Skip for now** opens the workspace; missing items are offered again at later launches. Progress and retry appear in the setup dialog, with Activity available if you continue into the workspace. You can reopen the guide from Models → **Set up downloads…**.

Qwen and embeddings are selected by default when missing. FEVER is optional and unselected because it is a much larger download and indexing job. A configured local GGUF counts as available. Model downloads verify pinned source metadata and checksums. Inference and imported data remain local after setup.

For a fresh developer checkout, stage the precompiled runtime once with `python scripts/prepare_runtime.py` before launching or packaging. This is a build/developer step, not an end-user setup download. The current workspace already has its matching Linux binary staged in `build/llama/bin`. Packaging refuses to omit the runtime or use a binary that does not match the pinned commit.

## Use the app

- **Conversations:** create/search/rename/delete chats, export them as human-readable Markdown, import JSON/JSONL posts, map their fields, and select attached cascades. Ask about factual claims, virality, or both. Responses show verdicts, evidence excerpts, and separate graph predictions.
- **Resources:** select a directory named `resources` or another local directory, then index TXT, Markdown, JSON, JSONL, or CSV. Reindex after edits/deletions. Semantic search combines NumPy cosine similarity with SQLite full-text results. Until embeddings are downloaded and indexed, the reported fallback is keyword search.
- **Skills:** enable/disable local skills or import folders containing instructions and a tool manifest. GraphSAGE is included. Skill files cannot execute arbitrary code.
- **Models:** reopen setup, download/retry weights, import a GGUF, select a trained GraphSAGE checkpoint, open the separate trainer, configure context/acceleration, and run **Verify generation & tools**. llama.cpp is already included; executable selection is an advanced override.
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

## Markdown conversation exports

Select a conversation and choose **Export conversation (.md)**. The report includes:

- User and assistant messages with timestamps and turn IDs.
- Recorded model-provided reasoning summaries, claim explanations, verdicts, and citations.
- Skill/tool names, request arguments, results, and errors, in execution order.
- RAG reference filenames, source paths, provenance, and retrieved excerpts.
- The application system-prompt snapshot for each new turn, enabled skills/instructions, available functions, model/settings, and graph provenance.

Exports are UTF-8 Markdown, with readable headings and bullets rather than an application JSON dump. Reasoning summaries are explicit answer explanations, not a reconstructed internal reasoning trace. Historical prompts/summaries that were never recorded are marked unavailable; current defaults are never passed off as historical prompts. The file is a snapshot of saved records at export time, so a response still streaming may not yet appear.

```bash
sentinel-x export CONVERSATION_ID conversation.md
```

## Separate GraphSAGE trainer

Training is no longer a page inside Sentinel-X. Start **Sentinel-X Trainer** as a separate application:

```bash
sentinel-x-trainer
# Or:
python -m sentinel_x.trainer
# Or:
sentinel-x trainer
```

You can also open it from Models → **Open Sentinel-X Trainer (separate app)**. Import native Twitter15/16 directories or their parent, choose the virality threshold/observation window, train, inspect metrics, and select a checkpoint. Defaults remain **100 recorded posts**, **60 minutes**, and seed **42**.

The two applications share their data/checkpoint directory but have separate job queues. Starting or closing the trainer does not interrupt assistant jobs. A selected trained checkpoint becomes available in the main app's Models view. Use the same `--data-dir` or `SENTINEL_X_DATA_DIR` override in both applications when relocating state. Training and chat can run independently and share the machine's memory/compute resources.

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

For headless CI, set `QT_QPA_PLATFORM=offscreen` (Linux: prefix the test command with that assignment). The tests include a real Qt window, the assistant and separate trainer windows, conversation/attachment interactions, NumPy semantic retrieval, precompiled-runtime archive verification, and real PyTorch GraphSAGE training/checkpoint/skill inference. See [validation status](docs/VALIDATION.md) for limitations.

## Package standalone applications

On each target operating system:

```bash
python -m pip install -r requirements-dev.txt
python scripts/lock_manifest.py
python scripts/package_app.py
```

By default the packager builds **both** native applications and puts them side by side in one platform archive. Windows contains `Sentinel-X/Sentinel-X.exe` and `Sentinel-X-Trainer/Sentinel-X-Trainer.exe`; macOS contains two `.app` bundles; Linux contains two application directories. Keep both directories together if you want the main app's launcher button to find the trainer. End users do not need Python.

The build prepares the verified precompiled llama.cpp artifact if it is not already staged, then **requires** the executable, companion libraries, and licenses inside Sentinel-X's bundle. The trainer does not need a runtime copy. Only model weights and optional evidence are downloaded by the running app.

Use `--llama-dir /path/to/precompiled/binaries` to supply an existing pinned runtime, or `--app assistant` / `--app trainer` to build one application. No CMake step is needed for normal packaging. `scripts/build_llama.py` remains an optional developer tool for custom source builds. The output is a ZIP on Windows/macOS or a tarball on Linux. PyInstaller freezes existing Python and installed binary libraries; it does not compile PyTorch or Qt.

`.github/workflows/package.yml` defines Windows x64, Linux x64, and macOS Apple Silicon builds. Signing/notarization and public distribution are separate release steps; installers have not been built here.

## Source and local data

Application code lives in `src/backend/sentinel_x`: `desktop.py` is the assistant UI, `trainer.py` is the separate trainer, and `service.py` connects them to the backend. `exporting.py` writes Markdown reports and `setup_dialog.py` provides the startup guide. Bundled configuration and the GraphSAGE skill are package assets under `src/backend/sentinel_x/assets`. Tests, build scripts, and documentation are in `tests`, `scripts`, and `docs`.

Local conversations, resources, model catalogs, checkpoints, indexes, and logs use the platform application-data directory. Open it from the sidebar or set `SENTINEL_X_DATA_DIR` to override it. Existing conversations/checkpoints remain compatible. Reindex an older HNSW evidence index to migrate it to NumPy shards.

When upgrading from the former Veritas name, Sentinel-X automatically reuses an existing legacy data directory/database. The former `VERITAS_DATA_DIR` override is also recognized; `SENTINEL_X_DATA_DIR` takes precedence.

Keep weights, datasets, credentials, and generated indexes out of Git. `AGENTS.md` remains unchanged. [Skill authoring](docs/SKILLS.md) describes the supported extension format.
