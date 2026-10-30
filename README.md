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

Packaged releases **include llama.cpp**, its shared libraries, and runtime licenses. Extract the release archive and keep the application directories intact. Optional GPU engines can be downloaded without compiling anything. The companion **Sentinel-X Trainer** is a separate executable/application beside Sentinel-X.

On opening Sentinel-X, a setup dialog offers Qwen3.5-9B Q4_K_M weights, BGE-small embeddings, and optional FEVER evidence. Nothing downloads before you choose **Download selected**. **Skip for now** opens the workspace; missing items are offered again at later launches unless you check **Don’t show setup at startup again**. That preference survives restarting. Progress and retry appear in the setup dialog, with Background work available if you continue into the workspace. You can reopen the guide from Settings → Models → **Set up downloads…**, including to re-enable startup prompts.

Qwen and embeddings are selected by default when missing. FEVER is optional and unselected because it is a much larger download and indexing job. A configured local GGUF counts as available. Model downloads verify pinned source metadata and checksums. Inference and imported data remain local after setup.

For a fresh developer checkout, stage the precompiled runtime once with `python scripts/prepare_runtime.py` before launching or packaging. This is a build/developer step, not an end-user setup download. The current workspace already has its matching Linux binary staged in `build/llama/bin`. Packaging refuses to omit the runtime or use a binary that does not match the pinned commit.

The setup dialog and **Settings → Models** also offer an engine dropdown: **Vulkan** for AMD/NVIDIA, **CUDA** for NVIDIA, **ROCm / HIP** for AMD, and **CPU / Apple Metal**. Options depend on the platform and the pinned [official llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b11960). Vulkan needs working graphics drivers; CUDA includes its matching downloadable libraries and still needs NVIDIA drivers; ROCm needs compatible AMD hardware/drivers. Choose **Download engine** to install a verified package. **Use after download** queues activation after current model work; **Use installed engine** switches an existing download. Failed device checks preserve the current engine. **Use included engine** restores the bundled runtime. Progress, cancellation, and retry are available in Background work. A model-load failure in a GPU engine retries using the installed CPU or included engine.

## Use the app

- **Chat:** create and search conversations, import posts, attach cascades, and ask about factual support or potential reach. Rename, export, and delete are under **Conversation actions**. The conversation list and dataset drawer collapse to give answers more space.
- **Evidence:** choose a local document directory and index TXT, Markdown, JSON, JSONL, or CSV. Reindex after edits/deletions. Semantic search combines NumPy cosine similarity with SQLite full-text results; keyword search remains available without downloaded embeddings.
- **Settings → Models:** reopen setup, import a GGUF, select a GraphSAGE checkpoint, or open the separate trainer. **Advanced settings** contains runtime selection, context, acceleration, and generation/tool verification.
- **Settings → Skills:** enable, disable, or import local skill folders. GraphSAGE is included. Skill files cannot execute arbitrary code.
- **Background work:** readable task names, current files, transferred bytes, speed, estimated time remaining, elapsed time, and progress bars. Select a task for details, cancellation, or download retry. Percentages describe the current file or phase; connecting, verification, and other unmeasured work use an animated indicator.

### Thinking and generation progress

The **Thinking · On / Off** switch beside the composer controls thinking for the next request. Off is the default; On may take longer. The selection is remembered per conversation and captured at submission, so changing it during a response affects the next request. Unsupported templates produce an explicit error rather than silently ignoring the selection. Structured field-mapping and graph-estimation helpers keep thinking off. Thinking is capped at 256 tokens by default; change **Maximum thinking tokens** under Settings → Models → Advanced settings (64–2,048 tokens).

A status panel shows elapsed time and actual backend stages, including evidence retrieval, model loading, request preparation, generation, tool execution, and citation checks. **Thinking** appears when the runtime reports reasoning activity; “Thinking enabled” describes the request setting. Tool stages can repeat. The local model’s thinking streams inside an expandable dropdown while it runs. On completion it collapses to **Thought for …**; click that row to view the saved trace again. New completed answers retain the most recent 64,000 characters locally with the message, including across restarts and in exports. Truncated traces are labeled. Older responses only retained duration, so their discarded traces cannot be recovered. If thinking was enabled but no separate trace was emitted, the answer says so instead of inventing one. Answer text also streams as a draft while generation is running, including with Thinking Off. Claims and citations are validated before the final response replaces the draft. **Stop** cancels the active response, and failed/cancelled requests offer **Restore request** for editing and resubmission.

Short connectivity checks such as “test” use a compact prompt without retrieval, history, or tools. They display the model's actual reply, without the former “online and ready” announcement. Regular answers are instructed to address the latest request directly without a connectivity preface. Factual answers that fail JSON validation receive one constrained formatting retry with Thinking Off; citation validation still applies. Empty or invalid server responses identify the failed operation instead of exposing a raw JSON parser error. Formatting retries are recorded in the conversation export.

Use **Ctrl+Enter** (or **Cmd+Enter**) to send; Enter inserts a newline. Draft text, pending attachments, and thinking settings survive switching conversations and restarting. Scrolling up to read older messages keeps your position during updates. Clicking a citation opens its source excerpt and provenance; **Evidence and original research details** opens the preserved analysis record.

### Edit messages

Every user message and assistant answer has **Edit**, with an inline editor and **Save changes / Cancel**. Empty messages are rejected; user messages support up to 6,000 characters. Edited messages show a revision-history link. Saving is blocked during that conversation’s generation; the editor keeps your text so you can stop or wait and retry.

Later messages stay visible and are marked **Based on an earlier message version**. Future model requests use edited text and omit outdated assistant answers from their history. Editing an assistant’s text does not change its recorded assessments, evidence, tool outputs, or original prompt snapshots; these remain clearly labeled as original research records.

**Regenerate from here** on a user message creates a separate conversation through that revised request and generates a fresh answer. The existing thread is retained. Markdown exports include current text, revisions, thinking settings and duration, and outdated status.

See the [workspace screenshots and interaction guide](docs/USABILITY.md).

For missing graph metadata, inspect the cascade, ask Qwen to propose links/timing, and review the proposal's tables and assumptions before accepting. Estimates remain prominently labeled experimental. No virality prediction is permitted for an unaccepted estimated graph.

Example post dataset:

```json
[
  {"id":"p1","text":"Mars has two moons.","author_id":"u1","timestamp":"2026-01-01T00:00:00Z"},
  {"id":"p2","text":"Sharing this claim","author_id":"u2","timestamp":"2026-01-01T00:03:00Z","parent_id":"p1"}
]
```

Numeric timestamps are Unix seconds; string timestamps require a timezone. Optional `cascade_id` groups posts. For a text-only bundle that should be estimated as one cascade, choose that grouping during import. Large bundles that exceed the model context are rejected rather than silently dropping posts.

## Performance and download checks

Chat/model work and maintenance tasks use separate workers. Evidence searches read committed data without waiting for the index writer. The runtime stays loaded across requests and reuses its prompt cache. Streamed text is kept in memory instead of rewriting the jobs database for every generated token. Automatic acceleration detects available devices; a CPU-only OpenVINO device uses native CPU execution instead of being treated as a GPU. A 9B model can still be slow on CPU. These changes are covered by protocol tests; a live speed comparison has not been run in the restricted development environment.

To check real download connectivity from your normal terminal:

```bash
python scripts/check_downloads.py
```

This uses the app's downloader to fetch and checksum-verify a small embedding configuration file, then requests the first four bytes of the Qwen model and Wikipedia archive. It prints a per-source report and exits nonzero if any source fails. It downloads no full model weights or Wikipedia archive. Full transfers, resume, cancellation, truncation detection, checksums, extraction, and indexing are covered with deterministic transport fixtures in the test suite. Live execution of this command in the current sandbox reports DNS failures for Hugging Face and FEVER.

To verify Thinking Off/On through the actual app runtime with your existing model:

```bash
python scripts/check_thinking.py --model /path/to/model.gguf
```

This uses temporary app state, starts the bundled local server, and sends both modes through the production streaming code. It checks the answer and whether nonempty thinking was observed. Exit 0 means both verified, 1 means a check failed, and 2 means Thinking On was accepted but emitted no separate trace. No model download is started. This command currently reports a blocked loopback port inside the development sandbox; it needs a normal environment that permits the local model server.

Thinking Off sends an explicit zero reasoning budget; On sends the configured limit. Nonempty thinking returned in Off mode produces a compatibility error. Structured import/estimation helpers also enforce Off. A nonempty trace, its streamed display, and its collapse are covered by protocol/UI tests; live server verification remains incomplete in this sandbox.

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

You can also open it from Settings → Models → **Open separate GraphSAGE trainer**. Import native Twitter15/16 directories or their parent, choose the virality threshold/observation window, train, inspect metrics, and select a checkpoint. Defaults remain **100 recorded posts**, **60 minutes**, and seed **42**.

The two applications share their data/checkpoint directory but have separate job queues. Starting or closing the trainer does not interrupt assistant jobs. A selected trained checkpoint becomes available in the main app's Models view. Use the same `--data-dir` or `SENTINEL_X_DATA_DIR` override in both applications when relocating state. Training and chat can run independently and share the machine's memory/compute resources.

## FEVER and prediction behavior

Optional FEVER setup defaults to **download only**: it keeps the Wikipedia ZIP and evaluation annotations without extracting or indexing the corpus. To use it later, open **Evidence → Build / resume FEVER index**. This action uses the saved archive without network access. Setup also has an unchecked option to build the index immediately. Saved archives from earlier versions are recognized. Cancel the current index in Background work to postpone it; the archive, verified extracted files, and committed passages remain available for resuming after restart.

Indexing reuses verified extracted files instead of rewriting the corpus on every attempt. It indexes only `wiki-pages` evidence, excludes metadata/license assets, and uses keyword search without full-corpus embeddings. Keyword inserts use batches of 2,048 passages, avoid unnecessary full-table scans for new files, and keep repeated progress samples from growing the job history indefinitely. Committed passages become searchable during indexing. A generic resources-folder update skips managed FEVER data; its separate index action remains explicit. BGE remains available for other local documents. Unindexed FEVER text cannot ground chat answers; chat continues using other indexed sources. Claim labels and evaluation annotations stay outside the evidence directory. Citations retain page/sentence provenance. Factual verdicts are **supported**, **refuted**, or **insufficient evidence**; a missing valid citation cannot support a definitive verdict.

GraphSAGE predicts recorded cascade growth, independently of factual support. Twitter15/16 rumor labels are not virality labels. Training derives its target from recorded cascade size and excludes future nodes/edges from observed features. Real-world reach and estimated-graph quality are not established by dataset test metrics. See [data and model details](docs/DATA_AND_MODELS.md).

Full FEVER indexing still needs disk space and time, but the default keyword-first path avoids creating two copies of every sentence embedding. Full-corpus semantic indexing would require much more storage and time; it is not part of FEVER setup. Qwen's 9B weights and context also need substantial RAM; CPU responses may be slow.

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

Application code lives in `src/backend/sentinel_x`: `desktop.py` assembles the assistant workspace, `conversation_ui.py` provides conversation presentation/editing/progress components, `trainer.py` is the separate trainer, and `service.py` connects them to the backend. `exporting.py` writes Markdown reports and `setup_dialog.py` provides the startup guide. Bundled configuration and the GraphSAGE skill are package assets under `src/backend/sentinel_x/assets`. Tests, build scripts, and documentation are in `tests`, `scripts`, and `docs`.

Local conversations, resources, model catalogs, checkpoints, indexes, and logs use the platform application-data directory. Open it from the sidebar or set `SENTINEL_X_DATA_DIR` to override it. Existing conversations/checkpoints remain compatible. Reindex an older HNSW evidence index to migrate it to NumPy shards.

When upgrading from the former Veritas name, Sentinel-X automatically reuses an existing legacy data directory/database. The former `VERITAS_DATA_DIR` override is also recognized; `SENTINEL_X_DATA_DIR` takes precedence.

Keep weights, datasets, credentials, and generated indexes out of Git. `AGENTS.md` remains unchanged. [Skill authoring](docs/SKILLS.md) describes the supported extension format.
