# Validation status

## Verified after the native Python conversion

All 144 Python tests pass with Python 3.14 and installed dependencies; none are skipped.

- Native PySide6/Qt Widgets window creation, the assistant’s three navigation pages and the separate trainer, conversations, and graph attachments using the offscreen platform plugin.
- Pure-Python wheel build with bundled assets, followed by native-window startup from an isolated extraction of that wheel.
- Visual inspection of a native-window screenshot.
- Real PyTorch GraphSAGE training on synthetic cascades, checkpoint save/load, batch inference, and invocation through the skill handler.
- Native Twitter15/16 and JSON graph parsing, estimate acceptance, future-feature isolation, split isolation, metrics, citations, and persisted conversations/jobs.
- Actual NumPy memory-mapped vector indexing, semantic retrieval with a deterministic test embedder, incremental file deletion, and old-index migration. No HNSW extension is used.
- Precompiled-runtime archive checksum verification, unpacking, executable selection, platform/architecture matching, and traversal rejection using an in-memory test artifact.
- Prebuilt-only pip policy and requirements/project-metadata consistency.
- Python compilation and CLI help. JavaScript, npm, Electron, React, FastAPI, and application HTTP-server files have been removed.

Synthetic training verifies mechanics rather than Twitter15/16 prediction quality. The test embedder validates the semantic-index pipeline rather than BGE model quality. Offscreen Qt tests do not establish every desktop interaction or display-server configuration.

Additional verified behavior:

- Markdown exports include exact recorded application prompts, messages, explanation summaries, tool arguments/results/errors, and RAG filenames/excerpts. Legacy records show missing history explicitly.
- Startup downloads require an explicit button press; skipping starts no download and leaves future prompts enabled. Optional GPU engines have separate download controls alongside weights/evidence.
- A separate trainer window completes real training and publishes its checkpoint to the assistant. Independent job namespaces prevent trainer startup/closure from interrupting assistant work.
- A cached official Linux llama.cpp binary was staged under `build/llama/bin` and its real `--version` output verified: build 11960, pinned commit `8642541b2`.
- Packaging orchestration was tested using a mocked freezer and fixture binaries: the release includes both apps and requires runtime libraries/licenses in the assistant. This is not a claim that actual frozen executables were built.

## Main application usability pass

The earlier 104-test usability pass included 55 added persistence, runtime-protocol, and Qt interaction tests. These verify message revision history and conflict rejection; outdated-message handling; regeneration into a separate conversation; draft/attachment/thinking persistence; send shortcuts; citation inspection; scroll and settings preservation; active-job editing guards; cancellation and request restoration; atomic background imports; tool stages; thinking output-limit errors; and asynchronous runtime shutdown without affecting a subsequent process.

Runtime tests use mocked local transport to verify thinking On/Off payloads, prompt-template consistency, unsupported-mode errors, reasoning-event separation, cold/warm startup, and CPU fallback. They do not establish live model output quality.

Offscreen screenshots were inspected at 1320 × 900 and 900 × 660, including setup, conversation, generation, and failure states. Checked-in screenshots use illustrative fixture messages, not live model responses. See [USABILITY.md](USABILITY.md).

The installed precompiled runtime's help confirms support for `--reasoning-format deepseek`. Local Qwen weights exist, but a live generation check is blocked because this sandbox rejects loopback socket binding (`Operation not permitted`). Ruff is not installed in this session; installing it failed because network DNS is unavailable. Ruff remains configured in CI, but its result for this pass is unverified. Python compilation and Git whitespace checks pass.

The Background work regression tests verify that the reparented job panel is visible on first opening and reopening, that progress refreshes preserve the selected job, and that details and cancellation work. The previous blank window was caused by the page retaining its explicit hidden state after removal from the navigation stack.

## Download progress and runtime performance pass

Additional tests cover the actual urllib request construction and HTTP response handling with deterministic transport fixtures: full-file transfer and checksum, resume after cancellation, ignored/invalid ranges, completed partial files, stale receipts, missing Content-Length, early EOF, HTTP/DNS failures, and HTTPS enforcement. Integration tests run fixture model installs and Wikipedia download/extraction/indexing, verify readiness is withheld on failure, remove invalid cached archives, and retry the original setup choices. No live multi-gigabyte transfer was performed.

`python scripts/check_downloads.py` was executed against the configured sources: embedding and Qwen failed to resolve `huggingface.co`; Wikipedia failed to resolve `fever.ai`. The command exits nonzero on these failures. Run it outside this sandbox to validate a real small checksum-verified download plus bounded probes of the large files. Its fixture path is verified in the suite.

Runtime tests verify thinking preview and collapse, persisted durations, real SSE reasoning/progress parsing, empty-choice progress packets, prompt-cache requests, token-write elimination, and the 256-token thinking budget. The initial version discarded traces; the later expandable-trace behavior is documented below. Device detection on this machine reports only CPU-backed OpenVINO; automatic mode now selects native CPU execution. Existing logs show around 3.3 generated tokens/second, but that is historical evidence, not a speed benchmark of the new configuration. Live inference remains blocked by loopback binding restrictions, so no measured speed-up is claimed. Runtime parameters were checked against the [pinned llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/tools/server/README.md).

Progress rows and thinking preview/completion were visually inspected with illustrative fixtures; screenshots are in [USABILITY.md](USABILITY.md).

## Thinking assurance pass

Twelve additional tests verify explicit per-request budgets (Off = 0, On = configured limit), mode changes on an already-loaded runtime, unexpected reasoning in Off mode, whitespace-only deltas, both reasoning field names, combined answer/reasoning packets, tool transitions, cumulative phase timing, failure/cancellation cleanup, helper-mode enforcement, and diagnostic success/failure/inconclusive results. Empty thinking is not counted as an observed trace. Answers clearly identify enabled thinking that emitted no separate trace.

The real installed GGUF template was read and rendered locally: On ends with an open `<think>` section; Off supplies a closed, empty section. The pinned [server implementation](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/tools/server/server-common.cpp) confirms Boolean `enable_thinking`, explicit `reasoning_format`, and `reasoning_budget_tokens` handling. These options are now sent on every production request, including zero-budget structured helpers.

Live attempts were made with the installed Qwen weights. Both the app's HTTP runtime and the newer `llama-cli` cannot open a loopback port in this sandbox. Offline `llama-completion` runs using prompts rendered from the actual GGUF succeeded in generating the arithmetic answer 391 with thinking disabled. Enabled-mode raw completions closed the thinking section without a nonempty trace, including a bounded multi-step probe and a sampled probe. Those runs do **not** verify live thinking On, API budget enforcement, or server streaming. No claim of a successful full live thinking check is made.

`python scripts/check_thinking.py --model /path/to/model.gguf` provides a direct check through the production API path outside this sandbox. It reports a blocked loopback port here and exits nonzero. Its verification logic is covered by deterministic streamed-response tests and returns an inconclusive result when On emits no trace.

## FEVER contention and model-load audit

Read-only copies of the real app state showed that FEVER's archive and extracted evidence already existed. The active job was embedding/indexing its first Wikipedia file. A previous attempt was cancelled after approximately 168,768 passages; the original whole-file transaction discarded that work. No committed evidence files or passages were visible in the index snapshot. The prior writer also held the retrieval lock for the full import, and chat shared the same worker. Together these could prevent a request from reaching Qwen at all, even with Thinking Off.

The implementation separates interactive and maintenance workers, permits WAL-backed keyword reads during indexing, commits resumable initial FEVER imports in batches, and builds FEVER keyword-first without the full-corpus embedding pass. The original 64-passage keyword batches have since increased to 2,048. The official archive's evidence subdirectory excludes its macOS metadata and license files. Regression tests prove that chat can retrieve committed passages and finish before an indexing writer is released, cancellation resumes without duplicates, changed/deleted partial files are handled, and the real archive layout completes without metadata parse errors. Existing complete-file update rollback tests still pass.

The installed embedding model was loaded locally with network access disabled and generated two finite, normalized 384-dimensional vectors. The configured llama-server executable reports the expected build 11960 / commit 8642541b2. Actual Qwen arithmetic generation was already exercised offline; live application-server inference remains unverified because this sandbox cannot bind the required port. Qwen's official model card confirms both thinking and non-thinking modes. An actual 500-page FEVER sample (1,668 passages) was successfully keyword-indexed in temporary storage without embeddings; its timing is not a full-corpus or production-disk benchmark.

Additional UI coverage verifies that an incomplete streamed JSON summary becomes a readable, explicitly unvalidated draft immediately; final validation replaces it. Escape sequences and partial Unicode are handled. Raw partial JSON is not exposed as chat text.

## Not yet verified

- Live build-time runtime/model downloads, Qwen thinking/function parsing through the application server, and full FEVER download/indexing.
- BGE weight loading and retrieval quality on real evidence.
- Standalone packaging, signing/notarization, and application execution on Windows/macOS. Native Linux UI creation is verified, but no standalone executable has been built here.
- Real GPU acceleration and held-out performance on the full Twitter15/16 datasets.

## Optional GPU engines and response-format recovery

Seventeen new regressions cover platform/backend selection, CUDA library pairing, mandatory published checksums, real archive extraction using transport fixtures, library search paths, version/device probes, cancelled installs, failed activation preserving the previous runtime, activation waiting for a model job, separate CPU fallback when a GPU loader fails, and UI download-to-activation progress. Software Vulkan devices and backend log lines do not count as GPUs. Answer checks cover display of a model-provided reply to “test” through a deterministic runtime fixture, retrieval/tool/history bypass, a single formatting retry with Thinking Off, citation downgrades after repair, helpful schema/server/streaming errors, and empty keep-alive frames. The exact retry instruction and mode are retained in exports.

The official Linux Vulkan archive (33.8 MB) and release metadata were downloaded with curl. Python's network path still reports DNS restrictions here, so the downloaded metadata and cached archive were supplied to the production installer. It verified the published SHA-256, unpacked the real binary, ran its version/device probes, and registered build 11960 / commit 8642541b2 in temporary storage. This verifies actual package installation, not an unrestricted end-to-end Python transfer. No GPU device is exposed in this sandbox; accelerated generation remains unverified.

The actual Qwen GGUF also generated `{"summary":"Connectivity check acknowledged.","claims":[]}` offline using the app's exact new “test” prompt, Thinking Off template, and an object-constrained grammar. The response passed the app's answer validator. This run took about 6.3 seconds including model load, with about 2.8 seconds of prompt/generation work; it is not an application-server latency benchmark. Starting the real server still fails because loopback binding is blocked. Updated setup and Models screenshots were visually inspected.

Standalone packaging uses PyInstaller on each target OS and prebuilt Python dependencies; llama.cpp is mandatory in assistant release bundles. No npm or native compilation step is part of normal setup. If a dependency wheel is unavailable for the user's Python/platform combination, pip fails rather than compiling it.

## Deferred FEVER indexing, expandable thinking, and setup preference

Thirteen additional tests verify archive-only setup/downloads, recognition of older saved ZIPs, restart and offline indexing, extraction reuse without modifying timestamps, CRC-based adoption/repair of older extracted files, cancellation after a 2,048-passage checkpoint followed by an offline resume without duplicates, missing-archive errors without a hidden download, and generic resource updates skipping managed FEVER. SQL tracing verifies bounded batch commits and the removal of full-table chunk scans for newly imported keyword files. Progress samples are coalesced while preserving stage transitions and file completions, so long jobs do not continually serialize a growing sample history.

Thinking tests verify collapsed defaults, HTML escaping, bounded 64,000-character retention, message persistence, and reopening/collapsing a completed trace through the real Qt action handler. Token updates remain in memory; the completed trace is saved with the message without duplication in job diagnostics. Older discarded traces are explicitly unavailable. Setup tests verify that startup suppression persists after reopening storage, manual setup still opens, unchecking restores prompts, and FEVER defaults to download-only with an explicit immediate-index option.

The new exact short-message prompt was run offline with the real installed Qwen GGUF and returned `{"summary":"Test received.","claims":[]}`, without the former connectivity/status announcement. An actual 5,000-page FEVER sample (16,461 passages) was successfully indexed using the new keyword path. A small local sample did not demonstrate a meaningful elapsed-time improvement, so no full-corpus speed-up or completion-time estimate is claimed. The reductions in repeated extraction, per-file table scans, small commits, and unbounded progress-history rewrites are verified independently. Full-corpus indexing remains optional and unbenchmarked. Setup, deferred FEVER, and expanded-thinking screenshots were visually inspected.

## Git history

The workspace's `.git` directory is empty and mounted read-only. Milestone commits are maintained in `/tmp/sentinel-x-development.git` with this workspace as the work tree and exported to `/tmp/sentinel-x-development.bundle`. An ignored copy named `sentinel-x-development.bundle` is retained at the workspace root. The GitHub repository remains `BluSnake57/543-research-project`; application naming does not rename it.

```bash
git bundle list-heads sentinel-x-development.bundle
git clone sentinel-x-development.bundle sentinel-x
```

This preserves source history without modifying the protected `.git` directory. The bundle excludes weights, datasets, and generated indexes.

`python scripts/restore_git.py` restores normal Git metadata in this project folder when run outside the restricted session. It preserves working files, restores the main branch from the bundle, configures the existing GitHub remote, and refuses to overwrite unknown metadata. Its integration tests use actual local Git repositories and bundles, including preserved local edits and an already-working checkout. Running `git init` directly in this session fails because the `.git` mount is read-only.

## Sentinel-X data migration and local-file protection

Ten additional tests verify the actual legacy-directory move and database upgrade, active model/runtime path rebasing, preservation of historical prompt snapshots, stable evidence citation IDs, complete-index reuse, partial-index resumption without duplicate passages, committed WAL data, retry after an interrupted migration, refusal to merge distinct existing data, and retention of original databases. Moving existing model files preserves their inode in the Linux fixture: weights are not copied or downloaded. Existing environment overrides remain compatible. Older assistant and trainer instances must be closed before the first upgraded launch. Live user storage was not migrated from this restricted session.

Actual Git ignore tests cover private environment/workspace settings, model weights, downloaded datasets/runtimes, imported graphs/skills, checkpoints, indexes, and application data. They also verify that source configuration, bundled skills, `.env.example`, and the resource-directory placeholder remain eligible for version control. Tracked files were inspected and contain no downloaded models, private settings, or user databases. Python compilation and Git whitespace checks pass. Ruff is not installed in this environment.
