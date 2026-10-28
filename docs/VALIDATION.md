# Validation status

## Verified in the initial workspace

- Python bytecode compilation for backend and packaging scripts.
- Dependency-light regression suite: graph parsing/validation, JSON mapping, estimated-graph acceptance, future-feature isolation, source-event split isolation, retrieval import/update/deletion, citation provenance, FEVER label exclusion, download resume/checksums/cancellation, skill restrictions, conversation tool loops, persistence, and queued jobs.
- Real PyTorch GraphSAGE training for three epochs on synthetic cascades, checkpoint save/load, and batch inference, using a locally cached PyTorch 2.14.1 CPU wheel with Python 3.14.8.
- Node tests for the local API route guard and restricted preload surface, plus JavaScript syntax checks.

The final Python run executes 31 tests: 28 pass, including real training and GraphSAGE skill inference; 3 HTTP integration tests skip because FastAPI is unavailable.

Synthetic training verifies mechanics, not prediction quality on Twitter15/16. No dataset accuracy claim is made.

## Not yet verified

- FastAPI HTTP integration tests: FastAPI is unavailable locally.
- TypeScript type-check, Vite build, Electron UI launch, and desktop interaction tests: frontend dependencies are not installed.
- Actual Qwen inference/tool calling: no GGUF or llama.cpp binary is available.
- Full FEVER download, embedding model execution, and HNSW semantic indexing: required assets/dependencies are unavailable. Keyword retrieval is tested.
- Packaging, application launch, signing, and acceleration checks on Windows, Linux, and macOS.

Shell network access fails DNS resolution in this environment, and npm has no suitable cached dependencies. CI definitions and reproducible commands are included to run these checks in a network-enabled build environment. Dependency-dependent Python tests report explicit skips.

## Git history

The workspace's `.git` directory was empty and mounted read-only. Normal `git init` failed with `Read-only file system`. Four milestone commits are maintained in `/tmp/veritas-development.git` with this workspace as the work tree, and exported to `/tmp/veritas-development.bundle`. A copy named `veritas-development.bundle` is retained at the workspace root (ignored by Git). This is a portable history, not a repair of the protected `.git` directory.

After copying the bundle to a normal writable location, inspect or clone it with:

```bash
git bundle list-heads /tmp/veritas-development.bundle
git clone /tmp/veritas-development.bundle veritas
```

The bundle contains source commits and does not contain downloaded weights, private datasets, or generated indexes.
