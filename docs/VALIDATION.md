# Validation status

## Verified after the native Python conversion

All 36 Python tests pass with Python 3.14 and installed dependencies; none are skipped.

- Native PySide6/Qt Widgets window creation, all six navigation pages, conversations, and graph attachments using the offscreen platform plugin.
- Pure-Python wheel build with bundled assets, followed by native-window startup from an isolated extraction of that wheel.
- Visual inspection of a native-window screenshot.
- Real PyTorch GraphSAGE training on synthetic cascades, checkpoint save/load, batch inference, and invocation through the skill handler.
- Native Twitter15/16 and JSON graph parsing, estimate acceptance, future-feature isolation, split isolation, metrics, citations, and persisted conversations/jobs.
- Actual NumPy memory-mapped vector indexing, semantic retrieval with a deterministic test embedder, incremental file deletion, and old-index migration. No HNSW extension is used.
- Precompiled-runtime archive checksum verification, unpacking, executable selection, platform/architecture matching, and traversal rejection using an in-memory test artifact.
- Prebuilt-only pip policy and requirements/project-metadata consistency.
- Python compilation and CLI help. JavaScript, npm, Electron, React, FastAPI, and application HTTP-server files have been removed.

Synthetic training verifies mechanics rather than Twitter15/16 prediction quality. The test embedder validates the semantic-index pipeline rather than BGE model quality. Offscreen Qt tests do not establish every desktop interaction or display-server configuration.

## Not yet verified

- Live official runtime/model downloads, actual Qwen generation/function parsing, and full FEVER download/indexing.
- BGE weight loading and retrieval quality on real evidence.
- Standalone packaging, signing/notarization, and application execution on Windows/macOS. Native Linux UI creation is verified, but no standalone executable has been built here.
- Real GPU acceleration and held-out performance on the full Twitter15/16 datasets.

Standalone packaging uses PyInstaller on each target OS and prebuilt Python dependencies. No npm or native compilation step is part of normal setup. If a dependency wheel is unavailable for the user's Python/platform combination, pip fails rather than compiling it.

## Git history

The workspace's `.git` directory is empty and mounted read-only. Milestone commits are maintained in `/tmp/veritas-development.git` with this workspace as the work tree and exported to `/tmp/sentinel-x-development.bundle`. An ignored copy named `sentinel-x-development.bundle` is retained at the workspace root.

```bash
git bundle list-heads sentinel-x-development.bundle
git clone sentinel-x-development.bundle sentinel-x
```

This preserves source history without modifying the protected `.git` directory. The bundle excludes weights, datasets, and generated indexes.
