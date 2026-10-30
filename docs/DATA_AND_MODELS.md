# Data and model behavior

## Twitter15/16

Accepted native layout:

```text
datasets/
  twitter15/
    tree/<source-tweet-id>.txt
    source_tweets.txt
    label.txt
  twitter16/
    tree/<source-tweet-id>.txt
    source_tweets.txt
    label.txt
```

An edge line contains `['uid', 'tweet ID', 'delay in minutes']->[...]`. Root sentinel tuples are omitted as graph nodes; `(user ID, tweet ID)` identifies a propagation node because retweets can share tweet IDs. Duplicate edges are removed, negative delays are clamped to zero with a warning, conflicting times keep the earliest, and malformed records are reported. Cyclic/disconnected cascades are rejected. Inspect import-job results for diagnostics rather than assuming every source file was accepted.

The original rumor labels are retained as metadata only. **Virality is a derived target**, not an original Twitter15/16 annotation: final *recorded* cascade size ≥ the configured threshold. This is not a measurement of worldwide lifetime reach. Missing posts and limited capture horizons constrain the target.

Only nodes at or before the observation window, and edges between them, enter features. Final cascade size is used solely for the target. No rumor label, user ID, tweet ID, or held-out statistic is a model feature. Node features are a constant, normalized observed delay, log indegree, log outdegree, log depth, and root indicator. Message passing uses bidirectional edges while the features retain propagation structure. Reply text is unnecessary because the native datasets do not include all replies.

Training uses two **mean-aggregation GraphSAGE** layers (32 hidden features), ReLU, global mean pooling, and a binary head, implemented in native PyTorch without optional compiled graph extensions. This is the GraphSAGE mean operator, not a generic text classifier. Default Adam learning rate is 0.003, maximum epochs 80, with early stopping after 12 epochs without validation-loss improvement. CPU training uses deterministic operations and fixed seeds; GPU numerical reproducibility is not guaranteed.

Source events are deduplicated across datasets and stratified into approximately 70/15/15 train/validation/test groups; at least three distinct events per class are required. The larger recorded version is retained for duplicate event IDs. Standardization uses only training-prefix features. The best weights are chosen by validation loss; the classification cutoff is chosen by validation F1. Test metrics are calculated once afterward. PR-AUC is reported as average precision with tied scores grouped. Compare F1 and PR-AUC with majority and validation-tuned observed-size baselines. No minimum quality is asserted before evaluating actual datasets.

Checkpoints consist of a `.pt` state dictionary and `.json` metadata containing configuration, feature version, training mean/scale, cutoff, split IDs, dataset fingerprint, history, and evaluation metrics. Loading uses `weights_only=True`; arbitrary pickle models are not accepted through the UI.

## Imported and estimated graphs

JSON arrays, `{ "posts": [...] }`, `{ "data": [...] }`, and JSONL are accepted. Normalize IDs, text, author, timestamp, parent/repost, and cascade fields. Numbers represent Unix seconds. Timestamps without timezones are rejected. Native Twitter delays are already minutes and follow their separate importer.

For observed graphs, imported metadata remains authoritative. Missing timing or relations marks the graph incomplete. If posts have no links, use explicit bundle grouping to request a single-cascade estimate; unrelated cascades should remain separate.

Qwen may propose missing relations and delays using only existing node IDs. The backend rejects cycles, unknown nodes, changed observed parents, negative/nonfinite delays, and disconnected estimated cascades. Candidates remain separate from active graphs until acceptance through the review screen. Acceptance persists the assumptions and inferred fields. Prediction always checks this acceptance flag and returns `estimated: true` plus warnings. Estimated-graph probabilities are experimental and have not been validated by observed-graph test metrics.

## FEVER and resource retrieval

FEVER claim JSONL contains labels and references to Wikipedia pages/sentences; it does not contain the referenced evidence text. Optional setup downloads the Wikipedia ZIP into `resources/fever` and annotations into a separate `datasets/fever` directory. By default it defers extraction and indexing. Evidence → Build / resume FEVER index works offline from the saved ZIP; setup can also explicitly request immediate indexing. The official archive’s `wiki-pages` subdirectory is the evidence source; license HTML and macOS metadata are excluded. Extracted files are CRC-verified once and reused while their size/mtime and archive CRC match. Older extracted files are adopted after a CRC check. Extraction writes a temporary file before replacing the target, preserving completed files on cancellation.

FEVER uses a resumable keyword-first import, committing each 2,048-passage batch so chat can retrieve the available subset during indexing. Cancellation keeps the archive and checkpoint. New files avoid full-table chunk scans; inserts use database batches. Repeated progress samples replace the previous sample for that phase, retaining stage transitions and file completions. A generic resources-folder index does not start managed FEVER indexing. Full FEVER embeddings are not generated by setup. FEVER-shaped labeled claim records are excluded even if manually copied into resources. Training/development labels are not evidence and are never used as verdict shortcuts.

Wikipedia page IDs and sentence IDs remain citation provenance. The verdict vocabulary is `supported`, `refuted`, and `insufficient_evidence`. A verdict without a valid retrieved citation is downgraded to insufficient evidence. A valid citation ID is a provenance check, not proof that the model's interpretation is correct; human review remains possible through the displayed excerpts. The corpus is historical, not live verification of current events.

Hybrid retrieval uses BGE-small English embeddings, SQLite FTS5, and reciprocal-rank fusion. Vectors are persisted in SQLite and memory-mapped NumPy shards of up to 100,000 sentences, scoring 4,096 vectors at a time during search. This bounds memory without a compiled ANN extension, but exact cosine search makes large-corpus retrieval slower. Full FEVER embedding/indexing can take substantial time and disk space. Embeddings and sentence batches use CPU. Reindex after embeddings finish downloading or after resource files change. Unsupported formats and parsing failures are reported; failed file updates preserve previous evidence.

## Runtime and downloads

Qwen runs text-only through an official precompiled llama.cpp release (or an explicitly selected local executable) and a Q4_K_M GGUF. Vision/OCR are outside this version. Context defaults to 8,192 tokens. The native Python app checks the rendered prompt token count before streamed generation and rejects oversized inputs. Token output and tool loops are bounded. Large post bundles must be split or use a larger context; posts are not silently sampled.

Use a machine with enough RAM for the roughly multi-gigabyte quantized weights, context state, and operating system. 16 GB or more is a practical starting target, not a guaranteed fit. Runtime startup logs and model verification report load or tool-parser failures. CPU fallback is slower; GPU acceleration depends on the compiled backend and installed drivers.

Model release manifests pin source revisions and artifact hashes. Local development catalogs resolve/persist embedding revisions on first setup. Models download from Hugging Face; FEVER downloads from its official site. No inference request or imported post data is sent to these services. There is no telemetry or live Twitter/X integration.

The desktop uses Qt Widgets directly from Python, without an application HTTP server or JavaScript runtime. Precompiled llama.cpp artifacts are selected from official release metadata, verified using the published SHA-256 digest, and pinned locally before use. Optional custom source builds retain the source commit in the bundled manifest.

## Application boundaries and audit exports

GraphSAGE training is provided by Sentinel-X Trainer, a separate process/application. The assistant retains inference and checkpoint selection. Both share checkpoint storage but use independent persisted job tables. The trainer does not start a language model or require llama.cpp.

Sentinel-X includes its precompiled llama.cpp runtime in the release bundle. Startup and Settings → Models also offer optional CPU/Metal, Vulkan, CUDA, and ROCm/HIP runtime packages from pinned release b11960, subject to platform availability. CUDA packages include the matching library archive. Both archives are checksum-verified; installation checks the executable's pinned version and available devices. GPU activation requires a detected compatible device and waits for current model work. The included engine remains available for fallback. Skipping starts no download and missing weights/evidence are offered at later launches.

Each new chat turn stores the assembled application system message, skill instructions/function definitions, generation settings, initial RAG evidence, graph provenance, and history message IDs. Tool requests/results remain chronological records. Markdown export includes these snapshots and explicit answer reasoning summaries; older missing context is labeled unavailable rather than reconstructed. Thinking mode remains disabled by default.
