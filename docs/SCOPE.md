# Scope branch: model downloads and placement

Models now offers both **Qwen3.5-9B + matching Scope SAE** and
**Qwen3.5-2B + matching Scope SAE** download buttons. Downloads are independent
of model selection and do not activate a model. Both pairs may coexist in app
storage. The selector determines which pair is loaded; 2B is an explicit
fallback, not a silent substitution after a failed 9B request.

Each pair uses layer 16, residual output, Top-K 50. 9B uses 32 layers, hidden
size 4096 and SAE width 65536; 2B uses 24 layers, hidden size 2048 and SAE width
32768. The committed artifact manifest pins revisions and checksums for both.
The weights downloaded are the official Transformers weights; NF4 quantization
with double quantization happens during loading, not during download. Download
size therefore exceeds the eventual quantized resident weight size.

Device choices are CPU and CUDA. CUDA placement choices are:

- No CPU offloading: Qwen and the Scope encoder run on CUDA.
- Scope SAE on CPU: generation stays on CUDA; bounded residual batches move to
  CPU for sparse feature extraction.
- Scope and selected Qwen modules on CPU: embeddings, output head and vision
  modules stay on CPU; decoder layers and final normalization stay on CUDA.
  CPU-dispatched modules may be unquantized and consume substantial RAM.

The offload preference is retained but inactive in CPU-only mode. Selection and
placement changes stop the old worker. All generation requires Scope; changing
placement never enables an uninstrumented chat runtime.

The target machine limits are 16 GiB system RAM and 6 GiB VRAM. Admission checks
reserve headroom, CUDA allocator usage is limited to 5.5 GiB, and the parent
monitors combined parent/worker RSS against 15 GiB. The RSS check is reactive,
not an OS-enforced hard limit, and cannot guarantee that a brief loading peak
will remain below the target. Estimates are not measured memory-fit results.
An incompatible configuration must be rejected; choose 2B, more offloading, or
less context. GPU driver repair is still necessary on the current laptop.

The Thinking panel contains only emitted thinking. Explanation · Scope contains
measured feature IDs, values and associated tokens, regardless of Thinking.
Completed panels start collapsed and expand independently. Feature descriptions
are not invented. Saved records include the actual model/SAE revisions,
quantization, device, offload choice, call boundary, token positions and capture
status. Edited responses retain analysis labeled as original-response analysis.
Markdown exports include the sparse records. Older messages gain no invented
measurements.

SAEs were trained on base checkpoints. Their application to quantized
conversational checkpoints is experimental; activations are not independently
validated causal explanations and do not explain GraphSAGE internals. Published
research metrics must not be assumed to transfer to either model configuration.

Settings reset clears model/device/offload preferences and preserves both pairs.
Factory reset removes downloaded pairs within application storage. Obsolete
GGUF and llama.cpp files are ignored and preserved during ordinary upgrades.

Validation: 149 tests plus 10 subtests passed offline, including model selection,
paired downloads, offload maps, activation math, capture, conversation storage,
exports and existing retrieval/GraphSAGE tests. A real bitsandbytes 0.50.2 NF4
linear layer executed on CPU. Neither complete 9B nor complete 2B loading and
inference has been measured against the memory targets. CUDA validation remains
blocked by driver access. The earlier frozen startup check failed on a PyTorch
shared-library dependency; release packaging remains unfinished.
