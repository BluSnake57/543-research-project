> Historical memory targets below are research validation targets, not runtime limits.
> Native memory admission checks, allocator caps, and RSS enforcement have been removed.

# Scope hardware investigation

Updated 2026-11-05. Required deployment limits: 16 GiB CPU RAM or 6 GiB GPU
VRAM for CUDA or ROCm. These supersede the original 9B/full-precision implementation assumptions.

## Latest selection

The user subsequently selected quantized 9B as the primary investigation, with
2B as fallback. Retain the official 9B Scope SAE for 9B and the official 2B SAE
for 2B. Do not cross-pair their weights. The 4B findings below remain context.

For 9B, separate input embeddings and output-head weights can consume substantial
unquantized memory. A four-bit parameter-count estimate alone cannot establish
fit on 6 GiB CUDA. Evaluate explicit CPU placement for the SAE and selected
unquantized modules if needed, while staying within the CPU budget too. Avoid
automatic unrestricted offloading that hides a RAM overrun. Select 2B if either
loading or generation exceeds the limits; record which model produced each turn.

## Model selection

Qwen3.5-4B is available in Transformers format, but the official Qwen-Scope
collection has no matching 4B SAE. Third-party 4B SAEs exist (including
caiovicentino1/Qwen3.5-4B-SAE-L18-topk); they are not Qwen-Scope and must not be
silently substituted or labeled as such. A 9B or 2B SAE cannot be applied to 4B:
the residual dimensions differ (4096, 2048, and 2560 respectively).

The user authorized investigating 2B if 4B does not work. The compatible official
pair is Qwen/Qwen3.5-2B and Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50.
The latter has 24 layers, hidden dimension 2048, width 32768, and Top-K 50.
Layer 16 is available. Its float32 encoder occupies 256 MiB, excluding biases;
the full encoder/decoder checkpoint is approximately 512 MiB. Only the encoder
and encoder bias are needed during feature measurement.

## Quantization candidate and acceptance

Investigate Transformers + bitsandbytes NF4 with double quantization. Current
upstream documentation supports NVIDIA CUDA, AMD ROCm, and AVX2 x86 CPU backends. That
support does not establish this specific model's correctness, latency, or peak
memory. Keep the SAE encoder float32 initially, and explicitly identify the
language model as quantized in saved analysis provenance. Quantization changes
activations; the base-trained SAE applied to a quantized conversational model
is not an independently validated causal explanation.

Start validation with batch size 1, 2048-token total context, bounded prompt
chunks, and no concurrent embedding or GraphSAGE model resident on the GPU.
Budget the entire application, worker, SAE, cache, temporary dequantization,
and loading peak, not just compressed linear weights. Embeddings and some
other weights remain unquantized. CPU backend temporary buffers can differ
from CUDA. No claim that either target fits is justified until measured.

Acceptance requires peak aggregate CPU memory below 16 GiB for the CPU profile,
and peak device usage below the usable portion of a 6 GiB GPU for CUDA or ROCm. Leave
headroom for the OS/display and reject oversized contexts. Test actual loading,
generation, sparse capture, tool calls, semantic extraction, cancellation,
reopening and export. Do not simply lower the existing preflight thresholds.

## Current implementation status

Both quantized profiles and CPU-offload choices are now implemented in the working
branch. See SCOPE.md for current offline validation; real-model memory validation
and release packaging remain incomplete. Offline
fixtures are not a substitute for the memory acceptance test. This machine's
NVIDIA driver remains unavailable to PyTorch. The native CPU advertises AVX2.

A source startup check passed. A frozen assistant was built, but its startup
check failed loading torch/lib/libtorch_global_deps.so; packaging needs repair
before release. Model artifacts were not downloaded for this investigation.

The native application now also has an explicit ROCm device choice with the same
offload policies and memory targets. See [ROCm setup and checks](ROCM.md). The
native Fedora 45 environment now uses PyTorch 2.14.1+rocm7.2. BF16 matrix
multiplication and the bitsandbytes NF4 diagnostic passed on the RX 9070 XT
(`gfx1201`). Full-model AMD inference and memory validation remain pending.

## Primary sources

- Official Scope inventory: https://huggingface.co/collections/Qwen/qwen-scope
- 4B architecture: https://huggingface.co/Qwen/Qwen3.5-4B
- 2B SAE: https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50
- 2B configuration: https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json
- Quantization: https://huggingface.co/docs/transformers/quantization/bitsandbytes
- Backend requirements: https://huggingface.co/docs/bitsandbytes/installation
