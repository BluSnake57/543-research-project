# ROCm for Qwen and Scope

Select **ROCm · AMD · 4-bit NF4** in Models to run the quantized Qwen model and
its paired Scope encoder on an AMD GPU. The 9B and 2B model pairs and all three
CPU-offload policies are available. Settings and saved Scope records identify
the backend as `rocm`.

The worker checks for a HIP-enabled PyTorch build, an available GPU, BF16 support,
and the existing memory budgets before loading. It also runs a small bitsandbytes
NF4 projection first, so a missing or incompatible kernel fails before loading
the full model. There is no automatic CPU fallback or change to model size.

## Prerequisites

Install a ROCm-capable PyTorch build in the application virtual environment using
[AMD's PyTorch installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)
for your GPU, operating system, and Python version. The repository's ordinary
dependency installation does not select a ROCm PyTorch build for you. Installing
the ROCm SDK alongside a CUDA PyTorch wheel does not enable AMD execution.

Keep the application's pinned `bitsandbytes==0.50.2`. Its
[installation guide](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation#amd-rocm)
lists compatible platforms and GPU targets. PyTorch and bitsandbytes must both
support your ROCm version and GPU. On Linux the application user needs access to
the AMD compute devices, typically `/dev/kfd` and the applicable `/dev/dri` render
device; follow the host's driver and group-permission setup.

PyTorch exposes HIP tensors through `torch.cuda` and the `cuda` tensor device
name, as documented in
[HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html).
The application translates its explicit `rocm` setting to this tensor interface.
A `cuda` tensor device in an error or diagnostic is therefore expected on ROCm;
the PyTorch HIP version distinguishes the installed backend.

## Configure and check

After installing ROCm PyTorch in `.venv`, install the remaining application
dependencies and save the device selection:

```sh
python run.py setup --device rocm --model 2b --cpu-offload sae
```

Setup preserves an installed PyTorch build satisfying `torch>=2.6,<3`. Recheck
the HIP version after dependency or environment changes. `--reset-venv` removes
that environment, so install ROCm PyTorch again if rebuilding it. Model downloads
remain separate: use the Models buttons or setup's `--download-models` flag.

Test device admission and an actual small NF4 calculation without model downloads:

```sh
.venv/bin/python scripts/check_scope_device.py --device rocm
```

The check exits nonzero on failure. Successful output includes the HIP version,
the chosen backend, `nf4_check: passed`, and `model_loaded: false`. CPU and CUDA
checks are also available with `--device cpu` or `--device cuda`. Passing this
check establishes small-kernel compatibility; it does not establish complete
Qwen loading, generation speed, or peak memory fit.

Start the application with `python run.py`, select the matching Qwen/SAE pair,
and run a model response. Offloading, cancellation, sparse capture, and exports
follow the same worker path as CUDA. Saved records retain `device: rocm`.

## Memory and validation

The branch's existing limits apply to both GPU backends: a 6 GiB GPU target,
5.5 GiB allocator allowance, and a 15 GiB combined application/worker RSS check
within the 16 GiB CPU target. These limits are intentionally unchanged. Select
2B, reduce context, or offload modules if admission fails. Allocator limits do
not cover every driver allocation; measure actual peaks during full-model runs.

Offline tests cover HIP/CUDA build mismatches, missing devices, BF16 and memory
failures, GPU placement with CPU offload, setup/UI persistence, and generation
provenance. The diagnostic's NF4 projection has run on CPU. This checkout's
environment has a CUDA PyTorch build, so actual ROCm kernels and full-model AMD
inference remain unvalidated. ROCm application support does not install host
drivers or add the Docker configuration from the separate Scope Docker branch.
