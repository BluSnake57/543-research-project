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
[AMD's PyTorch installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
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

Close the app before replacing its Python packages. Setup can replace PyTorch
from an explicit wheel index before installing the other dependencies. For a
Linux x86-64 host using ROCm 7.2, the following selects the official
[ROCm 7.2 wheel index](https://download.pytorch.org/whl/rocm7.2/torch/):

```sh
python run.py setup --device rocm --torch-index-url https://download.pytorch.org/whl/rocm7.2
```

Choose an index compatible with your host; this example is not a universal
ROCm version selection. Pip must have a wheel for the environment's Python and
platform. `--torch-index-url` explicitly reinstalls PyTorch, including replacing
a newer CUDA build with the ROCm build available at that index. It does not
install host drivers. No separate llama.cpp runner is needed on this branch.
Setup downloads only the torch wheel from that index, then installs that exact
wheel with ordinary dependencies from PyPI and backend dependencies from the
selected index. This avoids picking a newer CUDA wheel from PyPI or requiring
every dependency to be mirrored by the ROCm index.

Alternatively, after manually installing AMD's recommended wheels in `.venv`,
install the remaining application dependencies and save the device selection:

```sh
python run.py setup --device rocm --model 2b --cpu-offload sae
```

Without `--torch-index-url`, setup preserves an installed PyTorch build satisfying
`torch>=2.6,<3`. Explicit `--device rocm` checks for
HIP before installing ordinary dependencies and again after installation. A
missing or CUDA-only PyTorch build fails with instructions instead of silently
installing CUDA packages for an AMD request. `--skip-install --device rocm` also
checks the installed build. These setup checks do not require an available GPU;
use the diagnostic below to check the driver and kernels. Recheck
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
provenance. On 2026-11-05 the native Fedora 45 environment was repaired from
CUDA PyTorch to `torch==2.14.1+rocm7.2` with HIP `7.2.53211`, Python 3.15, and
`bitsandbytes==0.50.2`. On the Radeon RX 9070 XT (`gfx1201`), a BF16 identity
matrix multiplication and the diagnostic's NF4 projection both passed. The
dependency check also passed. This validates small GPU kernels on this host;
full-model AMD inference and peak memory remain unvalidated.
ROCm application support does not install host
drivers or add the Docker configuration from the separate Scope Docker branch.
