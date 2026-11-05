# Local Docker desktop with Scope

Build Sentinel-X locally and use its Xfce desktop through a browser. The image
contains Qt, noVNC, authenticated exports, the shipped GraphSAGE checkpoint, and
pinned Scope dependencies (Transformers, bitsandbytes, Accelerate, and PyTorch).
Qwen, Scope SAE, and embedding weights download only at runtime into persistent
application storage. They are never included in the image.

## Build and launch

Use a Linux x86-64 Docker Engine host with Docker Compose 2.24.4 or newer.
CUDA operation also requires a host NVIDIA driver compatible with CUDA 12.8 and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The host does not need Python application dependencies or the CUDA SDK; Python 3
is used for the password generator. CPU NF4 execution requires an AVX2-capable
x86-64 CPU. See [Scope hardware notes](SCOPE_HARDWARE.md) for validation limits.

From a clean checkout, create the secret and optional local resource settings:

```sh
python3 scripts/create_container_secret.py
cp .env.example .env
```

Build and launch **one** variant:

```sh
# NVIDIA CUDA: sentinel-x-desktop:local-cuda (PyTorch 2.9.1, CUDA 12.8)
docker compose build
docker compose up --no-build --wait -d

# CPU: sentinel-x-desktop:local-cpu (PyTorch 2.9.1 CPU wheels)
docker compose -f compose.yaml -f compose.cpu.yaml build
docker compose -f compose.yaml -f compose.cpu.yaml up --no-build --wait -d
```

Compose uses distinct local tags and [`pull_policy: never`](https://docs.docker.com/reference/compose-file/services/#pull_policy).
Build before the first launch; `up --no-build` requires that local image.
Base images, OS packages, and Python dependencies can still download during a
build. The container's offline network setting does not isolate the build process.
There is no application-image publishing workflow or registry login step.

Open <http://127.0.0.1:6080/> and choose **Open desktop**. Read the eight-character
password from `.secrets/vnc_password` and enter it in noVNC. The generator
preserves existing passwords. The private, Git-ignored secret is mounted
read-only; it is not embedded in the image or environment variables.

The service binds to the host's IPv4 loopback address. For remote use, tunnel it:

```sh
ssh -N -L 6080:127.0.0.1:6080 user@gpu-host
```

Sentinel-X starts automatically. Closing it leaves the desktop available; use
its launcher to reopen it. The Trainer has a separate launcher. Supervisor
restarts failed processes, and `tini` forwards signals and reaps orphan processes.

## Downloads, model selection, and offloading

The default internal network disables Internet downloads. Enable egress explicitly
for downloads, using the same variant files for every command. CUDA example:

```sh
docker compose down
docker compose -f compose.yaml -f compose.online.yaml up --no-build --wait -d
docker compose -f compose.yaml -f compose.online.yaml exec desktop python scripts/setup_environment.py --configure-only --model 9b --device cuda --cpu-offload sae --download-models
```

CPU example (selects the smaller, explicit 2B fallback):

```sh
docker compose -f compose.yaml -f compose.cpu.yaml down
docker compose -f compose.yaml -f compose.cpu.yaml -f compose.online.yaml up --no-build --wait -d
docker compose -f compose.yaml -f compose.cpu.yaml -f compose.online.yaml exec desktop python scripts/setup_environment.py --configure-only --model 2b --device cpu --download-models
```

The setup command downloads the selected Qwen model, its matching official Scope
layer-16 SAE, and embeddings. Use `--download-both` instead of `--download-models`
to download both pairs and embeddings. Omitting `--model`, `--device`, and
`--cpu-offload` preserves the corresponding saved choices. Normal container
startup runs setup without those flags and does not change them or download models.

For an already running app, prefer **Models**: its separate **Download Qwen3.5-9B**
and **Download Qwen3.5-2B** buttons do not change selection. Select the model and
device explicitly. If changing preferences with the setup CLI, close and reopen
the application afterward so any resident worker uses the new configuration.
The default model is 9B; 2B is never silently substituted.

CUDA CPU-offload choices in Models (CLI values in parentheses):

- **No CPU offloading** (`none`): Qwen and the SAE encoder use CUDA.
- **Scope SAE on CPU** (`sae`): Qwen uses CUDA; sparse extraction uses CPU.
- **Scope and selected Qwen modules on CPU** (`modules`): embeddings, output head,
  and vision modules also use CPU; decoder layers and final normalization use CUDA.

Offload preferences are retained but inactive in CPU mode. Models are loaded
with bitsandbytes NF4 double quantization; downloads contain the original weights
and need more disk space than quantized resident weights. All generation uses
Scope. See [Scope behavior and provenance](SCOPE.md), including the limitations
of applying base-trained SAEs to quantized conversational models.

After downloads, restore isolation by removing and recreating the network:

```sh
# CUDA; include compose.cpu.yaml for CPU sessions.
docker compose -f compose.yaml -f compose.online.yaml down
docker compose up --no-build --wait -d
```

Do not add `--volumes`: downloads and preferences must remain in the home volume.
Changing only the network YAML while the network exists does not reliably change
its isolation setting; use `down` before switching online/offline.

## CPU/CUDA switching and updates

Both tags can coexist and share `desktop-home`. Switching tags does not change the
saved Scope device. Stop the old service, build/start the desired variant, then
select **CPU** or **CUDA** in Models. The offload preference survives switching.
For example, CUDA to CPU:

```sh
docker compose down
docker compose -f compose.yaml -f compose.cpu.yaml build
docker compose -f compose.yaml -f compose.cpu.yaml up --no-build --wait -d
```

To return to CUDA, stop with both CPU files, then build/start using `compose.yaml`.
After obtaining updated source, repeat the build/start commands for the variant
you use. To refresh the Python base image as well, use `docker compose build --pull`
(with the CPU override if applicable). This pulls a **base** image; the application
is still built locally. Updating or recreating containers preserves the volume.

On a CUDA host, test an actual tensor operation on every exposed GPU:

```sh
docker compose exec desktop python /opt/container/check_cuda.py
```

A passing desktop health check does not establish CUDA availability or model fit.
The browser desktop uses software rendering independently of model computation.
GraphSAGE CLI training supports `--device cuda`; the Trainer GUI runs CPU training.

## Export conversations

Use **Conversation actions → Export Markdown** and save under
`/home/sentinel/Exports` (also linked from the desktop). On the browser landing
page choose **Download exported conversations**, or visit
<http://127.0.0.1:6080/exports/>. Log in as `sentinel` with the VNC password.

The endpoint exposes only Exports, rejects symlinks and uploads, and accepts
GET/HEAD. Files use `application/octet-stream` with `nosniff`; downloaded HTML
cannot execute as an application page. Download authentication is separate from
the noVNC session.

## Resource limits

Edit `.env`, then run the selected variant's `up --no-build --force-recreate --wait -d`.
Include the online override if keeping an online session.

| Variable | Default | Effect |
| --- | --- | --- |
| `CONTAINER_CPUS` | `8` | CPU quota in cores; fractions allowed; `0` removes quota |
| `CONTAINER_MEMORY` | `16g` | Hard container RAM limit; `0` removes it |
| `COMPUTE_THREADS` | `4` | OpenMP, MKL, and OpenBLAS thread defaults |
| `CONTAINER_SHM_SIZE` | `1gb` | `/dev/shm` capacity |
| `CONTAINER_TMP_SIZE` | `1g` | `/tmp` capacity; consumes the RAM allowance too |
| `CONTAINER_PIDS` | `512` | Maximum processes/threads |
| `DESKTOP_SIZE` | `1600x900` | Virtual display resolution |
| `NOVNC_PORT` | `6080` | Localhost browser port |

The 16 GiB default is a configurable limit, **not a guarantee that 9B fits**.
It includes the desktop, application, workers, caches, and temporary memory.
Scope additionally applies its own admission checks and RAM/VRAM budgets;
raising Docker's limit does not disable those application checks. CPU-offloaded
modules can remain unquantized and consume substantial RAM. Use 2B, more
appropriate offloading, or a smaller context if admission or loading fails.

Docker limits do not cap VRAM, GPU utilization, inference tokens, or persistent
volume disk usage. Thread defaults do not constrain every library. A full model
load, generation, Scope capture, and peak memory measurement are required to
establish fit within 16 GiB RAM / 6 GiB VRAM; container startup cannot establish it.

## Storage, reset, and host protections

`desktop-home` stores `/home/sentinel`, including Exports, desktop preferences,
caches, and application storage at `/home/sentinel/.local/share/sentinel-x`.
Models live in its `models/{qwen,scope,qwen_2b,scope_2b,embedding}` directories;
settings and conversations live in `sentinel-x.sqlite`.
Keep the same Compose project name/directory to reuse the same volume.
`down` preserves it. Back up important data before resetting anything.

For preference resets, stop the desktop first and run setup in a one-off container:

```sh
# CUDA configuration shown; include compose.cpu.yaml for a CPU image.
docker compose down
docker compose run --rm --no-deps --entrypoint python desktop scripts/setup_environment.py --reset-settings
docker compose up --no-build --wait -d
```

`--reset-settings` clears model/device/offload preferences while preserving
conversations and downloads. Replace it with `--factory-reset` only to delete
application storage, including downloaded models and conversations. Factory reset
leaves Exports and other home files intact; the next startup restores the shipped
GraphSAGE checkpoint. `docker compose down --volumes` deletes the **entire home
volume**, including exports, caches, settings, and all downloaded models.
Rebuilding an image is not a data reset.

The container retains an unprivileged UID/GID 1000, a read-only root filesystem,
all Linux capabilities dropped, and `no-new-privileges`. `/tmp` and `/run` are
bounded `noexec,nosuid,nodev` tmpfs mounts. Default seccomp and host container
security profiles remain active. No Docker socket, host home, privileged mode,
host networking, or host PID namespace is used. VNC listens on container loopback;
only HTTP/WebSocket access is published on host loopback.

An internal network is not a firewall against every host service or peer
container. The desktop user can modify its own data, and containers share the
host kernel and NVIDIA driver. Keep sensitive host directories unmounted; any
optional dataset mount should be read-only. The build context allows application
source, container support files, and exactly the shipped GraphSAGE `.pt`/`.json`
pair; local model weights, secrets, datasets, caches, and build outputs are excluded.

## Checks and validation status

Routine operation (include the same variant/online files used at launch):

```sh
docker compose ps
docker compose logs --tail=100 desktop
docker compose exec desktop python /opt/container/healthcheck.py
docker compose exec desktop python /opt/container/smoke_test.py
docker compose exec desktop python /opt/container/check_runtime.py --variant cuda
```

Use `--variant cpu` with the CPU image. Health checks cover the display, noVNC
assets, and VNC protocol; the app can be closed without making the desktop
unhealthy. Smoke tests also require a visible Sentinel-X window and check
WebSocket access, password rejection, authenticated exports, symlink blocking,
and container permissions. Runtime checks validate dependency versions/imports,
a small CPU NF4 operation, and spawned worker cleanup without downloading models.

Host checks after installing `requirements-dev.txt` as described in [setup.md](../setup.md):

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m ruff check src/backend scripts tests docker
.venv/bin/python -m ruff format --check docker scripts/check_container_config.py tests/test_container_integration.py scripts/create_container_secret.py tests/test_container_secret.py
bash -n docker/entrypoint.sh docker/with-display.sh
python3 scripts/check_container_config.py
git diff --check
```

The Compose checker validates CPU/CUDA with both offline/online networks and
local image selection. It also accepts `--compose /path/to/docker-compose` for
a standalone Compose binary and does not require a daemon.

The Container desktop CI workflow builds **both** local variants on CPU runners,
checks dependencies and the desktop, and recreates containers to check saved
model/device/offload settings, fixture files in all four model directories, and
exports. CUDA-wheel desktop testing on a CPU runner does not test GPU execution.
`check_persistence.py seed` overwrites preferences and is for disposable CI volumes
only; its marker files are not downloaded model weights.

Docker execution is unavailable in the integration workspace. Compose validation
and host checks can run here; image builds, container smoke tests, authentication,
and actual volume recreation must pass on a Docker host/CI before those paths are
considered verified. Real 9B/2B inference and CUDA memory acceptance also remain
unverified. See [integration validation](CONTAINER_VALIDATION.md) for results.
