# Docker desktop with noVNC and NVIDIA CUDA

The image runs Sentinel-X in an Xfce desktop accessed through a browser. It includes
PyTorch 2.9.1 with CUDA 12.8, Qt, the bundled GraphSAGE checkpoint, noVNC, and an
authenticated export download page. It does not download Qwen, embeddings, or a
llama.cpp runtime during the build.

## Setup and launch

Use a Linux Docker Engine host with Docker Compose 2.24.4 or newer, or Docker
Desktop with compatible NVIDIA GPU support. The host needs an NVIDIA driver
compatible with CUDA 12.8 and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The host does not need Python application dependencies or the CUDA SDK. Python 3
is needed only for the small password generator below. The image targets Linux
x86-64; GPU operation depends on the GPU and installed host driver.

From a clean checkout:

```sh
python3 scripts/create_container_secret.py
cp .env.example .env
docker compose up --build --wait -d
```

Open <http://127.0.0.1:6080/> and choose **Open desktop**. Read the password from
`.secrets/vnc_password` and enter it in noVNC. The generator preserves an existing
password; its containing directory is private and ignored by Git. The secret is
mounted read-only, rather than stored in the image or an environment variable.
Classic VNC authentication uses eight characters; use an SSH tunnel for remote
connections, rather than publishing this HTTP service on the LAN or Internet:

```sh
ssh -N -L 6080:127.0.0.1:6080 user@gpu-host
```

Then open the same local browser URL. HTTP and the VNC connection are carried in
the encrypted SSH tunnel. The default binding accepts connections only on the
Docker host's IPv4 loopback address.

Sentinel-X starts automatically. Closing the app leaves the desktop available;
use its desktop launcher to reopen it. The Trainer has its own launcher. Processes
that fail unexpectedly are restarted by Supervisor. `tini` forwards shutdown
signals and reaps orphan processes.

## Model downloads and CUDA

The default network is internal: no Internet downloads are available. To install
the model resources, explicitly enable outbound network access:

```sh
docker compose down
docker compose -f compose.yaml -f compose.online.yaml up --wait -d
docker compose exec desktop python scripts/setup_environment.py --configure-only --download-models --runtime cuda
```

The second command downloads Qwen, embeddings, and the CUDA llama.cpp runtime.
Alternatively, use the app's setup and runtime panels: choose **CUDA · NVIDIA**
and activate that runtime. GPU offload defaults to automatic when activated.
Check the displayed devices; installing CUDA PyTorch alone does not activate the
separate llama.cpp engine. GraphSAGE CLI training supports `--device cuda`;
the existing Trainer GUI currently labels and runs its training as CPU work.

After downloads finish, return to the isolated network without deleting storage:

```sh
docker compose down
docker compose up --wait -d
```

Use the same Compose files for starting and stopping an online session. Returning
offline requires removing the old network with `down`; do not assume an existing
network will change its isolation setting in place.

Verify CUDA with an actual tensor calculation on every exposed GPU:

```sh
docker compose exec desktop python /opt/container/check_cuda.py
```

This command fails when CUDA cannot initialize. The browser desktop itself uses
software rendering; this does not disable model compute acceleration. GPU access
uses Docker's NVIDIA device reservation, with `compute,utility` capabilities.
See [Docker GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/)
and [NVIDIA driver capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).

For a host without an NVIDIA GPU, build the CPU variation instead:

```sh
docker compose -f compose.yaml -f compose.cpu.yaml up --build --wait -d
```

It shares the same local image tag. Rebuild without `compose.cpu.yaml` before
switching back to CUDA. CUDA verification is expected to fail in the CPU variation.

## Download conversations

Use **Conversation actions → Export Markdown** in Sentinel-X and save the file
under `/home/sentinel/Exports`. The app initially opens its save dialog in this
directory; if you navigated elsewhere, select the **Exports** desktop shortcut.

On the browser landing page, choose **Download exported conversations**, or open
<http://127.0.0.1:6080/exports/>. Log in as `sentinel` using the same VNC password,
then select a file. The endpoint exposes only `Exports`, rejects symlinks, and
accepts GET/HEAD only. File contents use `application/octet-stream` with `nosniff`
so an exported HTML file cannot run as an app page. Downloads are authenticated
separately from the VNC session and do not require clipboard file transfer.

## Resource limits

Edit `.env` to fit the workload, then recreate the container:

```sh
docker compose up --force-recreate --wait -d
```

Include `-f compose.yaml -f compose.online.yaml` if keeping an online session.
For example, a larger research workstation might use:

```dotenv
CONTAINER_CPUS=12
CONTAINER_MEMORY=32g
COMPUTE_THREADS=8
CONTAINER_SHM_SIZE=2gb
CONTAINER_TMP_SIZE=2g
CONTAINER_PIDS=1024
```

| Variable | Default | Effect |
| --- | --- | --- |
| `CONTAINER_CPUS` | `8` | CPU time quota in cores; fractions allowed; `0` removes the quota |
| `CONTAINER_MEMORY` | `16g` | Hard RAM limit; `0` removes it; exceeding it can kill a process |
| `COMPUTE_THREADS` | `4` | OpenMP, MKL, and OpenBLAS thread defaults, not a hard quota |
| `CONTAINER_SHM_SIZE` | `1gb` | `/dev/shm` capacity for shared memory |
| `CONTAINER_TMP_SIZE` | `1g` | `/tmp` capacity; tmpfs usage also consumes the RAM allowance |
| `CONTAINER_PIDS` | `512` | Maximum processes/threads in the container |
| `DESKTOP_SIZE` | `1600x900` | Virtual display resolution |
| `NOVNC_PORT` | `6080` | Published localhost browser port |

CPU quotas allow bursts across cores within the configured time allowance. Thread
defaults do not constrain every library or llama.cpp job. Model context size and
GPU offload are configurable in the app. These Docker limits do **not** cap VRAM,
GPU utilization, inference tokens, or the size of the persistent volume. On a
shared GPU, use supported hardware partitioning or a separate GPU for firm GPU
resource isolation. Avoid setting process or memory limits so low that model
loading or CUDA initialization cannot complete.

## Host protections and their boundaries

- Fixed unprivileged UID/GID 1000, all Linux capabilities dropped, and
  `no-new-privileges`. No privileged mode, host PID namespace, host networking,
  Docker socket, or host home-directory mount.
- Read-only root filesystem. `/tmp` and `/run` are bounded tmpfs mounts with
  `noexec,nosuid,nodev`; only the dedicated home volume persists. Downloaded
  inference executables can run from that home volume.
- Docker's default seccomp profile and the host's normal container security
  profile remain enabled. No `seccomp=unconfined` override is needed.
- VNC listens only inside the container on loopback; only the HTTP/WebSocket
  service is published, on host loopback. Exports require authentication, reject
  symlinks, disallow uploads, and rate-limit requests.
- An internal bridge network is used by default. The online override explicitly
  allows egress. An internal network is **not** a firewall against every host
  service or another container on that network; configure host firewall rules or
  use a dedicated VM if untrusted code needs a stricter boundary.
- CPU, RAM, processes, temporary storage, shared memory, and container log
  rotation have limits. Core dumps are disabled. Persistent volume disk usage
  and GPU resources need separate host monitoring or quotas.

The desktop user can run application code and modify or delete its own models,
conversations, exports, and volume contents. This container shares the host
kernel and NVIDIA driver; it is not a VM security boundary or a safe place to
run arbitrary hostile code. Do not add sensitive host bind mounts. An optional
dataset mount should be read-only, for example
`./datasets:/home/sentinel/Datasets:ro`. App data is not included in Git or the
Docker build context. Keep backups of important exports.

## Storage and operation

The `desktop-home` named volume stores app settings, checkpoints, downloads,
conversations, caches, exports, and desktop preferences. `docker compose down`
preserves it. **`docker compose down --volumes` deletes that stored data.**

```sh
docker compose ps
docker compose logs --tail=100 desktop
docker compose exec desktop python /opt/container/healthcheck.py
docker compose exec desktop python /opt/container/smoke_test.py
docker compose stop
```

Health checks cover the X display, noVNC HTTP assets, and VNC protocol availability.
They do not require a GPU or an active model and remain healthy when you close
the app. The smoke test additionally requires a visible Sentinel-X window and
checks WebSocket connectivity, password rejection, authenticated downloads,
symlink blocking, an unprivileged process, and a read-only system.

## Development checks and files

The Dockerfile installs application dependencies using the tested versions in
`docker/requirements.lock`; host development continues to
use the existing setup workflow. For container-specific Python checks:

```sh
python3 -m unittest discover -s tests -p test_container_secret.py -v
python3 -m ruff check docker scripts/create_container_secret.py tests/test_container_secret.py
python3 -m ruff format --check docker scripts/create_container_secret.py tests/test_container_secret.py
bash -n docker/entrypoint.sh docker/with-display.sh
docker compose config --quiet
```

Install Ruff with the existing `requirements-dev.txt` when using a host development
environment. Container files live in `docker/`, the Compose manifests and Dockerfile
are at the repository root, password-generation tests live in `tests/`, and the
container CI workflow builds the CPU variation and checks desktop isolation and
persistence. A CUDA wheel build and desktop smoke test can run on a CPU host;
the actual CUDA calculation still requires NVIDIA hardware.

## Published images and Compose integration

The `Publish desktop image` GitHub Actions workflow builds and tests the CUDA image
before pushing it to the repository-linked GitHub Container Registry package:

```text
ghcr.io/blusnake57/543-research-project
```

Pushes to `main` publish `main`, `latest`, and a commit tag. Pushes to
`docker-novnc-desktop` publish the branch tag and a commit tag. Version tags
matching `v*` publish the version tag and attach checksummed Docker archives to
the corresponding GitHub Release. The workflow can also be run manually from
Actions. It uses `GITHUB_TOKEN`; no registry password belongs in the repo. It runs
only on pushes/manual dispatch, so pull requests do not receive package-write
permissions. Actual GPU execution must still be verified on an NVIDIA host.

Once an image is published, set this in `.env` (use `docker-novnc-desktop` while
testing the branch before it reaches main, or pin a release/commit tag):

```dotenv
SENTINEL_IMAGE=ghcr.io/blusnake57/543-research-project:latest
```

Then use the checked-in Compose file without rebuilding:

```sh
python3 scripts/create_container_secret.py
docker compose pull
docker compose up --no-build --wait -d
```

For an existing Compose project, merge the `desktop` service, `desktop-home`
volume, and `desktop` network definitions into its manifest. Keep the secret's
bind mount, isolation settings, and localhost port binding. Set `SENTINEL_IMAGE`
to the published reference; the `build` entry can be omitted when only pulling.
The container is a separate desktop service and needs no host X server or display
device mounts. Use an isolated network for it rather than automatically attaching
it to every service in the project.

New GHCR packages can initially be private. The repository owner may need to make
the linked package public in its GitHub package settings to permit anonymous
pulls, or consumers must log in with a token that can read the package. The GitHub
Release archives provide a separate download path through the repository's
**Releases** page. Download every archive part plus `SHA256SUMS` and `IMAGE.txt`,
then load the image:

```sh
sha256sum -c SHA256SUMS
cat sentinel-x-image.tar.gz.part-* | gzip -dc | docker image load
```

Set `SENTINEL_IMAGE` to the reference in `IMAGE.txt` and use `up --no-build`.
Each part is below GitHub's
[2 GiB release asset limit](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
The publish and release jobs take effect after the workflow is pushed; adding a
workflow locally does not publish an image or create a release.
