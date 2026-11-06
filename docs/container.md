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
build. The container network permits outbound connections and published-port access.
There is no application-image publishing workflow or registry login step.

Open <http://SERVER_IP:36080/> and choose **Open desktop**. Read the eight-character
password from `.secrets/vnc_password` and enter it in noVNC. The generator
preserves existing passwords. The private, Git-ignored secret is mounted
read-only; it is not embedded in the image or environment variables.

The service defaults to all host interfaces on port 36080. `127.0.0.1` in a
browser refers to the browser's own machine, not the remote server. For remote
browser access or Nginx Proxy Manager, use the configuration below. An SSH tunnel
is also available when binding the service to loopback:

```sh
ssh -N -L 36080:127.0.0.1:36080 user@gpu-host
```

Sentinel-X starts automatically. Closing it leaves the desktop available; use
its launcher to reopen it. The Trainer has a separate launcher. Supervisor
restarts failed processes, and `tini` forwards signals and reaps orphan processes.

## Remote access and Nginx Proxy Manager

The server runs the desktop and all model computation; the other person only
needs a browser. For a server publishing container port **6080** on host port
**36080**, set these values in the server checkout's `.env`:

```dotenv
NOVNC_BIND_ADDRESS=0.0.0.0
NOVNC_PORT=36080
```

You can use the server's specific private/LAN IP instead of `0.0.0.0` to bind
only that interface. This must be an address assigned to the Docker host.
With a `127.0.0.1` binding, changing only the port still leaves the
service inaccessible from other machines or a separate proxy container.
See [Docker's port publishing documentation](https://docs.docker.com/engine/network/port-publishing/).

Rebuild to include the updated browser links, then recreate the service. Use the
same CPU override file if applicable; CUDA example:

```sh
docker compose build
docker compose up --no-build --force-recreate --wait -d
docker compose port desktop 6080
```

The reported mapping should be `0.0.0.0:36080` (or your chosen private IP).
Open `http://SERVER_IP:36080/` from another machine on the same network. Use
**HTTP** for this direct check; TLS is supplied by the reverse proxy later.
Allow TCP 36080 from the client/proxy through the server/network firewall.
Keep the same Compose project and home volume so downloads and settings persist.

In **Nginx Proxy Manager → Hosts → Proxy Hosts → Add Proxy Host**, configure:

| Field | Value |
| --- | --- |
| Domain Names | Your desktop hostname, e.g. `sentinel.example.com` |
| Scheme | `http` |
| Forward Hostname / IP | The Docker server's private/LAN IP |
| Forward Port | `36080` (the host port) |
| Websockets Support | Enabled |
| SSL | Configure a certificate for the hostname and enable Force SSL |

For longer idle sessions, put this in the proxy host's **Advanced** configuration:

```nginx
proxy_read_timeout 3600s;
proxy_send_timeout 3600s;
proxy_buffering off;
```

If Nginx Proxy Manager itself runs in Docker, `127.0.0.1` refers to its own
container. Use the server's reachable IP above. If both containers instead share
a Docker network, forward to the desktop's service/network alias on **6080**;
the host mapping **36080** is only for connections through the host.
Point DNS at the proxy and route public HTTPS traffic to the proxy. Share
`https://sentinel.example.com/` and the desktop password with the other person.
Keep 36080 private to the LAN/proxy; HTTP and the export Basic authentication
need TLS at the public edge. Each container is one shared desktop, so users
connecting to the same container see the same session and files.

WebSocket forwarding is required even when the landing page loads correctly:
[NGINX WebSocket documentation](https://nginx.org/en/docs/http/websocket.html),
[Nginx Proxy Manager guide](https://nginxproxymanager.com/guide/).

### Optional path prefix

A dedicated hostname is simplest with Nginx Proxy Manager. The landing page also
supports a directory such as `https://example.com/sentinel/`: its desktop,
exports, and WebSocket links retain the public prefix. The proxy must redirect
`/sentinel` to `/sentinel/` and strip `/sentinel/` when forwarding. For a custom
NGINX server, the locations would be:

```nginx
location = /sentinel { return 301 /sentinel/; }
location /sentinel/ {
    proxy_pass http://SERVER_IP:36080/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;
    proxy_buffering off;
}
```

The trailing slash on `proxy_pass` is required to remove the prefix. Enter via
the landing page so it supplies the correct public connection URL to noVNC.

### If the server IP still does not respond

Run these on the Docker server (include your usual Compose override files):

```sh
docker compose ps
docker compose port desktop 6080
docker compose logs --tail=100 desktop
curl -I http://127.0.0.1:36080/
curl -I http://SERVER_IP:36080/
```

Replace `SERVER_IP` with the actual address. A loopback curl failure is expected
if you bound only a specific LAN IP. If both fail, check startup logs and the
published mapping. If the server can reach its own LAN IP but another machine
cannot, check routing and firewall rules. If the page loads but the desktop
doesn't connect, check WebSocket support: `/websockify` should return status
**101 Switching Protocols** in the browser's Network/WebSocket panel.

## Downloads, model selection, and offloading

Networking is enabled by default, including Internet access for model downloads.
Use the same variant files for every command. CUDA example:

```sh
docker compose down
docker compose -f compose.yaml up --no-build --wait -d
docker compose -f compose.yaml exec desktop python scripts/setup_environment.py --configure-only --model 9b --device cuda --cpu-offload sae --download-models
```

CPU example (selects the smaller, explicit 2B fallback):

```sh
docker compose -f compose.yaml -f compose.cpu.yaml down
docker compose -f compose.yaml -f compose.cpu.yaml up --no-build --wait -d
docker compose -f compose.yaml -f compose.cpu.yaml exec desktop python scripts/setup_environment.py --configure-only --model 2b --device cpu --download-models
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

### Migrating an existing isolated network

The online override has been removed; `compose.yaml` now uses `internal: false`.
An existing internal network must be recreated once. Keep the same project name
and include `compose.cpu.yaml` if applicable:

```sh
docker compose down
docker compose up --no-build --wait -d
```

Do not add `--volumes`: downloads and preferences remain in the home volume.
Remove `compose.online.yaml` from saved commands and `COMPOSE_FILE` settings.
Existing `.env` values still take precedence: use `NOVNC_BIND_ADDRESS=0.0.0.0`
and `NOVNC_PORT=36080` for direct LAN access.

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
<http://SERVER_IP:36080/exports/>. Log in as `sentinel` with the VNC password.

The endpoint exposes only Exports, rejects symlinks and uploads, and accepts
GET/HEAD. Files use `application/octet-stream` with `nosniff`; downloaded HTML
cannot execute as an application page. Download authentication is separate from
the noVNC session.

## Resource limits

Edit `.env`, then run the selected variant's `up --no-build --force-recreate --wait -d`.

| Variable | Default | Effect |
| --- | --- | --- |
| `CONTAINER_CPUS` | `8` | CPU quota in cores; fractions allowed; `0` removes quota |
| `CONTAINER_MEMORY` | `16g` | Hard container RAM limit; `0` removes it |
| `COMPUTE_THREADS` | `4` | OpenMP, MKL, and OpenBLAS thread defaults |
| `CONTAINER_SHM_SIZE` | `1gb` | `/dev/shm` capacity |
| `CONTAINER_TMP_SIZE` | `1g` | `/tmp` capacity; consumes the RAM allowance too |
| `CONTAINER_PIDS` | `512` | Maximum processes/threads |
| `DESKTOP_SIZE` | `1600x900` | Virtual display resolution |
| `NOVNC_BIND_ADDRESS` | `0.0.0.0` | Host bind address; use a private IP or `0.0.0.0` for remote/proxy access |
| `NOVNC_PORT` | `36080` | Published host browser port, e.g. `36080` |

The 16 GiB default is a configurable limit, **not a guarantee that 9B fits**.
It includes the desktop, application, workers, caches, and temporary memory.
Scope no longer applies a native inference memory admission limit; Docker's
container memory limit still applies. CPU-offloaded
modules can remain unquantized and consume substantial RAM. Use 2B, more
appropriate offloading, or a smaller context if loading fails.

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
only HTTP/WebSocket access is published, on all host interfaces by default or the
configured `NOVNC_BIND_ADDRESS`.

The online network permits access to reachable host services and peer containers.
The desktop user can modify its own data, and containers share the
host kernel and NVIDIA driver. Keep sensitive host directories unmounted; any
optional dataset mount should be read-only. The build context allows application
source, container support files, and exactly the shipped GraphSAGE `.pt`/`.json`
pair; local model weights, secrets, datasets, caches, and build outputs are excluded.

## Checks and validation status

The image pins matched PySide6/Shiboken 6.11.1 bindings. Version 6.12.0 caused
`none_dealloc: deallocating None` aborts on Python 3.11 during repeated progress
updates, stopping in-process downloads. Rebuild the image to apply the corrected
pins; recreating an old image alone does not update its Python packages. Retained
partial downloads can be resumed using **Retry / resume task** in Background work.
CI runs `/opt/container/check_qt.py` for 5,000 progress refreshes to catch this
native crash before dependency upgrades are accepted.

Routine operation (include the same variant files used at launch):

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

The Compose checker validates CPU/CUDA with online networking and
local image selection, including loopback and remote port bindings.
Browser URL regression checks use Node.js and run independently of the Python
application dependencies:

```sh
python3 -m unittest discover -s tests -p 'test_container_proxy.py' -v
```

The Compose checker also accepts `--compose /path/to/docker-compose` for
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
