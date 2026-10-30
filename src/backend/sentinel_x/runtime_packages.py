"""Verified optional CPU, Vulkan, CUDA and ROCm runtime packages."""
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import uuid

LABELS = {'cpu': 'CPU / Apple Metal', 'vulkan': 'Vulkan · AMD / NVIDIA',
          'cuda': 'CUDA · NVIDIA', 'rocm': 'ROCm / HIP · AMD'}
HELP = {
    'cpu': 'Portable CPU runtime; Apple Silicon builds also support Metal.',
    'vulkan': 'GPU acceleration for AMD and NVIDIA. Requires a working Vulkan graphics driver.',
    'cuda': 'NVIDIA GPU runtime. Downloads the matching CUDA libraries as well; NVIDIA drivers are required.',
    'rocm': 'AMD GPU runtime. Requires a supported GPU and compatible ROCm/HIP drivers and system libraries.',
}


def runtime_choices(system=None, machine=None):
    system = system or platform.system(); machine = (machine or platform.machine()).lower()
    if system == 'Darwin':
        return ['cpu']
    if system not in ('Linux', 'Windows'):
        return []
    if machine in ('amd64', 'x86_64'):
        return ['vulkan', 'cuda', 'rocm', 'cpu']
    if machine in ('arm64', 'aarch64'):
        return ['vulkan', 'cuda', 'cpu']
    return []


def asset_for(assets, system, machine, backend='cpu'):
    if backend not in LABELS:
        raise ValueError('Unknown runtime backend')
    architecture = {'amd64': 'x64', 'x86_64': 'x64', 'aarch64': 'arm64', 'arm64': 'arm64'}.get(machine.lower())
    systems = {'Linux': ('ubuntu', 'linux'), 'Windows': ('win', 'windows'), 'Darwin': ('macos',)}
    if not architecture or system not in systems:
        raise ValueError('No supported runtime for this operating system / architecture')
    if system == 'Darwin' and backend != 'cpu':
        raise ValueError('Use the Apple Metal runtime on macOS')
    choices = []
    for asset in assets:
        name = asset['name'].lower()
        if not name.endswith(('.zip', '.tar.gz', '.tgz')) or name.startswith('cudart-'):
            continue
        if not any(f'-{part}-' in name for part in systems[system]):
            continue
        if not re.search(r'-'+architecture+r'\.', name):
            continue
        kind = next((key for key in ('vulkan', 'cuda', 'rocm', 'hip', 'openvino', 'sycl', 'opencl', 'snapdragon')
                     if f'-{key}' in name), 'cpu')
        if kind == ('rocm' if backend == 'rocm' else backend) or (backend == 'rocm' and kind == 'hip'):
            choices.append(asset)
    if not choices:
        raise ValueError(f'The pinned release has no {LABELS[backend]} package for {system} {architecture}. '
                         'Choose another runtime or import a compatible local executable.')
    # CUDA 12 variants generally support more installed drivers than CUDA 13.
    return sorted(choices, key=lambda a: ('-cuda-12.' not in a['name'], a['name']))[0]


def package_catalog(models, backend):
    from .downloads import get_json, request
    system, machine = platform.system(), platform.machine()
    pin = models.manifest['llama_cpp']
    key = f'runtime_catalog_v2:{backend}:{system}:{machine}:{pin["commit"]}'
    saved = models.store.setting(key)
    if saved:
        return saved
    release = pin.get('binary_release', pin['release'])
    base = 'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/'
    metadata = get_json(base+release)
    if not pin.get('binary_release') and not any('-bin-' in a['name'] for a in metadata['assets']):
        pointer = next((a for a in metadata['assets'] if a['name'] == 'nightly-tag.txt'), None)
        if not pointer:
            raise ValueError('Release has no precompiled runtime artifacts')
        with request(pointer['browser_download_url']) as response:
            release = response.read(128).decode().strip()
        if not re.fullmatch(r'b\d+', release):
            raise ValueError('Invalid official binary release pointer')
        metadata = get_json(base+release)
    asset = asset_for(metadata['assets'], system, machine, backend)
    selected = [asset]
    if backend == 'cuda':
        suffix = asset['name'].split('-bin-', 1)[-1]
        companion = next((a for a in metadata['assets'] if a['name'].startswith('cudart-') and
                          a['name'].split('-bin-', 1)[-1] == suffix), None)
        if not companion:
            raise ValueError('Matching CUDA libraries are missing from this release. Choose Vulkan instead.')
        selected.append(companion)
    files = []
    for item in selected:
        digest = item.get('digest', '')
        if not re.fullmatch(r'sha256:[0-9a-fA-F]{64}', digest):
            raise ValueError('Official runtime artifact has no valid SHA-256 digest')
        files.append({'name': item['name'], 'url': item['browser_download_url'],
                      'checksum': digest[7:], 'size': item['size']})
    catalog = {'backend': backend, 'release': release, 'commit': pin['commit'],
               'platform': [system, machine], 'files': files}
    models.store.configure(key, catalog)
    return catalog


def library_directories(root, executable):
    directories = {str(Path(executable).parent.resolve())}
    for path in Path(root).rglob('*'):
        if path.is_file() and ('.so' in path.name or path.suffix.lower() in ('.dll', '.dylib')):
            directories.add(str(path.parent.resolve()))
    return sorted(directories)


def process_environment(directories):
    environment = os.environ.copy()
    variable = 'PATH' if os.name == 'nt' else 'DYLD_LIBRARY_PATH' if platform.system() == 'Darwin' else 'LD_LIBRARY_PATH'
    environment[variable] = os.pathsep.join([*directories, *([environment[variable]] if environment.get(variable) else [])])
    return environment


def runtime_environment(store, executable):
    entry = next((v for v in store.setting('installed_runtimes', {}).values()
                  if v.get('executable') == str(executable)), {})
    return process_environment(entry.get('library_dirs', [str(Path(executable).parent)]))


def gpu_devices(output, backend=None):
    devices = []
    for name, detail in re.findall(r'^\s*([\w.-]+):\s*(.+)$', output, re.MULTILINE):
        if not name.lower().startswith(('cuda', 'vulkan', 'rocm', 'hip', 'metal', 'sycl',
                                        'opencl', 'openvino', 'cann', 'musa', 'kompute', 'rpc')):
            continue
        if name.upper().startswith('CPU') or '=CPU' in detail.upper():
            continue
        if any(software in detail.lower() for software in ('llvmpipe', 'lavapipe', 'swiftshader')):
            continue
        if backend == 'vulkan' and not name.lower().startswith('vulkan'):
            continue
        if backend == 'cuda' and not name.lower().startswith('cuda'):
            continue
        if backend == 'rocm' and not name.lower().startswith(('rocm', 'hip')):
            continue
        devices.append(name+': '+detail)
    return devices


def probe_runtime(executable, commit, directories, backend):
    environment = process_environment(directories)
    try:
        result = subprocess.run([str(executable), '--version'], capture_output=True, text=True,
                                timeout=30, check=False, env=environment)
        version = (result.stdout+'\n'+result.stderr).strip()
        if result.returncode or commit[:9] not in version:
            raise ValueError('The runtime could not load or does not match the pinned llama.cpp version. '
                             'Check GPU libraries/drivers or choose Vulkan / CPU.')
        result = subprocess.run([str(executable), '--list-devices'], capture_output=True, text=True,
                                timeout=30, check=False, env=environment)
        devices = gpu_devices(result.stdout+'\n'+result.stderr, None if backend == 'cpu' else backend) if result.returncode == 0 else []
        return {'version': version, 'devices': devices}
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError('The runtime cannot start. Check its GPU drivers and libraries or choose CPU.') from exc


def install_runtime(models, job=None, backend='cpu', activate=True):
    from .downloads import download, extract_runtime, unpack_runtime_archive
    if job:
        job.emit('stage', stage='Finding official '+LABELS.get(backend, backend)+' runtime')
    catalog = package_catalog(models, backend)
    cache = models.store.root/'runtime'; packages = cache/'packages'/backend
    packages.mkdir(parents=True, exist_ok=True)
    installed = models.store.setting('installed_runtimes', {})
    previous = installed.get(backend)
    if previous and previous.get('release') == catalog['release'] and Path(previous['executable']).is_file():
        result = {**previous, **probe_runtime(previous['executable'], catalog['commit'], previous['library_dirs'], backend)}
    else:
        archives = []
        for number, item in enumerate(catalog['files'], 1):
            if job:
                job.emit('file', file=item['name'], index=number, count=len(catalog['files']))
            archives.append(download(item['url'], cache/item['name'], item['checksum'], item['size'], job))
        with tempfile.TemporaryDirectory(prefix='install-', dir=cache) as temporary:
            temporary = Path(temporary)
            if job:
                job.emit('stage', stage='Unpacking runtime and GPU libraries')
                job.check()
            executable = extract_runtime(archives[0], temporary)
            for companion in archives[1:]:
                unpack_runtime_archive(companion, temporary)
            if job:
                job.emit('stage', stage='Checking runtime version and available devices')
            probe = probe_runtime(executable, catalog['commit'], library_directories(temporary, executable), backend)
            if job:
                job.check()
            relative = executable.relative_to(temporary)
            destination = packages/(catalog['release']+'-'+uuid.uuid4().hex[:8])
            temporary.rename(destination)
        path = destination/relative
        result = {'executable': str(path), 'backend': backend, 'release': catalog['release'],
                  'commit': catalog['commit'], 'precompiled': True,
                  'library_dirs': library_directories(destination, path), **probe}
    if job:
        job.check()
    installed[backend] = result
    models.store.configure('installed_runtimes', installed)
    if activate:
        models.store.configure('llama_path', result['executable'])
    return result
