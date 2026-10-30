"""Prepare the included runtime; the app can additionally download optional GPU engines."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from .config import manifest
from .downloads import Models
from .state import Store


def runtime_executable(directory):
    directory = Path(directory)
    candidates = list(directory.rglob('llama-server'))+list(directory.rglob('llama-server.exe'))
    if len(candidates) != 1:
        raise ValueError('Runtime directory must contain exactly one llama-server executable')
    return candidates[0]


def verify_runtime(directory):
    executable = runtime_executable(directory)
    result = subprocess.run([str(executable.resolve()), '--version'], capture_output=True,
                            text=True, timeout=30, check=False)
    version = (result.stdout+'\n'+result.stderr).strip()
    commit = manifest()['llama_cpp']['commit']
    if result.returncode != 0 or commit[:9] not in version:
        raise ValueError(f'Bundled llama.cpp must run successfully and match pinned commit {commit[:9]}: {version}')
    return executable


def ensure_runtime(destination, source=None):
    """Use an existing binary or fetch the official verified artifact during the BUILD."""
    destination = Path(destination)
    if source is not None:
        verify_runtime(Path(source))
        return Path(source)
    if destination.exists() and (list(destination.rglob('llama-server')) or
                                 list(destination.rglob('llama-server.exe'))):
        verify_runtime(destination)
        return destination
    staging = destination.parent.parent/'runtime-build-cache'
    models = Models(Store(staging), manifest())
    result = models.install_runtime()
    directory = Path(result['executable']).parent
    verify_runtime(directory)
    shutil.copytree(directory, destination, dirs_exist_ok=True, symlinks=True)
    verify_runtime(destination)
    return destination


def bundle_runtime(source, application_directory):
    """A package without the executable, shared libraries and licenses is rejected."""
    source = Path(source)
    executable = verify_runtime(source)
    destination = Path(application_directory)/'llama'
    shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True)
    included = verify_runtime(destination)
    relative = executable.relative_to(source)
    if included.relative_to(destination) != relative:
        raise ValueError('Bundled runtime layout changed unexpectedly')
    (destination/'bundled-runtime.json').write_text(json.dumps({
        'source_commit': manifest()['llama_cpp']['commit'], 'version': manifest()['llama_cpp']['release'],
        'executable': str(relative), 'delivery': 'included with application'}, indent=2))
    return included
