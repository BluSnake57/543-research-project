"""Resumable downloads verified against immutable Hugging Face tree metadata."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile


def request(url, headers=None):
    if urllib.parse.urlparse(url).scheme != 'https':
        raise ValueError('Downloads require HTTPS')
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'Veritas/0.1', **(headers or {})}), timeout=60)


def get_json(url):
    with request(url) as response:
        return json.load(response)


def digest_file(path, algorithm='sha256', job=None):
    digest = hashlib.new('sha1' if algorithm == 'git-sha1' else algorithm)
    if algorithm == 'git-sha1':
        digest.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024):
            if job:
                job.check()
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target, expected=None, size=None, job=None, algorithm='sha256'):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and expected and digest_file(target, algorithm, job) == expected:
        return target
    receipt_path = target.with_name(target.name+'.checksum.json')
    if target.exists() and not expected and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('url') == url and digest_file(target, receipt['algorithm'], job) == receipt['digest']:
            return target
    partial = target.with_name(target.name+'.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if size and shutil.disk_usage(target.parent).free < max(0, size-offset) + 256*1024**2:
        raise ValueError('Not enough free disk space for download')
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    try:
        response = request(url, headers)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and size and offset == size:
            response = None
        else:
            raise
    if response is not None:
        with response:
            resumed = response.status == 206 and offset > 0
            if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                raise ValueError('Server returned an invalid resume offset')
            if not resumed:
                offset = 0
            length = int(response.headers.get('Content-Length', 0))
            total = size or (offset+length if length else None)
            if total and shutil.disk_usage(target.parent).free < total-offset+256*1024**2:
                raise ValueError('Not enough disk space')
            with partial.open('ab' if resumed else 'wb') as stream:
                while chunk := response.read(1024*1024):
                    if job:
                        job.check()
                    stream.write(chunk); offset += len(chunk)
                    if job:
                        job.emit('download', file=target.name, downloaded=offset, total=total)
                stream.flush(); os.fsync(stream.fileno())
    if size and partial.stat().st_size != size:
        raise ValueError('Download size mismatch; partial file retained for retry')
    digest = digest_file(partial, algorithm, job)
    if expected and digest != expected:
        partial.unlink()
        raise ValueError('Checksum mismatch; corrupt download removed')
    partial.replace(target)
    target.with_name(target.name+'.checksum.json').write_text(json.dumps({
        'algorithm': algorithm, 'digest': digest, 'url': url, 'upstream_verified': bool(expected)}))
    return target


class Models:
    def __init__(self, store, manifest):
        self.store = store
        self.manifest = manifest

    def catalog(self, key):
        pinned = self.store.root / 'models' / f'{key}.lock.json'
        if pinned.exists():
            return json.loads(pinned.read_text())
        config = self.manifest[key]
        if config.get('artifacts'):
            data = {'repo': config['repo'], 'revision': config['revision'],
                    'files': config['artifacts']}
            pinned.write_text(json.dumps(data, indent=2))
            return data
        repo = config['repo']; revision = config['revision']
        if not revision:
            revision = get_json(f'https://huggingface.co/api/models/{repo}')['sha']
        if len(revision) != 40:
            raise ValueError('Expected an immutable model revision')
        files = []
        for name in config['files']:
            path = str(Path(name).parent).replace('\\', '/')
            suffix = '' if path == '.' else '/'+path
            entries = get_json(f'https://huggingface.co/api/models/{repo}/tree/{revision}{suffix}')
            entry = next((e for e in entries if e['path'] == name), None)
            if not entry:
                if name in ('model.safetensors', 'config.json', 'tokenizer.json', 'modules.json',
                            'Qwen3.5-9B-Q4_K_M.gguf', '1_Pooling/config.json'):
                    raise ValueError(f'Required artifact missing: {name}')
                continue
            lfs = entry.get('lfs')
            checksum = lfs.get('oid') if lfs else entry['oid']
            if checksum.startswith('sha256:'):
                checksum = checksum.removeprefix('sha256:')
            files.append({'name': name, 'size': entry['size'], 'checksum': checksum,
                          'algorithm': 'sha256' if lfs else 'git-sha1',
                          'url': f'https://huggingface.co/{repo}/resolve/{revision}/{name}'})
        data = {'repo': repo, 'revision': revision, 'files': files}
        temporary = pinned.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2)); temporary.replace(pinned)
        return data

    def install(self, key, job=None):
        if key not in ('qwen', 'embedding'):
            raise ValueError('Unknown model')
        catalog = self.catalog(key)
        folder = self.store.root / 'models' / key
        # A ready marker is written only after every file has been verified.
        folder.mkdir(exist_ok=True)
        (folder / '.ready').unlink(missing_ok=True)
        for item in catalog['files']:
            download(item['url'], folder / item['name'], item['checksum'], item['size'], job,
                     item['algorithm'])
        (folder / '.ready').write_text(catalog['revision'])
        return {'model': key, 'revision': catalog['revision']}

    def status(self):
        return {key: {'ready': (self.store.root/'models'/key/'.ready').exists(),
                      'path': str(self.store.root/'models'/key)} for key in ('qwen', 'embedding')}

    def import_qwen(self, source):
        path = Path(source).expanduser().resolve()
        with path.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise ValueError('Not a GGUF model')
        self.store.configure('qwen_path', str(path))
        return {'path': str(path), 'warning': 'Local GGUF architecture and tool compatibility require a runtime smoke test'}

    def fever(self, job=None):
        root = self.store.root / 'resources' / 'fever'
        root.mkdir(exist_ok=True)
        archive = download(self.manifest['fever']['evidence'], root/'wiki-pages.zip', job=job)
        # FEVER does not publish a checksum here; record a local digest and rely on HTTPS + ZIP CRC.
        with zipfile.ZipFile(archive) as bundle:
            total = sum(item.file_size for item in bundle.infolist())
            if shutil.disk_usage(root).free < total+512*1024**2:
                raise ValueError('Not enough space to extract FEVER Wikipedia evidence')
            for item in bundle.infolist():
                if job:
                    job.check()
                destination = (root/item.filename).resolve()
                if not destination.is_relative_to(root.resolve()):
                    raise ValueError('Unsafe path in FEVER archive')
                if item.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, destination.open('wb') as target:
                    while block := source.read(1024*1024):
                        if job:
                            job.check()
                        target.write(block)
        # Claims are stored separately, outside the resources evidence directory.
        annotations = self.store.root / 'datasets' / 'fever'
        annotations.mkdir(parents=True, exist_ok=True)
        for name in ('train', 'development'):
            download(self.manifest['fever'][name], annotations/f'{name}.jsonl', job=job)
        return {'evidence_directory': str(root), 'annotations_directory': str(annotations),
                'checksum_policy': 'HTTPS and ZIP CRC; upstream FEVER checksum unavailable'}


    def install_runtime(self, job=None):
        import platform
        import re
        import tempfile
        catalog = self.store.setting('runtime_catalog')
        system, machine = platform.system(), platform.machine()
        if catalog and catalog.get('platform') != [system, machine]:
            catalog = None
        if not catalog:
            release = self.manifest['llama_cpp']['release']
            base = 'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/'
            metadata = get_json(base+release)
            try:
                asset = runtime_asset(metadata['assets'], system, machine)
            except ValueError:
                pointer = next((a for a in metadata['assets'] if a['name'] == 'nightly-tag.txt'), None)
                if not pointer:
                    raise ValueError('Release has no precompiled runtime artifacts')
                with request(pointer['browser_download_url']) as response:
                    release = response.read(128).decode().strip()
                if not re.fullmatch(r'b\d+', release):
                    raise ValueError('Invalid official binary release pointer')
                metadata = get_json(base+release)
                asset = runtime_asset(metadata['assets'], system, machine)
            digest = asset.get('digest', '')
            if not digest.startswith('sha256:'):
                raise ValueError('Official runtime artifact has no SHA-256 digest; choose a verified local executable instead')
            catalog = {'release': release, 'name': asset['name'], 'url': asset['browser_download_url'],
                       'checksum': digest[7:], 'size': asset['size'], 'platform': [system, machine]}
            self.store.configure('runtime_catalog', catalog)
        cache = self.store.root/'runtime'
        archive = download(catalog['url'], cache/catalog['name'], catalog['checksum'], catalog['size'], job)
        if job:
            job.check()
        with tempfile.TemporaryDirectory(prefix='unpack-', dir=cache) as temporary:
            executable = extract_runtime(archive, Path(temporary))
            relative = executable.relative_to(temporary)
            destination = cache/'llama'
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(temporary, destination)
        path = destination/relative
        self.store.configure('llama_path', str(path))
        return {'executable': str(path), 'release': catalog['release'], 'precompiled': True}

def runtime_asset(assets, system, machine):
    """Choose the official CPU/Metal binary, without guessing a download URL."""
    architecture = {'amd64': 'x64', 'x86_64': 'x64', 'aarch64': 'arm64', 'arm64': 'arm64'}.get(machine.lower())
    if not architecture:
        raise ValueError(f'Unsupported runtime architecture: {machine}')
    names = {'Linux': ('ubuntu', 'linux'), 'Darwin': ('macos',), 'Windows': ('win', 'windows')}.get(system)
    if not names:
        raise ValueError(f'Unsupported operating system: {system}')
    choices = []
    for asset in assets:
        name = asset['name'].lower()
        if not name.endswith(('.zip', '.tar.gz', '.tgz')):
            continue
        if not any(part in name for part in names) or architecture not in name:
            continue
        if any(part in name for part in ('cuda', 'vulkan', 'sycl', 'rocm', 'hip', 'cudart', 'opencl')):
            continue
        choices.append(asset)
    if not choices:
        raise ValueError('This release has no compatible precompiled runtime. Choose a local llama-server executable.')
    return sorted(choices, key=lambda item: ('cpu' not in item['name'].lower(), item['name']))[0]


def extract_runtime(archive, directory):
    import tarfile
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                target = (directory/item.filename).resolve()
                if not target.is_relative_to(directory):
                    raise ValueError('Unsafe runtime archive path')
            if shutil.disk_usage(directory).free < sum(x.file_size for x in bundle.infolist()):
                raise ValueError('Not enough disk space to unpack runtime')
            bundle.extractall(directory)
    else:
        with tarfile.open(archive) as bundle:
            if shutil.disk_usage(directory).free < sum(x.size for x in bundle.getmembers()):
                raise ValueError('Not enough disk space to unpack runtime')
            bundle.extractall(directory, filter='data')
    candidates = list(directory.rglob('llama-server')) + list(directory.rglob('llama-server.exe'))
    if len(candidates) != 1:
        raise ValueError('Runtime archive must contain one llama-server executable')
    for path in directory.rglob('*'):
        if path.is_file() and (path.name.startswith('llama-') or '.so' in path.name or path.suffix == '.dylib'):
            path.chmod(path.stat().st_mode | 0o755)
    return candidates[0]
