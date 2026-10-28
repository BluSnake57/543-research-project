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
