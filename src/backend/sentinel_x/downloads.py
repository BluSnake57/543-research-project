"""Resumable downloads verified against immutable Hugging Face tree metadata."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


class DownloadError(RuntimeError):
    """A download failure with a recovery instruction suitable for the UI."""


def request(url, headers=None):
    if urllib.parse.urlparse(url).scheme != 'https':
        raise ValueError('Downloads require HTTPS')
    host = urllib.parse.urlparse(url).hostname
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers={
            'User-Agent': 'Sentinel-X/0.1', **(headers or {})}), timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            raise  # The transfer handler can recover an invalid resume range.
        guidance = {
            401: 'The file requires access permission. Check the model source.',
            403: 'Access was denied. Check the model source and your network.',
            404: 'The file was not found. Check for an updated model catalog.',
            429: 'The server is busy. Wait a little and retry.',
        }.get(exc.code, 'The server could not complete the request. Retry later.')
        exc.close()
        raise DownloadError(f'{host} returned HTTP {exc.code}. {guidance} '
                            'Partial downloads are kept for retry.') from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DownloadError(f'Cannot connect to {host}. Check your internet connection, '
                            'DNS, and HTTPS access, then retry. Partial downloads are kept.') from exc


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
    def stage(text):
        if job:
            job.check()
            job.emit('stage', stage=text, file=target.name)
    stage('Checking saved file')
    if target.exists() and expected and digest_file(target, algorithm, job) == expected:
        stage('File already verified')
        return target
    receipt_path = target.with_name(target.name+'.checksum.json')
    if target.exists() and not expected and receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('url') == url and digest_file(target, receipt['algorithm'], job) == receipt['digest']:
                stage('File already verified')
                return target
        except (ValueError, KeyError):
            pass  # An invalid receipt is never used as proof of a complete file.
    partial = target.with_name(target.name+'.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if size and shutil.disk_usage(target.parent).free < max(0, size-offset) + 256*1024**2:
        raise ValueError('Not enough free disk space for download')
    stage('Connecting to '+str(urllib.parse.urlparse(url).hostname))
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    try:
        response = request(url, headers)
    except urllib.error.HTTPError as exc:
        if exc.code != 416:
            raise
        remote = re.fullmatch(r'bytes \*/(\d+)', exc.headers.get('Content-Range', '') if exc.headers else '')
        remote_size = int(remote[1]) if remote else None
        exc.close()
        if offset and ((size is not None and offset == size and remote_size in (None, size)) or
                       (size is None and offset == remote_size)):
            response = None
        else:
            # A stale/oversized partial cannot be resumed; start a fresh transfer.
            offset = 0
            response = request(url)
    total = size
    if response is not None:
        try:
            with response:
                resumed = response.status == 206 and offset > 0
                length = int(response.headers.get('Content-Length', 0))
                if response.status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset:
                        raise ValueError('Server returned an invalid resume offset. Retry the download.')
                    end, remote_size = int(match[2]), int(match[3])
                    if end < offset or end >= remote_size or (length and length != end-offset+1):
                        raise ValueError('Server returned an invalid download range.')
                    if size is not None and remote_size != size:
                        raise ValueError('Download size changed at the source. Check the model catalog.')
                    total = size if size is not None else remote_size
                else:
                    offset = 0
                    total = size if size is not None else (length or None)
                if total and shutil.disk_usage(target.parent).free < max(0, total-offset)+256*1024**2:
                    raise ValueError('Not enough disk space for this download')
                started = time.monotonic(); initial = offset; last_emit = started
                def progress(force=False):
                    nonlocal last_emit
                    now = time.monotonic()
                    if job and (force or now-last_emit >= .25):
                        speed = (offset-initial)/max(now-started, .001)
                        job.emit('download', file=target.name, downloaded=offset, total=total,
                                 speed=speed, eta=(max(0, total-offset)/speed if total and speed else None),
                                 resumed=resumed)
                        last_emit = now
                progress(True)
                with partial.open('ab' if resumed else 'wb') as stream:
                    while True:
                        if job:
                            job.check()
                        chunk = response.read(64*1024)
                        if not chunk:
                            break
                        stream.write(chunk); offset += len(chunk)
                        progress()
                    stream.flush(); os.fsync(stream.fileno())
                progress(True)
                if length and offset-initial != length:
                    raise DownloadError('Download was interrupted before all bytes arrived. Retry to resume.')
        except (OSError, http.client.IncompleteRead) as exc:
            raise DownloadError('The connection was interrupted. Retry to resume the partial download.') from exc
    if total is not None and (not partial.exists() or partial.stat().st_size != total):
        raise DownloadError('Download size mismatch; partial file retained for retry. Retry to resume.')
    stage('Verifying file integrity')
    digest = digest_file(partial, algorithm, job)
    if expected and digest != expected:
        partial.unlink()
        raise ValueError('Checksum mismatch; corrupt download removed. Retry to download a fresh copy.')
    if job:
        job.check()
    partial.replace(target)
    receipt_path.write_text(json.dumps({
        'algorithm': algorithm, 'digest': digest, 'url': url, 'upstream_verified': bool(expected)}))
    stage('File verified')
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
        if job:
            job.emit('stage', stage='Checking download sources')
        catalog = self.catalog(key)
        folder = self.store.root / 'models' / key
        # A ready marker is written only after every file has been verified.
        folder.mkdir(exist_ok=True)
        (folder / '.ready').unlink(missing_ok=True)
        for number, item in enumerate(catalog['files'], 1):
            if job:
                job.emit('file', file=item['name'], index=number, count=len(catalog['files']))
            download(item['url'], folder / item['name'], item['checksum'], item['size'], job,
                     item['algorithm'])
        if job:
            job.check()
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

    def fever_status(self):
        root = self.store.root/'resources'/'fever'
        archive = root/'wiki-pages.zip'
        downloaded = archive.is_file() and (self.store.setting('fever_downloaded', False) or
            archive.with_name(archive.name+'.checksum.json').is_file() or zipfile.is_zipfile(archive))
        return {'archive': str(archive), 'downloaded': bool(downloaded),
                'indexed': bool(self.store.setting('fever_ready', False))}

    def fever(self, job=None, extract=True, download_missing=True):
        root = self.store.root / 'resources' / 'fever'
        root.mkdir(exist_ok=True)
        archive = root/'wiki-pages.zip'
        if download_missing:
            archive = download(self.manifest['fever']['evidence'], archive, job=job)
        elif not archive.is_file():
            raise ValueError('Download FEVER first. Building its index uses the saved archive without network access.')
        # FEVER does not publish a checksum here; record a local digest and rely on HTTPS + ZIP CRC.
        try:
            with zipfile.ZipFile(archive) as bundle:
                # Validate paths even for download-only mode, without decompressing the corpus.
                for item in bundle.infolist():
                    destination = (root/item.filename).resolve()
                    if not destination.is_relative_to(root.resolve()):
                        raise ValueError('Unsafe path in FEVER archive')
                items = [item for item in bundle.infolist() if not item.is_dir() and
                         item.filename.endswith('.jsonl') and not any(
                             part.startswith('.') or part == '__MACOSX' for part in Path(item.filename).parts)]
                if not items:
                    raise ValueError('FEVER archive contains no Wikipedia evidence files')
                if any(item.filename.startswith('wiki-pages/') for item in items):
                    items = [item for item in items if item.filename.startswith('wiki-pages/')]
                self.store.configure('fever_downloaded', True)
                if extract:
                    self.extract_fever(bundle, items, root, job)
        except zipfile.BadZipFile as exc:
            archive.unlink(missing_ok=True)
            archive.with_name(archive.name+'.checksum.json').unlink(missing_ok=True)
            self.store.configure('fever_downloaded', False)
            raise DownloadError('Wikipedia archive failed integrity verification. The invalid archive was removed; retry to download a fresh copy.') from exc
        # Claims are stored separately, outside the resources evidence directory.
        annotations = self.store.root / 'datasets' / 'fever'
        annotations.mkdir(parents=True, exist_ok=True)
        if download_missing:
            for name in ('train', 'development'):
                download(self.manifest['fever'][name], annotations/f'{name}.jsonl', job=job)
        evidence = root/'wiki-pages' if any(item.filename.startswith('wiki-pages/') for item in items) else root
        return {'archive': str(archive), 'downloaded': True, 'index_deferred': not extract,
                'evidence_directory': str(evidence), 'annotations_directory': str(annotations),
                'checksum_policy': 'HTTPS and ZIP CRC; upstream FEVER checksum unavailable'}

    def extract_fever(self, bundle, items, root, job):
        import zlib
        receipt = root/'.extracted.json'
        try:
            saved = json.loads(receipt.read_text())
            if not isinstance(saved, dict): saved = {}
        except (OSError, ValueError):
            saved = {}
        total = sum(item.file_size for item in items)
        extracted = 0; last_progress = time.monotonic()
        if job:
            job.emit('stage', stage='Preparing saved Wikipedia files · verified files are reused')
        for item in items:
            if job: job.check()
            destination = (root/item.filename).resolve()
            reused = False
            if destination.is_file() and destination.stat().st_size == item.file_size:
                stamp = [item.CRC, item.file_size, destination.stat().st_mtime_ns]
                reused = saved.get(item.filename) == stamp
                if not reused:
                    # Adopt files extracted by older versions, verifying them once without rewriting.
                    crc = 0
                    with destination.open('rb') as stream:
                        while block := stream.read(1024*1024):
                            if job: job.check()
                            crc = zlib.crc32(block, crc)
                    reused = crc == item.CRC
            if not reused:
                if shutil.disk_usage(root).free < item.file_size+512*1024**2:
                    raise ValueError('Not enough space to extract the next FEVER evidence file; completed files are kept.')
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name+'.extracting')
                try:
                    with bundle.open(item) as source, temporary.open('wb') as target:
                        while block := source.read(1024*1024):
                            if job: job.check()
                            target.write(block); extracted += len(block)
                            if job and time.monotonic()-last_progress >= .25:
                                job.emit('extraction', file=item.filename, extracted=extracted, total=total)
                                last_progress = time.monotonic()
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                extracted += item.file_size
            saved[item.filename] = [item.CRC, item.file_size, destination.stat().st_mtime_ns]
            temporary_receipt = receipt.with_suffix('.tmp')
            temporary_receipt.write_text(json.dumps(saved))
            temporary_receipt.replace(receipt)
            if job:
                job.emit('extraction', file=item.filename, extracted=extracted, total=total, reused=reused)


    def install_runtime(self, job=None, backend='cpu', activate=True):
        from .runtime_packages import install_runtime
        return install_runtime(self, job, backend, activate)


def runtime_asset(assets, system, machine, backend='cpu'):
    from .runtime_packages import asset_for
    return asset_for(assets, system, machine, backend)


def unpack_runtime_archive(archive, directory):
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


def extract_runtime(archive, directory):
    directory = Path(directory).resolve()
    unpack_runtime_archive(archive, directory)
    candidates = list(directory.rglob('llama-server')) + list(directory.rglob('llama-server.exe'))
    if len(candidates) != 1:
        raise ValueError('Runtime archive must contain one llama-server executable')
    for path in directory.rglob('*'):
        if path.is_file() and (path.name.startswith('llama-') or '.so' in path.name or path.suffix == '.dylib'):
            path.chmod(path.stat().st_mode | 0o755)
    return candidates[0]
