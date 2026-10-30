"""Opt-in live checks using the application's actual HTTPS downloader.

Downloads and verifies a small embedding config; probes four bytes of large files.
No model weights or full Wikipedia archive are downloaded.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/backend'))
from sentinel_x.config import manifest
from sentinel_x.downloads import Models, download, request
from sentinel_x.state import Store


def check_source(models, source, directory):
    if source == 'embedding':
        item = next(x for x in models.catalog('embedding')['files'] if x['name'] == 'config.json')
        path = download(item['url'], directory/'embedding-config.json', item['checksum'],
                        item['size'], algorithm=item['algorithm'])
        if not isinstance(json.loads(path.read_text()), dict):
            raise ValueError('Downloaded config is not a JSON object')
        return 'Downloaded complete config and verified its size and checksum.'
    if source == 'qwen':
        item = next(x for x in models.catalog('qwen')['files'] if x['name'].endswith('.gguf'))
        url, magic = item['url'], b'GGUF'
    else:
        url, magic = models.manifest['fever']['evidence'], b'PK\x03\x04'
    with request(url, {'Range': 'bytes=0-3'}) as response:
        if response.read(4) != magic:
            raise ValueError('Download source did not return the expected file header')
    return 'Source reachable and file header valid; full multi-gigabyte transfer not tested.'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('all', 'embedding', 'qwen', 'fever'), default='all')
    options = parser.parse_args(argv)
    results = []; failed = False
    with tempfile.TemporaryDirectory(prefix='sentinel-download-check-') as temporary:
        directory = Path(temporary); models = Models(Store(directory), manifest())
        for source in (('embedding', 'qwen', 'fever') if options.source == 'all' else (options.source,)):
            try:
                results.append({'source': source, 'passed': True,
                                'detail': check_source(models, source, directory)})
            except Exception as exc:
                failed = True
                results.append({'source': source, 'passed': False, 'detail': str(exc)})
    print(json.dumps(results, indent=2))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
