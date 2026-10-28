"""Resolve model revisions and hashes for release builds, without downloading model weights."""
import json
from pathlib import Path
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src/backend'))
from veritas.downloads import Models
from veritas.state import Store

manifest = json.loads((root/'runtime-manifest.json').read_text())
with tempfile.TemporaryDirectory() as temporary:
    models = Models(Store(Path(temporary)), manifest)
    for key in ('qwen', 'embedding'):
        catalog = models.catalog(key)
        manifest[key]['revision'] = catalog['revision']
        manifest[key]['artifacts'] = catalog['files']
(root/'build').mkdir(exist_ok=True)
(root/'build/runtime-manifest.json').write_text(json.dumps(manifest, indent=2))
print('Pinned release model revisions and checksums in build/runtime-manifest.json')
