"""Packaged asset discovery shared by native app, CLI, and release scripts."""
import json
from pathlib import Path
import sys

ASSETS = Path(__file__).resolve().parent/'assets'


def manifest():
    if getattr(sys, 'frozen', False):
        release = Path(sys.executable).parent/'runtime-manifest.json'
        if release.exists():
            return json.loads(release.read_text())
    return json.loads((ASSETS/'runtime-manifest.json').read_text())


def bundled_runtime():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent/'llama'
    return Path(__file__).resolve().parents[3]/'build/llama/bin'
