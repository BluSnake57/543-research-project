"""Packaged asset discovery shared by native app, CLI, and release scripts."""

import json
from pathlib import Path
import sys

ASSETS = Path(__file__).resolve().parent / "assets"


def manifest():
    if getattr(sys, "frozen", False):
        release = Path(sys.executable).parent / "runtime-manifest.json"
        if release.exists():
            return json.loads(release.read_text())
    return json.loads((ASSETS / "runtime-manifest.json").read_text())


MODEL_PROFILES = {
    "9b": {
        "label": "Qwen3.5-9B",
        "keys": ("qwen", "scope"),
        "hidden": 4096,
        "width": 65536,
        "layers": 32,
    },
    "2b": {
        "label": "Qwen3.5-2B",
        "keys": ("qwen_2b", "scope_2b"),
        "hidden": 2048,
        "width": 32768,
        "layers": 24,
    },
}
MODEL_KEYS = ("qwen", "scope", "qwen_2b", "scope_2b", "embedding")


def model_profile(name):
    if name not in MODEL_PROFILES:
        raise ValueError("Select the 9B or 2B model and its matching Scope SAE")
    return MODEL_PROFILES[name]
