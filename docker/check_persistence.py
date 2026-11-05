"""Seed/check disposable CI storage across recreation; never run seed on user data."""

import argparse
from pathlib import Path

from sentinel_x.state import Store, data_directory

SETTINGS = {"scope_model": "2b", "scope_offload": "modules", "context": 2048}
MARKER = ".container-persistence-check"
CONTENTS = "preserved across container recreation\n"


def check_storage(root, home, *, seed=False, device="cpu"):
    store = Store(root)
    settings = {**SETTINGS, "scope_device": device}
    paths = [home / "Exports" / MARKER]
    paths += [root / "models" / key / MARKER for key in ("qwen", "scope", "qwen_2b", "scope_2b")]
    if seed:
        for key, value in settings.items():
            store.configure(key, value)
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(CONTENTS)
    else:
        for key, expected in settings.items():
            assert store.setting(key) == expected, f"Setting did not persist: {key}"
        for path in paths:
            assert path.read_text() == CONTENTS, f"File did not persist: {path}"
        assert store.setting("checkpoint"), "Bundled GraphSAGE checkpoint was not configured"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seed", "check"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    check_storage(data_directory(), Path.home(), seed=args.action == "seed", device=args.device)
    print(f"Persistence {args.action}: model/device/offload settings, model files, and exports.")


if __name__ == "__main__":
    main()
