"""Create a local environment and install optional Sentinel-X resources."""

from __future__ import annotations

import argparse
import filecmp
import json
import math
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/backend"))


def environment_python(directory, windows=None):
    windows = sys.platform == "win32" if windows is None else windows
    return directory / ("Scripts/python.exe" if windows else "bin/python")


def run(command):
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def validate_checkpoint(path):
    """Check preprocessing metadata and load only tensor weights on the CPU."""
    import torch
    from sentinel_x.graphs import FEATURE_VERSION
    from sentinel_x.learning import ALL_LLM_FEATURE_NAMES, torch_modules

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("id") != path.stem or not path.stem.isalnum():
        raise ValueError("Checkpoint id must match the alphanumeric filename")
    if data.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Checkpoint feature version is incompatible")
    config = data.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be an object")
    for field in ("hidden", "threshold"):
        if type(config.get(field)) is not int or config[field] <= 0:
            raise ValueError(f"config.{field} must be a positive integer")

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    if not number(config.get("window")) or config["window"] < 0:
        raise ValueError("config.window must be nonnegative and finite")
    if not number(data.get("cutoff")) or not 0 <= data["cutoff"] <= 1:
        raise ValueError("Checkpoint cutoff must be between zero and one")
    metrics = data.get("report", {}).get("test", {})
    if not all(number(metrics.get(key)) for key in ("f1", "precision", "recall", "pr_auc")):
        raise ValueError("Original report.test evaluation metrics are required by the UI")
    names = data.get("llm_feature_names", [])
    if (
        not isinstance(names, list)
        or any(name not in ALL_LLM_FEATURE_NAMES for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Invalid semantic feature names")
    fields = [("mean", 6), ("scale", 6)]
    if names:
        fields += [("llm_mean", len(names)), ("llm_scale", len(names))]
    for field, size in fields:
        values = data.get(field)
        if (
            not isinstance(values, list)
            or len(values) != size
            or not all(number(value) for value in values)
            or ("scale" in field and any(value <= 0 for value in values))
        ):
            raise ValueError(f"{field} requires {size} finite values (scales must be positive)")
    _, model_class = torch_modules(len(names))
    model = model_class(config["hidden"])
    model.load_state_dict(
        torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
    )
    return data


def install_checkpoints(source, store):
    """Install complete pairs; preserve existing files and settings on reruns."""
    installed, problems = [], []
    for weights in sorted(source.glob("*.pt")):
        metadata = weights.with_suffix(".json")
        if not metadata.is_file():
            problems.append(f"Missing {metadata.name}; obtain the original training metadata.")
            continue
        try:
            data = validate_checkpoint(metadata)
            destination = store.root / "checkpoints"
            for path in (weights, metadata):
                target = destination / path.name
                if target.exists() and not filecmp.cmp(path, target, shallow=False):
                    raise ValueError(f"Refusing to overwrite different existing file: {target}")
            # Metadata is published last so an interrupted copy is not selectable.
            for path in (weights, metadata):
                target = destination / path.name
                if not target.exists():
                    with tempfile.NamedTemporaryFile(dir=destination, delete=False) as stream:
                        temporary = Path(stream.name)
                    try:
                        shutil.copyfile(path, temporary)
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
            installed.append(data["id"])
            print(f"Checkpoint ready: {data['id']}", flush=True)
        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            OSError,
            RuntimeError,
            pickle.UnpicklingError,
            EOFError,
        ) as exc:
            problems.append(f"{metadata.name}: {exc}")
    if installed and not store.setting("checkpoint"):
        mapping = source / "checkpoint_map.json"
        if not mapping.is_file():
            mapping = ROOT / "results/final/checkpoint_map.json"
        preferred = None
        if mapping.is_file():
            try:
                preferred = json.loads(mapping.read_text(encoding="utf-8-sig")).get(
                    "twitter15_emotional_language"
                )
            except (ValueError, AttributeError):
                pass
        if preferred in installed or len(installed) == 1:
            store.configure("checkpoint", preferred if preferred in installed else installed[0])
        else:
            print("Multiple checkpoints installed: select one in Settings > Models.")
    return installed, problems


class Progress:
    def check(self):
        pass

    def emit(self, event, **values):
        if event in ("stage", "file"):
            print(values.get("stage") or values.get("file", event), flush=True)


def configure(args):
    from sentinel_x.config import manifest
    from sentinel_x.downloads import Models
    from sentinel_x.state import Store, data_directory

    store = Store(args.data_dir or data_directory())
    models = Models(store, manifest())
    print(f"Application data: {store.root}", flush=True)
    installed, problems = install_checkpoints(args.weights_dir, store)
    local_qwen = args.weights_dir / "Qwen3.5-9B-Q4_K_M.gguf"
    if local_qwen.is_file():
        models.import_qwen(local_qwen)
        print(f"Using local Qwen: {local_qwen} (keep this file in place)")
    if args.runtime:
        from sentinel_x.runtime_packages import runtime_choices

        if args.runtime not in runtime_choices():
            raise ValueError(f"{args.runtime} runtime is unavailable on this platform; try cpu")
        models.install_runtime(Progress(), backend=args.runtime)
    if args.download_models:
        for key in ("qwen", "embedding"):
            if key == "qwen" and store.setting("qwen_path"):
                continue
            models.install(key, Progress())
    for problem in problems:
        print(f"CHECKPOINT INCOMPLETE: {problem}", file=sys.stderr)
    if not installed:
        print("No new GraphSAGE checkpoint installed. Add a matching .pt/.json pair and rerun.")
    print("Environment configured. See setup.md for launch and remaining application setup.")
    return 2 if problems else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "model weights")
    parser.add_argument("--data-dir", type=Path, help="Override application storage for this run")
    parser.add_argument("--dev", action="store_true", help="Include testing and packaging tools")
    parser.add_argument(
        "--skip-install", action="store_true", help="Reuse an installed environment"
    )
    parser.add_argument(
        "--download-models", action="store_true", help="Download Qwen and embeddings"
    )
    parser.add_argument(
        "--runtime",
        choices=("cpu", "cuda", "vulkan", "rocm"),
        help="Download and configure a llama.cpp runtime (cpu includes Apple Metal)",
    )
    parser.add_argument("--configure-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not (3, 11) <= sys.version_info[:2] < (3, 16):
        parser.error("Run this script with Python 3.11 through 3.15; see setup.md.")
    args.venv = args.venv.expanduser().absolute()
    args.weights_dir = args.weights_dir.expanduser().resolve()
    if args.data_dir:
        args.data_dir = args.data_dir.expanduser().resolve()
    if not args.weights_dir.is_dir():
        parser.error(f"Weights directory does not exist: {args.weights_dir}")
    if args.configure_only:
        return configure(args)
    python = environment_python(args.venv)
    if not python.is_file():
        if args.skip_install:
            parser.error("--skip-install requires an existing environment; omit it for first setup")
        if args.venv.exists() and any(args.venv.iterdir()):
            parser.error(
                "Environment directory is nonempty but has no Python; choose another --venv"
            )
        print(f"Creating environment: {args.venv}", flush=True)
        venv.EnvBuilder(with_pip=True).create(args.venv)
    if not (args.venv / "pyvenv.cfg").is_file():
        parser.error("--venv must point to a Python virtual environment")
    if not args.skip_install:
        run([python, "-m", "pip", "install", "--upgrade", "pip"])
        requirements = "requirements-dev.txt" if args.dev else "requirements.txt"
        run([python, "-m", "pip", "install", "-r", ROOT / requirements])
    run([python, "-m", "pip", "check"])
    command = [
        python,
        Path(__file__).resolve(),
        "--configure-only",
        "--weights-dir",
        args.weights_dir,
    ]
    if args.data_dir:
        command += ["--data-dir", args.data_dir]
    if args.runtime:
        command += ["--runtime", args.runtime]
    if args.download_models:
        command += ["--download-models"]
    result = subprocess.run([str(part) for part in command], cwd=ROOT)
    print(
        f'Launch with: "{python}" -m sentinel_x.cli '
        + (f'--data-dir "{args.data_dir}" ' if args.data_dir else "")
        + "gui"
    )
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
