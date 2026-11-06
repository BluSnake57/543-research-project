"""Create a local environment and install optional Sentinel-X resources."""

from __future__ import annotations

import argparse
from contextlib import closing
import filecmp
import json
import math
import os
import pickle
from pathlib import Path
import shutil
import sqlite3
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


def verify_rocm_python(python):
    """Check the target interpreter, not the Python running the installer."""
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import json, torch; print(json.dumps({'torch': torch.__version__, "
            "'hip': torch.version.hip}))",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    try:
        installed = json.loads(result.stdout) if result.returncode == 0 else {}
    except ValueError:
        installed = {}
    if not installed.get("hip"):
        detail = installed.get("torch") or result.stderr.strip() or "PyTorch could not be inspected"
        raise ValueError(
            f"ROCm-enabled PyTorch is required in {python}. Detected: {detail}. "
            "Rerun setup --device rocm --torch-index-url URL using the official ROCm "
            "wheel index compatible with your GPU, OS and Python (see docs/ROCM.md), "
            "or install the AMD-provided wheels into this environment first. "
            "Selecting ROCm alone does not install it. No separate llama.cpp runner is used."
        )
    print(f"ROCm PyTorch ready: {installed['torch']} (HIP {installed['hip']})", flush=True)


def install_torch(python, index_url):
    """Select torch only from the requested index; resolve its dependencies separately."""
    with tempfile.TemporaryDirectory(prefix="sentinel-torch-") as directory:
        run(
            [
                python,
                "-m",
                "pip",
                "download",
                "--no-deps",
                "--only-binary=:all:",
                "--dest",
                directory,
                "torch>=2.6,<3",
                "--index-url",
                index_url,
            ]
        )
        wheels = list(Path(directory).glob("torch-*.whl"))
        if len(wheels) != 1:
            raise ValueError("Expected exactly one PyTorch wheel from the selected index")
        # Pin the wheel by path: a newer CUDA version on PyPI cannot win resolution.
        # Keep PyPI available for dependencies missing from vendor indexes.
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                str(wheels[0]),
                "--extra-index-url",
                index_url,
            ]
        )


def application_directory():
    """Find storage without creating it or running migrations during reset preflight."""
    override = os.environ.get("SENTINEL_X_DATA_DIR") or os.environ.get("VERITAS_DATA_DIR")
    if override:
        return Path(override).expanduser().absolute()
    if sys.platform == "win32":
        parent = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        current, legacy = parent / "Sentinel-X", parent / "Veritas"
    elif sys.platform == "darwin":
        parent = Path.home() / "Library/Application Support"
        current, legacy = parent / "Sentinel-X", parent / "Veritas"
    else:
        parent = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        current, legacy = parent / "sentinel-x", parent / "veritas"
    if not current.exists() and legacy.exists():
        return legacy
    return current


def validate_reset_path(path, protected=()):
    if path.is_symlink():
        raise ValueError(f"Refusing to reset a symlink: {path}")
    resolved = path.resolve()
    for keep in (ROOT, Path.home(), Path(sys.prefix), Path(sys.base_prefix), *protected):
        if Path(keep).resolve().is_relative_to(resolved):
            raise ValueError(f"Refusing to reset protected directory: {path}")
    return resolved


def validate_environment_reset(path, protected=()):
    if Path(sys.prefix).resolve().is_relative_to(path.resolve()):
        raise ValueError(
            "Cannot reset the running Python environment. Deactivate it and run "
            "setup with a system Python, or choose a different --venv."
        )
    validate_reset_path(path, protected)
    if path.exists() and not (path / "pyvenv.cfg").is_file():
        raise ValueError(f"Refusing to reset a directory without pyvenv.cfg: {path}")


def application_database(path):
    if not path.exists():
        return None
    for name in ("sentinel-x.sqlite", "veritas.sqlite"):
        database = path / name
        if database.is_symlink():
            raise ValueError(f"Refusing to reset a linked database: {database}")
        if database.is_file():
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            if {"settings", "conversations", "messages", "graphs"} <= tables:
                return database
    if any(path.iterdir()):
        raise ValueError(f"Refusing to reset unrecognized application storage: {path}")
    return None


def reset_application(path, factory=False):
    """Caller preflights protected paths. Never follow configured external resource paths."""
    database = application_database(path)
    if factory:
        if path.exists():
            shutil.rmtree(path)
        # Leave a fresh database so old legacy storage cannot be migrated back on launch.
        from sentinel_x.state import Store

        Store(path)
        print(f"Factory reset complete: {path}. Downloads, conversations and app data removed.")
    elif database:
        preferences = (
            "resources",
            "context",
            "gpu_layers",
            "llama_path",
            "qwen_path",
            "checkpoint",
            "thinking_budget",
            "scope_device",
            "scope_model",
            "scope_offload",
            "setup_started",
            "hide_setup_at_startup",
        )
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.executemany("DELETE FROM settings WHERE key=?", [(k,) for k in preferences])
            connection.execute("DELETE FROM settings WHERE key LIKE 'ui:%' OR key LIKE 'skill:%'")
        print(f"Preferences reset: {path}. Conversations, drafts and downloads preserved.")
    else:
        print(f"No saved preferences to reset: {path}")


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
    if args.model:
        store.configure("scope_model", args.model)
    if args.cpu_offload:
        store.configure("scope_offload", args.cpu_offload)
    if args.device:
        store.configure("scope_device", args.device)
    if args.download_models or args.download_both:
        from sentinel_x.config import model_profile, MODEL_KEYS

        keys = (
            MODEL_KEYS
            if args.download_both
            else (*model_profile(store.setting("scope_model", "9b"))["keys"], "embedding")
        )
        for key in keys:
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
        "--torch-index-url",
        help="Replace PyTorch using this wheel index before installing app dependencies",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation; still check dependencies and configure resources",
    )
    parser.add_argument(
        "--reset-venv",
        action="store_true",
        help="Delete and recreate --venv, then reinstall dependencies",
    )
    resets = parser.add_mutually_exclusive_group()
    resets.add_argument(
        "--reset-settings",
        action="store_true",
        help="Reset app preferences only; preserve downloads and conversations",
    )
    resets.add_argument(
        "--factory-reset",
        action="store_true",
        help="Delete all app storage, including models, FEVER, runtime and chats",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download Qwen, Scope layer 16, and embeddings",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "rocm"),
        help="Scope device; CPU is an explicit, slower fallback requiring more memory",
    )
    parser.add_argument(
        "--model", choices=("9b", "2b"), help="Select a model; omitted preserves the current choice"
    )
    parser.add_argument(
        "--cpu-offload",
        choices=("none", "sae", "modules"),
        help="CUDA/ROCm CPU-offload policy; omitted preserves the current choice",
    )
    parser.add_argument(
        "--download-both",
        action="store_true",
        help="Download both model/SAE pairs and embeddings without changing selection",
    )
    parser.add_argument("--configure-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not (3, 11) <= sys.version_info[:2] < (3, 16):
        parser.error("Run this script with Python 3.11 through 3.15; see setup.md.")
    args.venv = args.venv.expanduser().absolute()
    args.weights_dir = args.weights_dir.expanduser().resolve()
    if args.data_dir:
        args.data_dir = args.data_dir.expanduser().absolute()
    resetting_app = args.reset_settings or args.factory_reset
    if args.torch_index_url and (args.skip_install or args.configure_only or resetting_app):
        parser.error(
            "--torch-index-url cannot be combined with skip-install, configure-only or resets"
        )
    if args.reset_venv and args.skip_install:
        parser.error("--reset-venv cannot be combined with --skip-install")
    if resetting_app and (
        args.download_models
        or args.download_both
        or args.device
        or args.model
        or args.cpu_offload
        or args.configure_only
    ):
        parser.error("Application resets cannot be combined with downloads or --configure-only")
    if not resetting_app and not args.weights_dir.is_dir():
        parser.error(f"Weights directory does not exist: {args.weights_dir}")
    reset_root = args.data_dir or application_directory()
    # Validate every deletion target before making any changes.
    if args.reset_venv:
        validate_environment_reset(args.venv, (args.weights_dir, reset_root))
    if resetting_app:
        reset_root = validate_reset_path(reset_root, (args.weights_dir, args.venv))
        application_database(reset_root)
        reset_application(reset_root, factory=args.factory_reset)
        if not args.reset_venv:
            return 0
    if args.reset_venv and args.venv.exists():
        print(f"Removing Python environment: {args.venv}", flush=True)
        shutil.rmtree(args.venv)
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
        if args.torch_index_url:
            install_torch(python, args.torch_index_url)
        # Fail before ordinary pip installation can fetch a CUDA wheel for an AMD request.
        if args.device == "rocm":
            verify_rocm_python(python)
        requirements = "requirements-dev.txt" if args.dev else "requirements.txt"
        run([python, "-m", "pip", "install", "-r", ROOT / requirements])
    run([python, "-m", "pip", "check"])
    if args.device == "rocm":
        verify_rocm_python(python)
    if resetting_app:
        print("Python environment rebuilt. Run setup --skip-install to install local checkpoints.")
        return 0
    command = [
        python,
        Path(__file__).resolve(),
        "--configure-only",
        "--weights-dir",
        args.weights_dir,
    ]
    if args.model:
        command += ["--model", args.model]
    if args.cpu_offload:
        command += ["--cpu-offload", args.cpu_offload]
    if args.download_both:
        command += ["--download-both"]
    if args.data_dir:
        command += ["--data-dir", args.data_dir]
    if args.device:
        command += ["--device", args.device]
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
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        sqlite3.Error,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
