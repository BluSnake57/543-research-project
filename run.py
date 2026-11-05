"""Launch Sentinel-X using the project environment, without shell activation."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    options, arguments = parser.parse_known_args(argv)
    environment = options.venv.expanduser().absolute()
    if arguments and arguments[0] == "setup":
        return subprocess.call(
            [
                sys.executable,
                str(ROOT / "scripts/setup_environment.py"),
                "--venv",
                str(environment),
                *arguments[1:],
            ],
            cwd=ROOT,
        )
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file() or not (environment / "pyvenv.cfg").is_file():
        print("Run python run.py setup first, or pass --venv PATH.", file=sys.stderr)
        return 1
    return subprocess.call([str(python), "-m", "sentinel_x.cli", *(arguments or ["gui"])], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
