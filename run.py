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
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file() or not (environment / "pyvenv.cfg").is_file():
        print(
            "Run python scripts/setup_environment.py first, or pass --venv PATH.", file=sys.stderr
        )
        return 1
    return subprocess.call([str(python), "-m", "sentinel_x.cli", *(arguments or ["gui"])], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
