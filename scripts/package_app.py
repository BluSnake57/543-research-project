"""Freeze the native Python app on its target OS. No JavaScript or C++ compilation."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--llama-dir', type=Path, help='Optional directory of existing precompiled llama.cpp binaries')
args = parser.parse_args()
command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--windowed',
           '--name', 'Veritas', '--paths', str(ROOT/'src/backend'),
           '--collect-data', 'veritas', '--collect-all', 'sentence_transformers',
           '--collect-all', 'transformers', '--collect-all', 'tokenizers',
           '--distpath', str(ROOT/'dist'), '--workpath', str(ROOT/'build/pyinstaller'),
           '--specpath', str(ROOT/'build'), str(ROOT/'scripts/desktop_entry.py')]
subprocess.run(command, check=True, cwd=ROOT)
output = ROOT/'dist/Veritas'
if sys.platform == 'darwin':
    output = ROOT/'dist/Veritas.app/Contents/MacOS'
if args.llama_dir:
    server = 'llama-server.exe' if sys.platform == 'win32' else 'llama-server'
    if not any(args.llama_dir.rglob(server)):
        parser.error('--llama-dir must contain a precompiled llama-server executable')
    shutil.copytree(args.llama_dir, output/'llama', dirs_exist_ok=True)
release = ROOT/'build/runtime-manifest.json'
if release.exists():
    shutil.copy2(release, output/'runtime-manifest.json')
archive_root = ROOT/'dist'
if sys.platform == 'darwin':
    shutil.make_archive(str(ROOT/'dist/Veritas-macOS'), 'zip', archive_root, 'Veritas.app')
elif sys.platform == 'win32':
    shutil.make_archive(str(ROOT/'dist/Veritas-Windows'), 'zip', archive_root, 'Veritas')
else:
    shutil.make_archive(str(ROOT/'dist/Veritas-Linux'), 'gztar', archive_root, 'Veritas')
print(f'Packaged native application in {output}')
