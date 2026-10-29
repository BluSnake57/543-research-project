"""Package assistant and trainer, including llama.cpp with the assistant at build time."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/backend'))
from sentinel_x.bundling import bundle_runtime, ensure_runtime


def package(app='both', llama_dir=None):
    targets = [('Sentinel-X', 'desktop_entry.py'), ('Sentinel-X-Trainer', 'trainer_entry.py')]
    if app != 'both': targets = [targets[0] if app == 'assistant' else targets[1]]
    runtime = ensure_runtime(ROOT/'build/llama/bin', llama_dir) if app != 'trainer' else None
    built = []
    for name, entry in targets:
        command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--windowed',
                   '--name', name, '--paths', str(ROOT/'src/backend'),
                   '--collect-data', 'sentinel_x', '--collect-all', 'sentence_transformers',
                   '--collect-all', 'transformers', '--collect-all', 'tokenizers',
                   '--distpath', str(ROOT/'dist'), '--workpath', str(ROOT/'build/pyinstaller'/name),
                   '--specpath', str(ROOT/'build'), str(ROOT/'scripts'/entry)]
        subprocess.run(command, check=True, cwd=ROOT)
        directory = ROOT/'dist'/name
        if sys.platform == 'darwin': directory = ROOT/'dist'/f'{name}.app/Contents/MacOS'
        if not directory.is_dir(): raise ValueError(f'Packager did not produce {directory}')
        if name == 'Sentinel-X': bundle_runtime(runtime, directory)
        release = ROOT/'build/runtime-manifest.json'
        if release.exists(): shutil.copy2(release, directory/'runtime-manifest.json')
        built.append(name+'.app' if sys.platform == 'darwin' else name)
    # Both application folders are siblings, so the assistant can launch the companion trainer.
    with tempfile.TemporaryDirectory(prefix='release-', dir=ROOT/'build') as temporary:
        for name in built: shutil.copytree(ROOT/'dist'/name, Path(temporary)/name, symlinks=True)
        platform = 'macOS' if sys.platform == 'darwin' else 'Windows' if sys.platform == 'win32' else 'Linux'
        name = 'Sentinel-X-Trainer' if app == 'trainer' else 'Sentinel-X'
        archive = shutil.make_archive(str(ROOT/'dist'/f'{name}-{platform}'),
                                      'gztar' if platform == 'Linux' else 'zip', temporary)
    print(f'Packaged {", ".join(built)} in {archive}')
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', choices=['both', 'assistant', 'trainer'], default='both')
    parser.add_argument('--llama-dir', type=Path, help='Existing precompiled pinned runtime; otherwise prepared during build')
    args = parser.parse_args()
    package(args.app, args.llama_dir)
