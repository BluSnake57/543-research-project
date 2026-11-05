"""Package assistant and trainer, including the isolated Transformers worker."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/backend'))



def package(app='both'):
    targets = [('Sentinel-X', 'desktop_entry.py'), ('Sentinel-X-Trainer', 'trainer_entry.py')]
    if app != 'both': targets = [targets[0] if app == 'assistant' else targets[1]]
    built = []
    for name, entry in targets:
        command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--windowed',
                   '--name', name, '--paths', str(ROOT/'src/backend'),
                   '--collect-data', 'sentinel_x', '--collect-all', 'sentence_transformers',
                   '--collect-all', 'transformers', '--collect-all', 'tokenizers',
                   '--collect-all', 'bitsandbytes', '--collect-all', 'accelerate', '--collect-all', 'safetensors',
                   '--hidden-import', 'sentinel_x.scope_worker',
                   '--distpath', str(ROOT/'dist'), '--workpath', str(ROOT/'build/pyinstaller'/name),
                   '--specpath', str(ROOT/'build'), str(ROOT/'scripts'/entry)]
        subprocess.run(command, check=True, cwd=ROOT)
        directory = ROOT/'dist'/name
        if sys.platform == 'darwin': directory = ROOT/'dist'/f'{name}.app/Contents/MacOS'
        if not directory.is_dir(): raise ValueError(f'Packager did not produce {directory}')
        release = ROOT/'src/backend/sentinel_x/assets/runtime-manifest.json'
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
    args = parser.parse_args()
    package(args.app)
