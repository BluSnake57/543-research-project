"""Build on the target OS; optionally use an already built pinned llama.cpp runtime."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--llama-dir', type=Path, default=ROOT/'build/llama/bin')
args = parser.parse_args()
server = 'llama-server.exe' if sys.platform == 'win32' else 'llama-server'
if not (args.llama_dir/server).exists():
    parser.error('Build llama.cpp first with scripts/build_llama.py or pass --llama-dir')
subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
    '--name', 'veritas-worker', '--paths', str(ROOT/'src/backend'),
    '--collect-all', 'torch_geometric', '--collect-all', 'sentence_transformers',
    '--collect-all', 'transformers', '--collect-all', 'tokenizers',
    '--collect-all', 'hnswlib', '--collect-submodules', 'uvicorn',
    '--distpath', str(ROOT/'build/frozen'), '--workpath', str(ROOT/'build/pyinstaller'),
    '--specpath', str(ROOT/'build'), str(ROOT/'scripts/worker_entry.py')], check=True, cwd=ROOT)
output = ROOT/'build/worker'
if output.exists():
    shutil.rmtree(output)
shutil.copytree(ROOT/'build/frozen/veritas-worker', output)
shutil.copytree(args.llama_dir, output/'llama', ignore=shutil.ignore_patterns('*.a', '*.lib', '*.exp', '*.pdb'))
print(f'Worker and llama.cpp runtime assembled in {output}')
