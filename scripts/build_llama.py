"""Build the exact llama.cpp commit recorded in the release manifest."""
import argparse
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['cpu', 'cuda', 'metal', 'vulkan'], default='cpu')
args = parser.parse_args()
manifest = json.loads((root/'runtime-manifest.json').read_text())
source = root/'build/llama-source'
source.parent.mkdir(exist_ok=True)
if not (source/'.git').exists():
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/ggml-org/llama.cpp', str(source)], check=True)
commit = manifest['llama_cpp']['commit']
subprocess.run(['git', '-C', str(source), 'fetch', 'origin', commit], check=True)
subprocess.run(['git', '-C', str(source), 'checkout', '--detach', commit], check=True)
options = ['-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DGGML_VULKAN=OFF',
           '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DGGML_NATIVE=OFF', '-DBUILD_SHARED_LIBS=ON']
if args.backend != 'cpu':
    options.append(f'-DGGML_{args.backend.upper()}=ON')
subprocess.run(['cmake', '-S', str(source), '-B', str(root/'build/llama'),
                '-DCMAKE_BUILD_TYPE=Release', *options], check=True)
subprocess.run(['cmake', '--build', str(root/'build/llama'), '--config', 'Release',
                '--target', 'llama-server', '-j', '2'], check=True)
# Multi-configuration Windows generators place output below Release.
release = root/'build/llama/bin/Release'
if release.exists():
    import shutil
    for path in release.iterdir():
        if path.is_file():
            shutil.copy2(path, root/'build/llama/bin'/path.name)
