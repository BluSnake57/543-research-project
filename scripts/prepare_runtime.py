"""Developer/build step: stage the precompiled runtime BEFORE distributing the app."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/backend'))
from sentinel_x.bundling import ensure_runtime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, help='Existing verified precompiled runtime directory')
    args = parser.parse_args()
    directory = ensure_runtime(ROOT/'build/llama/bin', args.source)
    print(f'Precompiled llama.cpp ready for packaging: {directory}')
