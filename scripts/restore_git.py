"""Restore this checkout's Git metadata from its local recovery bundle, preserving files."""
import argparse
from pathlib import Path
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = 'https://github.com/BluSnake57/543-research-project.git'


def git(root, *arguments, check=True):
    return subprocess.run(['git', '-C', str(root), *arguments], check=check,
                          capture_output=True, text=True)


def restore(root, bundle):
    root = Path(root).resolve(); bundle = Path(bundle).resolve()
    existing = git(root, 'rev-parse', '--show-toplevel', check=False)
    if existing.returncode == 0 and Path(existing.stdout.strip()).resolve() == root:
        return 'This directory already has working Git metadata; nothing changed.'
    if not bundle.is_file():
        raise ValueError(f'Recovery bundle is missing: {bundle}')
    metadata = root/'.git'
    if metadata.exists() and (not metadata.is_dir() or any(metadata.iterdir())):
        raise ValueError('The .git path contains unrecognized metadata. It was left untouched.')
    if metadata.exists():
        try:
            metadata.chmod(metadata.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
        except OSError as exc:
            raise ValueError('The .git directory is mounted read-only. Run this script in your normal terminal outside the restricted session.') from exc
    git(root, 'init', '--initial-branch=main')
    git(root, 'fetch', str(bundle), 'refs/heads/main')
    # --mixed restores the branch and index, while retaining every working file/edit.
    git(root, 'reset', '--mixed', 'FETCH_HEAD')
    git(root, 'remote', 'add', 'origin', REMOTE)
    git(root, 'config', 'branch.main.remote', 'origin')
    git(root, 'config', 'branch.main.merge', 'refs/heads/main')
    return f'Git restored in {root}. Remote: {REMOTE}. Working files were preserved.'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT/'sentinel-x-development.bundle')
    arguments = parser.parse_args(argv)
    try:
        print(restore(ROOT, arguments.bundle))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc)+'\n')


if __name__ == '__main__':
    main()
