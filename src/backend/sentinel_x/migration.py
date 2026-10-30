"""Move legacy application storage without copying downloaded weights or rebuilding evidence."""
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3
import uuid


@contextmanager
def migration_lock(parent):
    parent.mkdir(parents=True, exist_ok=True)
    with (parent/'.sentinel-x-migration.lock').open('a+b') as stream:
        if not stream.tell(): stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt': msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(stream, fcntl.LOCK_UN)


def rebase(value, old, new):
    if isinstance(value, str):
        if value == str(old): return str(new)
        if value.startswith(str(old)+os.sep): return str(new)+value[len(str(old)):]
    if isinstance(value, list): return [rebase(item, old, new) for item in value]
    if isinstance(value, dict): return {key: rebase(item, old, new) for key, item in value.items()}
    return value


def migrate_storage(current, legacy=None):
    """Called at startup after closing older app instances; never merge distinct databases."""
    current = Path(current)
    legacy = Path(legacy) if legacy else None
    origin_file = current/'.migration-origin.json'
    old_database = current/'veritas.sqlite'
    database = current/'sentinel-x.sqlite'
    needs_move = legacy is not None and legacy != current and any(
        (legacy/name).is_file() for name in ('veritas.sqlite', 'sentinel-x.sqlite'))
    if not needs_move and not origin_file.exists() and not (old_database.exists() and not database.exists()):
        return current
    with migration_lock(current.parent):
        if needs_move and not database.exists():
            if current.exists():
                if any(current.iterdir()):
                    # A prior interrupted move is completed below; other data is never overwritten.
                    if not origin_file.exists():
                        raise ValueError(f'Cannot migrate to {current}: it already contains files. '
                                         'Keep both directories and select one using SENTINEL_X_DATA_DIR.')
                else:
                    current.rmdir()
            if not current.exists():
                (legacy/'.migration-origin.json').write_text(json.dumps({'root': str(legacy.resolve())}))
                legacy.rename(current)
        origin = json.loads(origin_file.read_text())['root'] if origin_file.exists() else None
        if database.exists() and not origin:
            return current
        source = database if database.exists() else old_database
        if not source.exists():
            raise ValueError(f'Legacy migration database is missing from {current}')
        temporary = current/('.migration-'+uuid.uuid4().hex+'.sqlite')
        try:
            # SQLite backup includes committed WAL transactions; the original database is retained.
            with closing(sqlite3.connect(source)) as old, closing(sqlite3.connect(temporary)) as new:
                old.backup(new)
                new.execute('PRAGMA journal_mode=DELETE')
                if origin:
                    for key, raw in new.execute('SELECT key,value FROM settings').fetchall():
                        if key in ('qwen_path', 'llama_path', 'resources', 'installed_runtimes'):
                            new.execute('UPDATE settings SET value=? WHERE key=?',
                                        (json.dumps(rebase(json.loads(raw), origin, current)), key))
                    existing = new.execute('SELECT value FROM settings WHERE key=?', ('legacy_data_roots',)).fetchone()
                    roots = json.loads(existing[0]) if existing else []
                    if origin not in roots: roots.append(origin)
                    new.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', ('legacy_data_roots', json.dumps(roots)))
                new.commit()
                if new.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise ValueError('Migrated database failed integrity verification; the original is retained')
            if source == database:
                backup = current/'migration-backup'; backup.mkdir(exist_ok=True)
                if not (backup/'before-rename.sqlite').exists():
                    with closing(sqlite3.connect(source)) as old, closing(sqlite3.connect(backup/'before-rename.sqlite')) as copy:
                        old.backup(copy)
            temporary.replace(database)
            origin_file.unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)
    return current
