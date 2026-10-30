from __future__ import annotations

import contextlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import sys
import uuid
from .migration import migrate_storage


def data_directory() -> Path:
    override = os.environ.get('SENTINEL_X_DATA_DIR') or os.environ.get('VERITAS_DATA_DIR')
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == 'win32':
        parent = Path(os.environ.get('LOCALAPPDATA', Path.home()))
        current, legacy = parent/'Sentinel-X', parent/'Veritas'
    elif sys.platform == 'darwin':
        parent = Path.home()/'Library/Application Support'
        current, legacy = parent/'Sentinel-X', parent/'Veritas'
    else:
        parent = Path(os.environ.get('XDG_DATA_HOME', Path.home()/'.local/share'))
        current, legacy = parent/'sentinel-x', parent/'veritas'
    return migrate_storage(current, legacy)


class Store:
    def __init__(self, root: Path):
        root = Path(root).expanduser().resolve()
        self.root = migrate_storage(root)
        for name in ('resources', 'models', 'graphs', 'checkpoints', 'skills', 'runtime', 'index'):
            (root / name).mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    created TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, conversation TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE, role TEXT, content TEXT, data TEXT,
                    created TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS graphs(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            ''')

    @contextlib.contextmanager
    def connection(self):
        path = self.root/'sentinel-x.sqlite'
        if not path.exists() and (self.root/'veritas.sqlite').exists():
            path = self.root/'veritas.sqlite'
        db = sqlite3.connect(path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def setting(self, key, default=None):
        with self.connection() as db:
            row = db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def live_path(self, value):
        """Display/access moved files while preserving historical database identities."""
        path = Path(value)
        for legacy in self.setting('legacy_data_roots', []):
            if path.is_relative_to(legacy):
                return self.root/path.relative_to(legacy)
        return path

    def path_aliases(self, value):
        path = Path(value)
        if not path.is_relative_to(self.root): return [str(path)]
        return [str(path), *(str(Path(old)/path.relative_to(self.root))
                             for old in self.setting('legacy_data_roots', []))]

    def configure(self, key, value):
        with self.connection() as db:
            db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', (key, json.dumps(value)))

    def conversations(self, query=''):
        with self.connection() as db:
            return [dict(r) for r in db.execute('''SELECT DISTINCT c.* FROM conversations c
                LEFT JOIN messages m ON m.conversation=c.id
                WHERE c.title LIKE ? OR m.content LIKE ? ORDER BY c.created DESC''',
                (f'%{query}%', f'%{query}%'))]

    def create_conversation(self, title='New conversation'):
        ident = uuid.uuid4().hex
        with self.connection() as db:
            db.execute('INSERT INTO conversations(id,title) VALUES (?,?)', (ident, title[:200]))
        return {'id': ident, 'title': title[:200]}

    def messages(self, ident):
        with self.connection() as db:
            rows = db.execute('SELECT * FROM messages WHERE conversation=? ORDER BY id', (ident,))
            return [{**dict(r), 'data': json.loads(r['data'] or '{}')} for r in rows]

    def message(self, ident, role, content, data=None):
        with self.connection() as db:
            cursor = db.execute('INSERT INTO messages(conversation,role,content,data) VALUES (?,?,?,?)',
                                (ident, role, content, json.dumps(data or {})))
            return cursor.lastrowid

    def message_data(self, ident, data):
        with self.connection() as db:
            db.execute('UPDATE messages SET data=? WHERE id=?', (json.dumps(data), ident))

    def edit_message(self, conversation, ident, content, revision):
        """Change visible text atomically, retaining original execution provenance."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Message cannot be empty')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM messages WHERE id=? AND conversation=?',
                             (ident, conversation)).fetchone()
            if not row or row['role'] not in ('user', 'assistant'):
                raise ValueError('Editable message not found in this conversation')
            data = json.loads(row['data'] or '{}')
            if data.get('revision', 0) != revision:
                raise ValueError('This message changed. Reopen the editor to use its latest version.')
            if content == row['content']:
                return
            now = datetime.now(timezone.utc).isoformat()
            data.setdefault('revisions', []).append({'content': row['content'],
                'revision': revision, 'replaced_at': now})
            data.update(revision=revision+1, edited_at=now)
            db.execute('UPDATE messages SET content=?, data=? WHERE id=?',
                       (content, json.dumps(data), ident))
            for later in db.execute('SELECT id,data FROM messages WHERE conversation=? AND id>?',
                                    (conversation, ident)).fetchall():
                metadata = json.loads(later['data'] or '{}')
                causes = metadata.setdefault('outdated_by', [])
                if ident not in causes:
                    causes.append(ident)
                db.execute('UPDATE messages SET data=? WHERE id=?',
                           (json.dumps(metadata), later['id']))

    def branch_at(self, conversation, ident):
        """Copy history before a user request; caller resubmits the revised request."""
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            original = db.execute('SELECT title FROM conversations WHERE id=?',
                                  (conversation,)).fetchone()
            target = db.execute('SELECT * FROM messages WHERE conversation=? AND id=?',
                                (conversation, ident)).fetchone()
            if not original or not target or target['role'] != 'user':
                raise ValueError('User message not found')
            branch = uuid.uuid4().hex
            db.execute('INSERT INTO conversations(id,title) VALUES (?,?)',
                       (branch, original['title']+' (regenerated)'))
            mapping = {}
            for row in db.execute('SELECT * FROM messages WHERE conversation=? AND id<? ORDER BY id',
                                  (conversation, ident)).fetchall():
                data = json.loads(row['data'] or '{}')
                data['copied_from'] = {'conversation': conversation, 'message': row['id']}
                data['outdated_by'] = [mapping[x] for x in data.get('outdated_by', []) if x in mapping]
                # Audit snapshots describe the original execution and remain immutable.
                cursor = db.execute('INSERT INTO messages(conversation,role,content,data,created) '
                    'VALUES (?,?,?,?,?)', (branch, row['role'], row['content'], json.dumps(data), row['created']))
                mapping[row['id']] = cursor.lastrowid
        return branch

    def save_graph(self, graph):
        with self.connection() as db:
            db.execute('INSERT OR REPLACE INTO graphs VALUES (?,?)',
                       (graph['id'], json.dumps(graph)))

    def graph(self, ident):
        with self.connection() as db:
            row = db.execute('SELECT data FROM graphs WHERE id=?', (ident,)).fetchone()
        if not row:
            raise ValueError('Graph not found')
        return json.loads(row[0])

    def graphs(self):
        with self.connection() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT data FROM graphs')]
