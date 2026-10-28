from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import uuid


def data_directory() -> Path:
    if os.environ.get('VERITAS_DATA_DIR'):
        return Path(os.environ['VERITAS_DATA_DIR']).expanduser().resolve()
    if sys.platform == 'win32':
        return Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'Veritas'
    if sys.platform == 'darwin':
        return Path.home() / 'Library/Application Support/Veritas'
    return Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'veritas'


class Store:
    def __init__(self, root: Path):
        self.root = root
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
        db = sqlite3.connect(self.root / 'veritas.sqlite', timeout=30)
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
            db.execute('INSERT INTO messages(conversation,role,content,data) VALUES (?,?,?,?)',
                       (ident, role, content, json.dumps(data or {})))

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
