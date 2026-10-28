"""Incremental, provenance-preserving hybrid retrieval with a disk-backed HNSW index."""
from __future__ import annotations

import csv
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading

EXTENSIONS = {'.txt', '.md', '.json', '.jsonl', '.csv'}


def claim_record(value):
    return isinstance(value, dict) and 'claim' in value and ('label' in value or 'evidence' in value)


def documents(path):
    suffix = path.suffix.lower()
    if path.name.lower() == 'label.txt':
        return
    if suffix in ('.txt', '.md'):
        text = path.read_text(encoding='utf-8-sig')
        for index, start in enumerate(range(0, len(text), 1600)):
            yield str(index), text[max(0, start-200):start+1600], {'chunk': index}
        return
    if suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for number, row in enumerate(csv.DictReader(stream), 2):
                if not claim_record(row):
                    yield str(number), json.dumps(row, ensure_ascii=False), {'row': number}
        return
    if suffix == '.jsonl':
        def records():
            with path.open(encoding='utf-8-sig') as stream:
                for number, line in enumerate(stream, 1):
                    if line.strip():
                        yield number, json.loads(line)
    else:
        def records():
            value = json.loads(path.read_text(encoding='utf-8-sig'))
            for number, row in enumerate(value if isinstance(value, list) else [value], 1):
                yield number, row
    for number, row in records():
        if claim_record(row):
            continue
        if isinstance(row, dict) and 'id' in row and 'lines' in row:
            # FEVER Wikipedia JSONL: line ID<TAB>sentence<TAB>optional hyperlink columns.
            for line in row['lines'].splitlines():
                parts = line.split('\t')
                if len(parts) >= 2 and parts[1].strip():
                    yield f'{number}:{parts[0]}', parts[1], {'page': row['id'], 'sentence': parts[0]}
        else:
            text = row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)
            for start in range(0, len(text), 1600):
                yield f'{number}:{start}', text[max(0, start-200):start+1600], {'record': number}


class Retrieval:
    def __init__(self, store):
        self.store = store
        self.root = store.root / 'index'
        self.path = self.root / 'evidence.sqlite'
        self.lock = threading.RLock()
        self.embedder = None
        with closing(self.connection()) as db, db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, hash TEXT, root TEXT);
                CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, path TEXT, key TEXT,
                    text TEXT, provenance TEXT, vector BLOB);
                CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(text, content='chunks', content_rowid='id');
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO search(rowid,text) VALUES(new.id,new.text); END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO search(search,rowid,text) VALUES('delete',old.id,old.text); END;
            ''')

    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def embedding(self):
        path = self.store.root / 'models' / 'embedding'
        if not (path / '.ready').exists():
            return None
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(str(path), local_files_only=True, device='cpu')
        return self.embedder

    def index(self, directory: Path, job=None):
        directory = directory.expanduser().resolve()
        if not directory.is_dir():
            raise ValueError('Resources directory does not exist')
        result = {'indexed': 0, 'unchanged': 0, 'deleted': 0, 'errors': [], 'excluded': 0}
        embedding = self.embedding()
        seen = set()
        with self.lock:
            db = self.connection()
            try:
                for path in sorted(directory.rglob('*')):
                    if job:
                        job.check()
                    if not path.is_file() or path.is_symlink():
                        continue
                    if path.name.endswith('.checksum.json') or path.name == 'wiki-pages.zip':
                        continue
                    if path.suffix.lower() not in EXTENSIONS:
                        result['errors'].append({'path': str(path), 'error': 'Unsupported file type'})
                        continue
                    name = str(path.resolve()); seen.add(name)
                    try:
                        with path.open('rb') as stream:
                            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                        previous = db.execute('SELECT hash FROM files WHERE path=?', (name,)).fetchone()
                        missing_vectors = db.execute('SELECT 1 FROM chunks WHERE path=? AND vector IS NULL LIMIT 1', (name,)).fetchone()
                        if previous and previous[0] == digest and not (embedding and missing_vectors):
                            result['unchanged'] += 1
                            continue
                        # A savepoint keeps the old file index intact if parsing fails mid-file.
                        db.execute('SAVEPOINT importing')
                        db.execute('DELETE FROM chunks WHERE path=?', (name,))
                        batch = []
                        count = 0

                        def flush():
                            if not batch:
                                return
                            vectors = embedding.encode([r[1] for r in batch], normalize_embeddings=True) if embedding else [None]*len(batch)
                            for (key, text, provenance), vector in zip(batch, vectors):
                                db.execute('INSERT INTO chunks(path,key,text,provenance,vector) VALUES (?,?,?,?,?)',
                                    (name, key, text, json.dumps(provenance), vector.astype('float32').tobytes() if vector is not None else None))
                            batch.clear()
                        for record in documents(path):
                            if job:
                                job.check()
                            batch.append(record); count += 1
                            if len(batch) >= 64:
                                flush()
                        flush()
                        db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?)', (name, digest, str(directory)))
                        db.execute('RELEASE importing'); db.commit()
                        result['indexed'] += 1
                        if not count:
                            result['excluded'] += 1
                        if job:
                            job.emit('indexing', path=name, chunks=count)
                    except Exception as exc:
                        from .jobs import Cancelled
                        if isinstance(exc, Cancelled):
                            db.rollback()
                            raise
                        db.rollback()
                        result['errors'].append({'path': name, 'error': str(exc)})
                for row in db.execute('SELECT path FROM files WHERE root=?', (str(directory),)).fetchall():
                    if row[0] not in seen:
                        db.execute('DELETE FROM chunks WHERE path=?', (row[0],))
                        db.execute('DELETE FROM files WHERE path=?', (row[0],)); result['deleted'] += 1
                db.commit()
                self.rebuild_vectors(db, job)
            finally:
                db.close()
        result['mode'] = 'hybrid' if embedding else 'keyword-only: download embeddings, then reindex'
        return result

    def rebuild_vectors(self, db, job=None):
        # Persist bounded-size shards rather than allocating the entire FEVER index in RAM.
        import shutil
        import uuid
        count = db.execute('SELECT count(*) FROM chunks WHERE vector IS NOT NULL').fetchone()[0]
        manifest = self.root / 'vectors.json'
        old = json.loads(manifest.read_text()) if manifest.exists() else {'shards': []}
        shards = []
        generation = self.root / ('vectors-' + uuid.uuid4().hex)
        generation.mkdir()
        try:
            if count:
                import hnswlib
                import numpy as np
                cursor = db.execute('SELECT id,vector FROM chunks WHERE vector IS NOT NULL ORDER BY id')
                remaining = count
                while remaining:
                    capacity = min(100_000, remaining)
                    index = hnswlib.Index(space='cosine', dim=384)
                    index.init_index(max_elements=capacity, ef_construction=150, M=16, random_seed=42)
                    loaded = 0
                    while loaded < capacity:
                        if job:
                            job.check()
                        rows = cursor.fetchmany(min(2048, capacity-loaded))
                        index.add_items(np.stack([np.frombuffer(r['vector'], dtype=np.float32) for r in rows]),
                                        [r['id'] for r in rows])
                        loaded += len(rows)
                    target = generation / f'{len(shards)}.hnsw'
                    index.save_index(str(target))
                    shards.append(str(target.relative_to(self.root)))
                    remaining -= loaded
                    del index
                    if job:
                        job.emit('vector_index', indexed=count-remaining, total=count)
            temporary = self.root / 'vectors-manifest.tmp'
            temporary.write_text(json.dumps({'shards': shards}))
            temporary.replace(manifest)
        except BaseException:
            shutil.rmtree(generation, ignore_errors=True)
            raise
        for name in old['shards']:
            old_directory = (self.root/name).parent
            if old_directory != self.root:
                shutil.rmtree(old_directory, ignore_errors=True)
        if not shards:
            generation.rmdir()

    def search(self, query, limit=8):
        words = re.findall(r'\w+', query, flags=re.UNICODE)[:32]
        if not words:
            return []
        with self.lock:
            db = self.connection()
            try:
                rows = db.execute('SELECT rowid FROM search WHERE search MATCH ? ORDER BY rank LIMIT 40',
                                  (' OR '.join('"'+w+'"' for w in words),)).fetchall()
                ranks = {r[0]: 1/(60+i) for i, r in enumerate(rows, 1)}
                mode = 'keyword'
                manifest = self.root / 'vectors.json'
                if manifest.exists() and self.embedding():
                    import hnswlib
                    vector = self.embedding().encode([query], normalize_embeddings=True)
                    candidates = []
                    for shard in json.loads(manifest.read_text())['shards']:
                        index = hnswlib.Index(space='cosine', dim=384)
                        index.load_index(str(self.root/shard)); index.set_ef(80)
                        count = index.get_current_count()
                        if count:
                            labels, distances = index.knn_query(vector, k=min(40, count))
                            candidates.extend(zip(distances[0].tolist(), labels[0].tolist()))
                        del index
                    for i, (_, ident) in enumerate(sorted(candidates)[:40], 1):
                        ranks[int(ident)] = ranks.get(int(ident), 0) + 1/(60+i)
                    mode = 'hybrid'
                result = []
                for ident in sorted(ranks, key=ranks.get, reverse=True)[:limit]:
                    row = db.execute('SELECT * FROM chunks WHERE id=?', (ident,)).fetchone()
                    if row:
                        provenance = json.loads(row['provenance'])
                        result.append({'citation': f'E{ident}', 'text': row['text'],
                                       'path': row['path'], 'provenance': provenance, 'retrieval': mode})
                return result
            finally:
                db.close()
