"""Persisted jobs with a single heavyweight worker and replayable event cursors."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import uuid


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, manager, kind):
        self.manager = manager
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.data = {'id': uuid.uuid4().hex, 'kind': kind, 'status': 'queued',
                     'events': [], 'created': time.time()}

    def emit(self, event, **payload):
        with self.lock:
            item = {'seq': len(self.data['events']), 'event': event, **payload}
            self.data['events'].append(item)
            self.manager.persist(self.data)
        return item

    def check(self):
        if self.cancelled.is_set():
            raise Cancelled('Job cancelled')


class Jobs:
    def __init__(self, store):
        self.store = store
        self.live = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='veritas-work')
        with store.connection() as db:
            for row in db.execute('SELECT id,data FROM jobs').fetchall():
                data = json.loads(row['data'])
                if data['status'] in ('queued', 'running'):
                    data.update(status='interrupted', error='Application exited before completion')
                    db.execute('UPDATE jobs SET data=? WHERE id=?', (json.dumps(data), row['id']))

    def persist(self, data):
        with self.store.connection() as db:
            db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?)', (data['id'], json.dumps(data)))

    def submit(self, kind, function):
        job = Job(self, kind)
        self.live[job.data['id']] = job
        self.persist(job.data)

        def run():
            try:
                job.check()
                job.data['status'] = 'running'
                job.emit('status', status='running')
                result = function(job)
                job.check()
                job.data.update(status='completed', result=result)
                job.emit('done', result=result)
            except Cancelled:
                job.data['status'] = 'cancelled'
                job.emit('cancelled')
            except Exception as exc:
                if job.cancelled.is_set():
                    job.data['status'] = 'cancelled'
                    job.emit('cancelled')
                else:
                    job.data.update(status='failed', error=str(exc))
                    job.emit('error', message=str(exc))
        self.pool.submit(run)
        return job.data['id']

    def get(self, ident):
        with self.store.connection() as db:
            row = db.execute('SELECT data FROM jobs WHERE id=?', (ident,)).fetchone()
        if not row:
            raise ValueError('Job not found')
        return json.loads(row[0])

    def list(self):
        with self.store.connection() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT data FROM jobs ORDER BY rowid DESC LIMIT 30')]

    def cancel(self, ident):
        self.get(ident)
        if ident in self.live:
            self.live[ident].cancelled.set()
