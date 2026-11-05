"""Persisted jobs with separate interactive and maintenance workers."""
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
        self.thinking_text = ''
        self.thinking_trace = ''
        self.thinking_trace_truncated = False
        self.thinking_started = None
        self.answer_text = ''
        self.data = {'id': uuid.uuid4().hex, 'kind': kind, 'status': 'queued',
                     'events': [], 'created': time.time()}

    def emit(self, event, **payload):
        with self.lock:
            events = self.data['events']
            previous = events[-1] if events else {}
            # Repeated samples describe the current phase, rather than an unbounded history
            # that gets serialized and rewritten for every update during a multi-hour import.
            replace = (event in ('download', 'extraction', 'indexing', 'vector_index', 'prompt_progress', 'scope_progress') and
                previous.get('event') == event and not previous.get('file_complete') and
                not payload.get('file_complete') and all(previous.get(key) == payload.get(key)
                                                         for key in ('path', 'file')))
            item = {'seq': previous['seq'] if replace else len(events),
                    'event': event, 'time': time.time(), **payload}
            if replace:
                events[-1] = item
            else:
                events.append(item)
            self.manager.persist(self.data)
        return item

    def check(self):
        if self.cancelled.is_set():
            raise Cancelled('Job cancelled')

    def stream_piece(self, text, reasoning=False):
        """Transient preview: avoid rewriting SQLite for every generated token."""
        if reasoning:
            with self.lock:
                if self.thinking_started is None and not text.strip():
                    return
                if self.thinking_started is None:
                    self.thinking_started = time.monotonic()
                    if self.thinking_trace:
                        self.thinking_trace += '\n\n'
                self.data['thinking_characters'] = self.data.get('thinking_characters', 0)+len(text.strip())
                self.thinking_text = (self.thinking_text+text)[-32000:]
                self.thinking_trace += text
                if len(self.thinking_trace) > 64000:
                    self.thinking_trace = self.thinking_trace[-64000:]
                    self.thinking_trace_truncated = True
        else:
            self.finish_thinking()
            with self.lock:
                self.answer_text = (self.answer_text+text)[-32000:]

    def begin_generation(self):
        with self.lock:
            self.answer_text = ''

    def clear_answer(self):
        with self.lock:
            self.answer_text = ''

    def finish_thinking(self):
        with self.lock:
            if self.thinking_started is None:
                return
            seconds = time.monotonic()-self.thinking_started
            self.thinking_started = None
            self.thinking_text = ''
            self.data['thinking_seconds'] = self.data.get('thinking_seconds', 0)+seconds
        self.emit('thinking_complete', seconds=self.data['thinking_seconds'])

    def preview(self):
        with self.lock:
            seconds = self.data.get('thinking_seconds', 0)
            if self.thinking_started is not None:
                seconds += time.monotonic()-self.thinking_started
            return {'text': self.thinking_text, 'active': self.thinking_started is not None,
                    'trace': self.thinking_trace, 'trace_truncated': self.thinking_trace_truncated,
                    'seconds': seconds, 'answer': self.answer_text,
                    **({'scope': self.data['scope']} if self.data.get('scope') else {})}


class Jobs:
    def __init__(self, store, table='jobs'):
        if table not in ('jobs', 'trainer_jobs'):
            raise ValueError('Unknown job namespace')
        self.store = store
        self.table = table
        self.live = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sentinel_x-work')
        self.maintenance = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sentinel_x-index') if table == 'jobs' else None
        with store.connection() as db:
            db.execute(f'CREATE TABLE IF NOT EXISTS {table}(id TEXT PRIMARY KEY, data TEXT NOT NULL)')
            for row in db.execute(f'SELECT id,data FROM {table}').fetchall():
                data = json.loads(row['data'])
                if data['status'] in ('queued', 'running'):
                    data.update(status='interrupted', error='Application exited before completion')
                    db.execute(f'UPDATE {table} SET data=? WHERE id=?', (json.dumps(data), row['id']))

    def persist(self, data):
        with self.store.connection() as db:
            db.execute(f'INSERT OR REPLACE INTO {self.table} VALUES (?,?)', (data['id'], json.dumps(data)))

    def submit(self, kind, function, request=None):
        job = Job(self, kind)
        if request is not None:
            job.data['request'] = request
        job.emit('stage', stage='Queued')
        self.live[job.data['id']] = job
        self.persist(job.data)

        def run():
            try:
                job.check()
                job.data['status'] = 'running'
                job.emit('status', status='running')
                result = function(job)
                job.check()
                job.finish_thinking()
                job.clear_answer()
                job.data.update(status='completed', result=result, finished=time.time())
                job.emit('done', result=result)
            except Cancelled:
                job.finish_thinking()
                job.clear_answer()
                job.data.update(status='cancelled', finished=time.time())
                job.emit('cancelled')
            except Exception as exc:
                job.finish_thinking()
                job.clear_answer()
                job.data['finished'] = time.time()
                if job.cancelled.is_set():
                    job.data['status'] = 'cancelled'
                    job.emit('cancelled')
                else:
                    job.data.update(status='failed', error=str(exc))
                    job.emit('error', message=str(exc))
        interactive = kind.startswith(('chat:', 'runtime-activate:')) or kind in (
            'map-posts', 'estimate-graph', 'qwen-smoke', 'thinking-check', 'search')
        executor = self.pool if interactive or self.maintenance is None else self.maintenance
        executor.submit(run)
        return job.data['id']

    def shutdown(self, wait=True, cancel_futures=False):
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        if self.maintenance is not None:
            self.maintenance.shutdown(wait=wait, cancel_futures=cancel_futures)

    def get(self, ident):
        with self.store.connection() as db:
            row = db.execute(f'SELECT data FROM {self.table} WHERE id=?', (ident,)).fetchone()
        if not row:
            raise ValueError('Job not found')
        return json.loads(row[0])

    def list(self):
        with self.store.connection() as db:
            return [json.loads(r[0]) for r in db.execute(f'SELECT data FROM {self.table} ORDER BY rowid DESC LIMIT 30')]

    def cancel(self, ident):
        self.get(ident)
        if ident in self.live:
            job = self.live[ident]
            if job.data['status'] in ('running', 'queued'):
                job.cancelled.set()
                job.data['cancelling'] = True
                if job.data['status'] == 'queued':
                    job.data.update(status='cancelled', finished=time.time())
                job.emit('cancellation_requested')
