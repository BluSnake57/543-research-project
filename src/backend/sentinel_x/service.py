"""Direct Python application services. Qt and CLI require no web server or IPC bridge."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import shutil

from .assistant import Assistant
from .config import ASSETS, bundled_runtime, manifest
from .downloads import Models
from .graphs import (estimate_graph, native_twitter, normalize_posts, read_posts,
                     suggest_mapping, validate_graph)
from .jobs import Jobs
from .retrieval import Retrieval
from .runtime import Llama
from .runtime_packages import LABELS, runtime_choices, probe_runtime
from .skills import Skills
from .state import Store, data_directory


class Service:
    def __init__(self, root=None, role='assistant'):
        if role not in ('assistant', 'trainer'):
            raise ValueError('Unknown application role')
        self.store = Store(Path(root) if root else data_directory())
        self.jobs = Jobs(self.store, table='trainer_jobs' if role == 'trainer' else 'jobs')
        self.retrieval = Retrieval(self.store)
        self.runtime = Llama(self.store, bundled_runtime())
        self.models = Models(self.store, manifest())
        self.skills = Skills([ASSETS/'skills', self.store.root/'skills'], self.store)
        self.assistant = Assistant(self.store, self.retrieval, self.runtime, self.skills)

    def close(self):
        for job in self.jobs.live.values():
            job.cancelled.set()
        self.runtime.stop(wait=False)
        self.jobs.shutdown(wait=False, cancel_futures=True)

    def cancel(self, ident):
        data = self.jobs.get(ident)
        self.jobs.cancel(ident)
        if data['status'] == 'running' and (data['kind'].startswith('chat:') or
                data['kind'] in ('map-posts', 'estimate-graph', 'qwen-smoke', 'thinking-check')):
            self.runtime.stop(wait=False)

    def status(self):
        executable, model = self.runtime.configured()
        return {'root': str(self.store.root), 'models': self.models.status(),
                'resources': self.store.setting('resources', str(self.store.root/'resources')),
                'llama_path': executable, 'qwen_path': model,
                'checkpoint': self.store.setting('checkpoint'),
                'context': self.store.setting('context', 8192),
                'gpu_layers': str(self.store.setting('gpu_layers', 'auto')),
                'thinking_budget': self.store.setting('thinking_budget', 256),
                'installed_runtimes': self.store.setting('installed_runtimes', {}),
                'fever': self.models.fever_status(),
                'running_runtime': self.runtime.active,
                'first_run': not self.store.setting('setup_started', False)}

    def configure(self, **values):
        if set(values) - {'resources', 'context', 'gpu_layers', 'llama_path', 'checkpoint', 'thinking_budget'}:
            raise ValueError('Unknown setting')
        for key, value in values.items():
            if key == 'thinking_budget' and (not isinstance(value, int) or not 64 <= value <= 2048):
                raise ValueError('Thinking budget must be between 64 and 2048 tokens')
            if key == 'context' and (not isinstance(value, int) or not 2048 <= value <= 32768):
                raise ValueError('Context must be between 2048 and 32768')
            if key == 'gpu_layers' and str(value) != 'auto' and (not str(value).isdigit() or int(value) > 999):
                raise ValueError('GPU layers must be auto or 0–999')
            if key == 'checkpoint' and value and (not isinstance(value, str) or not value.isalnum() or
                    not (self.store.root/'checkpoints'/f'{value}.json').is_file()):
                raise ValueError('Checkpoint not found')
            if key in ('llama_path', 'resources') and not isinstance(value, str):
                raise ValueError('Path must be a string')
        for key, value in values.items():
            self.store.configure(key, value)
        if set(values) & {'context', 'gpu_layers', 'llama_path', 'thinking_budget'}:
            self.runtime.stop()

    def setup(self, keys=('qwen', 'embedding'), fever=False, index_fever=False):
        keys = tuple(keys)
        if set(keys) - {'qwen', 'embedding'}:
            raise ValueError('Only model weights are downloaded during setup')
        self.store.configure('setup_started', True)
        def work(job):
            result = {'models': [self.models.install(key, job) for key in keys]}
            if fever:
                result['fever'] = self.models.fever(job, extract=index_fever)
                if index_fever:
                    result['fever']['index'] = self.index_fever(result['fever']['evidence_directory'], job)
            return result
        return self.jobs.submit('setup', work,
            request={'keys': list(keys), 'fever': fever, 'index_fever': index_fever})

    def download(self, key):
        if key not in ('qwen', 'embedding'):
            raise ValueError('Unknown download')
        return self.jobs.submit('download:'+key, lambda job: self.models.install(key, job))

    def download_runtime(self, backend, use_after_download=True):
        if backend not in runtime_choices():
            raise ValueError('This runtime is not supported on the current platform')
        def work(job):
            result = self.models.install_runtime(job, backend=backend, activate=False)
            job.check()
            if use_after_download:
                result = {**result, 'activation_job': self.activate_runtime(backend)}
            return result
        return self.jobs.submit('runtime-download:'+backend, work,
            request={'backend': backend, 'use_after_download': use_after_download})

    def activate_runtime(self, backend):
        if backend != 'bundled' and backend not in LABELS:
            raise ValueError('Unknown runtime')
        def work(job):
            job.emit('stage', stage='Checking runtime before switching engines')
            if backend == 'bundled':
                folder = self.runtime.bundled
                candidates = list(folder.rglob('llama-server'))+list(folder.rglob('llama-server.exe')) if folder else []
                if not candidates:
                    raise ValueError('Bundled runtime is missing')
                path = str(candidates[0]); directories = [str(candidates[0].parent)]
            else:
                entry = self.store.setting('installed_runtimes', {}).get(backend)
                if not entry or not Path(entry['executable']).is_file():
                    raise ValueError('Download this runtime first')
                path = entry['executable']; directories = entry['library_dirs']
            checked = probe_runtime(path, manifest()['llama_cpp']['commit'], directories,
                                    'cpu' if backend == 'bundled' else backend)
            if backend not in ('cpu', 'bundled') and not checked['devices']:
                raise ValueError('Runtime installed, but no compatible GPU was detected. '
                                 'Check graphics drivers or use the bundled CPU runtime.')
            job.check()
            self.runtime.stop()
            self.store.configure('llama_path', path)
            self.store.configure('gpu_layers', 'auto' if backend != 'cpu' or platform.system() == 'Darwin' else '0')
            return {'backend': backend, 'executable': path, 'devices': checked['devices'], 'activated': True}
        return self.jobs.submit('runtime-activate:'+backend, work)

    def retry_download(self, ident):
        job = self.jobs.get(ident)
        if job['status'] not in ('failed', 'cancelled', 'interrupted'):
            raise ValueError('Only stopped or failed downloads can be retried.')
        if job['kind'] in ('download:qwen', 'download:embedding'):
            return self.download(job['kind'].split(':', 1)[1])
        if job['kind'].startswith('runtime-download:'):
            return self.download_runtime(job['request']['backend'], job['request']['use_after_download'])
        if job['kind'] == 'fever':
            return self.fever(job.get('request', {}).get('index', False))
        if job['kind'] == 'fever-index':
            return self.build_fever_index()
        if job['kind'] == 'setup' and job.get('request'):
            return self.setup(job['request']['keys'], job['request']['fever'], job['request'].get('index_fever', False))
        raise ValueError('Reopen the original action to retry this task.')

    def import_model(self, path):
        self.runtime.stop()
        return self.models.import_qwen(path)

    def smoke(self):
        def work(job):
            self.runtime.start(job)
            if self.runtime.structured('Return {"ready":true}.', job).get('ready') is not True:
                raise ValueError('Generation smoke check failed')
            tools = [{'type': 'function', 'function': {'name': 'smoke_echo',
                      'description': 'Echo a message', 'parameters': {'type': 'object',
                      'properties': {'message': {'type': 'string'}}, 'required': ['message']}}}]
            result = self.runtime.stream([{'role': 'user', 'content':
                                          'Call smoke_echo with message hello. Do not answer directly.'}], tools, job)
            calls = result.get('tool_calls', [])
            if not calls or calls[0]['function']['name'] != 'smoke_echo':
                raise ValueError('Qwen function calling is incompatible with this runtime/model')
            json.loads(calls[0]['function']['arguments'])
            return {'generation': True, 'function_calling': True}
        return self.jobs.submit('qwen-smoke', work)

    def chat(self, conversation, text, graph_ids, thinking=False):
        if not isinstance(thinking, bool):
            raise ValueError('Thinking must be On or Off')
        if not text.strip() or len(text) > 6000:
            raise ValueError('Enter a message of 1–6000 characters')
        if len(graph_ids) > 100:
            raise ValueError('Attach at most 100 graphs')
        if not any(c['id'] == conversation for c in self.store.conversations()):
            raise ValueError('Conversation not found')
        if any(j.data['kind'] == f'chat:{conversation}' and j.data['status'] in ('queued', 'running')
               for j in self.jobs.live.values()):
            raise ValueError('Wait for the current response or cancel it first')
        return self.jobs.submit(f'chat:{conversation}', lambda job:
            self.assistant.chat(conversation, text, graph_ids, job, thinking),
            request={'conversation': conversation, 'text': text,
                     'graph_ids': list(graph_ids), 'thinking_enabled': thinking})

    def ensure_idle(self, conversation):
        if any(j.data['kind'] == f'chat:{conversation}' and
               j.data['status'] in ('queued', 'running') for j in self.jobs.live.values()):
            raise ValueError('Stop the active response or wait for it to finish before editing.')

    def edit_message(self, conversation, ident, text, revision=0):
        self.ensure_idle(conversation)
        if any(m['id'] == ident and m['role'] == 'user' for m in self.store.messages(conversation)):
            if len(text) > 6000:
                raise ValueError('User messages support at most 6000 characters.')
        self.store.edit_message(conversation, ident, text, revision)

    def regenerate(self, conversation, ident):
        self.ensure_idle(conversation)
        message = next((m for m in self.store.messages(conversation) if m['id'] == ident), None)
        if not message or message['role'] != 'user':
            raise ValueError('User message not found')
        text = message['content']
        if not 1 <= len(text.strip()) <= 6000:
            raise ValueError('Regeneration requires a request of 1–6000 characters')
        branch = self.store.branch_at(conversation, ident)
        job = self.chat(branch, text, message['data'].get('graph_ids', []),
                        message['data'].get('thinking_enabled', False))
        return branch, job

    def rename(self, ident, title):
        with self.store.connection() as db:
            db.execute('UPDATE conversations SET title=? WHERE id=?', (title[:200], ident))

    def export(self, ident, path):
        from .exporting import export_conversation
        return export_conversation(self.store, ident, path)

    def delete_conversation(self, ident):
        if any(j.data['kind'] == f'chat:{ident}' and j.data['status'] in ('queued', 'running')
               for j in self.jobs.live.values()):
            raise ValueError('Cancel the active chat and wait for it to finish before deletion')
        with self.store.connection() as db:
            db.execute('DELETE FROM conversations WHERE id=?', (ident,))

    def index(self, path):
        self.configure(resources=str(path))
        return self.jobs.submit('index', lambda job: self.retrieval.index(Path(path), job))

    def search(self, query):
        return self.jobs.submit('search', lambda job: self.retrieval.search(query[:2000]))

    def fever(self, index=False):
        def work(job):
            result = self.models.fever(job, extract=index)
            if index:
                result['index'] = self.index_fever(result['evidence_directory'], job)
            return result
        return self.jobs.submit('fever', work, request={'index': index})

    def build_fever_index(self):
        existing = next((j for j in list(self.jobs.live.values()) if j.data['kind'] == 'fever-index' and
                         j.data['status'] in ('queued', 'running')), None)
        if existing:
            return existing.data['id']
        def work(job):
            self.store.configure('fever_ready', False)
            result = self.models.fever(job, extract=True, download_missing=False)
            result['index'] = self.index_fever(result['evidence_directory'], job)
            return result
        return self.jobs.submit('fever-index', work)

    def index_fever(self, directory, job):
        job.emit('stage', stage='Building resumable Wikipedia keyword index · chat remains available')
        result = self.retrieval.index(Path(directory), job, semantic=False, resumable=True)
        self.store.configure('fever_ready', not result['errors'])
        if result['errors']:
            raise ValueError(f'Evidence downloaded, but {len(result["errors"])} files could not be indexed. '
                             'Retry the task. First error: '+result['errors'][0]['error'])
        return result

    def import_posts(self, path, mapping=None, combine=False):
        rows = read_posts(Path(path)); mapping = mapping or suggest_mapping(rows)
        if combine:
            for row in rows:
                row['__sentinel_x_cascade'] = 'bundle'
            mapping = {**mapping, 'cascade_id': '__sentinel_x_cascade'}
        graphs = normalize_posts(rows, mapping, str(path))
        for graph in graphs:
            self.store.save_graph(graph)
        return [g['id'] for g in graphs]

    def preview_posts(self, path):
        def work(job):
            job.emit('stage', stage='Reading dataset')
            rows = read_posts(Path(path))
            job.check()
            return {'count': len(rows), 'mapping': suggest_mapping(rows),
                    'keys': sorted(set().union(*(r.keys() for r in rows[:100]))),
                    'samples': rows[:3]}
        return self.jobs.submit('posts-preview', work)

    def import_posts_background(self, path, mapping, combine=False):
        if not mapping.get('id') or not mapping.get('text'):
            raise ValueError('Choose fields for Post ID and Text before importing.')
        def work(job):
            job.emit('stage', stage='Validating and importing posts')
            rows = read_posts(Path(path))
            selected = dict(mapping)
            for number, row in enumerate(rows, 1):
                if any(row.get(selected[field]) is None or not str(row[selected[field]]).strip()
                       for field in ('id', 'text')):
                    raise ValueError(f'Post {number} needs a nonempty Post ID and Text.')
            if combine:
                for row in rows:
                    row['__sentinel_x_cascade'] = 'bundle'
                selected['cascade_id'] = '__sentinel_x_cascade'
            graphs = normalize_posts(rows, selected, str(path))
            job.check()
            with self.store.connection() as db:
                for graph in graphs:
                    job.check()
                    db.execute('INSERT OR REPLACE INTO graphs VALUES (?,?)',
                               (graph['id'], json.dumps(graph)))
            return [g['id'] for g in graphs]
        return self.jobs.submit('posts-import', work)

    def map_posts(self, path):
        def work(job):
            rows = read_posts(Path(path))
            result = self.runtime.structured('Map these fields to id,text,author_id,timestamp,parent_id,cascade_id. '
                'Return {"mapping":{"id":"actual_field",...}}, null for missing fields. '
                'Do not invent values. Samples:\n'+json.dumps(rows[:3])[:10000], job)
            mapping = result.get('mapping'); allowed = suggest_mapping(rows)
            keys = set().union(*(r.keys() for r in rows[:100]))
            if not isinstance(mapping, dict) or set(mapping) != set(allowed):
                raise ValueError('Qwen returned an incomplete field mapping')
            if any(value is not None and (not isinstance(value, str) or value not in keys)
                   for value in mapping.values()):
                raise ValueError('Qwen mapped a field absent from the input')
            return mapping
        return self.jobs.submit('map-posts', work)

    def import_twitter(self, path):
        directory = Path(path)
        folders = [directory] if (directory/'tree').is_dir() else [directory/name
                   for name in ('twitter15', 'twitter16') if (directory/name/'tree').is_dir()]
        if not folders:
            raise ValueError('Choose a Twitter15/16 directory or their parent')
        def work(job):
            result = {'imported': 0, 'errors': []}
            existing = {(g['source'], g['cascade_id']): g['id'] for g in self.store.graphs()}
            for folder in folders:
                for graph in native_twitter(folder):
                    job.check()
                    if 'error' in graph:
                        result['errors'].append(graph['error'])
                    else:
                        graph['native_twitter'] = True
                        if (folder.name, graph['cascade_id']) in existing:
                            graph['id'] = existing[(folder.name, graph['cascade_id'])]
                        self.store.save_graph(graph); result['imported'] += 1
                    if result['imported'] % 25 == 0:
                        job.emit('importing', **result)
            return result
        return self.jobs.submit('twitter-import', work)

    def estimate(self, ident):
        graph = self.store.graph(ident)
        if len(graph['nodes']) > 100:
            raise ValueError('Estimate at most 100 posts at once')
        def work(job):
            prompt = ('Propose missing propagation metadata; this is an estimate, not observations. '
                'Return {"edges":[["parent_id","child_id"]],"delays":{"node_id":minutes},'
                '"assumptions":"explain all inferred links and timing"}. '
                'Use existing IDs only. Preserve observed edges/times. Make a single acyclic cascade. '
                'Delays are minutes since root. Data:\n'+json.dumps(graph))
            if len(prompt) > 24000:
                raise ValueError('Graph estimation input is too large; split the bundle')
            proposed = estimate_graph(graph, self.runtime.structured(prompt, job))
            self.store.configure('proposal:'+ident, proposed)
            return proposed
        return self.jobs.submit('estimate-graph', work)

    def accept(self, ident):
        proposed = self.store.setting('proposal:'+ident)
        if not proposed:
            raise ValueError('No estimated graph proposal to review')
        validate_graph(proposed); proposed['accepted'] = True
        self.store.save_graph(proposed); self.store.configure('proposal:'+ident, None)
        return proposed

    def checkpoints(self):
        return [json.loads(p.read_text()) for p in sorted((self.store.root/'checkpoints').glob('*.json'))]

    def train(self, threshold=100, window=60., epochs=80):
        def work(job):
            from .learning import train
            self.runtime.stop()
            graphs = [g for g in self.store.graphs() if g.get('native_twitter') or
                      g['source'].lower() in ('twitter15', 'twitter16')]
            result = train(graphs, self.store.root/'checkpoints',
                           {'threshold': threshold, 'window': window, 'epochs': epochs}, job)
            self.configure(checkpoint=result['id'])
            return result
        return self.jobs.submit('training', work)

    def import_skill(self, path):
        source = Path(path)
        data = json.loads((source/'tools.json').read_text())
        ident = data['id']
        if not isinstance(ident, str) or not ident.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Invalid skill ID')
        if any(s['id'] == ident for s in self.skills.list()):
            raise ValueError('Skill ID already exists')
        destination = self.store.root/'skills'/ident
        destination.mkdir()
        try:
            for name in ('SKILL.md', 'tools.json'):
                shutil.copyfile(source/name, destination/name)
            result = next(s for s in self.skills.list() if s['id'] == ident)
            if result.get('error'):
                raise ValueError(result['error'])
        except BaseException:
            shutil.rmtree(destination)
            raise
        return result
