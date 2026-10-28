from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
import secrets
import shutil

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .assistant import Assistant
from .downloads import Models
from .graphs import (estimate_graph, native_twitter, normalize_posts, read_posts, suggest_mapping,
                     validate_graph)
from .jobs import Jobs
from .retrieval import Retrieval
from .runtime import Llama
from .skills import Skills
from .state import Store


class PathInput(BaseModel):
    path: str


class PostInput(PathInput):
    mapping: dict[str, str | None] | None = None
    combine: bool = False


class ChatInput(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    graph_ids: list[str] = Field(default_factory=list, max_length=100)


class TrainInput(BaseModel):
    graph_ids: list[str] = Field(default_factory=list)
    threshold: int = Field(default=100, ge=2)
    window: float = Field(default=60., gt=0)
    epochs: int = Field(default=80, ge=1, le=2000)
    seed: int = 42
    device: str = 'cpu'


def create_app(root: Path, token: str, manifest_path: Path, skills_path: Path, runtime_path=None):
    store = Store(root); jobs = Jobs(store); retrieval = Retrieval(store)
    runtime = Llama(store, runtime_path)
    models = Models(store, json.loads(manifest_path.read_text()))
    skills = Skills([skills_path, root/'skills'], store)
    assistant = Assistant(store, retrieval, runtime, skills)

    @asynccontextmanager
    async def lifespan(app):
        yield
        for job in jobs.live.values():
            job.cancelled.set()
        runtime.stop()
        jobs.pool.shutdown(wait=False, cancel_futures=True)

    def auth(authorization: str = Header(default='')):
        if not secrets.compare_digest(authorization, f'Bearer {token}'):
            raise HTTPException(401, 'Invalid worker authentication')

    app = FastAPI(title='Veritas local API', version='1.0', dependencies=[Depends(auth)],
                  lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store; app.state.jobs = jobs

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={'detail': str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def file_error(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={'detail': str(exc)})

    @app.get('/v1/health')
    def health():
        return {'ready': True, 'version': '0.1.0'}

    @app.get('/v1/status')
    def status():
        executable, model = runtime.configured()
        return {'root': str(root), 'resources': store.setting('resources', str(root/'resources')),
                'models': models.status(), 'llama_path': executable, 'qwen_path': model,
                'checkpoint': store.setting('checkpoint'),
                'context': store.setting('context', 8192), 'gpu_layers': store.setting('gpu_layers', 'auto'),
                'first_run': not store.setting('setup_started', False)}

    @app.post('/v1/setup')
    def setup():
        store.configure('setup_started', True)
        def work(job):
            return [models.install(key, job) for key in ('qwen', 'embedding')]
        return {'job_id': jobs.submit('setup', work)}

    @app.patch('/v1/settings')
    def settings(values: dict):
        allowed = {'resources', 'context', 'gpu_layers', 'llama_path', 'checkpoint'}
        if set(values) - allowed:
            raise ValueError('Unknown setting')
        for key, value in values.items():
            if key in ('resources', 'llama_path') and not isinstance(value, str):
                raise ValueError('Path must be a string')
            if key == 'context' and (not isinstance(value, int) or not 2048 <= value <= 32768):
                raise ValueError('Context must be 2048–32768')
            if key == 'gpu_layers' and str(value) != 'auto' and (not str(value).isdigit() or int(value) > 999):
                raise ValueError('Invalid GPU layer count')
            if key == 'checkpoint' and value and (not isinstance(value, str) or not value.isalnum() or
                    not (root/'checkpoints'/f'{value}.json').exists()):
                raise ValueError('Checkpoint does not exist')
        for key, value in values.items():
            store.configure(key, value)
        if set(values) & {'context', 'gpu_layers', 'llama_path'}:
            runtime.stop()
        return {'saved': True}

    @app.get('/v1/conversations')
    def conversations(q: str = ''):
        return store.conversations(q)

    @app.post('/v1/conversations')
    def new_conversation(values: dict):
        return store.create_conversation(str(values.get('title', 'New conversation')))

    @app.patch('/v1/conversations/{ident}')
    def rename(ident: str, values: dict):
        with store.connection() as db:
            db.execute('UPDATE conversations SET title=? WHERE id=?',
                       (str(values.get('title', 'Conversation'))[:200], ident))
        return {'saved': True}

    @app.delete('/v1/conversations/{ident}')
    def delete_conversation(ident: str):
        # A queued chat must not recreate or write to a deleted conversation.
        if any(j.data['status'] in ('running', 'queued') and j.data['kind'] == f'chat:{ident}'
               for j in jobs.live.values()):
            raise ValueError('Cancel the active chat and wait for it to finish before deletion')
        with store.connection() as db:
            db.execute('DELETE FROM conversations WHERE id=?', (ident,))
        return {'deleted': True}

    @app.get('/v1/conversations/{ident}/messages')
    def messages(ident: str):
        return store.messages(ident)

    @app.post('/v1/conversations/{ident}/chat')
    def chat(ident: str, body: ChatInput):
        if not any(c['id'] == ident for c in store.conversations()):
            raise ValueError('Conversation not found')
        return {'job_id': jobs.submit(f'chat:{ident}', lambda job:
                    assistant.chat(ident, body.text, body.graph_ids, job))}

    @app.get('/v1/jobs')
    def list_jobs():
        return jobs.list()

    @app.get('/v1/jobs/{ident}')
    def get_job(ident: str, after: int = -1):
        data = jobs.get(ident)
        data['events'] = [e for e in data['events'] if e['seq'] > after]
        return data

    @app.post('/v1/jobs/{ident}/cancel')
    def cancel_job(ident: str):
        jobs.cancel(ident)
        return {'cancelling': True}

    @app.post('/v1/models/{key}/download')
    def download_model(key: str):
        if key not in ('qwen', 'embedding'):
            raise ValueError('Unknown model')
        return {'job_id': jobs.submit('download:'+key, lambda job: models.install(key, job))}

    @app.post('/v1/models/import')
    def import_model(body: PathInput):
        runtime.stop()
        return models.import_qwen(body.path)

    @app.post('/v1/models/smoke')
    def smoke_model():
        def work(job):
            runtime.start(job)
            answer = runtime.structured('Return {"ready":true}.', job)
            if answer.get('ready') is not True:
                raise ValueError('Qwen structured output smoke check failed')
            # Exercise actual server-side function parsing, without executing a side effect.
            tools = [{'type': 'function', 'function': {'name': 'smoke_echo',
                'description': 'Echo the supplied message.', 'parameters': {'type': 'object',
                    'properties': {'message': {'type': 'string'}}, 'required': ['message']}}}]
            response = runtime.stream([{'role': 'user', 'content':
                                       'Call smoke_echo with message hello. Do not answer directly.'}], tools, job)
            calls = response.get('tool_calls', [])
            if not calls or calls[0]['function']['name'] != 'smoke_echo':
                raise ValueError('Qwen function-calling smoke check failed; runtime/model combination is incompatible')
            json.loads(calls[0]['function']['arguments'])
            return {'generation': True, 'function_calling': True}
        return {'job_id': jobs.submit('qwen-smoke', work)}

    @app.post('/v1/resources/index')
    def index_resources(body: PathInput):
        store.configure('resources', body.path)
        return {'job_id': jobs.submit('index', lambda job: retrieval.index(Path(body.path), job))}

    @app.post('/v1/resources/fever')
    def download_fever():
        def work(job):
            result = models.fever(job)
            result['index'] = retrieval.index(Path(result['evidence_directory']), job)
            return result
        return {'job_id': jobs.submit('fever', work)}

    @app.get('/v1/resources/search')
    def search(q: str):
        return retrieval.search(q[:2000])

    @app.get('/v1/skills')
    def list_skills():
        return skills.list()

    @app.patch('/v1/skills/{ident}')
    def toggle_skill(ident: str, values: dict):
        store.configure('skill:'+ident, bool(values.get('enabled', True)))
        return {'saved': True}

    @app.post('/v1/skills/import')
    def import_skill(body: PathInput):
        source = Path(body.path).resolve()
        data = json.loads((source/'tools.json').read_text())
        ident = data['id']
        if not isinstance(ident, str) or not ident.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Invalid skill ID')
        if any(s['id'] == ident for s in skills.list()):
            raise ValueError('Skill ID already exists')
        destination = root/'skills'/ident
        destination.mkdir()
        try:
            for name in ('SKILL.md', 'tools.json'):
                shutil.copyfile(source/name, destination/name)
            imported = next(s for s in skills.list() if s['id'] == ident)
            if imported.get('error'):
                raise ValueError(imported['error'])
        except Exception:
            shutil.rmtree(destination)
            raise
        return imported

    @app.get('/v1/graphs')
    def list_graphs():
        return [{'id': g['id'], 'cascade_id': g['cascade_id'], 'source': g['source'],
                 'nodes': len(g['nodes']), 'estimated': g.get('estimated'),
                 'accepted': g.get('accepted'), 'needs_estimation': g.get('needs_estimation'),
                 'warnings': g.get('warnings', [])} for g in store.graphs()]

    @app.get('/v1/graphs/{ident}')
    def get_graph(ident: str):
        return store.graph(ident)

    @app.post('/v1/posts/preview')
    def preview_posts(body: PathInput):
        rows = read_posts(Path(body.path))
        return {'mapping': suggest_mapping(rows), 'sample': rows[:3], 'count': len(rows)}

    @app.post('/v1/posts/map')
    def map_posts(body: PathInput):
        rows = read_posts(Path(body.path))
        return {'job_id': jobs.submit('map-posts', lambda job: runtime.structured(
            'Map these post fields to id,text,author_id,timestamp,parent_id,cascade_id. '
            'Return {"mapping":{"id":"actual_field",...}}; use null for missing fields. '
            'Do not invent values. Samples:\n'+json.dumps(rows[:3])[:10000], job))}

    @app.post('/v1/posts/import')
    def import_posts(body: PostInput):
        rows = read_posts(Path(body.path)); mapping = body.mapping or suggest_mapping(rows)
        if body.combine:
            # Explicit grouping for text-only bundles, without inventing their edges.
            for row in rows:
                row['__veritas_cascade'] = 'bundle'
            mapping = {**mapping, 'cascade_id': '__veritas_cascade'}
        graphs = normalize_posts(rows, mapping, body.path)
        for graph in graphs:
            store.save_graph(graph)
        return {'graph_ids': [g['id'] for g in graphs], 'count': len(graphs)}

    @app.post('/v1/twitter/import')
    def import_twitter(body: PathInput):
        directory = Path(body.path)
        directories = [directory] if (directory/'tree').is_dir() else [directory/x for x in ('twitter15', 'twitter16') if (directory/x/'tree').is_dir()]
        if not directories:
            raise ValueError('Choose a twitter15/twitter16 directory or their parent directory')
        def work(job):
            result = {'imported': 0, 'errors': []}
            for folder in directories:
                for graph in native_twitter(folder):
                    job.check()
                    if 'error' in graph:
                        result['errors'].append(graph['error'])
                    else:
                        # Re-importing replaces the existing event rather than duplicating it.
                        existing = next((g for g in store.graphs() if g['source'] == folder.name and
                                         g['cascade_id'] == graph['cascade_id']), None)
                        if existing:
                            graph['id'] = existing['id']
                        store.save_graph(graph); result['imported'] += 1
                    if result['imported'] % 25 == 0:
                        job.emit('importing', **result)
            return result
        return {'job_id': jobs.submit('twitter-import', work)}

    @app.post('/v1/graphs/{ident}/estimate')
    def estimate(ident: str):
        graph = store.graph(ident)
        def work(job):
            if len(graph['nodes']) > 100:
                raise ValueError('Estimate at most 100 posts at once')
            prompt = ('Propose a plausible propagation graph for these posts. This is an estimate, not observed data. '
                'Return {"edges":[["parent_id","child_id"]],"delays":{"node_id":minutes},'
                '"assumptions":"explain all invented links and timing"}. '
                'Use existing IDs only, preserve observed edges and times, form a single acyclic cascade. '
                'Times are minutes since root. Data:\n'+json.dumps(graph))
            proposal = runtime.structured(prompt[:24000], job)
            estimated = estimate_graph(graph, proposal)
            # Store a candidate separately; it does not replace observed data until acceptance.
            store.configure('proposal:'+ident, estimated)
            return estimated
        return {'job_id': jobs.submit('estimate-graph', work)}

    @app.post('/v1/graphs/{ident}/accept')
    def accept(ident: str, values: dict):
        if values.get('accept') is not True:
            raise ValueError('Explicit acceptance is required')
        proposal = store.setting('proposal:'+ident)
        if not proposal:
            raise ValueError('No estimated graph proposal to review')
        validate_graph(proposal)
        proposal['accepted'] = True
        store.save_graph(proposal); store.configure('proposal:'+ident, None)
        return proposal

    @app.get('/v1/graphs/{ident}/proposal')
    def proposal(ident: str):
        return store.setting('proposal:'+ident)

    @app.get('/v1/checkpoints')
    def checkpoints():
        return [json.loads(path.read_text()) for path in sorted((root/'checkpoints').glob('*.json'))]

    @app.post('/v1/training')
    def training(body: TrainInput):
        def work(job):
            from .learning import train
            runtime.stop()
            graphs = [store.graph(ident) for ident in body.graph_ids] if body.graph_ids else [
                g for g in store.graphs() if g['source'] in ('twitter15', 'twitter16')]
            config = body.model_dump(exclude={'graph_ids'})
            result = train(graphs, root/'checkpoints', config, job)
            store.configure('checkpoint', result['id'])
            return result
        return {'job_id': jobs.submit('training', work)}

    @app.post('/v1/predict')
    def predict_graphs(values: dict):
        ids = values.get('graph_ids', [])
        return {'job_id': jobs.submit('predict', lambda job: assistant.invoke(
            'graphsage_predict', values, ids, []))}

    return app
