import json
from pathlib import Path


class Skills:
    def __init__(self, folders, store):
        self.folders = folders
        self.store = store

    def list(self):
        result = []; ids = set()
        for folder in self.folders:
            for manifest in sorted(Path(folder).glob('*/tools.json')):
                try:
                    data = json.loads(manifest.read_text(encoding='utf-8'))
                    if data['id'] in ids:
                        raise ValueError('Duplicate skill ID')
                    ids.add(data['id'])
                    for tool in data['tools']:
                        if tool['handler'] not in ('graphsage.predict', 'retrieval.search'):
                            raise ValueError('Unknown tool handler; skill files cannot register executable code')
                        if tool['function']['name'] not in ('graphsage_predict', 'evidence_search'):
                            raise ValueError('Unknown tool function')
                    data.update(path=str(manifest.parent),
                        instructions=(manifest.parent/'SKILL.md').read_text(encoding='utf-8'),
                        enabled=self.store.setting('skill:'+data['id'], True))
                    result.append(data)
                except (ValueError, KeyError, OSError, TypeError) as exc:
                    result.append({'id': manifest.parent.name, 'name': manifest.parent.name,
                                   'enabled': False, 'error': str(exc)})
        return result

    def tools(self):
        tools = [{'type': 'function', 'function': {'name': 'evidence_search',
                  'description': 'Search local factual evidence. Returns excerpts with citation identifiers.',
                  'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}},
                                 'required': ['query'], 'additionalProperties': False}}}]
        for skill in self.list():
            if skill.get('enabled') and not skill.get('error'):
                for tool in skill['tools']:
                    if not any(t['function']['name'] == tool['function']['name'] for t in tools):
                        tools.append({'type': 'function', 'function': tool['function']})
        return tools

    def instructions(self):
        return '\n\n'.join(s.get('instructions', '') for s in self.list() if s.get('enabled'))
