"""Human-readable Markdown conversation reports, based only on persisted records."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile

from . import __version__


def quote(value):
    return '\n'.join('> '+line if line else '>' for line in str(value).splitlines()) or '> (empty)'


def code(value):
    text = str(value)
    fence = '`' * (max((len(m.group()) for m in re.finditer(r'`+', text)), default=0)+1)
    return f'{fence} {text} {fence}'


def block(value):
    text = str(value)
    fence = '`' * max(3, max((len(m.group())+1 for m in re.finditer(r'`+', text)), default=3))
    return f'{fence}text\n{text}\n{fence}'


def readable(value, depth=0):
    """Nested bullets rather than raw JSON for tool parameters, results and settings."""
    indent = '  '*depth
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            name = str(key).replace('_', ' ').capitalize()
            if isinstance(item, (dict, list)) and item:
                lines.append(f'{indent}- **{name}:**')
                lines.extend(readable(item, depth+1))
            else:
                rendered = 'Not recorded' if item is None else 'None' if item in ([], {}) else str(item)
                lines.append(f'{indent}- **{name}:** {rendered.replace(chr(10), chr(10)+indent+"  ")}')
        return lines
    if isinstance(value, list):
        lines = []
        for i, item in enumerate(value, 1):
            if isinstance(item, (dict, list)):
                lines.append(f'{indent}- **Item {i}:**'); lines.extend(readable(item, depth+1))
            else:
                lines.append(f'{indent}- {str(item).replace(chr(10), chr(10)+indent+"  ")}')
        return lines or [indent+'- None']
    return [indent+'- '+str(value)]


def evidence_section(records):
    if not records:
        return ['No RAG references were recorded.']
    lines = []
    seen = set()
    for record in records:
        key = (record.get('citation'), record.get('path'), record.get('text'))
        if key in seen: continue
        seen.add(key)
        path = str(record.get('path') or 'Not recorded')
        filename = PureWindowsPath(path).name if '\\' in path else Path(path).name
        lines += [f'##### Reference {code(record.get("citation", "unassigned"))}', '',
                  f'- **File name:** {code(filename)}', f'- **Source path:** {code(path)}',
                  f'- **Retrieval:** {record.get("retrieval", "Not recorded")}', '']
        if record.get('provenance'):
            lines += readable(record['provenance']) + ['']
        lines += ['**Retrieved excerpt:**', '', quote(record.get('text', 'Not recorded')), '']
    return lines


def tool_record(message):
    data = message.get('data', {})
    if data.get('name'):
        value = dict(data)
        if value.get('arguments') is None:
            raw = data.get('wire', {}).get('function', {}).get('arguments')
            if raw is not None:
                try: value['arguments'] = json.loads(raw)
                except ValueError: value['arguments'] = raw
        return value
    try:
        content = json.loads(message['content'])
        if not isinstance(content, dict): return {'name': 'Unknown tool', 'result': message['content']}
    except (ValueError, TypeError):
        return {'name': 'Unknown tool', 'result': message['content']}
    wire = data.get('wire', {}).get('function', {})
    arguments = content.get('arguments')
    if arguments is None and wire.get('arguments'):
        try: arguments = json.loads(wire['arguments'])
        except ValueError: arguments = wire['arguments']
    return {**content, 'arguments': arguments}


def render_conversation(store, ident):
    # Read title and messages in one transaction for a consistent saved-message snapshot.
    with store.connection() as db:
        db.execute('BEGIN')
        conversation = db.execute('SELECT * FROM conversations WHERE id=?', (ident,)).fetchone()
        if not conversation:
            raise ValueError('Conversation not found')
        messages = [{**dict(row), 'data': json.loads(row['data'] or '{}')} for row in db.execute(
            'SELECT * FROM messages WHERE conversation=? ORDER BY id', (ident,))]
    exported = datetime.now(timezone.utc).isoformat(timespec='seconds')
    lines = ['# Sentinel-X Conversation Export', '', '## Conversation', '',
             '**Title:**', '', quote(conversation['title']), '',
             f'- **Conversation ID:** {code(ident)}',
             f'- **Created (UTC):** {conversation["created"]}',
             f'- **Exported (UTC):** {exported}', f'- **Export format:** Markdown (.md)',
             f'- **Application version at export:** {__version__}',
             f'- **Saved message records:** {len(messages)}', '',
             'This report contains saved messages and recorded application context. '
             'Reasoning sections contain explicit answer explanations or a model-provided reasoning summary; '
             'they are not a reconstructed internal reasoning trace. Historical context that was not logged is marked as unavailable.', '',
             '## Messages and execution record', '']
    if not messages:
        lines += ['This conversation has no saved messages.', '']
    current_turn_tools = []
    for index, message in enumerate(messages, 1):
        data = message['data']; role = message['role']
        label = {'user': 'User', 'assistant': 'Sentinel-X', 'tool': 'Skill / tool execution'}.get(role, role.title())
        lines += [f'### {index}. {label}', '', f'- **Recorded (UTC):** {message["created"]}',
                  f'- **Message ID:** {message["id"]}']
        if data.get('turn_id'): lines += [f'- **Turn ID:** {code(data["turn_id"])}']
        lines += ['']
        if role == 'tool':
            tool = tool_record(message)
            current_turn_tools.append(tool)
            name = tool.get('name', 'Unknown tool')
            skill = 'GraphSAGE virality skill' if name == 'graphsage_predict' else 'Core RAG evidence search' if name == 'evidence_search' else 'Tool'
            lines += [f'**Capability used:** {skill}', '', f'**Function:** {code(name)}', '',
                      '#### Request arguments', '']
            lines += readable(tool.get('arguments', 'Not recorded'))+['', '#### Result / error', '']
            lines += readable(tool.get('result', 'Not recorded'))+['']
            continue
        lines += [quote(message['content']), '']
        if role == 'user':
            current_turn_tools = []
            if data.get('graph_ids'):
                lines += ['#### Attached graph IDs', '']+readable(data['graph_ids'])+['']
            audit = data.get('audit')
            lines += ['#### Application system prompt', '']
            if audit and audit.get('system_prompt'):
                lines += ['Exact application system-message snapshot prepared for this turn:', '', block(audit['system_prompt']), '',
                          '#### Model and generation settings', '']
                lines += readable({k: audit.get(k) for k in ('model', 'model_path', 'runtime_path', 'selected_checkpoint')})
                lines += readable(audit.get('generation', {}))+['', '#### Enabled skills (available, not necessarily invoked)', '']
                lines += readable(audit.get('enabled_skills', []))+['']
                if audit.get('skill_instructions'):
                    lines += ['**Skill instructions included in the system prompt:**', '', block(audit['skill_instructions']), '']
                lines += ['**Available function definitions:**', '']+readable(audit.get('tool_definitions', []))+['']
                lines += ['#### Graph provenance', '']+readable(audit.get('graphs', []))+['']
                lines += ['#### Initial RAG references', '']+evidence_section(audit.get('initial_evidence', []))+['']
                lines += ['**Conversation history message IDs supplied as context:**', '']+readable(audit.get('history_message_ids', []))+['']
            else:
                lines += ['Not recorded for this historical turn. The current prompt is not substituted for an unknown historical prompt.', '']
        elif role == 'assistant':
            lines += ['#### Reasoning summary and explanations', '']
            summary = data.get('reasoning_summary')
            lines += [quote(summary) if summary else 'No separate model-provided reasoning summary was recorded.', '']
            for number, claim in enumerate(data.get('claims', []), 1):
                lines += [f'##### Claim {number}', '', quote(claim.get('claim', '')), '',
                          f'**Verdict:** {str(claim.get("verdict", "Not recorded")).replace("_", " ")}', '',
                          '**Explanation:**', '', quote(claim.get('explanation', 'Not recorded')), '',
                          '**Citations:** '+', '.join(code(c) for c in claim.get('citations', [])), '']
            lines += ['#### RAG references for this answer', '']+evidence_section(data.get('evidence', []))+['']
            # Older records can have summary tool results without a standalone tool message.
            missing = data.get('tools', [])[len(current_turn_tools):]
            if missing:
                lines += ['#### Recorded skill usage', '']+readable(missing)+['']
    return '\n'.join(lines).rstrip()+'\n'


def export_conversation(store, ident, destination):
    path = Path(destination).expanduser()
    if path.suffix.lower() != '.md':
        raise ValueError('Choose an export filename ending in .md')
    content = render_conversation(store, ident)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                prefix='.sentinel-x-export-', suffix='.tmp', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
    return path
