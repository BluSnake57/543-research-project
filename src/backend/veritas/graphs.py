"""Canonical propagation graphs and native Twitter15/16 ingestion."""
from __future__ import annotations

import ast
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import uuid

FEATURE_VERSION = 1
FIELDS = ('id', 'text', 'author_id', 'timestamp', 'parent_id', 'cascade_id')
ALIASES = {'id': ('id', 'post_id', 'tweet_id'), 'text': ('text', 'content', 'body'),
           'author_id': ('author_id', 'user_id', 'uid'),
           'timestamp': ('timestamp', 'created_at', 'time'),
           'parent_id': ('parent_id', 'reply_to', 'in_reply_to_status_id', 'repost_of'),
           'cascade_id': ('cascade_id', 'thread_id', 'root_id')}


def read_posts(path: Path):
    if path.suffix.lower() == '.jsonl':
        with path.open(encoding='utf-8-sig') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(rows, dict):
            rows = rows.get('posts', rows.get('data'))
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError('Expected a nonempty JSON array of posts or one post per JSONL line')
    if len(rows) > 100_000:
        raise ValueError('Import at most 100,000 posts per file')
    return rows


def suggest_mapping(rows):
    keys = set().union(*(r.keys() for r in rows[:100]))
    return {field: next((k for k in aliases if k in keys), None)
            for field, aliases in ALIASES.items()}


def timestamp(value):
    if value is None or value == '':
        return None
    try:
        if isinstance(value, (int, float)) or str(value).replace('.', '', 1).isdigit():
            result = float(value)
        else:
            dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
            if dt.tzinfo is None:
                raise ValueError('Timestamps must include a timezone')
            result = dt.timestamp()
        if not math.isfinite(result):
            raise ValueError('Timestamp must be finite')
        return result
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Invalid timestamp: {value}') from exc


def validate_graph(graph):
    nodes = graph.get('nodes', [])
    if not nodes or len(nodes) > 100_000:
        raise ValueError('Graph must contain 1–100,000 nodes')
    ids = [str(n['id']) for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate graph node IDs')
    known = set(ids)
    adjacency = defaultdict(list)
    indegrees = dict.fromkeys(ids, 0)
    edges = set()
    for edge in graph.get('edges', []):
        if len(edge) != 2:
            raise ValueError('An edge needs a parent and child')
        a, b = map(str, edge)
        if a not in known or b not in known or a == b:
            raise ValueError('Edges must reference distinct existing nodes')
        if (a, b) not in edges:
            edges.add((a, b))
            adjacency[a].append(b)
            indegrees[b] += 1
    queue = [node for node, degree in indegrees.items() if degree == 0]
    roots = list(queue)
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in adjacency[node]:
            indegrees[child] -= 1
            if indegrees[child] == 0:
                queue.append(child)
    if visited != len(nodes):
        raise ValueError('Propagation graph contains a cycle')
    for node in nodes:
        delay = node.get('delay')
        if delay is not None and (not isinstance(delay, (int, float)) or
                                  not math.isfinite(delay) or delay < 0):
            raise ValueError('Delays must be finite nonnegative minutes')
    graph['edges'] = [list(e) for e in sorted(edges)]
    graph['roots'] = roots
    return graph


def normalize_posts(rows, mapping, source='import'):
    if any(k not in FIELDS for k in mapping):
        raise ValueError('Unrecognized mapping field')
    nodes = {}
    for index, row in enumerate(rows):
        item = {f: row.get(mapping.get(f)) for f in FIELDS}
        ident = str(item['id']) if item['id'] is not None else f'generated-{index}'
        if ident in nodes:
            raise ValueError(f'Duplicate post ID: {ident}')
        nodes[ident] = {'id': ident, 'text': str(item['text'] or ''),
                        'author_id': str(item['author_id'] or ''),
                        'timestamp': timestamp(item['timestamp']),
                        'parent_id': str(item['parent_id']) if item['parent_id'] is not None else None,
                        'cascade_id': str(item['cascade_id']) if item['cascade_id'] is not None else None,
                        'original': row}
    groups = defaultdict(list)
    for node in nodes.values():
        trail = set()
        current = node
        while current['parent_id'] in nodes:
            if current['id'] in trail:
                raise ValueError('Post relationships contain a cycle')
            trail.add(current['id'])
            current = nodes[current['parent_id']]
        # Explicit cascade IDs keep records with missing links in the same candidate graph.
        group = node['cascade_id'] or current['cascade_id'] or current['id']
        groups[group].append(node)
    graphs = []
    for cascade, members in groups.items():
        ids = {m['id'] for m in members}
        times = [m['timestamp'] for m in members if m['timestamp'] is not None]
        origin = min(times) if times else None
        edges = [[m['parent_id'], m['id']] for m in members if m['parent_id'] in ids]
        missing = []
        if any(m['timestamp'] is None for m in members):
            missing.append('Missing timestamps')
        if len(members) > 1 and len(edges) < len(members) - 1:
            missing.append('Missing propagation links')
        if any(m['parent_id'] and m['parent_id'] not in ids for m in members):
            missing.append('Parents absent from imported records')
        for member in members:
            member['delay'] = ((member['timestamp'] - origin) / 60
                               if member['timestamp'] is not None and origin is not None else None)
        graph = {'id': uuid.uuid4().hex, 'cascade_id': cascade, 'source': source,
                 'nodes': members, 'edges': edges, 'estimated': False,
                 'accepted': False, 'warnings': missing,
                 'needs_estimation': bool(missing), 'source_hash': fingerprint(rows)}
        graphs.append(validate_graph(graph))
    return graphs


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def estimate_graph(original, proposal):
    """Only add missing information. Acceptance is a separate explicit UI action."""
    graph = json.loads(json.dumps(original))
    known = {n['id']: n for n in graph['nodes']}
    proposed_delays = proposal.get('delays', {})
    inferred = []
    for ident, node in known.items():
        if node.get('delay') is None:
            if ident not in proposed_delays:
                raise ValueError(f'Estimation omitted time for {ident}')
            node['delay'] = float(proposed_delays[ident])
            inferred.append({'node': ident, 'field': 'delay', 'value': node['delay']})
    parents = {b for _, b in graph['edges']}
    for edge in proposal.get('edges', []):
        a, b = map(str, edge)
        if b in parents:
            if [a, b] not in graph['edges']:
                raise ValueError('Cannot overwrite an observed parent')
            continue
        graph['edges'].append([a, b])
        parents.add(b)
        inferred.append({'field': 'edge', 'value': [a, b]})
    graph.update(estimated=True, accepted=False, needs_estimation=False,
                 inferred=inferred, assumptions=str(proposal.get('assumptions', '')))
    graph['warnings'] = list(set(graph['warnings'] + [
        'Estimated graph — experimental prediction; inferred links/times are not observations']))
    validate_graph(graph)
    if len(graph['roots']) != 1:
        raise ValueError('An estimated cascade must have exactly one root; review grouping first')
    return graph


def native_twitter(directory: Path):
    """Yield complete observed cascades; native rumor labels are retained only as metadata."""
    sources = {}
    with (directory / 'source_tweets.txt').open(encoding='utf-8-sig') as stream:
        for line in stream:
            ident, _, text = line.rstrip('\n').partition('\t')
            sources[ident] = text
    labels = {}
    for line in (directory / 'label.txt').read_text(encoding='utf-8-sig').splitlines():
        label, sep, ident = line.partition(':')
        if sep:
            labels[ident.strip()] = label.strip()
    for file in sorted((directory / 'tree').glob('*.txt')):
        nodes, edges, warnings = {}, [], []
        root_id = None
        for line_number, line in enumerate(file.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            try:
                left, right = line.split('->', 1)
                pair = [ast.literal_eval(left.strip()), ast.literal_eval(right.strip())]
                parsed = []
                for raw in pair:
                    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                        raise ValueError('Expected [user ID, tweet ID, delay]')
                    uid, tid, delay = raw
                    if str(uid).upper() == 'ROOT' or str(tid).upper() == 'ROOT':
                        parsed.append(None)
                        continue
                    ident = f'{uid}:{tid}'
                    delay = float(delay)
                    if not math.isfinite(delay):
                        raise ValueError('Nonfinite delay')
                    if delay < 0:
                        warnings.append(f'Line {line_number}: negative delay clamped to zero')
                    value = {'id': ident, 'tweet_id': str(tid), 'author_id': str(uid),
                             'delay': max(0., delay), 'text': sources.get(str(tid), '')}
                    if ident in nodes and nodes[ident]['delay'] != value['delay']:
                        warnings.append(f'Line {line_number}: conflicting time; kept earliest')
                        value['delay'] = min(value['delay'], nodes[ident]['delay'])
                    nodes[ident] = value
                    parsed.append(ident)
                a, b = parsed
                if a is None:
                    root_id = b
                elif b is not None and a != b:
                    edges.append([a, b])
            except (ValueError, SyntaxError, TypeError) as exc:
                warnings.append(f'Line {line_number}: {exc}')
        graph = {'id': uuid.uuid4().hex, 'cascade_id': file.stem,
                 'source': directory.name, 'nodes': list(nodes.values()), 'edges': edges,
                 'estimated': False, 'accepted': True, 'needs_estimation': False,
                 'warnings': warnings, 'rumor_label': labels.get(file.stem),
                 'source_text': sources.get(file.stem, ''), 'root_id': root_id}
        try:
            validate_graph(graph)
            if len(graph['roots']) != 1:
                raise ValueError('Expected a single connected cascade')
            graph['source_hash'] = fingerprint({'source_id': file.stem})
            yield graph
        except ValueError as exc:
            yield {'error': f'{file.name}: {exc}'}


def observed_prefix(graph, minutes):
    if graph.get('needs_estimation'):
        raise ValueError('Graph needs metadata estimation and review before prediction')
    if graph.get('estimated') and not graph.get('accepted'):
        raise ValueError('Review and accept the estimated graph before prediction')
    nodes = [n for n in graph['nodes'] if n.get('delay') is not None and n['delay'] <= minutes]
    if not nodes:
        raise ValueError('No posts fall within the model observation window')
    ids = {n['id'] for n in nodes}
    prefix = {**graph, 'nodes': nodes,
              'edges': [e for e in graph['edges'] if e[0] in ids and e[1] in ids]}
    return validate_graph(prefix)


def node_features(graph, window):
    """All features depend exclusively on the already-truncated observed prefix."""
    parents, children = defaultdict(list), defaultdict(list)
    for a, b in graph['edges']:
        children[a].append(b)
        parents[b].append(a)
    depth = dict.fromkeys(graph['roots'], 0)
    queue = list(graph['roots'])
    for ident in queue:
        for child in children[ident]:
            if child not in depth:
                depth[child] = depth[ident] + 1
                queue.append(child)
    return [[1., min(n['delay'] / max(window, 1e-6), 1.),
             math.log1p(len(parents[n['id']])), math.log1p(len(children[n['id']])),
             math.log1p(depth.get(n['id'], 0)), float(not parents[n['id']])]
            for n in graph['nodes']]
