"""Reproducible graph-level GraphSAGE binary virality training."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import uuid

import numpy as np

from .graphs import FEATURE_VERSION, fingerprint, node_features, observed_prefix


def split_graphs(graphs, threshold=100, seed=42):
    groups = {}
    for graph in graphs:
        if graph.get('estimated'):
            raise ValueError('Training requires observed Twitter15/16 cascades')
        key = graph.get('cascade_id', graph['id'])
        if key in groups:
            # Keep the largest recorded version of an event; never split duplicate events.
            if len(graph['nodes']) > len(groups[key]['nodes']):
                groups[key] = graph
        else:
            groups[key] = graph
    buckets = {0: [], 1: []}
    for graph in groups.values():
        buckets[int(len(graph['nodes']) >= threshold)].append(graph)
    rng = random.Random(seed)
    result = {'train': [], 'validation': [], 'test': []}
    for label, bucket in buckets.items():
        bucket.sort(key=lambda g: g.get('cascade_id', g['id']))
        rng.shuffle(bucket)
        if len(bucket) < 3:
            raise ValueError(f'Need at least 3 distinct cascades per class; class {label} has {len(bucket)}')
        n_test = max(1, round(len(bucket) * .15))
        n_val = max(1, round(len(bucket) * .15))
        result['test'].extend(bucket[:n_test])
        result['validation'].extend(bucket[n_test:n_test+n_val])
        result['train'].extend(bucket[n_test+n_val:])
    return result


def scores(labels, probabilities, cutoff=.5):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = p >= cutoff
    tp = int(np.sum((y == 1) & pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tn = int(np.sum((y == 0) & ~pred))
    precision = tp / max(tp+fp, 1); recall = tp / max(tp+fn, 1)
    # Average precision with tied scores evaluated as a group.
    ap = 0.; previous_recall = 0.
    for level in sorted(set(p.tolist()), reverse=True):
        chosen = p >= level
        count = int(np.sum((y == 1) & chosen))
        current_recall = count / max(int(sum(y)), 1)
        ap += (current_recall - previous_recall) * count / max(int(sum(chosen)), 1)
        previous_recall = current_recall
    return {'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/max(precision+recall, 1e-12), 'pr_auc': ap,
            'accuracy': (tp+tn)/max(len(y), 1), 'confusion_matrix': [[tn, fp], [fn, tp]],
            'cutoff': float(cutoff), 'count': len(y)}


def best_cutoff(labels, probabilities):
    options = sorted(set([.5, 0., 1.] + list(probabilities)))
    return max(options, key=lambda c: (scores(labels, probabilities, c)['f1'], -abs(c-.5)))


def torch_modules():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Training requires PyTorch; install the backend dependencies') from exc

    class SAGEConv(torch.nn.Module):
        """Mean GraphSAGE: W_neighbor mean(neighbors) + W_root x, without optional extensions."""
        def __init__(self, inputs, outputs):
            super().__init__()
            self.neighbor = torch.nn.Linear(inputs, outputs)
            self.root = torch.nn.Linear(inputs, outputs, bias=False)

        def forward(self, x, edges):
            neighbors = torch.zeros_like(x)
            degrees = torch.zeros((len(x), 1), dtype=x.dtype, device=x.device)
            if edges.shape[1]:
                source, destination = edges
                neighbors.index_add_(0, destination, x[source])
                degrees.index_add_(0, destination, torch.ones((len(destination), 1),
                                                             dtype=x.dtype, device=x.device))
            return self.neighbor(neighbors / degrees.clamp(min=1)) + self.root(x)

    def global_mean_pool(x, batch):
        count = int(batch.max()) + 1
        pooled = torch.zeros((count, x.shape[1]), dtype=x.dtype, device=x.device)
        pooled.index_add_(0, batch, x)
        sizes = torch.bincount(batch, minlength=count).to(x.dtype).clamp(min=1)
        return pooled / sizes.unsqueeze(1)

    class GraphSAGE(torch.nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.conv1 = SAGEConv(6, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.head = torch.nn.Linear(hidden, 1)

        def forward(self, x, edges, batch):
            x = torch.relu(self.conv1(x, edges))
            x = torch.relu(self.conv2(x, edges))
            return self.head(global_mean_pool(x, batch)).view(-1)
    return torch, GraphSAGE


def tensor_graph(torch, graph, window, mean, scale, device):
    prefix = observed_prefix(graph, window)
    features = (np.asarray(node_features(prefix, window), dtype=np.float32) - mean) / scale
    indices = {n['id']: i for i, n in enumerate(prefix['nodes'])}
    pairs = [(indices[a], indices[b]) for a, b in prefix['edges']]
    # Bidirectional message passing preserves propagation features in x.
    pairs += [(b, a) for a, b in pairs]
    edges = torch.tensor(pairs, dtype=torch.long, device=device).reshape(-1, 2).T.contiguous()
    return (torch.tensor(features, dtype=torch.float32, device=device), edges,
            torch.zeros(len(features), dtype=torch.long, device=device))


def train(graphs, output: Path, config=None, job=None):
    config = {'threshold': 100, 'window': 60., 'epochs': 80, 'seed': 42,
              'hidden': 32, 'learning_rate': .003, 'device': 'cpu', **(config or {})}
    if config['threshold'] < 2 or not math.isfinite(config['window']) or config['window'] <= 0 or not 1 <= config['epochs'] <= 2000:
        raise ValueError('Require threshold >=2, positive window, and 1–2000 epochs')
    if config['device'] not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Device must be cpu, cuda, or mps')
    splits = split_graphs(graphs, config['threshold'], config['seed'])
    torch, model_class = torch_modules()
    torch.manual_seed(config['seed']); random.seed(config['seed']); np.random.seed(config['seed'])
    if config['device'] == 'cpu':
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
        torch.use_deterministic_algorithms(True)
    device = torch.device(config['device'])
    prefix_features = np.concatenate([np.array(node_features(observed_prefix(g, config['window']),
                                                             config['window']), dtype=np.float32)
                                      for g in splits['train']])
    mean = prefix_features.mean(axis=0); scale = prefix_features.std(axis=0)
    scale[scale < 1e-6] = 1.
    tensors = {key: [(tensor_graph(torch, g, config['window'], mean, scale, device),
                      float(len(g['nodes']) >= config['threshold'])) for g in value]
               for key, value in splits.items()}
    model = model_class(config['hidden']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    positive = sum(label for _, label in tensors['train'])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        (len(tensors['train'])-positive)/max(positive, 1), device=device))
    best_loss = float('inf'); best_state = None; history = []; patience = 0
    for epoch in range(config['epochs']):
        if job:
            job.check()
        model.train(); total_loss = 0.
        order = list(range(len(tensors['train']))); random.shuffle(order)
        for i in order:
            if job:
                job.check()
            args, label = tensors['train'][i]
            optimizer.zero_grad()
            loss = criterion(model(*args), torch.tensor([label], device=device))
            loss.backward(); optimizer.step(); total_loss += float(loss.item())
        model.eval()
        with torch.no_grad():
            validation_loss = sum(float(criterion(model(*args), torch.tensor([label], device=device)))
                                  for args, label in tensors['validation']) / len(tensors['validation'])
        history.append({'epoch': epoch+1, 'loss': total_loss/len(order),
                        'validation_loss': validation_loss})
        if job:
            job.emit('training', **history[-1])
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss; patience = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 12:
            break
    model.load_state_dict(best_state); model.eval()

    def evaluate(key):
        with torch.no_grad():
            return [label for _, label in tensors[key]], [float(torch.sigmoid(model(*args))[0])
                                                          for args, _ in tensors[key]]
    val_y, val_p = evaluate('validation'); cutoff = best_cutoff(val_y, val_p)
    test_y, test_p = evaluate('test')
    majority = int(positive >= len(tensors['train'])/2)
    size_p = [min(len(observed_prefix(g, config['window'])['nodes']) / config['threshold'], 1.)
              for g in splits['test']]
    # Tune the observed-size baseline only on validation data too.
    val_sizes = [min(len(observed_prefix(g, config['window'])['nodes']) / config['threshold'], 1.)
                 for g in splits['validation']]
    report = {'validation': scores(val_y, val_p, cutoff), 'test': scores(test_y, test_p, cutoff),
              'majority_baseline': scores(test_y, [majority]*len(test_y)),
              'observed_size_baseline': scores(test_y, size_p, best_cutoff(val_y, val_sizes)),
              'history': history,
              'warning': 'Target is recorded cascade size, not lifetime worldwide reach. '
                         'Estimated graphs have not been validated by this evaluation.'}
    ident = uuid.uuid4().hex
    metadata = {'id': ident, 'feature_version': FEATURE_VERSION, 'config': config,
                'mean': mean.tolist(), 'scale': scale.tolist(), 'cutoff': cutoff,
                'splits': {key: [g.get('cascade_id', g['id']) for g in value]
                           for key, value in splits.items()},
                'dataset_fingerprint': fingerprint(sorted(fingerprint({k: g[k] for k in
                    ('cascade_id', 'nodes', 'edges')}) for g in graphs)), 'report': report}
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / f'{ident}.pt')
    (output / f'{ident}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def predict(graphs, checkpoint: Path):
    metadata = json.loads(checkpoint.read_text(encoding='utf-8'))
    if metadata['feature_version'] != FEATURE_VERSION:
        raise ValueError('Checkpoint feature version is incompatible')
    torch, model_class = torch_modules()
    model = model_class(metadata['config']['hidden'])
    model.load_state_dict(torch.load(checkpoint.with_suffix('.pt'), map_location='cpu', weights_only=True))
    model.eval(); outputs = []
    with torch.no_grad():
        for graph in graphs:
            args = tensor_graph(torch, graph, metadata['config']['window'],
                                np.array(metadata['mean']), np.array(metadata['scale']), 'cpu')
            probability = float(torch.sigmoid(model(*args))[0])
            outputs.append({'graph_id': graph['id'], 'cascade_id': graph['cascade_id'],
                'viral_probability': probability, 'viral': probability >= metadata['cutoff'],
                'decision_cutoff': metadata['cutoff'], 'model_id': metadata['id'],
                'observation_minutes': metadata['config']['window'],
                'size_threshold': metadata['config']['threshold'],
                'estimated': graph.get('estimated', False), 'warnings': graph.get('warnings', []),
                'provenance': {'source': graph['source'], 'source_hash': graph.get('source_hash')}})
    return outputs
