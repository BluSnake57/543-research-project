"""Reproducible graph-level GraphSAGE binary virality training."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import uuid

import numpy as np

from .semantic import feature_cache_protocol
from .graphs import FEATURE_VERSION, fingerprint, node_features, observed_prefix


ALL_LLM_FEATURE_NAMES = (
    'urgency',
    'sensationalism',
    'emotional_language',
    'authority_framing',
    'conspiracy_framing',
)


def split_graphs(graphs, threshold=100, seed=42):
    groups = {}

    for graph in graphs:
        if graph.get('estimated'):
            raise ValueError(
                'Training requires observed Twitter15/16 cascades'
            )

        key = graph.get('cascade_id', graph['id'])

        if key in groups:
            # Keep the largest recorded version of an event.
            if len(graph['nodes']) > len(groups[key]['nodes']):
                groups[key] = graph
        else:
            groups[key] = graph

    buckets = {0: [], 1: []}

    for graph in groups.values():
        label = int(
            len(graph['nodes']) >= threshold
        )
        buckets[label].append(graph)

    rng = random.Random(seed)

    result = {
        'train': [],
        'validation': [],
        'test': [],
    }

    for label, bucket in buckets.items():
        bucket.sort(
            key=lambda g: g.get(
                'cascade_id',
                g['id'],
            )
        )

        rng.shuffle(bucket)

        if len(bucket) < 3:
            raise ValueError(
                f'Need at least 3 distinct cascades per class; '
                f'class {label} has {len(bucket)}'
            )

        n_test = max(
            1,
            round(len(bucket) * .15),
        )

        n_val = max(
            1,
            round(len(bucket) * .15),
        )

        result['test'].extend(
            bucket[:n_test]
        )

        result['validation'].extend(
            bucket[n_test:n_test + n_val]
        )

        result['train'].extend(
            bucket[n_test + n_val:]
        )

    return result


def scores(labels, probabilities, cutoff=.5):
    y = np.asarray(
        labels,
        dtype=int,
    )

    p = np.asarray(
        probabilities,
        dtype=float,
    )

    pred = p >= cutoff

    tp = int(
        np.sum((y == 1) & pred)
    )

    fp = int(
        np.sum((y == 0) & pred)
    )

    fn = int(
        np.sum((y == 1) & ~pred)
    )

    tn = int(
        np.sum((y == 0) & ~pred)
    )

    precision = (
        tp / max(tp + fp, 1)
    )

    recall = (
        tp / max(tp + fn, 1)
    )

    # Average precision with tied scores
    # evaluated as a group.
    ap = 0.
    previous_recall = 0.

    for level in sorted(
        set(p.tolist()),
        reverse=True,
    ):
        chosen = p >= level

        count = int(
            np.sum(
                (y == 1) & chosen
            )
        )

        current_recall = (
            count
            / max(int(sum(y)), 1)
        )

        ap += (
            current_recall
            - previous_recall
        ) * (
            count
            / max(
                int(sum(chosen)),
                1,
            )
        )

        previous_recall = (
            current_recall
        )

    return {
        'precision': precision,
        'recall': recall,
        'f1': (
            2
            * precision
            * recall
            / max(
                precision + recall,
                1e-12,
            )
        ),
        'pr_auc': ap,
        'brier': float(np.mean((p-y)**2)) if len(y) else None,
        'reliability': [
            {'lower': i/10, 'upper': (i+1)/10, 'count': int(mask.sum()),
             'mean_probability': float(p[mask].mean()) if mask.any() else None,
             'observed_frequency': float(y[mask].mean()) if mask.any() else None}
            for i in range(10)
            for mask in [(p >= i/10) & ((p < (i+1)/10) if i < 9 else (p <= 1))]
        ],
        'accuracy': (
            (tp + tn)
            / max(len(y), 1)
        ),
        'confusion_matrix': [
            [tn, fp],
            [fn, tp],
        ],
        'cutoff': float(cutoff),
        'count': len(y),
    }


def best_cutoff(
    labels,
    probabilities,
):
    options = sorted(
        set(
            [0.5, 0., 1.]
            + list(probabilities)
        )
    )

    return max(
        options,
        key=lambda c: (
            scores(
                labels,
                probabilities,
                c,
            )['f1'],
            -abs(c - .5),
        ),
    )


def load_llm_features(
    path,
    feature_names,
):
    if path is None:
        return None

    path = Path(path)

    if not path.is_file():
        raise ValueError(
            f'LLM feature file not found: {path}'
        )

    raw = json.loads(
        path.read_text(
            encoding='utf-8'
        )
    )

    if not isinstance(raw, dict):
        raise ValueError(
            'LLM feature file must contain '
            'a JSON object'
        )

    result = {}

    for tweet_id, record in raw.items():
        if not isinstance(
            record,
            dict,
        ):
            raise ValueError(
                f'Invalid LLM record for '
                f'{tweet_id}'
            )

        features = record.get(
            'features'
        )

        if not isinstance(
            features,
            dict,
        ):
            raise ValueError(
                f'Missing features for '
                f'{tweet_id}'
            )

        values = []

        for name in feature_names:
            if name not in features:
                raise ValueError(
                    f'{tweet_id} missing '
                    f'LLM feature {name}'
                )

            value = float(
                features[name]
            )

            if not math.isfinite(
                value
            ):
                raise ValueError(
                    f'Nonfinite LLM feature '
                    f'for {tweet_id}'
                )

            if not 0. <= value <= 1.:
                raise ValueError(
                    f'LLM feature outside '
                    f'[0,1] for {tweet_id}'
                )

            values.append(value)

        result[str(tweet_id)] = (
            np.asarray(
                values,
                dtype=np.float32,
            )
        )

    return result


def graph_llm_features(
    graph,
    feature_map,
):
    if feature_map is None:
        return None

    candidates = [
        str(
            graph.get(
                'cascade_id',
                '',
            )
        ),
        str(
            graph.get(
                'root_id',
                '',
            )
        ),
    ]

    for candidate in candidates:
        if (
            candidate
            and candidate in feature_map
        ):
            return feature_map[
                candidate
            ]

    root_id = graph.get(
        'root_id'
    )

    if root_id:
        for node in graph.get(
            'nodes',
            [],
        ):
            if (
                node.get('id')
                == root_id
            ):
                tweet_id = str(
                    node.get(
                        'tweet_id',
                        '',
                    )
                )

                if (
                    tweet_id
                    in feature_map
                ):
                    return (
                        feature_map[
                            tweet_id
                        ]
                    )

    raise ValueError(
        'No Qwen features found for '
        f"cascade "
        f"{graph.get('cascade_id', graph.get('id'))}"
    )


def torch_modules(
    llm_feature_count=0,
):
    try:
        import torch

    except ImportError as exc:
        raise RuntimeError(
            'Training requires PyTorch; '
            'install the backend dependencies'
        ) from exc

    class SAGEConv(
        torch.nn.Module
    ):
        """Mean GraphSAGE aggregation."""

        def __init__(
            self,
            inputs,
            outputs,
        ):
            super().__init__()

            self.neighbor = (
                torch.nn.Linear(
                    inputs,
                    outputs,
                )
            )

            self.root = (
                torch.nn.Linear(
                    inputs,
                    outputs,
                    bias=False,
                )
            )

        def forward(
            self,
            x,
            edges,
        ):
            neighbors = (
                torch.zeros_like(x)
            )

            degrees = torch.zeros(
                (len(x), 1),
                dtype=x.dtype,
                device=x.device,
            )

            if edges.shape[1]:
                (
                    source,
                    destination,
                ) = edges

                neighbors.index_add_(
                    0,
                    destination,
                    x[source],
                )

                degrees.index_add_(
                    0,
                    destination,
                    torch.ones(
                        (
                            len(
                                destination
                            ),
                            1,
                        ),
                        dtype=x.dtype,
                        device=x.device,
                    ),
                )

            return (
                self.neighbor(
                    neighbors
                    / degrees.clamp(
                        min=1
                    )
                )
                + self.root(x)
            )

    def global_mean_pool(
        x,
        batch,
    ):
        count = (
            int(batch.max())
            + 1
        )

        pooled = torch.zeros(
            (
                count,
                x.shape[1],
            ),
            dtype=x.dtype,
            device=x.device,
        )

        pooled.index_add_(
            0,
            batch,
            x,
        )

        sizes = (
            torch.bincount(
                batch,
                minlength=count,
            )
            .to(x.dtype)
            .clamp(min=1)
        )

        return (
            pooled
            / sizes.unsqueeze(1)
        )

    class GraphSAGE(
        torch.nn.Module
    ):
        def __init__(
            self,
            hidden=32,
        ):
            super().__init__()

            self.conv1 = SAGEConv(
                6,
                hidden,
            )

            self.conv2 = SAGEConv(
                hidden,
                hidden,
            )

            self.head = (
                torch.nn.Linear(
                    hidden
                    + llm_feature_count,
                    1,
                )
            )

        def forward(
            self,
            x,
            edges,
            batch,
            llm_features=None,
        ):
            x = torch.relu(
                self.conv1(
                    x,
                    edges,
                )
            )

            x = torch.relu(
                self.conv2(
                    x,
                    edges,
                )
            )

            pooled = (
                global_mean_pool(
                    x,
                    batch,
                )
            )

            if (
                llm_feature_count
                > 0
            ):
                if (
                    llm_features
                    is None
                ):
                    raise ValueError(
                        'Qwen features '
                        'required for '
                        'this model'
                    )

                if (
                    llm_features.ndim
                    == 1
                ):
                    llm_features = (
                        llm_features
                        .unsqueeze(0)
                    )

                pooled = torch.cat(
                    [
                        pooled,
                        llm_features,
                    ],
                    dim=1,
                )

            return (
                self.head(
                    pooled
                )
                .view(-1)
            )

    return torch, GraphSAGE


def tensor_graph(
    torch,
    graph,
    window,
    mean,
    scale,
    device,
    llm_features=None,
    llm_mean=None,
    llm_scale=None,
):
    prefix = observed_prefix(
        graph,
        window,
    )

    features = np.asarray(
        node_features(
            prefix,
            window,
        ),
        dtype=np.float32,
    )

    features = (
        features - mean
    ) / scale

    indices = {
        node['id']: index
        for index, node
        in enumerate(
            prefix['nodes']
        )
    }

    pairs = [
        (
            indices[a],
            indices[b],
        )
        for a, b
        in prefix['edges']
    ]

    # Bidirectional message passing.
    pairs += [
        (b, a)
        for a, b
        in pairs
    ]

    edges = torch.tensor(
        pairs,
        dtype=torch.long,
        device=device,
    ).reshape(
        -1,
        2,
    ).T.contiguous()

    result = [
        torch.tensor(
            features,
            dtype=torch.float32,
            device=device,
        ),
        edges,
        torch.zeros(
            len(features),
            dtype=torch.long,
            device=device,
        ),
    ]

    if llm_features is not None:
        values = (
            llm_features
            - llm_mean
        ) / llm_scale

        result.append(
            torch.tensor(
                values,
                dtype=torch.float32,
                device=device,
            )
        )

    return tuple(result)


def train(
    graphs,
    output: Path,
    config=None,
    job=None,
):
    config = {
        'threshold': 100,
        'window': 60.,
        'epochs': 80,
        'seed': 42,
        'hidden': 32,
        'learning_rate': .003,
        'device': 'cpu',
        'llm_features': None,
        'llm_feature': None,
        **(config or {}),
    }

    if (
        config['threshold'] < 2
        or not math.isfinite(
            config['window']
        )
        or config['window'] <= 0
        or not (
            1
            <= config['epochs']
            <= 2000
        )
    ):
        raise ValueError(
            'Require threshold >=2, '
            'positive window, and '
            '1-2000 epochs'
        )

    if config['device'] not in (
        'cpu',
        'cuda',
        'mps',
    ):
        raise ValueError(
            'Device must be cpu, '
            'cuda, or mps'
        )

    selected_feature = (
        config.get(
            'llm_feature'
        )
    )

    if selected_feature:
        if (
            selected_feature
            not in
            ALL_LLM_FEATURE_NAMES
        ):
            raise ValueError(
                'Unknown LLM feature: '
                f'{selected_feature}'
            )

        feature_names = (
            selected_feature,
        )

    else:
        feature_names = (
            ALL_LLM_FEATURE_NAMES
        )

    use_llm = bool(
        config.get(
            'llm_features'
        )
    )

    if (
        selected_feature
        and not use_llm
    ):
        raise ValueError(
            '--llm-feature requires '
            '--llm-features'
        )

    if not use_llm:
        feature_names = ()

    feature_map = (
        load_llm_features(
            config.get(
                'llm_features'
            ),
            feature_names,
        )
        if use_llm
        else None
    )

    llm_feature_count = len(
        feature_names
    )

    splits = split_graphs(
        graphs,
        config['threshold'],
        config.get('split_seed', config['seed']),
    )

    torch, model_class = (
        torch_modules(
            llm_feature_count
        )
    )

    torch.manual_seed(
        config['seed']
    )

    random.seed(
        config['seed']
    )

    np.random.seed(
        config['seed']
    )

    if (
        config['device']
        == 'cpu'
    ):
        torch.set_num_threads(
            max(
                1,
                min(
                    torch.get_num_threads(),
                    4,
                ),
            )
        )

        torch.use_deterministic_algorithms(
            True
        )

    device = torch.device(
        config['device']
    )

    prefix_features = (
        np.concatenate([
            np.asarray(
                node_features(
                    observed_prefix(
                        graph,
                        config[
                            'window'
                        ],
                    ),
                    config[
                        'window'
                    ],
                ),
                dtype=np.float32,
            )
            for graph
            in splits['train']
        ])
    )

    mean = (
        prefix_features
        .mean(axis=0)
    )

    scale = (
        prefix_features
        .std(axis=0)
    )

    scale[
        scale < 1e-6
    ] = 1.

    llm_mean = None
    llm_scale = None

    if use_llm:
        training_llm = (
            np.asarray([
                graph_llm_features(
                    graph,
                    feature_map,
                )
                for graph
                in splits['train']
            ], dtype=np.float32)
        )

        llm_mean = (
            training_llm
            .mean(axis=0)
        )

        llm_scale = (
            training_llm
            .std(axis=0)
        )

        llm_scale[
            llm_scale < 1e-6
        ] = 1.

    tensors = {}

    for (
        key,
        values,
    ) in splits.items():

        tensors[key] = []

        for graph in values:

            semantic = (
                graph_llm_features(
                    graph,
                    feature_map,
                )
                if use_llm
                else None
            )

            args = tensor_graph(
                torch,
                graph,
                config['window'],
                mean,
                scale,
                device,
                semantic,
                llm_mean,
                llm_scale,
            )

            label = float(
                len(
                    graph['nodes']
                )
                >= config[
                    'threshold'
                ]
            )

            tensors[key].append(
                (
                    args,
                    label,
                )
            )

    model = model_class(
        config['hidden']
    ).to(device)

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=config[
                'learning_rate'
            ],
        )
    )

    positive = sum(
        label
        for _, label
        in tensors['train']
    )

    criterion = (
        torch.nn
        .BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                (
                    len(
                        tensors[
                            'train'
                        ]
                    )
                    - positive
                )
                / max(
                    positive,
                    1,
                ),
                device=device,
            )
        )
    )

    best_loss = float(
        'inf'
    )

    best_state = None
    history = []
    patience = 0

    for epoch in range(
        config['epochs']
    ):
        if job:
            job.check()

        model.train()

        total_loss = 0.

        order = list(
            range(
                len(
                    tensors[
                        'train'
                    ]
                )
            )
        )

        random.shuffle(
            order
        )

        for i in order:
            if job:
                job.check()

            (
                args,
                label,
            ) = tensors[
                'train'
            ][i]

            optimizer.zero_grad()

            loss = criterion(
                model(*args),
                torch.tensor(
                    [label],
                    device=device,
                ),
            )

            loss.backward()

            optimizer.step()

            total_loss += float(
                loss.item()
            )

        model.eval()

        with torch.no_grad():
            validation_loss = (
                sum(
                    float(
                        criterion(
                            model(
                                *args
                            ),
                            torch.tensor(
                                [label],
                                device=device,
                            ),
                        )
                    )
                    for (
                        args,
                        label,
                    )
                    in tensors[
                        'validation'
                    ]
                )
                / len(
                    tensors[
                        'validation'
                    ]
                )
            )

        history.append({
            'epoch':
                epoch + 1,
            'loss':
                total_loss
                / len(order),
            'validation_loss':
                validation_loss,
        })

        if job:
            job.emit(
                'training',
                **history[-1],
            )

        if (
            validation_loss
            < best_loss - 1e-6
        ):
            best_loss = (
                validation_loss
            )

            patience = 0

            best_state = {
                key: value
                .detach()
                .cpu()
                .clone()
                for (
                    key,
                    value,
                )
                in model
                .state_dict()
                .items()
            }

        else:
            patience += 1

        if patience >= 12:
            break

    if best_state is None:
        raise RuntimeError(
            'Training failed to '
            'produce a checkpoint'
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    def evaluate(key):
        with torch.no_grad():

            labels = [
                label
                for _, label
                in tensors[key]
            ]

            probabilities = [
                float(
                    torch.sigmoid(
                        model(*args)
                    )[0]
                )
                for args, _
                in tensors[key]
            ]

        return (
            labels,
            probabilities,
        )

    (
        val_y,
        val_p,
    ) = evaluate(
        'validation'
    )

    cutoff = best_cutoff(
        val_y,
        val_p,
    )

    (
        test_y,
        test_p,
    ) = evaluate(
        'test'
    )

    majority = int(
        positive
        >= (
            len(
                tensors[
                    'train'
                ]
            )
            / 2
        )
    )

    size_p = [
        min(
            len(
                observed_prefix(
                    graph,
                    config[
                        'window'
                    ],
                )[
                    'nodes'
                ]
            )
            / config[
                'threshold'
            ],
            1.,
        )
        for graph
        in splits['test']
    ]

    val_sizes = [
        min(
            len(
                observed_prefix(
                    graph,
                    config[
                        'window'
                    ],
                )[
                    'nodes'
                ]
            )
            / config[
                'threshold'
            ],
            1.,
        )
        for graph
        in splits[
            'validation'
        ]
    ]

    report = {
        'validation': scores(
            val_y,
            val_p,
            cutoff,
        ),

        'test': scores(
            test_y,
            test_p,
            cutoff,
        ),

        'majority_baseline':
            scores(
                test_y,
                [majority]
                * len(test_y),
            ),

        'observed_size_baseline':
            scores(
                test_y,
                size_p,
                best_cutoff(
                    val_y,
                    val_sizes,
                ),
            ),

        'history': history,

        'warning': (
            'Target is recorded '
            'cascade size, not '
            'lifetime worldwide '
            'reach. Estimated '
            'graphs have not been '
            'validated by this '
            'evaluation.'
        ),
    }

    ident = uuid.uuid4().hex

    config_saved = dict(
        config
    )

    if use_llm:
        config_saved[
            'llm_features'
        ] = str(
            config[
                'llm_features'
            ]
        )

    if not use_llm:
        model_variant = (
            'graphsage'
        )

    elif (
        len(feature_names)
        == 1
    ):
        model_variant = (
            'graphsage_qwen_ablation'
        )

    else:
        model_variant = (
            'graphsage_qwen'
        )

    metadata = {
        'feature_protocol': feature_cache_protocol(config['llm_features']) if use_llm else None,
        'id': ident,

        'feature_version':
            FEATURE_VERSION,

        'model_variant':
            model_variant,

        'llm_feature_names':
            list(
                feature_names
            ),

        'config':
            config_saved,

        'mean':
            mean.tolist(),

        'scale':
            scale.tolist(),

        'llm_mean': (
            llm_mean.tolist()
            if use_llm
            else None
        ),

        'llm_scale': (
            llm_scale.tolist()
            if use_llm
            else None
        ),

        'cutoff':
            cutoff,

        'splits': {
            key: [
                graph.get(
                    'cascade_id',
                    graph['id'],
                )
                for graph
                in value
            ]
            for (
                key,
                value,
            )
            in splits.items()
        },

        'dataset_fingerprint':
            fingerprint(
                sorted(
                    fingerprint({
                        key:
                            graph[key]
                        for key in (
                            'cascade_id',
                            'nodes',
                            'edges',
                        )
                    })
                    for graph
                    in graphs
                )
            ),

        'report':
            report,
    }

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights_temp = (
        output
        / f'{ident}.pt.tmp'
    )

    metadata_temp = (
        output
        / f'{ident}.json.tmp'
    )

    torch.save(
        best_state,
        weights_temp,
    )

    metadata_temp.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding='utf-8',
    )

    weights_temp.replace(
        output
        / f'{ident}.pt'
    )

    metadata_temp.replace(
        output
        / f'{ident}.json'
    )

    return metadata


def predict(
    graphs,
    checkpoint: Path,
    llm_features=None,
):
    metadata = json.loads(
        checkpoint.read_text(
            encoding='utf-8'
        )
    )

    if (
        metadata[
            'feature_version'
        ]
        != FEATURE_VERSION
    ):
        raise ValueError(
            'Checkpoint feature '
            'version is incompatible'
        )

    feature_names = tuple(
        metadata.get(
            'llm_feature_names',
            [],
        )
    )

    use_llm = bool(
        feature_names
    )

    feature_map = None

    if use_llm:
        source = (
            llm_features
            or metadata[
                'config'
            ].get(
                'llm_features'
            )
        )

        if not source:
            raise ValueError(
                'This checkpoint '
                'requires Qwen '
                'semantic features'
            )

        feature_map = (
            load_llm_features(
                source,
                feature_names,
            )
        )

    torch, model_class = (
        torch_modules(
            len(feature_names)
        )
    )

    model = model_class(
        metadata[
            'config'
        ]['hidden']
    )

    model.load_state_dict(
        torch.load(
            checkpoint.with_suffix(
                '.pt'
            ),
            map_location='cpu',
            weights_only=True,
        )
    )

    model.eval()

    compatibility = []
    if use_llm:
        incoming = feature_cache_protocol(source)
        if metadata.get('feature_protocol') is None or incoming is None:
            compatibility.append('Semantic extraction provenance is unknown for a legacy cache/checkpoint; compatibility is unverified.')
        elif incoming != metadata['feature_protocol']:
            compatibility.append('Semantic extraction protocol differs from checkpoint training; treat this run as exploratory.')
    outputs = []

    with torch.no_grad():

        for graph in graphs:

            semantic = (
                graph_llm_features(
                    graph,
                    feature_map,
                )
                if use_llm
                else None
            )

            args = tensor_graph(
                torch,
                graph,
                metadata[
                    'config'
                ]['window'],
                np.asarray(
                    metadata[
                        'mean'
                    ]
                ),
                np.asarray(
                    metadata[
                        'scale'
                    ]
                ),
                'cpu',
                semantic,
                (
                    np.asarray(
                        metadata[
                            'llm_mean'
                        ]
                    )
                    if use_llm
                    else None
                ),
                (
                    np.asarray(
                        metadata[
                            'llm_scale'
                        ]
                    )
                    if use_llm
                    else None
                ),
            )

            probability = float(
                torch.sigmoid(
                    model(*args)
                )[0]
            )

            prefix = observed_prefix(graph, metadata['config']['window'])
            outputs.append({
                'observed_nodes': len(prefix['nodes']),
                'observed_edges': len(prefix['edges']),
                'semantic_features': dict(zip(feature_names, map(float, semantic))) if use_llm else {},
                'graph_id':
                    graph['id'],

                'cascade_id':
                    graph[
                        'cascade_id'
                    ],

                'viral_probability':
                    probability,

                'viral': (
                    probability
                    >= metadata[
                        'cutoff'
                    ]
                ),

                'decision_cutoff':
                    metadata[
                        'cutoff'
                    ],

                'model_id':
                    metadata['id'],

                'model_variant':
                    metadata.get(
                        'model_variant',
                        'graphsage',
                    ),

                'llm_feature_names':
                    list(
                        feature_names
                    ),

                'observation_minutes':
                    metadata[
                        'config'
                    ]['window'],

                'size_threshold':
                    metadata[
                        'config'
                    ]['threshold'],

                'estimated':
                    graph.get(
                        'estimated',
                        False,
                    ),

                'warnings':
                    graph.get(
                        'warnings',
                        [],
                    ) + compatibility,

                'provenance': {
                    'source':
                        graph[
                            'source'
                        ],

                    'source_hash':
                        graph.get(
                            'source_hash'
                        ),
                },
            })

    return outputs