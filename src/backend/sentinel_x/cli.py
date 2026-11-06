from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .state import Store, data_directory


LLM_FEATURE_CHOICES = [
    'urgency',
    'sensationalism',
    'emotional_language',
    'authority_framing',
    'conspiracy_framing',
]


def main():
    parser = argparse.ArgumentParser(
        prog='sentinel-x'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=data_directory(),
    )

    sub = parser.add_subparsers(
        dest='command'
    )

    sub.add_parser('gui')
    sub.add_parser('trainer')

    export = sub.add_parser(
        'export'
    )

    export.add_argument(
        'conversation'
    )

    export.add_argument(
        'output',
        type=Path,
    )

    train = sub.add_parser(
        'train'
    )

    train.add_argument(
        'dataset',
        type=Path,
    )

    train.add_argument(
        '--threshold',
        type=int,
        default=100,
    )

    train.add_argument(
        '--window',
        type=float,
        default=60.,
    )

    train.add_argument(
        '--epochs',
        type=int,
        default=80,
    )

    train.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    train.add_argument(
        '--device',
        choices=[
            'cpu',
            'cuda',
            'mps',
        ],
        default='cpu',
    )

    train.add_argument(
        '--llm-features',
        type=Path,
        default=None,
        help=(
            'Qwen semantic feature JSON. '
            'If omitted, train the original '
            'graph-only GraphSAGE model.'
        ),
    )

    train.add_argument(
        '--llm-feature',
        choices=LLM_FEATURE_CHOICES,
        default=None,
        help=(
            'Use only one Qwen semantic '
            'feature for ablation testing. '
            'Requires --llm-features.'
        ),
    )

    train.add_argument('--hidden', type=int, choices=(32, 64, 128), default=32)
    train.add_argument('--split-seed', type=int, default=42)

    predict = sub.add_parser(
        'predict'
    )

    predict.add_argument(
        'graph',
        type=Path,
    )

    predict.add_argument(
        'checkpoint',
        type=Path,
    )

    predict.add_argument(
        '--llm-features',
        type=Path,
        default=None,
        help=(
            'Qwen feature JSON required '
            'when using a Qwen-enhanced '
            'checkpoint.'
        ),
    )

    index = sub.add_parser(
        'index'
    )

    index.add_argument(
        'directory',
        type=Path,
    )

    args = parser.parse_args()

    if args.command in (
        None,
        'gui',
    ):
        from .desktop import (
            main as desktop
        )

        return desktop(
            args.data_dir
        )

    if args.command == 'trainer':
        from .trainer import (
            main as trainer
        )

        return trainer(
            args.data_dir
        )

    store = Store(
        args.data_dir
    )

    if args.command == 'export':

        from .exporting import (
            export_conversation
        )

        print(
            export_conversation(
                store,
                args.conversation,
                args.output,
            )
        )

    elif args.command == 'train':

        from .graphs import (
            native_twitter
        )

        from .learning import train

        if (
            args.llm_feature
            and not args.llm_features
        ):
            parser.error(
                '--llm-feature requires '
                '--llm-features'
            )

        if (
            args.dataset
            / 'tree'
        ).is_dir():

            folders = [
                args.dataset
            ]

        else:

            folders = [
                args.dataset
                / name

                for name in (
                    'twitter15',
                    'twitter16',
                )

                if (
                    args.dataset
                    / name
                    / 'tree'
                ).is_dir()
            ]

        if not folders:
            parser.error(
                'No native Twitter15/16 '
                'directories found'
            )

        graphs = []

        for folder in folders:

            for graph in (
                native_twitter(
                    folder
                )
            ):

                if (
                    'error'
                    in graph
                ):
                    print(
                        graph[
                            'error'
                        ],
                        file=sys.stderr,
                    )

                else:
                    graphs.append(
                        graph
                    )

        config = {
            'threshold':
                args.threshold,

            'window':
                args.window,

            'hidden': args.hidden,
            'split_seed': args.split_seed,
            'epochs':
                args.epochs,

            'seed':
                args.seed,

            'device':
                args.device,

            'llm_features': (
                str(
                    args.llm_features
                )
                if args.llm_features
                else None
            ),

            'llm_feature':
                args.llm_feature,
        }

        result = train(
            graphs,
            store.root
            / 'checkpoints',
            config,
        )

        store.configure(
            'checkpoint',
            result['id'],
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    elif args.command == 'predict':

        from .learning import (
            predict
        )

        value = json.loads(
            args.graph.read_text(
                encoding='utf-8'
            )
        )

        graphs = (
            value
            if isinstance(
                value,
                list,
            )
            else [value]
        )

        result = predict(
            graphs,
            args.checkpoint,
            llm_features=(
                args.llm_features
                if args.llm_features
                else None
            ),
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    elif args.command == 'index':

        from .retrieval import (
            Retrieval
        )

        print(
            json.dumps(
                Retrieval(
                    store
                ).index(
                    args.directory
                ),
                indent=2,
            )
        )


if __name__ == '__main__':
    main()