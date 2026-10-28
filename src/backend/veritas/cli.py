from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

from .state import Store, data_directory


def main():
    parser = argparse.ArgumentParser(prog='veritas')
    parser.add_argument('--data-dir', type=Path, default=data_directory())
    sub = parser.add_subparsers(dest='command', required=True)
    worker = sub.add_parser('worker')
    worker.add_argument('--port', type=int, default=0)
    worker.add_argument('--manifest', type=Path, required=True)
    worker.add_argument('--skills', type=Path, required=True)
    worker.add_argument('--runtime', type=Path)
    train = sub.add_parser('train')
    train.add_argument('dataset', type=Path)
    train.add_argument('--threshold', type=int, default=100)
    train.add_argument('--window', type=float, default=60.)
    train.add_argument('--epochs', type=int, default=80)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    predict = sub.add_parser('predict')
    predict.add_argument('graph', type=Path)
    predict.add_argument('checkpoint', type=Path)
    index = sub.add_parser('index'); index.add_argument('directory', type=Path)
    args = parser.parse_args()
    store = Store(args.data_dir)
    if args.command == 'worker':
        import uvicorn
        from .api import create_app
        token = os.environ.get('VERITAS_WORKER_TOKEN')
        if not token or len(token) < 32:
            parser.error('VERITAS_WORKER_TOKEN must contain at least 32 characters')
        app = create_app(args.data_dir, token, args.manifest, args.skills, args.runtime)
        sock = socket.socket(); sock.bind(('127.0.0.1', args.port)); sock.listen(128)
        print(json.dumps({'port': sock.getsockname()[1]}), flush=True)
        uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False)).run(sockets=[sock])
    elif args.command == 'train':
        from .graphs import native_twitter
        from .learning import train
        folders = [args.dataset] if (args.dataset/'tree').is_dir() else [
            args.dataset/name for name in ('twitter15', 'twitter16') if (args.dataset/name/'tree').is_dir()]
        if not folders:
            parser.error('No native Twitter15/16 directories found')
        graphs = []
        for folder in folders:
            for graph in native_twitter(folder):
                if 'error' in graph:
                    print(graph['error'], file=sys.stderr)
                else:
                    graphs.append(graph)
        result = train(graphs, store.root/'checkpoints', {key: getattr(args, key) for key in
            ('threshold', 'window', 'epochs', 'seed', 'device')})
        store.configure('checkpoint', result['id'])
        print(json.dumps(result, indent=2))
    elif args.command == 'predict':
        from .learning import predict
        value = json.loads(args.graph.read_text())
        print(json.dumps(predict(value if isinstance(value, list) else [value], args.checkpoint), indent=2))
    elif args.command == 'index':
        from .retrieval import Retrieval
        print(json.dumps(Retrieval(store).index(args.directory), indent=2))


if __name__ == '__main__':
    main()
