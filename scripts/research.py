"""Audits, FEVER-only evaluation, feature extraction, and training comparisons."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/backend"))
from sentinel_x.research.audit import audit, audit_report
from sentinel_x.research.evidence import freeze_fever
from sentinel_x.research.reporting import score_run
from sentinel_x.research.corpus import prepare_corpus, cross_evaluate
from sentinel_x.research.download import fetch
from sentinel_x.research.execution import execute
from sentinel_x.research.runner import evaluate, extract, training_matrix, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("audit")
    a.add_argument("--cases", type=Path, default=Path("datasets/twitter16-model-cases"))
    a.add_argument("--sources", type=Path, default=Path(".cache/research/sources"))
    a.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model weights/fe44637dfc4d49fe9cd402c4ce6cc21d.json"),
    )
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--source-lock", type=Path, default=Path("docs/research/sources.json"))
    f = commands.add_parser("freeze-fever")
    f.add_argument("--index", type=Path, required=True)
    f.add_argument("--cases", type=Path, default=Path("datasets/twitter16-model-cases"))
    f.add_argument("--output", type=Path, required=True)
    for name in ("evaluate", "extract", "train-matrix", "cross-evaluate"):
        c = commands.add_parser(name)
        c.add_argument("--config", type=Path, required=True)
        c.add_argument("--output", type=Path, required=True)
    r = commands.add_parser("report")
    r.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--cases", type=Path, default=Path("datasets/twitter16-model-cases"))
    score.add_argument("--reviewers", nargs=3, type=Path)
    corpus = commands.add_parser("prepare-corpus")
    corpus.add_argument("--sources", type=Path, default=Path(".cache/research/sources"))
    corpus.add_argument("--output", type=Path, required=True)
    corpus.add_argument("--dataset", choices=["twitter15", "twitter16"], required=True)
    corpus.add_argument(
        "--provenance", type=Path, default=Path("datasets/twitter16-model-cases/PROVENANCE.json")
    )
    fetch_parser = commands.add_parser("fetch-sources")
    fetch_parser.add_argument("--lock", type=Path, default=Path("docs/research/sources.json"))
    fetch_parser.add_argument("--output", type=Path, default=Path(".cache/research/sources"))
    corpus.add_argument("--source-lock", type=Path, default=Path("docs/research/sources.json"))
    args = parser.parse_args()
    if args.command == "fetch-sources":
        print(json.dumps(fetch(args.lock, args.output)))
        return 0
    if args.command == "prepare-corpus":
        print(
            json.dumps(
                prepare_corpus(
                    args.sources, args.output, args.dataset, args.provenance, args.source_lock
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "audit":
        result = audit(args.cases, args.sources, args.checkpoint, args.source_lock)
        audit_report(result, args.output)
        print(result["status"])
        return int(result["status"] != "passed")
    if args.command == "freeze-fever":
        result = freeze_fever(args.index, args.cases, args.output)
        print(f"Frozen {len(result['passages'])} FEVER passages; human annotation pending")
        return 0
    if args.command == "score":
        print(json.dumps(score_run(args.output, args.cases, args.reviewers), indent=2))
        return 0
    if args.command == "report":
        result = summarize(args.output)
    else:
        config = json.loads(args.config.read_text())
        operation = {
            "evaluate": evaluate,
            "extract": extract,
            "train-matrix": training_matrix,
            "cross-evaluate": cross_evaluate,
        }[args.command]
        result = execute(args.command, config, args.output, operation)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
