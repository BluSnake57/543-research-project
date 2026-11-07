"""Collect all 15 development cases sequentially; score the saved results later."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/backend"))

from sentinel_x.research.audit import write_json
from sentinel_x.research.execution import execute
from sentinel_x.research.runner import evaluate


def configuration(budget=512, timeout=1200, model_data=None):
    config = json.loads((ROOT / "docs/research/full.json").read_text())
    cases = json.loads((ROOT / config["cases"] / "manifest.json").read_text())
    config.update(
        case_names=sorted(row["case"] for row in cases),
        budgets=[budget],
        timeout_seconds=timeout,
        purpose="Single-pass all-case development baseline; score saved outputs after collection",
    )
    if model_data is not None:
        config["model_data"] = str(Path(model_data).expanduser().resolve())
    return config


def collect(config, output):
    path = Path(output) / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Saved configuration differs; choose a new output directory")
    write_json(path, config)
    return evaluate(config, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Directory relative to the repository root")
    parser.add_argument("--thinking-budget", type=int, choices=[256, 512, 1024], default=512)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--model-data", help="Override the local Sentinel-X model directory")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without loading models")
    parser.add_argument("--resume", action="store_true", help="Skip successes; retry failures/interrupted cases, preserving all attempts")
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    config = configuration(args.thinking_budget, args.timeout_seconds, args.model_data)
    os.chdir(ROOT)
    output = args.output or Path(".cache/research") / (
        "all-cases-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output = output.expanduser().resolve()
    if args.dry_run:
        print(json.dumps({"output": str(output), "config": config}, indent=2))
        return 0
    if output.exists() and not args.resume:
        parser.error("Output directory already exists; choose a new one or explicitly use --resume")
    print(f"Collecting {len(config['case_names'])} cases sequentially, Thinking budget {args.thinking_budget}.", flush=True)
    print("One pass; the next case starts after the previous case finishes. Failures are recorded and execution continues.", flush=True)
    print("Output:", output, flush=True)
    # 'collect' retains logs/status/CSV but deliberately defers reference scoring.
    result = execute("collect", config, output, collect)
    print(f"Collection finished: {result['latest_cases']} cases recorded; {result['completed']} accepted answers.", flush=True)
    print("Saved results.csv and attempts/*.json. Reference scoring and review remain pending.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
