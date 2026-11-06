import argparse
import json
from pathlib import Path
import sys
import time

from sentinel_x.service import Service


from sentinel_x.semantic import FEATURES, build_prompt, validate_features, protocol, cache_compatible
import hashlib


class SimpleJob:
    """Minimal job interface required by ScopeRuntime.structured()."""

    thinking_enabled = True

    def __init__(self):
        self.data = {}

    def emit(self, *args, **kwargs):
        pass

    def check(self):
        pass


def load_source_tweets(dataset_dir):
    path = dataset_dir / "source_tweets.txt"

    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")

    tweets = {}

    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            tweet_id, separator, text = line.rstrip("\n").partition("\t")

            if separator and tweet_id:
                tweets[tweet_id] = text

    return tweets


def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        help="Twitter15/16 dataset directory",
    )

    parser.add_argument(
        "--output",
        default="results/llm_features.json",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only N tweets for testing",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_path = Path(args.output)

    tweets = load_source_tweets(dataset_dir)

    if output_path.exists():
        results = json.loads(
            output_path.read_text(encoding="utf-8")
        )
    else:
        results = {}

    service = Service()
    expected = protocol(service.store, FEATURES, True)
    if any(not cache_compatible(record, tweets.get(key, ''), expected) for key, record in results.items()):
        service.close()
        raise ValueError('Legacy or incompatible feature cache: choose a new output file; the original is preserved')
    job = SimpleJob()

    print(f"Dataset: {dataset_dir}")
    print(f"Source tweets: {len(tweets)}")
    print(f"Already processed: {len(results)}")

    pending = [
        (tweet_id, text)
        for tweet_id, text in tweets.items()
        if tweet_id not in results
    ]

    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"Processing: {len(pending)}")

    try:
        for index, (tweet_id, text) in enumerate(pending, 1):

            if not text.strip():
                print(
                    f"[{index}/{len(pending)}] "
                    f"{tweet_id}: EMPTY TEXT"
                )

                continue

            prompt = build_prompt(text)

            success = False

            for attempt in range(args.retries + 1):
                try:
                    job = SimpleJob()
                    raw = service.runtime.structured(
                        prompt,
                        job,
                    )

                    features = validate_features(raw)

                    results[tweet_id] = {
                        "text": text,
                        "protocol": expected,
                        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "features": features,
                        "scope": job.data.get("scope", []),
                    }

                    save_results(
                        output_path,
                        results,
                    )

                    print(
                        f"[{index}/{len(pending)}] "
                        f"{tweet_id}: OK "
                        f"{features}"
                    )

                    success = True
                    break

                except Exception as exc:
                    print(
                        f"[{index}/{len(pending)}] "
                        f"{tweet_id}: attempt "
                        f"{attempt + 1} failed: {exc}"
                    )

                    time.sleep(1)

            if not success:
                print(
                    f"{tweet_id}: FAILED after retries"
                )

    finally:
        save_results(
            output_path,
            results,
        )

        service.close()

    print()
    print("Finished.")
    print(f"Saved features: {len(results)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()