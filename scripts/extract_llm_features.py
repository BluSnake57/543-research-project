import argparse
import json
from pathlib import Path
import sys
import time

from sentinel_x.service import Service


FEATURES = (
    "urgency",
    "sensationalism",
    "emotional_language",
    "authority_framing",
    "conspiracy_framing",
)


class SimpleJob:
    """Minimal job interface required by Llama.structured()."""

    thinking_enabled = False

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


def build_prompt(text):
    return f"""
Analyze the social-media post below.

Return ONLY a JSON object containing exactly these five numerical fields:

{{
  "urgency": 0.0,
  "sensationalism": 0.0,
  "emotional_language": 0.0,
  "authority_framing": 0.0,
  "conspiracy_framing": 0.0
}}

Every value must be a number from 0.0 to 1.0.

Definitions:

urgency:
Pressure to react, share, click, believe, or act immediately.

sensationalism:
Exaggerated, shocking, dramatic, provocative, or attention-grabbing presentation.

emotional_language:
Use of wording intended to trigger strong emotions such as fear, anger,
outrage, excitement, or sympathy.

authority_framing:
Reliance on experts, officials, institutions, celebrities, scientists,
government figures, or other authority figures to strengthen the message.

conspiracy_framing:
Suggestions of secrecy, suppression, cover-ups, hidden actors,
coordinated deception, or information being deliberately concealed.

Do not determine whether the post is true or false.
Do not retrieve evidence.
Do not explain your answer.
Return only the five numerical fields.

POST:
{text}
""".strip()


def validate_features(result):
    if not isinstance(result, dict):
        raise ValueError("Qwen response is not a JSON object")

    output = {}

    for feature in FEATURES:
        if feature not in result:
            raise ValueError(f"Missing feature: {feature}")

        value = result[feature]

        if isinstance(value, bool):
            raise ValueError(f"{feature} must be numeric")

        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{feature} is not numeric")

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{feature} must be between 0.0 and 1.0"
            )

        output[feature] = value

    return output


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
                    raw = service.runtime.structured(
                        prompt,
                        job,
                    )

                    features = validate_features(raw)

                    results[tweet_id] = {
                        "text": text,
                        "features": features,
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