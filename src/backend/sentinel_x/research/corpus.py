"""Pinned native corpus preparation and exploratory cross-dataset evaluation."""

import json
from pathlib import Path
from difflib import SequenceMatcher
import zipfile

from ..graphs import native_twitter, observed_prefix
from ..learning import predict, scores
from ..semantic import feature_cache_protocol
from .audit import sha, write_json, source_texts, normalized, digest
from .runner import freeze_run
from .reporting import reliability_svg


def prepare_corpus(sources, output, dataset, provenance, source_lock=None):
    sources, output = Path(sources), Path(output)
    expected = json.loads(Path(provenance).read_text())["download_sha256"]
    archive = sources / "twitter-datasets.zip"
    if sha(archive) != expected["twitter-datasets.zip"]:
        raise ValueError("Archive checksum mismatch")
    if output.exists():
        raise ValueError("Choose a new corpus output; existing data is never overwritten")
    if source_lock:
        lock = json.loads(Path(source_lock).read_text())
        for split in ("train", "dev", "test"):
            name = dataset + "." + split
            if sha(sources / name) != lock[name]["sha256"]:
                raise ValueError("Source text checksum mismatch: " + name)
    texts = source_texts(sources, dataset)
    output.mkdir(parents=True)
    hashes = {p.name: sha(p) for p in sources.glob(dataset + ".*")}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            parts = Path(name).parts
            if len(parts) < 3 or parts[1] != dataset or name.endswith("/"):
                continue
            relative = Path(*parts[2:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive path")
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(name))
    ids = {p.stem for p in (output / "tree").glob("*.txt")}
    if ids - set(texts):
        raise ValueError("Missing source tweets")
    (output / "source_tweets.txt").write_text(
        "".join(f"{key}\t{texts[key]}\n" for key in sorted(ids))
    )
    write_json(
        output / "provenance.json",
        {"archive_sha256": sha(archive), "source_hashes": hashes, "dataset": dataset},
    )
    graphs = list(native_twitter(output))
    report = {
        "total": len(graphs),
        "usable": sum("error" not in g for g in graphs),
        "excluded": [g for g in graphs if "error" in g],
    }
    write_json(output / "import-report.json", report)
    return report


def cross_evaluate(config, output):
    output = Path(output)
    checkpoint = Path(config["checkpoint"])
    metadata = json.loads(checkpoint.read_text())
    graphs = list(native_twitter(Path(config["dataset"])))
    training = list(native_twitter(Path(config["training_dataset"])))
    exclude = set(metadata["splits"]["train"] + metadata["splits"]["validation"])
    known = {
        g["cascade_id"]: normalized(g["source_text"])
        for g in training
        if "error" not in g and g["cascade_id"] in exclude
    }
    features = config.get("feature_cache")
    if metadata.get("llm_feature_names") and (
        not features
        or feature_cache_protocol(features) != metadata.get("feature_protocol")
        or metadata.get("feature_protocol") is None
    ):
        raise ValueError("Cross-dataset hybrid evaluation requires matching versioned features")
    freeze_run(
        output,
        {
            "config": config,
            "checkpoint_hash": sha(checkpoint),
            "weights_hash": sha(checkpoint.with_suffix(".pt")),
            "features_hash": sha(features) if features else None,
            "dataset_hash": digest(
                [{k: g.get(k) for k in ("cascade_id", "nodes", "edges", "error")} for g in graphs]
            ),
        },
    )
    eligible, excluded = [], []
    for graph in graphs:
        if "error" in graph:
            excluded.append(graph)
            continue
        text = normalized(graph["source_text"])
        matches = [
            key
            for key, value in known.items()
            if value == text or SequenceMatcher(None, text, value).ratio() >= 0.85
        ]
        if graph["cascade_id"] in exclude or matches:
            excluded.append(
                {
                    "cascade_id": graph["cascade_id"],
                    "reason": "ID/text/event overlap candidate",
                    "matches": matches,
                }
            )
        else:
            eligible.append(graph)
    write_json(output / "excluded.json", excluded)
    if not eligible:
        raise ValueError("No eligible events after overlap screening")
    predictions = predict(eligible, checkpoint, features)
    write_json(output / "predictions.json", predictions)
    labels = [int(len(g["nodes"]) >= metadata["config"]["threshold"]) for g in eligible]
    metrics = scores(labels, [p["viral_probability"] for p in predictions], metadata["cutoff"])
    train_graphs = [
        g for g in training if "error" not in g and g["cascade_id"] in metadata["splits"]["train"]
    ]
    majority = int(
        sum(len(g["nodes"]) >= metadata["config"]["threshold"] for g in train_graphs)
        >= len(train_graphs) / 2
    )
    size_probabilities = [
        min(
            len(observed_prefix(g, metadata["config"]["window"])["nodes"])
            / metadata["config"]["threshold"],
            1.0,
        )
        for g in eligible
    ]
    report = {
        "interpretation": "EXPLORATORY: event overlap candidates excluded; human duplicate audit pending",
        "target": metadata["config"]["threshold"],
        "model": metrics,
        "majority_baseline": scores(labels, [majority] * len(labels)),
        "observed_size_baseline": scores(
            labels, size_probabilities, metadata["report"]["observed_size_baseline"]["cutoff"]
        ),
    }
    write_json(output / "scores.json", report)
    reliability_svg(metrics, output / "reliability.svg")
    return report
