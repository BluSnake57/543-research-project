"""Partition a downloaded Twitter16 ZIP into 15 deterministic native-format cases."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/backend"))
from sentinel_x.graphs import native_twitter, observed_prefix


def prepare(archive, output, source_files):
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}; choose a fresh output directory")
    output.mkdir(parents=True)
    raw = output / "original/twitter16"
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            parts = name.split("/")
            if len(parts) < 3 or parts[1] != "twitter16" or name.endswith("/"):
                continue
            relative = Path(*parts[2:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive path")
            target = raw / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(name))
    sources = {}
    for source in source_files:
        for line in source.read_text().splitlines():
            ident, text, label = line.split("\t", 2)
            if ident in sources and sources[ident] != text:
                raise ValueError(f"Conflicting source text: {ident}")
            sources[ident] = text
    ids = {p.stem for p in (raw / "tree").glob("*.txt")}
    if ids - sources.keys():
        raise ValueError(f"Missing source text: {sorted(ids - sources.keys())}")
    (raw / "source_tweets.txt").write_text("".join(f"{i}\t{sources[i]}\n" for i in sorted(ids)))
    graphs = list(native_twitter(raw))
    errors = [g for g in graphs if "error" in g]
    graphs = [g for g in graphs if "error" not in g]
    sources = dict(
        line.split("\t", 1) for line in (raw / "source_tweets.txt").read_text().splitlines()
    )
    labels = {
        line.split(":", 1)[1].strip(): line.split(":", 1)[0]
        for line in (raw / "label.txt").read_text().splitlines()
    }
    groups = defaultdict(list)
    for graph in graphs:
        groups[(graph["rumor_label"], len(graph["nodes"]) >= 475)].append(graph)
    batches = [[] for _ in range(15)]
    for key in sorted(groups):
        for graph in sorted(groups[key], key=lambda g: (-len(g["nodes"]), g["cascade_id"])):
            batch = min(
                range(15),
                key=lambda i: (
                    sum((g["rumor_label"], len(g["nodes"]) >= 475) == key for g in batches[i]),
                    len(batches[i]),
                    i,
                ),
            )
            batches[batch].append(graph)
    manifest = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "window_minutes": 60,
        "virality_threshold": 475,
        "total_events": len(graphs),
        "cases": [],
    }
    seen = set()
    for index, batch in enumerate(batches, 1):
        folder = output / f"case-{index:02d}"
        (folder / "tree").mkdir(parents=True)
        expected = []
        for graph in sorted(batch, key=lambda g: g["cascade_id"]):
            ident = graph["cascade_id"]
            assert ident not in seen
            seen.add(ident)
            shutil.copyfile(raw / "tree" / f"{ident}.txt", folder / "tree" / f"{ident}.txt")
            prefix = observed_prefix(graph, 60)
            expected.append(
                {
                    "cascade_id": ident,
                    "rumor_label_metadata_only": graph["rumor_label"],
                    "final_recorded_nodes": len(graph["nodes"]),
                    "observed_nodes_60min": len(prefix["nodes"]),
                    "observed_edges_60min": len(prefix["edges"]),
                    "viral_at_475": len(graph["nodes"]) >= 475,
                    "viral_at_shipped_checkpoint_464": len(graph["nodes"]) >= 464,
                    "expected_import_warnings": graph["warnings"],
                }
            )
        ids = [row["cascade_id"] for row in expected]
        (folder / "source_tweets.txt").write_text(
            "".join(f"{ident}\t{sources[ident]}\n" for ident in ids)
        )
        (folder / "label.txt").write_text("".join(f"{labels[ident]}:{ident}\n" for ident in ids))
        (folder / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
        actual = list(native_twitter(folder))
        assert len(actual) == len(batch) and all("error" not in g for g in actual)
        assert {g["cascade_id"] for g in actual} == set(ids)
        manifest["cases"].append(
            {
                "case": folder.name,
                "events": len(batch),
                "viral_at_475": sum(r["viral_at_475"] for r in expected),
                "warnings": sum(bool(r["expected_import_warnings"]) for r in expected),
            }
        )
    for index, error in enumerate(errors):
        ident = error["error"].split(".txt:", 1)[0]
        folder = output / f"case-{15 - index:02d}"
        shutil.copyfile(raw / "tree" / f"{ident}.txt", folder / "tree" / f"{ident}.txt")
        with (folder / "source_tweets.txt").open("a") as stream:
            stream.write(f"{ident}\t{sources[ident]}\n")
        with (folder / "label.txt").open("a") as stream:
            stream.write(f"{labels[ident]}:{ident}\n")
        path = folder / "expected.json"
        rows = json.loads(path.read_text())
        rows.append({"cascade_id": ident, "expected_import_error": error["error"]})
        path.write_text(json.dumps(rows, indent=2) + "\n")
        manifest["cases"][14 - index]["expected_rejections"] = 1
    manifest["expected_rejections"] = errors
    manifest["total_events"] = len(graphs) + len(errors)
    assert len(seen) + len(errors) == len(list((raw / "tree").glob("*.txt")))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/twitter16-validation"))
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.output, args.sources)
