"""Independent native-tree accounting and source/provenance checks."""

import ast
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

from ..graphs import native_twitter, observed_prefix


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def normalized(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def independent_tree(path, window=60):
    times, raw_edges, roots = {}, [], []
    repairs = Counter()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        ends = []
        for side in line.split("->"):
            row = ast.literal_eval(side.strip())
            if len(row) != 3:
                raise ValueError(f"Line {number}: invalid triple")
            user, tweet, delay = row
            if "ROOT" in (str(user).upper(), str(tweet).upper()):
                ends.append(None)
                continue
            key, delay = f"{user}:{tweet}", float(delay)
            if not math.isfinite(delay):
                raise ValueError(f"Line {number}: nonfinite time")
            if delay < 0:
                repairs["negative_time"] += 1
            delay = max(delay, 0.0)
            if key in times and times[key] != delay:
                repairs["conflicting_time"] += 1
            times[key] = min(times.get(key, delay), delay)
            ends.append(key)
        if len(ends) != 2:
            raise ValueError(f"Line {number}: invalid edge")
        a, b = ends
        if a is None:
            roots.append(b)
        elif b is not None and a != b:
            raw_edges.append((a, b))
        elif a == b:
            repairs["self_edge"] += 1
    if len(set(roots)) != 1 or roots[0] is None:
        raise ValueError("Expected one declared root")
    adjacency, seen, accepted = defaultdict(set), set(), []
    for a, b in sorted(raw_edges, key=lambda e: (times[e[1]], times[e[0]], *e)):
        if (a, b) in seen:
            repairs["duplicate"] += 1
            continue
        seen.add((a, b))
        if times[b] < times[a]:
            repairs["backward-time"] += 1
            continue
        reachable, pending = set(), [b]
        while pending:
            node = pending.pop()
            if node not in reachable:
                reachable.add(node)
                pending.extend(adjacency[node])
        if a in reachable:
            repairs["cycle"] += 1
            continue
        adjacency[a].add(b)
        accepted.append((a, b))
    indegree = Counter(b for a, b in accepted)
    actual_roots = {n for n in times if not indegree[n]}
    if actual_roots != {roots[0]}:
        raise ValueError("Disconnected graph or declared root mismatch")
    early = {n for n, t in times.items() if t <= window}
    return {
        "root_id": roots[0],
        "final_recorded_nodes": len(times),
        "observed_nodes_60min": len(early),
        "observed_edges_60min": sum(a in early and b in early for a, b in accepted),
        "normalized_edges": len(accepted),
        "raw_edges": len(raw_edges),
        "repairs": dict(repairs),
        "viral_target_464": len(times) >= 464,
        "viral_target_475": len(times) >= 475,
    }


def source_texts(folder, dataset):
    rows = {}
    for split in ("train", "dev", "test"):
        for line in (Path(folder) / f"{dataset}.{split}").read_text().splitlines():
            ident, text, label = line.split("\t", 2)
            if ident in rows and rows[ident] != text:
                raise ValueError(f"Conflicting source {ident}")
            rows[ident] = text
    return rows


def audit(cases, sources, checkpoint, source_lock=None):
    cases, sources = Path(cases), Path(sources)
    provenance = json.loads((cases / "PROVENANCE.json").read_text())
    failures = []
    for name, expected in provenance["files_sha256"].items():
        if not (cases / name).is_file() or sha(cases / name) != expected:
            failures.append(f"Case hash mismatch: {name}")
    for name, expected in provenance["download_sha256"].items():
        if not (sources / name).is_file() or sha(sources / name) != expected:
            failures.append(f"Source hash mismatch/missing: {name}")
    if source_lock:
        for name, record in json.loads(Path(source_lock).read_text()).items():
            if not (sources / name).is_file() or sha(sources / name) != record["sha256"]:
                failures.append(f"Pinned source mismatch: {name}")
    texts = source_texts(sources, "twitter16")
    training_texts = source_texts(sources, "twitter15")
    cp = json.loads(Path(checkpoint).read_text())
    excluded = set(cp["splits"]["train"] + cp["splits"]["validation"])
    rows = []
    with zipfile.ZipFile(sources / "twitter-datasets.zip") as archive:
        names = archive.namelist()
        labels_path = next(n for n in names if n.endswith("/twitter16/label.txt"))
        labels = dict(
            reversed(line.split(":", 1)) for line in archive.read(labels_path).decode().splitlines()
        )
        for item in json.loads((cases / "manifest.json").read_text()):
            folder, ident = cases / item["case"], item["source_id"]
            errors = []
            try:
                inputs = folder / "input"
                tree = inputs / "tree" / f"{ident}.txt"
                original = next(n for n in names if n.endswith(f"/twitter16/tree/{ident}.txt"))
                if tree.read_bytes() != archive.read(original):
                    errors.append("Tree differs from pinned source")
                source_id, text = (
                    (inputs / "source_tweets.txt").read_text().rstrip("\n").split("\t", 1)
                )
                if source_id != ident or texts.get(ident) != text:
                    errors.append("Source ID/text mismatch")
                if (inputs / "label.txt").read_text().strip():
                    errors.append("Rumor labels leaked into model input")
                if set(p.name for p in inputs.iterdir()) != {
                    "tree",
                    "source_tweets.txt",
                    "label.txt",
                }:
                    errors.append("Unexpected model input file")
                reference = json.loads((folder / "EVALUATOR_ONLY.json").read_text())
                if reference["source_id"] != ident or reference[
                    "original_rumor_label"
                ] != labels.get(ident):
                    errors.append("Reference ID/original annotation mismatch")
                counts = independent_tree(tree)
                for key in (
                    "final_recorded_nodes",
                    "observed_nodes_60min",
                    "viral_target_464",
                    "viral_target_475",
                ):
                    if counts[key] != reference[key]:
                        errors.append(f"Reference mismatch: {key}")
                graph = next(native_twitter(inputs))
                if "error" in graph:
                    raise ValueError(graph["error"])
                prefix = observed_prefix(graph, 60)
                if (
                    len(graph["nodes"]),
                    len(prefix["nodes"]),
                    len(prefix["edges"]),
                    graph["root_id"],
                ) != (
                    counts["final_recorded_nodes"],
                    counts["observed_nodes_60min"],
                    counts["observed_edges_60min"],
                    counts["root_id"],
                ):
                    errors.append("Independent calculation disagrees with importer")
                overlaps = [
                    key
                    for key, value in training_texts.items()
                    if key in excluded and normalized(value) == normalized(text)
                ]
                possible = [
                    key
                    for key, value in training_texts.items()
                    if key in excluded
                    and key not in overlaps
                    and SequenceMatcher(None, normalized(value), normalized(text)).ratio() >= 0.85
                ]
                rows.append(
                    {
                        **item,
                        **counts,
                        "errors": errors,
                        "exact_text_overlap": overlaps,
                        "possible_event_overlap": possible,
                        "id_overlap": ident in excluded,
                        "heldout_eligible": not (ident in excluded or overlaps or possible),
                        "status": "failed" if errors else "passed",
                    }
                )
            except (ValueError, KeyError, StopIteration, OSError, SyntaxError) as exc:
                rows.append({**item, "status": "failed", "errors": [str(exc)]})
    return {
        "version": 1,
        "status": "passed"
        if not failures and all(r["status"] == "passed" for r in rows)
        else "failed",
        "source_failures": failures,
        "cases": rows,
        "checkpoint_sha256": sha(checkpoint),
        "case_manifest_sha256": digest(provenance),
        "overlap_review": "Pending human event-level review; lexical matching is not semantic adjudication.",
    }


def audit_report(report, output):
    output = Path(output)
    write_json(output / "audit.json", report)
    lines = [
        "# Evaluation data audit",
        "",
        f"Status: {report['status']}",
        "",
        "| Case | Status | Final nodes | First-hour nodes / edges | Repairs |",
        "|---|---|---:|---:|---|",
    ]
    for r in report["cases"]:
        lines.append(
            f"| {r['case']} | {r['status']} | {r.get('final_recorded_nodes', '?')} | {r.get('observed_nodes_60min', '?')} / {r.get('observed_edges_60min', '?')} | {r.get('repairs', {})} |"
        )
        write_json(output / (r["case"] + ".json"), r)
    lines += ["", "## Overlap eligibility", ""]
    for r in report["cases"]:
        if not r.get("heldout_eligible", False):
            lines.append(
                f"- {r['case']}: excluded from held-out metrics. Exact text matches: {r.get('exact_text_overlap', [])}; near matches: {r.get('possible_event_overlap', [])}."
            )
    lines += ["", report["overlap_review"], "", *report["source_failures"]]
    (output / "audit.md").write_text("\n".join(lines) + "\n")
