"""Resumable isolated evaluations. No selected checkpoint or user chat is modified."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

from ..graphs import native_twitter
from ..jobs import Job, Cancelled
from ..service import Service
from ..semantic import protocol, build_prompt, validate_features, cache_compatible
from .. import learning
from .audit import audit, audit_report, sha, digest, write_json
from .evidence import validate_pack, install_pack


class ResearchJob(Job):
    def __init__(self, manager, seconds=1200):
        super().__init__(manager, "research")
        self.deadline = time.monotonic() + seconds
        self.thinking_enabled = True

    def emit(self, event, **payload):
        if event == "stage":
            print("STAGE", payload.get("stage"), flush=True)
        return super().emit(event, **payload)

    def check(self):
        super().check()
        if time.monotonic() > self.deadline:
            raise TimeoutError("Research call exceeded configured timeout")


def freeze_run(output, manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Run configuration/data changed; choose a new output directory")
    write_json(path, manifest)


def prompt_for(folder):
    task = (folder / "USER_TASK.md").read_text()
    if "## Primary prompt" in task:
        request = task.split(
            "Copy the source post above and the following request into the conversation:\n\n", 1
        )[1]
    else:
        request = task.split("and ask:\n\n", 1)[1]
    return request.split("\n\nSave the complete response", 1)[0]


def isolated_service(root, model_data, checkpoint, budget, pack=None):
    service = Service(root)
    source = Path(model_data).expanduser().resolve()
    for key in ("qwen", "scope"):
        (service.store.root / "models" / key).symlink_to(
            source / "models" / key, target_is_directory=True
        )
    for suffix in (".json", ".pt"):
        shutil.copy2(Path(checkpoint).with_suffix(suffix), service.store.root / "checkpoints")
    service.store.configure("checkpoint", Path(checkpoint).stem)
    for key, value in {
        "scope_model": "9b",
        "scope_device": "rocm",
        "scope_offload": "sae",
        "context": 8192,
        "thinking_budget": budget,
    }.items():
        service.store.configure(key, value)
    if pack:
        install_pack(service, pack)
    return service


def close(service):
    service.runtime.stop()
    service.jobs.shutdown(wait=True)
    service.close()


def compact_scope(records):
    fields = (
        "kind",
        "status",
        "model_revision",
        "sae_revision",
        "generated_tokens",
        "thinking_budget_reached",
        "output_truncated",
        "truncated",
        "finish_reason",
        "peak_gpu_allocated_bytes",
        "peak_gpu_reserved_bytes",
    )
    return [{k: r.get(k) for k in fields} for r in records]


def summarize(output):
    output = Path(output)
    rows = [json.loads(p.read_text()) for p in sorted(output.glob("attempts/*.json"))]
    fields = (
        "case",
        "budget",
        "attempt",
        "status",
        "seconds",
        "probability",
        "predicted_class",
        "error",
    )
    with (output / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    latest = {}
    for row in rows:
        key = (row["case"], row["budget"])
        if key not in latest or latest[key]["attempt"] < row["attempt"]:
            latest[key] = row
    completed = [r for r in latest.values() if r["status"] == "completed"]
    summary = {
        "attempts": len(rows),
        "latest_cases": len(latest),
        "completed": len(completed),
        "completion_rate": len(completed) / len(latest) if latest else None,
        "human_quality": "PENDING independent reviewer annotations",
        "budgets": {},
        "scope_budget_hits": sum(
            bool(s["thinking_budget_reached"]) for r in completed for s in r["scope"]
        ),
    }
    for budget in sorted({r["budget"] for r in completed}):
        group = [r for r in completed if r["budget"] == budget]
        summary["budgets"][str(budget)] = {
            "runs": len(group),
            "mean_seconds": sum(r["seconds"] for r in group) / len(group),
        }
    summary["manifest_sha256"] = (
        digest(json.loads((output / "manifest.json").read_text()))
        if (output / "manifest.json").exists()
        else None
    )
    for budget in sorted({r["budget"] for r in latest.values()}):
        group = [r for r in latest.values() if r["budget"] == budget]
        entry = summary["budgets"].setdefault(str(budget), {})
        calls = [call for r in group for call in r.get("scope", [])]
        peaks = [
            call["peak_gpu_allocated_bytes"]
            for call in calls
            if call.get("peak_gpu_allocated_bytes") is not None
        ]
        entry.update(
            attempted_cases=len(group),
            failed=sum(r["status"] == "failed" for r in group),
            budget_hits=sum(bool(c.get("thinking_budget_reached")) for c in calls),
            peak_gpu_allocated_bytes=max(peaks) if peaks else None,
        )
    stability = {}
    for r in latest.values():
        prediction = r.get("prediction") or {}
        for name, value in prediction.get("semantic_features", {}).items():
            stability.setdefault(r["case"] + ":" + name, []).append(value)
    summary["semantic_score_ranges"] = {
        key: {"min": min(v), "max": max(v), "runs": len(v)} for key, v in stability.items()
    }
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(
        "# Research run\n\n```json\n"
        + json.dumps(summary, indent=2)
        + "\n```\n\nAll attempts, including failures, remain in results.csv. Pilot cases are developmental.\n"
    )
    return summary


def evaluate(config, output):
    output = Path(output)
    cases = Path(config["cases"])
    report = audit(cases, config["sources"], config["checkpoint"], config.get("source_lock"))
    audit_report(report, output / "data-audit")
    if report["status"] != "passed":
        raise ValueError("Data audit failed; resolve discrepancies before evaluation")
    pack = validate_pack(config["evidence_pack"])
    manifest = {
        "config": config,
        "audit_hash": digest(report),
        "pack_hash": pack["content_sha256"],
        "weights_hash": sha(Path(config["checkpoint"]).with_suffix(".pt")),
        "harness_hash": digest({p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}),
        "model_manifest_hash": sha(
            Path(__file__).resolve().parents[1] / "assets/runtime-manifest.json"
        ),
        "implementation_hash": digest(
            {str(p.name): sha(p) for p in sorted(Path(__file__).resolve().parents[1].glob("*.py"))}
        ),
        "prompt_hashes": {
            name: sha(cases / name / "USER_TASK.md") for name in config["case_names"]
        },
    }
    freeze_run(output, manifest)
    for budget in config["budgets"]:
        service = None
        try:
            for name in config["case_names"]:
                previous = sorted((output / "attempts").glob(f"{name}-{budget}-*.json"))
                if any(json.loads(p.read_text())["status"] == "completed" for p in previous):
                    continue
                if service is None:
                    service = isolated_service(
                        output / "sessions" / uuid.uuid4().hex,
                        config["model_data"],
                        config["checkpoint"],
                        budget,
                        pack,
                    )
                row = {
                    "case": name,
                    "budget": budget,
                    "attempt": len(previous) + 1,
                    "status": "running",
                }
                path = output / "attempts" / f"{name}-{budget}-{len(previous) + 1:03d}.json"
                write_json(path, row)
                print("START CASE", name, "Thinking budget", budget, flush=True)
                start = time.monotonic()
                job = ResearchJob(service.jobs, config.get("timeout_seconds", 1200))
                try:
                    graph = next(native_twitter(cases / name / "input"))
                    graph["id"] = hashlib.sha256((name + graph["cascade_id"]).encode()).hexdigest()[
                        :32
                    ]
                    service.store.save_graph(graph)
                    conv = service.store.create_conversation(f"Research {name} budget {budget}")[
                        "id"
                    ]
                    row["session"] = str(service.store.root)
                    row["conversation"] = conv
                    row["job_id"] = job.data["id"]
                    write_json(path, row)
                    result = service.assistant.chat(
                        conv,
                        graph["source_text"] + "\n\n" + prompt_for(cases / name),
                        [graph["id"]],
                        job,
                        True,
                    )
                    row.update(
                        status="completed",
                        summary=result["summary"],
                        claims=result["claims"],
                        scope=compact_scope(job.data.get("scope", [])),
                        evidence=result.get("evidence", []),
                        prediction=result.get("prediction"),
                        analysis=result.get("analysis"),
                    )
                    if result.get("prediction"):
                        row.update(
                            probability=result["prediction"]["viral_probability"],
                            predicted_class=result["prediction"]["viral"],
                        )
                except (KeyboardInterrupt, Cancelled):
                    row.update(status="interrupted", error="Cancelled by user")
                    raise
                except Exception as exc:
                    row.update(
                        status="failed",
                        error=str(exc),
                        scope=compact_scope(job.data.get("scope", [])),
                    )
                finally:
                    # Keep readable outputs next to every result, including rejected answers.
                    # Large activation arrays remain in the isolated session database.
                    output_fields = (
                        "call_id", "kind", "status", "output_text", "thinking_text",
                        "finish_reason", "generated_tokens", "output_truncated", "error",
                    )
                    row["model_outputs"] = [
                        {key: capture[key] for key in output_fields if key in capture}
                        for capture in job.data.get("scope", [])
                    ]
                    # A failed Qwen explanation must not discard a computed GraphSAGE result.
                    if not row.get("prediction"):
                        measured = [
                            e.get("result")
                            for e in job.data.get("events", [])
                            if e.get("event") == "tool_result"
                            and e.get("name") == "graphsage_predict"
                        ]
                        if measured and measured[-1]:
                            row["prediction"] = measured[-1][0]
                            row["probability"] = row["prediction"]["viral_probability"]
                            row["predicted_class"] = row["prediction"]["viral"]
                    row["seconds"] = time.monotonic() - start
                    write_json(path, row)
                    summarize(output)
                    print(name, budget, row["status"], round(row["seconds"], 1), flush=True)
        finally:
            if service:
                close(service)
    return summarize(output)


def extract(config, output):
    output = Path(output)
    graphs = list(native_twitter(Path(config["dataset"])))
    if any("error" in g for g in graphs):
        write_json(output / "excluded.json", [g for g in graphs if "error" in g])
        graphs = [g for g in graphs if "error" not in g]
    manifest = {
        "config": config,
        "sources": {
            g["cascade_id"]: hashlib.sha256(g["source_text"].encode()).hexdigest() for g in graphs
        },
    }
    freeze_run(output, manifest)
    service = isolated_service(
        output / "sessions" / uuid.uuid4().hex,
        config["model_data"],
        config["checkpoint"],
        config.get("budget", 256),
    )
    path = output / "features.json"
    features = json.loads(path.read_text()) if path.exists() else {}
    names = config.get("features", ["emotional_language"])
    expected = protocol(service.store, names, True)
    try:
        if any(
            not cache_compatible(
                r, next(g["source_text"] for g in graphs if g["cascade_id"] == key), expected
            )
            for key, r in features.items()
        ):
            raise ValueError("Incompatible cache; select a new output directory")
        for graph in graphs:
            key, text = graph["cascade_id"], graph["source_text"]
            if key in features:
                continue
            job = ResearchJob(service.jobs, config.get("timeout_seconds", 1200))
            try:
                values = validate_features(
                    service.runtime.structured(build_prompt(text, names), job), names
                )
                features[key] = {
                    "text": text,
                    "features": values,
                    "protocol": expected,
                    "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
                write_json(path, features)
            except (KeyboardInterrupt, Cancelled):
                write_json(
                    output / "failures" / f"{key}-{uuid.uuid4().hex}.json",
                    {"status": "interrupted"},
                )
                raise
            except Exception as exc:
                write_json(
                    output / "failures" / f"{key}-{uuid.uuid4().hex}.json",
                    {"status": "failed", "error": str(exc)},
                )
            finally:
                write_json(
                    output / "captures" / f"{key}-{uuid.uuid4().hex}.json",
                    job.data.get("scope", []),
                )
    finally:
        close(service)
    return {"completed": len(features), "expected": len(graphs)}


def training_matrix(config, output):
    output = Path(output)
    graphs = list(native_twitter(Path(config["dataset"])))
    write_json(output / "excluded.json", [g for g in graphs if "error" in g])
    graphs = [g for g in graphs if "error" not in g]
    feature_path = Path(config["feature_cache"])
    from ..semantic import feature_cache_protocol

    if feature_cache_protocol(feature_path) is None:
        raise ValueError(
            "Full comparisons require versioned regenerated features, not legacy caches"
        )
    freeze_run(
        output,
        {
            "config": config,
            "features_sha256": sha(feature_path),
            "dataset_sha256": digest(
                [{k: g[k] for k in ("cascade_id", "nodes", "edges")} for g in graphs]
            ),
        },
    )
    for width in (32, 64, 128):
        for hybrid in (False, True):
            for seed in (42, 43, 44):
                key = f"w{width}-hybrid{int(hybrid)}-seed{seed}"
                path = output / (key + ".json")
                if path.exists() and json.loads(path.read_text())["status"] == "completed":
                    continue
                attempt = output / "attempts" / (key + "-" + uuid.uuid4().hex + ".json")
                row = {"status": "running", "key": key}
                write_json(attempt, row)
                start = time.monotonic()
                try:
                    metadata = learning.train(
                        graphs,
                        output / "checkpoints",
                        {
                            "threshold": 464,
                            "window": 60.0,
                            "epochs": 30,
                            "hidden": width,
                            "seed": seed,
                            "split_seed": 42,
                            "device": config.get("device", "cuda"),
                            "llm_features": str(feature_path) if hybrid else None,
                            "llm_feature": "emotional_language" if hybrid else None,
                        },
                    )
                    row.update(status="completed", checkpoint=metadata)
                except KeyboardInterrupt:
                    row.update(status="interrupted")
                    raise
                except Exception as exc:
                    row.update(status="failed", error=str(exc))
                finally:
                    row["seconds"] = time.monotonic() - start
                    write_json(path, row)
                    write_json(attempt, row)
    rows = [json.loads(p.read_text()) for p in output.glob("w*.json")]
    write_json(output / "matrix.json", rows)
    # Rank configurations by mean validation F1 only, never by test results.
    candidates = []
    for width in (32, 64, 128):
        for hybrid in (False, True):
            group = [
                r
                for r in rows
                if r["status"] == "completed"
                and r["checkpoint"]["config"]["hidden"] == width
                and bool(r["checkpoint"]["llm_feature_names"]) == hybrid
            ]
            if len(group) == 3:
                candidates.append(
                    {
                        "hidden": width,
                        "hybrid": hybrid,
                        "validation_f1": sum(
                            r["checkpoint"]["report"]["validation"]["f1"] for r in group
                        )
                        / 3,
                        "checkpoints": [r["checkpoint"]["id"] for r in group],
                    }
                )
    candidates.sort(key=lambda r: (-r["validation_f1"], r["hidden"], r["hybrid"]))
    write_json(output / "validation-ranking.json", candidates)
    with (output / "matrix.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["configuration", "status", "seconds", "validation_f1", "test_f1"])
        for r in rows:
            m = r.get("checkpoint", {}).get("report", {})
            writer.writerow(
                [
                    r["key"],
                    r["status"],
                    r["seconds"],
                    m.get("validation", {}).get("f1"),
                    m.get("test", {}).get("f1"),
                ]
            )
    return {"completed": sum(r["status"] == "completed" for r in rows), "total": 18}
