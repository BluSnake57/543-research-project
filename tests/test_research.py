"""Data-audit independence, protocol compatibility, evidence isolation and resumability."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from sentinel_x.service import Service
from sentinel_x.semantic import build_prompt, validate_features, protocol, cache_compatible
from sentinel_x.research.audit import independent_tree, audit, sha, write_json
from sentinel_x.research.evidence import freeze_fever, validate_pack, install_pack
from sentinel_x.research.runner import freeze_run, summarize, prompt_for
from sentinel_x.learning import scores, train
from sentinel_x.graphs import native_twitter

ROOT = Path(__file__).resolve().parents[1]


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def tree(self):
        p = self.root / "tree.txt"
        p.write_text(
            "['ROOT','ROOT',0]->['u','100',0]\n['u','100',0]->['v','101',60]\n['u','100',0]->['w','102',61]\n['u','100',0]->['v','101',60]\n['v','101',60]->['u','100',0]\n"
        )
        return p

    def fixture(self):
        cases, sources = self.root / "cases", self.root / "sources"
        inputs = cases / "case-01/input"
        (inputs / "tree").mkdir(parents=True)
        sources.mkdir()
        shutil.copyfile(self.tree(), inputs / "tree/100.txt")
        (inputs / "source_tweets.txt").write_text("100\tA test source\n")
        (inputs / "label.txt").write_text("")
        write_json(cases / "manifest.json", [{"case": "case-01", "source_id": "100"}])
        write_json(
            cases / "case-01/EVALUATOR_ONLY.json",
            {
                "source_id": "100",
                "original_rumor_label": "false",
                "final_recorded_nodes": 3,
                "observed_nodes_60min": 2,
                "viral_target_464": False,
                "viral_target_475": False,
            },
        )
        with zipfile.ZipFile(sources / "twitter-datasets.zip", "w") as z:
            z.write(self.tree(), "archive/twitter16/tree/100.txt")
            z.writestr("archive/twitter16/label.txt", "false:100\n")
        for dataset in ("twitter15", "twitter16"):
            for split in ("train", "dev", "test"):
                (sources / f"{dataset}.{split}").write_text(
                    "100\tA test source\tfalse\n" if split == "train" else ""
                )
        write_json(
            cases / "PROVENANCE.json",
            {
                "files_sha256": {
                    str(p.relative_to(cases)): sha(p) for p in cases.rglob("*") if p.is_file()
                },
                "download_sha256": {p.name: sha(p) for p in sources.iterdir()},
            },
        )
        checkpoint = self.root / "checkpoint.json"
        write_json(checkpoint, {"splits": {"train": ["100"], "validation": []}})
        return cases, sources, checkpoint

    def test_independent_boundary_and_repairs(self):
        result = independent_tree(self.tree())
        self.assertEqual(result["final_recorded_nodes"], 3)
        self.assertEqual(result["observed_nodes_60min"], 2)
        self.assertEqual(result["observed_edges_60min"], 1)
        self.assertEqual(result["repairs"], {"duplicate": 1, "backward-time": 1})

    def test_invalid_root_disconnected_nonfinite_and_cycle(self):
        for text in (
            "['u','100',0]->['v','101',1]",
            "['ROOT','ROOT',0]->['a','1',0]\n['b','2',1]->['c','3',2]",
            "['ROOT','ROOT',0]->['u','100','nan']",
        ):
            self.tree().write_text(text)
            with self.assertRaises(ValueError):
                independent_tree(self.root / "tree.txt")
        p = self.tree()
        p.write_text(
            "['ROOT','ROOT',0]->['u','100',0]\n['u','100',0]->['v','101',0]\n['v','101',0]->['u','100',0]\n"
        )
        self.assertEqual(independent_tree(p)["repairs"]["cycle"], 1)

    def test_audit_flags_training_overlap_without_rewriting_cases(self):
        args = self.fixture()
        result = audit(*args)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["cases"][0]["heldout_eligible"])
        self.assertEqual(result["cases"][0]["exact_text_overlap"], ["100"])

    def test_bad_hash_reference_source_and_label_leak_fail(self):
        args = self.fixture()
        ref = args[0] / "case-01/EVALUATOR_ONLY.json"
        data = json.loads(ref.read_text())
        data["final_recorded_nodes"] = 999
        write_json(ref, data)
        (args[0] / "case-01/input/label.txt").write_text("false:100")
        (args[0] / "case-01/input/source_tweets.txt").write_text("999\twrong source")
        result = audit(*args)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["source_failures"])
        self.assertIn("Reference mismatch: final_recorded_nodes", result["cases"][0]["errors"])
        self.assertIn("Rumor labels leaked into model input", result["cases"][0]["errors"])
        self.assertIn("Source ID/text mismatch", result["cases"][0]["errors"])

    def test_shared_batch_prompt_and_validation(self):
        spec = importlib.util.spec_from_file_location(
            "batch", ROOT / "scripts/extract_llm_features.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.build_prompt("hello"), build_prompt("hello"))
        for value in (True, float("nan"), 1.1, None):
            with self.assertRaises(ValueError):
                validate_features({"emotional_language": value}, ["emotional_language"])
        self.assertFalse(cache_compatible({"text": "hello"}, "hello", {}))

    def test_fever_snapshot_excludes_unrelated_resources_and_reviewer_notes(self):
        service = Service(self.root / "source")
        self.addCleanup(service.jobs.shutdown, wait=True)
        self.addCleanup(service.close)
        cases, _, _ = self.fixture()
        with service.retrieval.connection() as db:
            for folder, text in (
                ("fever", "A test source is a test."),
                ("private", "A test source private note."),
            ):
                db.execute(
                    "INSERT INTO chunks(path,key,text,provenance) VALUES(?,?,?,?)",
                    (
                        str(self.root / folder / "wiki.jsonl"),
                        "0",
                        text,
                        json.dumps({"page": "Test", "sentence": "0"}),
                    ),
                )
        pack = freeze_fever(service.retrieval.path, cases, self.root / "pack")
        self.assertEqual(len(pack["passages"]), 1)
        verified = validate_pack(self.root / "pack/pack.json")
        dest = Service(self.root / "dest")
        self.addCleanup(dest.jobs.shutdown, wait=True)
        self.addCleanup(dest.close)
        install_pack(dest, verified)
        self.assertNotIn("private", str(dest.retrieval.search("test")))
        self.assertEqual(
            json.loads((self.root / "pack/reviewers/reviewer-a.json").read_text())["cases"][0][
                "factual_verdict"
            ],
            None,
        )
        verified["passages"][0]["text"] = "tampered"
        write_json(self.root / "pack/pack.json", verified)
        with self.assertRaises(ValueError):
            validate_pack(self.root / "pack/pack.json")

    def test_manifest_changes_require_new_run_and_failed_attempts_retained(self):
        output = self.root / "run"
        freeze_run(output, {"budget": 256})
        freeze_run(output, {"budget": 256})
        with self.assertRaises(ValueError):
            freeze_run(output, {"budget": 512})
        for attempt, status in ((1, "failed"), (2, "completed")):
            write_json(
                output / "attempts" / f"case-01-256-{attempt}.json",
                {
                    "case": "case-01",
                    "budget": 256,
                    "attempt": attempt,
                    "status": status,
                    "seconds": 1.0,
                    "scope": [],
                },
            )
        result = summarize(output)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["completed"], 1)
        self.assertIn("failed", (output / "results.csv").read_text())

    def test_metrics_and_prompt_never_supply_reference(self):
        result = scores([0, 1], [0.2, 0.8])
        self.assertAlmostEqual(result["brier"], 0.04)
        self.assertEqual(sum(r["count"] for r in result["reliability"]), 2)
        prompt = prompt_for(ROOT / "datasets/twitter16-model-cases/case-01")
        self.assertNotIn("EVALUATOR_ONLY", prompt)
        self.assertNotIn("564", prompt)

    def test_fixed_split_seed_across_training_seeds(self):
        graphs = [
            next(native_twitter(p))
            for p in sorted((ROOT / "datasets/twitter16-model-cases").glob("case-*/input"))
        ]
        a = train(
            graphs, self.root / "a", {"epochs": 1, "seed": 42, "split_seed": 42, "threshold": 464}
        )
        b = train(
            graphs, self.root / "b", {"epochs": 1, "seed": 43, "split_seed": 42, "threshold": 464}
        )
        self.assertEqual(a["splits"], b["splits"])
        self.assertIn("brier", a["report"]["test"])

    def test_scoring_keeps_outcomes_out_of_run_and_missing_humans_pending(self):
        from sentinel_x.research.reporting import score_run, reviewer_rows

        output = self.root / "run"
        cases, _, _ = self.fixture()
        write_json(output / "manifest.json", {"pack_hash": "p"})
        write_json(
            output / "data-audit/audit.json",
            {"cases": [{"case": "case-01", "heldout_eligible": False}]},
        )
        write_json(
            output / "attempts/case-01-256-001.json",
            {
                "case": "case-01",
                "budget": 256,
                "attempt": 1,
                "status": "completed",
                "prediction": {
                    "size_threshold": 464,
                    "viral_probability": 0.2,
                    "decision_cutoff": 0.7,
                },
            },
        )
        result = score_run(output, cases)
        self.assertAlmostEqual(result["prediction_metrics"]["256-all_development"]["brier"], 0.04)
        self.assertNotIn("256-overlap_screened", result["prediction_metrics"])
        self.assertIn("PENDING", result["human_review"]["status"])
        self.assertTrue((output / "reliability-256-all_development.svg").exists())
        p = self.root / "reviewer.json"
        write_json(p, {"reviewer_id": None, "pack_sha256": "p", "cases": []})
        self.assertEqual(reviewer_rows(p, "p"), {})
        with self.assertRaises(ValueError):
            reviewer_rows(p, "other")

    def test_training_matrix_resumes_completed_configurations_and_ranks_validation(self):
        from sentinel_x.research.runner import training_matrix

        cases, _, _ = self.fixture()
        cache = self.root / "features.json"
        write_json(
            cache, {"100": {"protocol": {"version": 1}, "features": {"emotional_language": 0.5}}}
        )
        output = self.root / "matrix"
        config = {
            "dataset": str(cases / "case-01/input"),
            "feature_cache": str(cache),
            "device": "cpu",
        }
        calls = []

        def fake_train(graphs, destination, settings):
            calls.append(settings)
            return {
                "id": str(len(calls)),
                "config": settings,
                "llm_feature_names": ["emotional_language"] if settings["llm_features"] else [],
                "report": {
                    "validation": {"f1": 0.8 if settings["hidden"] == 64 else 0.5},
                    "test": {"f1": 0.9 if settings["hidden"] == 128 else 0.4},
                },
            }

        with patch("sentinel_x.research.runner.learning.train", side_effect=fake_train):
            result = training_matrix(config, output)
            training_matrix(config, output)
        self.assertEqual(result["completed"], 18)
        self.assertEqual(len(calls), 18)
        self.assertEqual({c["split_seed"] for c in calls}, {42})
        ranking = json.loads((output / "validation-ranking.json").read_text())
        self.assertEqual(ranking[0]["hidden"], 64)

    def test_pinned_corpus_preparation_rejects_hash_change_and_preserves_existing(self):
        from sentinel_x.research.corpus import prepare_corpus

        cases, sources, _ = self.fixture()
        out = self.root / "native"
        result = prepare_corpus(sources, out, "twitter16", cases / "PROVENANCE.json")
        self.assertEqual(result["usable"], 1)
        with self.assertRaises(ValueError):
            prepare_corpus(sources, out, "twitter16", cases / "PROVENANCE.json")
        (sources / "twitter-datasets.zip").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare_corpus(sources, self.root / "next", "twitter16", cases / "PROVENANCE.json")

    def test_evaluator_resume_and_capture_preservation_without_live_model(self):
        from sentinel_x.research.runner import evaluate

        cases, sources, checkpoint = self.fixture()
        (cases / "case-01/USER_TASK.md").write_text(
            "Import `input/`, select this cascade, and ask:\n\nAssess spread.\n\nSave the complete response"
        )
        checkpoint.with_suffix(".pt").write_bytes(b"fixture weights")
        service = Service(self.root / "service")
        self.addCleanup(service.jobs.shutdown, wait=True)
        self.addCleanup(service.close)

        class FakeAssistant:
            def chat(self, conv, text, ids, job, thinking):
                self.calls = getattr(self, "calls", 0) + 1
                if self.calls == 1:
                    raise ValueError("fixture failure")
                return {
                    "summary": "Finished",
                    "claims": [],
                    "prediction": {"viral_probability": 0.2, "viral": False},
                }

        service.assistant = FakeAssistant()
        config = {
            "cases": str(cases),
            "sources": str(sources),
            "checkpoint": str(checkpoint),
            "evidence_pack": "fixture",
            "case_names": ["case-01"],
            "budgets": [256],
            "model_data": "fixture",
        }
        output = self.root / "evaluation"
        with (
            patch(
                "sentinel_x.research.runner.audit",
                return_value={
                    "status": "passed",
                    "cases": [],
                    "source_failures": [],
                    "overlap_review": "pending",
                },
            ),
            patch(
                "sentinel_x.research.runner.validate_pack",
                return_value={"content_sha256": "fixture"},
            ),
            patch("sentinel_x.research.runner.isolated_service", return_value=service),
            patch("sentinel_x.research.runner.close"),
        ):
            evaluate(config, output)
            evaluate(config, output)
            evaluate(config, output)
        self.assertEqual(service.assistant.calls, 2)
        rows = [json.loads(p.read_text()) for p in sorted((output / "attempts").glob("*.json"))]
        self.assertEqual([r["status"] for r in rows], ["failed", "completed"])

    def test_output_lock_releases_after_failure(self):
        from sentinel_x.research.locking import run_lock

        output = self.root / "locked"
        with run_lock(output):
            with self.assertRaises(ValueError):
                with run_lock(output):
                    pass
        with self.assertRaises(RuntimeError):
            with run_lock(output):
                raise RuntimeError("cancel")
        with run_lock(output):
            pass

    def test_extract_records_failure_then_resumes_compatible_features(self):
        from sentinel_x.research.runner import extract

        cases, _, _ = self.fixture()
        service = Service(self.root / "service")
        self.addCleanup(service.jobs.shutdown, wait=True)
        self.addCleanup(service.close)
        config = {
            "dataset": str(cases / "case-01/input"),
            "model_data": "fixture",
            "checkpoint": "fixture",
        }
        output = self.root / "extraction"
        with (
            patch("sentinel_x.research.runner.isolated_service", return_value=service),
            patch("sentinel_x.research.runner.close"),
            patch.object(
                service.runtime,
                "structured",
                side_effect=[ValueError("failed"), {"emotional_language": 0.5}],
            ) as model,
        ):
            self.assertEqual(extract(config, output)["completed"], 0)
            self.assertEqual(extract(config, output)["completed"], 1)
            self.assertEqual(extract(config, output)["completed"], 1)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(len(list((output / "failures").glob("*.json"))), 1)
        record = json.loads((output / "features.json").read_text())["100"]
        self.assertTrue(record["protocol"]["thinking"])
        self.assertEqual(record["features"], {"emotional_language": 0.5})

    def test_unattended_execution_logs_and_records_failure(self):
        from sentinel_x.research.execution import execute

        output = self.root / "unattended"

        def fail(config, root):
            print("durable output", flush=True)
            raise ValueError("fixture failure")

        with self.assertRaises(ValueError):
            execute("extract", {}, output, fail)
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")
        self.assertIn("durable output", (output / "run.log").read_text())
        execute("extract", {}, output, lambda c, o: {"completed": 1, "expected": 2})
        self.assertEqual(
            json.loads((output / "status.json").read_text())["status"], "completed_with_failures"
        )

    def test_cross_dataset_report_uses_saved_cutoff_and_observed_baseline(self):
        from sentinel_x.research.corpus import cross_evaluate

        cases, _, _ = self.fixture()
        dataset = cases / "case-01/input"
        training = self.root / "training"
        shutil.copytree(dataset, training)
        (training / "tree/100.txt").rename(training / "tree/999.txt")
        (training / "source_tweets.txt").write_text("999\tCompletely different earthquake news\n")
        checkpoint = self.root / "trained.json"
        write_json(
            checkpoint,
            {
                "splits": {"train": ["999"], "validation": []},
                "llm_feature_names": [],
                "config": {"threshold": 464, "window": 60},
                "cutoff": 0.7,
                "report": {"observed_size_baseline": {"cutoff": 0.2}},
            },
        )
        checkpoint.with_suffix(".pt").write_bytes(b"fixture")
        output = self.root / "cross"
        with patch("sentinel_x.research.corpus.predict", return_value=[{"viral_probability": 0.2}]):
            report = cross_evaluate(
                {
                    "checkpoint": str(checkpoint),
                    "dataset": str(dataset),
                    "training_dataset": str(training),
                },
                output,
            )
        self.assertEqual(report["model"]["cutoff"], 0.7)
        self.assertAlmostEqual(report["model"]["brier"], 0.04)
        self.assertEqual(report["observed_size_baseline"]["cutoff"], 0.2)
        self.assertTrue((output / "reliability.svg").exists())
