"""Measured results stay fixed while Qwen completes the research interpretation."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from sentinel_x.assistant import AnswerFormatError, validate_prediction_answer
from sentinel_x.answer_contract import answer_contract
from sentinel_x.graphs import native_twitter
from sentinel_x.jobs import Job
from sentinel_x.prediction_reporting import scope_report
from sentinel_x.service import Service

ROOT = Path(__file__).resolve().parents[1]


def complete_answer():
    return {
        "summary": "Model prediction and evidence assessment.",
        "claims": [],
        "analysis": {
            "claim": "The post implies an unspecified reaction to a food offer.",
            "framing": 'The word "unreal" signals surprise.',
            "evidence": "No relevant evidence was retrieved; original media is needed.",
            "limitations": "Only the initial propagation is observed; calibration is unknown.",
            "assessment": "The eventual outcome is withheld; the claim is unverified.",
        },
    }


class PredictionAnswerValidationTests(unittest.TestCase):
    def test_contract_has_one_complete_nested_example(self):
        contract = answer_contract(prediction=True)
        example = json.loads(contract.splitlines()[1])
        validate_prediction_answer(json.dumps(example), [])
        self.assertEqual(set(example['analysis']), set(complete_answer()['analysis']))
        self.assertNotIn('JSON AND', contract)

    def test_missing_or_empty_sections_fail_instead_of_silently_completing(self):
        for field in complete_answer()["analysis"]:
            for invalid in (None, "", "  ", [], 135):
                with self.subTest(field=field, invalid=invalid):
                    answer = complete_answer()
                    answer["analysis"][field] = invalid
                    with self.assertRaisesRegex(AnswerFormatError, "five analysis sections"):
                        validate_prediction_answer(json.dumps(answer), [])
        answer = complete_answer()
        del answer["analysis"]
        with self.assertRaises(AnswerFormatError):
            validate_prediction_answer(json.dumps(answer), [])

    def test_unsupported_verdict_requires_retry_including_cascade_citations(self):
        for citations in (["643103859729166337"], ["invented"], [], None, "E1"):
            for verdict in ("supported", "refuted"):
                with self.subTest(citations=citations, verdict=verdict):
                    answer = complete_answer()
                    answer["claims"] = [
                        {
                            "claim": "The underlying event happened.",
                            "verdict": verdict,
                            "citations": citations,
                        }
                    ]
                    with self.assertRaisesRegex(AnswerFormatError, "cascade IDs are not evidence"):
                        validate_prediction_answer(json.dumps(answer), [{"citation": "E1"}])

    def test_valid_evidence_and_explicit_insufficiency_preserve_the_full_analysis(self):
        answer = complete_answer()
        answer["claims"] = [
            {"claim": "An evidenced claim.", "verdict": "supported", "citations": ["E1"]},
            {"claim": "An unknown claim.", "verdict": "insufficient_evidence", "citations": []},
        ]
        final = validate_prediction_answer(json.dumps(answer), [{"citation": "E1"}])
        self.assertEqual(final["analysis"], answer["analysis"])
        self.assertIn("Framing:", final["summary"])
        self.assertIn("Assessment:", final["summary"])
        self.assertEqual(
            [c["verdict"] for c in final["claims"]], ["supported", "insufficient_evidence"]
        )

    def test_live_pilot_count_conflation_requires_correction(self):
        measured = [{'observed_nodes': 135, 'observed_edges': 136}]
        for text in ('135 nodes/edges', '135 observed nodes and edges', '136 nodes and 135 edges'):
            with self.subTest(text=text):
                answer = complete_answer()
                answer['analysis']['limitations'] = text
                with self.assertRaisesRegex(AnswerFormatError, 'misstates observed graph counts'):
                    validate_prediction_answer(json.dumps(answer), [], measured)
        answer = complete_answer()
        answer['analysis']['limitations'] = '135 nodes and 136 edges were observed.'
        result = validate_prediction_answer(json.dumps(answer), [], measured)
        self.assertEqual(result['analysis']['limitations'], answer['analysis']['limitations'])

    def test_measurement_check_preserves_source_numbers_and_equal_counts(self):
        answer = complete_answer()
        answer['analysis']['claim'] = 'The source talks about 500 nodes.'
        answer['analysis']['limitations'] = 'There are 135 nodes/edges.'
        validate_prediction_answer(json.dumps(answer), [], [{'observed_nodes': 135, 'observed_edges': 135}])
        # Prose attribution for multiple cascades is not guessed by this check.
        validate_prediction_answer(json.dumps(answer), [], [
            {'observed_nodes': 1, 'observed_edges': 2}, {'observed_nodes': 135, 'observed_edges': 136}])

    def test_scope_summary_uses_saved_measurements_and_discloses_capture_limits(self):
        report = scope_report(
            [
                {
                    "call_id": "recorded-call",
                    "kind": "response",
                    "status": "complete",
                    "model": "recorded-model",
                    "sae": "recorded-sae",
                    "layer": 16,
                    "tokens": [{"position": 0}, {"position": 1}],
                    "truncated": True,
                    "thinking_budget_reached": True,
                }
            ]
        )
        self.assertIn("saved token records=2", report)
        self.assertIn("Capture truncated=True", report)
        self.assertIn("Thinking budget reached=True", report)
        self.assertIn("model=recorded-model", report)
        self.assertIn("Propagation node/edge counts are separate", report)
        self.assertIn("unavailable", scope_report([]))


class PredictionReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = Service(Path(self.temp.name))
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.service.jobs.shutdown, wait=True)
        self.addCleanup(self.service.close)
        self.graph = next(native_twitter(ROOT / "datasets/twitter16-model-cases/case-01/input"))
        # A future post must not be included in the language-model prompt or retrieval query.
        self.graph["nodes"][-1]["text"] = "FUTURE_ONLY_MARKER"
        self.graph["rumor_label"] = "HIDDEN_RUMOR_LABEL"
        self.service.store.save_graph(self.graph)
        self.checkpoint = "fe44637dfc4d49fe9cd402c4ce6cc21d"
        for suffix in (".pt", ".json"):
            shutil.copy2(
                ROOT / "model weights" / (self.checkpoint + suffix),
                self.service.store.root / "checkpoints",
            )
        self.service.store.configure("checkpoint", self.checkpoint)

    def test_case01_prediction_remains_honest_and_explanation_runs(self):
        conversation = self.service.store.create_conversation()["id"]
        job = Job(self.service.jobs, "test")
        captured = []

        def respond(messages, tools, current_job):
            captured.extend(messages)
            self.assertTrue(current_job.thinking_enabled)
            self.assertNotIn("graphsage_predict", [t["function"]["name"] for t in tools])
            return {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "summary": 'The post uses "unreal" to frame an unspecified event. No relevant evidence is available.',
                        "reasoning_summary": "The probability does not establish truth or a causal explanation.",
                        "claims": [],
                        "analysis": {
                            k: "The post uses unreal; evidence is unavailable."
                            for k in (
                                "claim",
                                "framing",
                                "evidence",
                                "limitations",
                                "scope",
                                "assessment",
                            )
                        }
                        | {"scope": "INCORRECT: 135 graph nodes are Scope activations."},
                    }
                ),
            }

        with (
            patch.object(
                self.service.runtime, "structured", return_value={"emotional_language": 0.95}
            ),
            patch.object(self.service.runtime, "stream", side_effect=respond) as stream,
            patch.object(self.service.retrieval, "search", return_value=[]) as search,
        ):
            result = self.service.assistant.chat(
                conversation,
                "Assess early spread and explain the evidence.",
                [self.graph["id"]],
                job,
            )
        stream.assert_called_once()
        p = result["prediction"]
        self.assertEqual(p["observed_nodes"], 135)
        self.assertFalse(
            p["viral"]
        )  # This pilot miss must not be made to pass by changing the answer.
        self.assertAlmostEqual(p["viral_probability"], 0.21661721169948578, places=5)
        self.assertAlmostEqual(p["semantic_features"]["emotional_language"], 0.95, places=5)
        self.assertIn("135 observed nodes", result["summary"])
        self.assertIn("Qwen interpretation", result["summary"])
        self.assertIn("No Scope records were saved", result["analysis"]["scope"])
        self.assertNotIn("INCORRECT", result["summary"])
        self.assertIn("unreal", result["summary"])
        system = captured[0]["content"]
        self.assertIn('"observed_nodes": 135', system)
        self.assertNotIn("55 of 637", system)
        self.assertIn("Full-import repair counts", system)
        self.assertIn("55 of 637", result["summary"])
        for hidden in (
            "total_posts",
            "final_recorded_nodes",
            "rumor_label",
            "HIDDEN_RUMOR_LABEL",
            "FUTURE_ONLY_MARKER",
        ):
            self.assertNotIn(hidden, system)
        self.assertEqual(search.call_args.args[0], self.graph["source_text"])
        self.assertIn('"semantic_features"', system)

    def test_failed_explanation_does_not_report_completed_analysis(self):
        conv = self.service.store.create_conversation()["id"]
        with (
            patch.object(
                self.service.runtime, "structured", return_value={"emotional_language": 0.95}
            ),
            patch.object(
                self.service.runtime, "stream", side_effect=RuntimeError("generation failed")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                self.service.assistant.chat(
                    conv, "Assess spread", [self.graph["id"]], Job(self.service.jobs, "test")
                )
        self.assertEqual([m["role"] for m in self.service.store.messages(conv)], ["user", "tool"])

    def test_incomplete_prediction_analysis_retries_once_without_recomputing_prediction(self):
        conv = self.service.store.create_conversation()["id"]
        job = Job(self.service.jobs, "test")
        replies = [
            {"summary": "Only the numbers.", "claims": []},
            complete_answer(),
        ]

        def respond(messages, tools, current_job):
            if len(replies) == 1:
                self.assertIn("five analysis sections", messages[-1]["content"])
                self.assertEqual([m['role'] for m in messages], ['system', 'user'])
                self.assertIn('"analysis": {', messages[0]['content'])
                self.assertNotIn('JSON AND an analysis', messages[0]['content'])
                self.assertIn('failed_draft', messages[-1]['content'])
                self.assertFalse(current_job.thinking_enabled)
                self.assertEqual(tools, [])
            return {"role": "assistant", "content": json.dumps(replies.pop(0))}

        with (
            patch.object(
                self.service.runtime, "structured", return_value={"emotional_language": 0.95}
            ) as extract,
            patch.object(self.service.runtime, "stream", side_effect=respond) as stream,
            patch.object(self.service.retrieval, "search", return_value=[]),
        ):
            result = self.service.assistant.chat(conv, "Assess spread", [self.graph["id"]], job)
        self.assertEqual(stream.call_count, 2)
        extract.assert_called_once()
        self.assertTrue(job.thinking_enabled)
        self.assertIn("135 observed nodes", result["summary"])
        for key, value in complete_answer()["analysis"].items():
            self.assertEqual(result["analysis"][key], value)

    def test_pilot_split_objects_and_graph_citations_get_fresh_repair_and_strict_validation(self):
        for draft in ('split_objects', 'graph_citation', 'count_mismatch'):
            with self.subTest(draft=draft):
                conv = self.service.store.create_conversation()['id']
                job = Job(self.service.jobs, 'test')
                first = complete_answer()
                if draft == 'split_objects':
                    analysis = first.pop('analysis')
                    raw = json.dumps(first) + '\n' + json.dumps(analysis)
                elif draft == 'graph_citation':
                    first['claims'] = [{'claim': 'GraphSAGE predicts nonviral',
                                        'verdict': 'supported', 'citations': [self.graph['id']]}]
                    raw = json.dumps(first)
                else:
                    first['analysis']['limitations'] = '135 nodes/edges were observed.'
                    raw = json.dumps(first)
                captured = []

                def respond(messages, tools, current_job):
                    captured.append(messages)
                    return {'role': 'assistant', 'content': raw if len(captured) == 1 else json.dumps(complete_answer())}

                with (
                    patch.object(self.service.runtime, 'structured', return_value={'emotional_language': 0.95}),
                    patch.object(self.service.runtime, 'stream', side_effect=respond),
                    patch.object(self.service.retrieval, 'search', return_value=[]),
                ):
                    result = self.service.assistant.chat(conv, 'Assess spread', [self.graph['id']], job)
                self.assertEqual(len(captured), 2)
                self.assertEqual(len(captured[1]), 2)
                self.assertEqual(captured[1][1]['role'], 'user')
                payload = json.loads(captured[1][1]['content'].split('\n', 1)[1])
                self.assertEqual(payload['failed_draft'], raw)
                self.assertEqual(payload['retrieved_evidence'], [])
                if draft == 'split_objects':
                    self.assertIn('Extra data', payload['parser_detail'])
                elif draft == 'graph_citation':
                    self.assertIn('cascade IDs are not evidence', payload['validation_error'])
                else:
                    self.assertIn('misstates observed graph counts', payload['validation_error'])
                self.assertFalse(result['prediction']['viral'])
                self.assertEqual(result['claims'], [])
                self.assertTrue(job.thinking_enabled)
                audit = self.service.store.messages(conv)[0]['data']['audit']['format_retry']
                self.assertEqual(audit['status'], 'completed')

    def test_failed_repair_retains_raw_draft_and_does_not_erase_missing_analysis(self):
        conv = self.service.store.create_conversation()['id']
        job = Job(self.service.jobs, 'test')
        raw = json.dumps({'summary': 'Only measurements', 'claims': []})
        with (
            patch.object(self.service.runtime, 'structured', return_value={'emotional_language': 0.95}),
            patch.object(self.service.runtime, 'stream', return_value={'role': 'assistant', 'content': raw}) as stream,
            patch.object(self.service.retrieval, 'search', return_value=[]),
        ):
            with self.assertRaisesRegex(AnswerFormatError, 'five analysis sections'):
                self.service.assistant.chat(conv, 'Assess spread', [self.graph['id']], job)
        self.assertEqual(stream.call_count, 2)
        self.assertTrue(job.thinking_enabled)
        messages = self.service.store.messages(conv)
        self.assertNotIn('assistant', [m['role'] for m in messages])
        audit = messages[0]['data']['audit']['format_retry']
        self.assertEqual(audit['status'], 'failed')
        self.assertIn('five analysis sections', audit['error'])

    def test_case01_prompt_regeneration_matches_versioned_file(self):
        spec = importlib.util.spec_from_file_location(
            "prepare_cases", ROOT / "scripts/prepare_twitter16_workflows.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        title, prompt = module.SCENARIOS[0]
        rendered = module.user_task(
            0, title, self.graph["cascade_id"], self.graph["source_text"], prompt
        )
        self.assertEqual(
            rendered, (ROOT / "datasets/twitter16-model-cases/case-01/USER_TASK.md").read_text()
        )
        for hidden in ("564", "135", "viral_target_464", "0.95", "original_rumor_label"):
            self.assertNotIn(hidden, prompt)
