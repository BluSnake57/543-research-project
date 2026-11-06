"""Offline Scope math, generation alignment, lifecycle and persistence acceptance."""

import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from sentinel_x.config import manifest
from sentinel_x.conversation_ui import transcript_html
from sentinel_x.jobs import Cancelled, Job
from sentinel_x.scope import Capture, resolve_hook, sparse_topk, strongest, validate_sae
from sentinel_x.scope_worker import Engine, parse_output, preflight
from sentinel_x.service import Service
from sentinel_x.state import Store


class Tokenizer:
    # Qwen's tokenizer ends a chat turn with a different token from the model
    # config's end-of-text token when no generation_config.json is installed.
    eos_token_id = 12

    def decode(self, ids, **kwargs):
        return "".join(
            {
                3: "reason",
                4: "answer",
                5: '{"score":0.95}',
                6: "\n",
                8: "</think>",
                9: "<end>",
                12: "<|im_end|>",
                13: "<tool_call><function=lookup><parameter=query>Mars</parameter></function></tool_call>",
                14: '{"text":"<|im_end|>"}',
            }.get(i, f"[{i}]")
            for i in ids
        )

    def convert_tokens_to_ids(self, text):
        return {"</think>": 8, "<think>": 7}[text]

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.options = messages, kwargs
        return "prompt"

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[10, 11]])}


def fixture_engine(tokens):
    engine = Engine.__new__(Engine)
    engine.config = {"context": 8192, "thinking_budget": 256}
    engine.catalog = manifest()
    engine.device = "cpu"
    engine.weight = torch.arange(120, dtype=torch.float32).reshape(60, 2) / 120
    engine.bias = torch.zeros(60)
    engine.tokenizer = Tokenizer()
    engine.layer = torch.nn.Identity()

    class Model:
        generation_config = SimpleNamespace(eos_token_id=9)

        def __init__(self):
            self.index = 0
            self.inputs = []

        def __call__(self, input_ids, **kwargs):
            self.inputs.extend(input_ids[0].tolist())
            engine.layer(torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1))
            logits = torch.full((1, 1, 20), -100.0)
            logits[0, 0, tokens[min(self.index, len(tokens) - 1)]] = 100
            self.index += 1
            return SimpleNamespace(logits=logits, past_key_values=object())

    engine.model = Model()
    return engine


def record():
    cap = Capture(torch.ones(60, 2), torch.zeros(60), Tokenizer(), manifest(), "response")
    cap.prepare([4], ["answer"])
    cap.hook(None, None, torch.tensor([[[2.0, 3.0]]]))
    return cap.finish()


class ScopeMathTests(unittest.TestCase):
    def test_topk_matches_independent_scalar_reference_in_bounded_batches(self):
        rng = np.random.default_rng(17)
        x, w, b = rng.normal(size=(9, 4)), rng.normal(size=(7, 4)), rng.normal(size=7)
        batches = list(sparse_topk(torch.tensor(x), torch.tensor(w), torch.tensor(b), 3, 2))
        self.assertEqual([len(indices) for indices, _ in batches], [2, 2, 2, 2, 1])
        actual = [(i, v) for ids, vals in batches for i, v in zip(ids, vals)]
        for row, (ids, values) in zip(x, actual):
            reference = [
                sum(float(a) * float(c) for a, c in zip(row, weight)) + float(bias)
                for weight, bias in zip(w, b)
            ]
            expected = sorted(range(len(reference)), key=lambda i: reference[i], reverse=True)[:3]
            self.assertEqual(ids, expected)
            np.testing.assert_allclose(values, [reference[i] for i in expected], rtol=1e-6)
        negatives = list(
            sparse_topk(torch.zeros(1, 2), torch.ones(3, 2), -torch.arange(1.0, 4.0), 2)
        )
        self.assertEqual(negatives[0][1], [[-1.0, -2.0]])

    def test_dimensions_nonfinite_hook_location_and_checkpoint_shapes(self):
        with self.assertRaises(ValueError):
            list(sparse_topk(torch.ones(2, 3), torch.ones(5, 4), torch.zeros(5), 2))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            list(sparse_topk(torch.full((1, 2), float("nan")), torch.ones(4, 2), torch.zeros(4), 2))
        with self.assertRaises(ValueError):
            validate_sae({"W_enc": torch.zeros(1)})
        layer = type("Qwen3_5DecoderLayer", (), {})()
        model = SimpleNamespace(
            config=SimpleNamespace(
                model_type="qwen3_5",
                text_config=SimpleNamespace(hidden_size=4096, num_hidden_layers=32),
            ),
            model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer] * 32)),
        )
        self.assertIs(resolve_hook(model), layer)
        model.config.text_config.hidden_size = 2
        with self.assertRaises(ValueError):
            resolve_hook(model)

    def test_capture_truncation_call_separation_and_no_dense_storage(self):
        captures = [
            Capture(torch.ones(60, 2), torch.zeros(60), Tokenizer(), manifest(), kind, limit=1)
            for kind in ("response", "auxiliary")
        ]
        for capture in captures:
            capture.prepare([10, 11], ["prompt", "prompt"])
            capture.hook(None, None, torch.ones(1, 2, 2))
            result = capture.finish()
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["observed_tokens"], 2)
            self.assertEqual(len(result["tokens"]), 1)
            self.assertEqual(len(result["tokens"][0]["values"]), 50)
            json.dumps(result)  # No tensors survive in the persisted record.
        self.assertNotEqual(captures[0].record["call_id"], captures[1].record["call_id"])
        self.assertEqual(len(strongest([c.record for c in captures])), 20)


class WorkerGenerationTests(unittest.TestCase):
    def generate(self, tokens, thinking=False, cancelled=None, max_tokens=10, device="cpu"):
        engine = fixture_engine(tokens)
        engine.device = device
        events = []
        engine.generate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
                "thinking": thinking,
                "kind": "response",
                "max_tokens": max_tokens,
            },
            lambda e: events.append(copy.deepcopy(e)),
            cancelled or threading.Event(),
        )
        self.assertEqual(len(engine.layer._forward_hooks), 0)
        return engine, events

    def test_tokenizer_turn_end_stops_before_control_token_reaches_json_parser(self):
        engine, events = self.generate([5, 12, 6, 9])
        answer = events[-1]["message"]["content"]
        self.assertEqual(json.loads(answer), {"score": 0.95})
        self.assertEqual(engine.model.inputs, [10, 11, 5, 12])
        saved = [e["record"] for e in events if e["event"] == "scope"][-1]
        self.assertEqual(saved["tokens"][-1]["phase"], "boundary")
        self.assertEqual(saved["tokens"][-1]["token_id"], 12)
        self.assertFalse(saved["output_truncated"])
        self.assertEqual(saved["finish_reason"], "stop")
        self.assertEqual(saved["output_text"], answer)
        self.assertEqual(saved["thinking_text"], "")
        self.assertEqual(saved["stop_token_ids"], [9, 12])
        self.assertFalse(any("<|im_end|>" in e["text"] for e in events if e["event"] == "piece"))

    def test_turn_end_preserves_thinking_boundary(self):
        _, events = self.generate([3, 8, 5, 12], thinking=True)
        self.assertEqual(json.loads(events[-1]["message"]["content"]), {"score": 0.95})
        self.assertEqual(
            "".join(e["text"] for e in events if e["event"] == "piece" and e["reasoning"]), "reason"
        )

    def test_thinking_does_not_consume_answer_allowance(self):
        _, events = self.generate([3, 3, 3, 8, 5, 12], thinking=True, max_tokens=2)
        self.assertEqual(json.loads(events[-1]['message']['content']), {'score': 0.95})
        record = [e['record'] for e in events if e['event']=='scope'][-1]
        self.assertFalse(record['output_truncated'])
        self.assertEqual(record['answer_token_budget'], 2)
        self.assertGreater(record['generated_tokens'], 2)

    def test_literal_control_token_text_inside_json_is_not_removed(self):
        _, events = self.generate([14, 12, 9])
        self.assertEqual(json.loads(events[-1]["message"]["content"]), {"text": "<|im_end|>"})

    def test_turn_end_does_not_strip_tool_call_tags(self):
        engine = fixture_engine([13, 12, 9])
        events = []
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        engine.generate(
            {
                "messages": [{"role": "user", "content": "lookup Mars"}],
                "tools": tools,
                "thinking": False,
                "kind": "response",
                "max_tokens": 5,
            },
            lambda event: events.append(copy.deepcopy(event)),
            threading.Event(),
        )
        message = events[-1]["message"]
        self.assertEqual(message["content"], "")
        self.assertEqual(
            json.loads(message["tool_calls"][0]["function"]["arguments"]), {"query": "Mars"}
        )

    def test_generation_eos_list_is_combined_with_tokenizer_eos(self):
        for stop in (9, 12):
            engine = fixture_engine([5, stop, 6])
            engine.model.generation_config.eos_token_id = [9]
            events = []
            engine.generate(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [],
                    "thinking": False,
                    "kind": "auxiliary",
                    "max_tokens": 4,
                },
                lambda event: events.append(copy.deepcopy(event)),
                threading.Event(),
            )
            self.assertEqual(json.loads(events[-1]["message"]["content"]), {"score": 0.95})

    def test_rocm_generation_uses_cuda_tensors_but_saves_rocm_provenance(self):
        original_tensor, original_to = torch.tensor, torch.Tensor.to
        requested = []

        def tensor(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                requested.append("cuda")
                kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)

        def move(value, *args, **kwargs):
            if args and args[0] == "cuda":
                requested.append("cuda")
                args = ("cpu", *args[1:])
            return original_to(value, *args, **kwargs)

        with (patch("torch.tensor", side_effect=tensor), patch.object(torch.Tensor, "to", move),
              patch("torch.cuda.reset_peak_memory_stats"),
              patch("torch.cuda.max_memory_allocated", return_value=123),
              patch("torch.cuda.max_memory_reserved", return_value=456)):
            _, events = self.generate([4, 9], device="rocm")
        record = [e["record"] for e in events if e["event"] == "scope"][-1]
        self.assertEqual(record["device"], "rocm")
        self.assertEqual(record["peak_gpu_allocated_bytes"], 123)
        self.assertEqual(record["peak_gpu_reserved_bytes"], 456)
        self.assertEqual(record["status"], "complete")
        self.assertGreaterEqual(len(requested), 2)
        self.assertEqual(events[-1]["message"]["content"], "answer")

    def test_same_execution_token_alignment_and_thinking_disabled(self):
        engine, events = self.generate([4, 9])
        result = next(e["record"] for e in reversed(events) if e["event"] == "scope")
        self.assertEqual(engine.model.inputs, [10, 11, 4, 9])
        self.assertEqual([t["token_id"] for t in result["tokens"]], engine.model.inputs)
        self.assertEqual(
            [t["phase"] for t in result["tokens"]], ["prompt", "prompt", "answer", "boundary"]
        )
        self.assertEqual(result["status"], "complete")
        self.assertFalse(engine.tokenizer.options["enable_thinking"])
        self.assertEqual(events[-1]["message"]["content"], "answer")
        self.assertFalse(any(e.get("reasoning") for e in events))

    def test_thinking_boundaries_and_output_limit(self):
        _, events = self.generate([3, 8, 4, 9], thinking=True)
        result = [e["record"] for e in events if e["event"] == "scope"][-1]
        self.assertEqual(
            [t["phase"] for t in result["tokens"]][2:],
            ["thinking", "boundary", "answer", "boundary"],
        )
        self.assertTrue(any(e.get("reasoning") for e in events))
        _, events = self.generate([4, 4], max_tokens=1)
        result = [e["record"] for e in events if e["event"] == "scope"][-1]
        self.assertTrue(result["output_truncated"])
        self.assertEqual(result["status"], "incomplete")

    def test_analysis_failure_preserves_answer_and_cancel_removes_hook(self):
        with patch("sentinel_x.scope.sparse_topk", side_effect=ValueError("analysis failed")):
            _, events = self.generate([4, 9])
        self.assertEqual(events[-1]["message"]["content"], "answer")
        result = [e["record"] for e in events if e["event"] == "scope"][-1]
        self.assertEqual(result["status"], "failed")
        cancelled = threading.Event()
        cancelled.set()
        _, events = self.generate([4, 9], cancelled=cancelled)
        self.assertEqual(events[-1]["record"]["status"], "cancelled")
        self.assertFalse(any(e["event"] == "result" for e in events))

    def test_xml_tools_malformed_tools_and_template_arguments(self):
        tools = [
            {
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "ids": {"type": "array"}},
                        "required": ["query"],
                    },
                }
            }
        ]
        text = "<tool_call><function=lookup><parameter=query>Mars</parameter><parameter=ids>[1,2]</parameter></function></tool_call>"
        parsed = parse_output(text, tools)
        self.assertEqual(
            json.loads(parsed["tool_calls"][0]["function"]["arguments"]),
            {"query": "Mars", "ids": [1, 2]},
        )
        for bad in (
            "<tool_call>",
            text.replace("lookup", "unknown"),
            text.replace("query", "bad"),
            "<tool_call><function=lookup></function></tool_call>",
        ):
            with self.assertRaises(ValueError):
                parse_output(bad, tools)

    def test_real_transformers_tiny_hybrid_cache_matches_full_forward(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        config = Qwen3_5Config(
            text_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "vocab_size": 32,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                "max_position_embeddings": 128,
            },
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "depth": 1,
                "num_heads": 2,
                "out_hidden_size": 32,
            },
            image_token_id=30,
            video_token_id=31,
        )
        torch.manual_seed(11)
        model = Qwen3_5ForConditionalGeneration(config).eval()
        outputs = []
        hook = model.model.language_model.layers[2].register_forward_hook(
            lambda module, inputs, output: outputs.append(output.detach().clone())
        )
        with torch.inference_mode():
            full = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True, logits_to_keep=1)
            first = model(input_ids=torch.tensor([[1, 2]]), use_cache=True, logits_to_keep=1)
            last = model(
                input_ids=torch.tensor([[3, 4]]),
                past_key_values=first.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        hook.remove()
        torch.testing.assert_close(full.logits, last.logits, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(outputs[0], torch.cat(outputs[1:], dim=1), atol=1e-6, rtol=1e-5)
        self.assertEqual(last.past_key_values.get_seq_length(), 4)

    def test_preflight_checks_backend_without_estimated_memory_rejection(self):
        with patch("torch.version.hip", None), patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                preflight("cuda")
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=8 * 1024**3)):
            self.assertEqual(preflight("cpu"), torch.float32)
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 1024**3)):
            self.assertEqual(preflight("cpu"), torch.float32)


class ScopeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = Service(Path(self.temp.name))
        self.job = Job(self.service.jobs, "test")

    def tearDown(self):
        self.service.close()
        self.service.jobs.shutdown(wait=True)
        self.temp.cleanup()

    def test_worker_protocol_streaming_analysis_failure_and_mandatory_record(self):
        runtime = self.service.runtime
        runtime.connection = Mock()
        runtime.cancelled = threading.Event()
        failed = record()
        failed.update(status="failed", error="SAE failed")
        events = [
            {"event": "scope", "record": failed},
            {"event": "piece", "text": "answer", "reasoning": False},
            {"event": "result", "message": {"role": "assistant", "content": "answer"}},
        ]
        with patch.object(runtime, "start"), patch.object(runtime, "receive", side_effect=events):
            self.assertEqual(runtime.stream([], [], self.job)["content"], "answer")
        self.assertEqual(self.job.preview()["answer"], "answer")
        self.assertEqual(self.job.data["scope"][0]["status"], "failed")
        with (
            patch.object(runtime, "start"),
            patch.object(
                runtime,
                "receive",
                return_value={"event": "result", "message": {"content": "answer"}},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "without Scope"):
                runtime.stream([], [], self.job)

    def test_actual_allocation_error_is_reported_from_worker(self):
        runtime = self.service.runtime
        runtime.process = Mock()
        runtime.connection = Mock()
        runtime.connection.poll.return_value = True
        runtime.connection.recv.return_value = {
            "event": "error",
            "message": "HIP out of memory. Tried to allocate 2 GiB.",
        }
        with self.assertRaisesRegex(ValueError, "HIP out of memory"):
            runtime.receive(self.job)
        runtime.process = runtime.connection = None

    def test_invalid_or_empty_ipc_is_actionable(self):
        runtime = self.service.runtime
        runtime.process = Mock(pid=__import__("os").getpid())
        runtime.connection = Mock()
        runtime.connection.poll.return_value = True
        runtime.connection.recv.return_value = None
        with self.assertRaisesRegex(RuntimeError, "invalid IPC"):
            runtime.receive(self.job)
        runtime.connection.recv.side_effect = EOFError()
        with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
            runtime.receive(self.job)
        runtime.process = None

    def test_structured_output_accepts_fences_and_retries_malformed_output(self):
        runtime = self.service.runtime
        with patch.object(
            runtime,
            "_generate",
            side_effect=[
                {"content": "{bad"},
                {"content": '```json\n{"value":1}\n```'},
            ],
        ) as generate:
            self.assertEqual(runtime.structured("extract", self.job), {"value": 1})
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(generate.call_args.args[0][1]["content"], "extract")
        self.assertEqual(self.job.data["structured_output_retries"][-1]["reason"], "invalid_json")

    def test_structured_json_validation_and_auxiliary_boundary(self):
        runtime = self.service.runtime
        for text in ("[]", "", "{bad"):
            with patch.object(runtime, "_generate", return_value={"content": text}):
                with self.assertRaisesRegex(ValueError, "invalid JSON"):
                    runtime.structured("extract", self.job)
        with patch.object(
            runtime, "_generate", return_value={"content": '{"value":1}'}
        ) as generate:
            self.assertEqual(runtime.structured("extract", self.job), {"value": 1})
        self.assertEqual(generate.call_args.kwargs["kind"], "auxiliary")
        self.assertTrue(generate.call_args.kwargs["thinking"])

    def test_startup_failure_reaps_real_spawned_worker(self):
        # Invalid device rejects before artifacts or GPU are accessed, on every CI machine.
        self.service.store.configure("scope_device", "invalid")
        with self.assertRaisesRegex(ValueError, "Choose CUDA"):
            self.service.runtime.start(self.job)
        self.assertIsNone(self.service.runtime.process)
        self.assertIsNone(self.service.runtime.connection)

    def test_cancel_stops_worker_and_labels_record(self):
        runtime = self.service.runtime
        runtime.connection = Mock()
        runtime.cancelled = threading.Event()
        running = record()
        running["status"] = "running"
        with (
            patch.object(runtime, "start"),
            patch.object(
                runtime,
                "receive",
                side_effect=[{"event": "scope", "record": running}, Cancelled("cancelled")],
            ),
        ):
            with self.assertRaises(Cancelled):
                runtime.stream([], [], self.job)
        self.assertEqual(self.job.data["scope"][0]["status"], "cancelled")
        self.assertIsNone(runtime.connection)

    def test_pins_ignore_legacy_locks_and_verify_hashes(self):
        models = self.service.models
        (self.service.store.root / "models/qwen.lock.json").write_text('{"repo":"old-gguf"}')
        self.assertEqual(models.catalog("qwen")["repo"], "Qwen/Qwen3.5-9B")
        for key in ("qwen", "scope", "embedding"):
            data = models.catalog(key)
            self.assertEqual(len(data["revision"]), 40)
            self.assertTrue(all(a["checksum"] for a in data["files"]))
        with self.assertRaisesRegex(ValueError, "Missing or corrupt"):
            models.verify("scope")
        with self.assertRaises(ValueError):
            self.service.configure(llama_path="obsolete")

    def test_persistence_export_editing_regeneration_and_independent_panels(self):
        conversation = self.service.store.create_conversation("Scope test")["id"]

        def stream(messages, tools, job):
            job.data["scope"] = [record()]
            return {"role": "assistant", "content": '{"summary":"answer","claims":[]}'}

        def wait(ident):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                saved = self.service.jobs.get(ident)
                if saved["status"] in ("completed", "failed"):
                    self.assertEqual(saved["status"], "completed", saved.get("error"))
                    return
                time.sleep(0.01)
            self.fail("Job timed out")

        with patch.object(self.service.runtime, "stream", side_effect=stream):
            wait(self.service.chat(conversation, "hello", [], False))
            user, answer = self.service.store.messages(conversation)
            self.assertTrue(answer["data"]["scope"])
            original_call = answer["data"]["scope"][0]["call_id"]
            answer["data"].update(thinking_trace="reason trace", thinking_seconds=1)
            closed = transcript_html([answer])
            self.assertNotIn("Feature ", closed)
            opened = transcript_html([answer], scope_expanded={str(answer["id"]): True})
            self.assertIn("Activity inside the model", opened)
            self.assertNotIn("Feature ID</th>", opened)
            self.assertNotIn("reason trace", opened)
            thinking = transcript_html([answer], thinking_expanded={str(answer["id"]): True})
            self.assertIn("reason trace", thinking)
            self.assertNotIn("Feature ", thinking)
            self.service.edit_message(conversation, answer["id"], "edited")
            reopened = Store(self.service.store.root).messages(conversation)[-1]
            self.assertIn("original response", transcript_html([reopened]))
            output = self.service.store.root / "export.md"
            self.service.export(conversation, output)
            self.assertIn("Explanation · Scope", output.read_text())
            self.assertIn(manifest()["scope"]["revision"], output.read_text())
            branch, ident = self.service.regenerate(conversation, user["id"])
            wait(ident)
            regenerated = self.service.store.messages(branch)[-1]
            self.assertNotEqual(original_call, regenerated["data"]["scope"][0]["call_id"])
        old = {"id": 999, "role": "assistant", "content": "old", "data": {}}
        self.assertNotIn("Explanation · Scope", transcript_html([old]))
