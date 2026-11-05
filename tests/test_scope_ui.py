"""Readable Scope views remain grounded in saved tokens and independent UI state."""

import copy
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import unittest
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication
from sentinel_x.scope_ui import render_scope
from sentinel_x.desktop import Window
from sentinel_x.service import Service


def example(call="call-a", kind="response"):
    def token(text, phase, position, value):
        return {
            "token": text,
            "phase": phase,
            "position": position,
            "feature_ids": [123],
            "values": [value],
        }

    return {
        "call_id": call,
        "kind": kind,
        "status": "complete",
        "layer": 16,
        "observed_tokens": 7,
        "model": "example model",
        "tokens": [
            token("<|im_start|>", "prompt", 0, 100),
            token("system instruction", "prompt", 1, 80),
            token("thinking text", "thinking", 2, 50),
            token("Mars", "answer", 3, 5),
            token(" has", "answer", 4, 4),
            token(" two moons.", "answer", 5, 3),
            token("<|im_end|>", "answer", 6, 90),
        ],
    }


class ScopeViewTests(unittest.TestCase):
    def test_answer_default_filters_prompt_markers_and_hides_raw_values(self):
        r = example()
        original = copy.deepcopy(r)
        html = render_scope("1", [r], True)
        self.assertIn("Mars", html)
        self.assertNotIn("system instruction", html)
        self.assertNotIn("thinking text", html)
        self.assertNotIn("im_end", html)
        self.assertNotIn("call-a", html)
        self.assertNotIn("Feature ID</th>", html)
        self.assertEqual(r, original)
        self.assertIn("No supporting evidence", html)
        self.assertIn("details:1", render_scope("1", [r], True, evidence=True))

    def test_phase_formatting_and_technical_views_are_scoped(self):
        r = example()
        prompt = render_scope("1", [r], True, state={"phase": "prompt", "technical": True})
        self.assertIn("system instruction", prompt)
        self.assertNotIn("im_start", prompt)
        self.assertNotIn("Mars", prompt)
        formatted = render_scope("1", [r], True, state={"phase": "prompt", "formatting": True})
        self.assertIn("&lt;|im_start|&gt;", formatted)
        selected = render_scope("1", [r], True, state={"token": 3, "technical": True})
        self.assertIn("Selected text piece", selected)
        self.assertIn("Feature ID</th>", selected)
        self.assertIn("<td>5</td>", selected)
        self.assertNotIn("<td>90</td>", selected)
        self.assertIn("not percentages", selected)

    def test_calls_empty_states_and_partial_measurements(self):
        helper = example("helper", "auxiliary")
        response = example("final")
        helper["tokens"][3]["token"] = "helper-only"
        html = render_scope("1", [response, helper], True)
        self.assertNotIn("helper-only", html)
        self.assertIn("helper-only", render_scope("1", [response, helper], True, state={"call": 1}))
        response["tokens"] = []
        response["status"] = "failed"
        html = render_scope("1", [response], True, edited=True)
        self.assertIn("Measurement failed", html)
        self.assertIn("partial or unavailable", html)
        self.assertIn("No saved text activity", html)
        self.assertIn("original response", html)
        response["status"] = "running"
        self.assertIn("Measuring…", render_scope("1", [response], True))

    def test_untrusted_token_text_is_escaped(self):
        r = example()
        r["tokens"][3]["token"] = "<img src=x onerror=alert(1)>"
        html = render_scope("1", [r], True, state={"token": 3, "technical": True})
        self.assertNotIn("<img", html)
        self.assertIn("&lt;img", html)

    def test_qt_switches_survive_refresh_and_do_not_expand_thinking(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            service = Service(Path(directory))
            window = Window(service, auto_setup=False)
            try:
                conversation = service.store.create_conversation("Scope")["id"]
                ident = service.store.message(
                    conversation,
                    "assistant",
                    "Mars has two moons.",
                    {
                        "scope": [example()],
                        "thinking_trace": "private trace",
                        "thinking_seconds": 1,
                    },
                )
                window.conversation = conversation
                window.message_action(QUrl(f"scope:{ident}"))
                self.assertIn("Explore up to", window.transcript.toPlainText())
                window.message_action(QUrl(f"scopeview:{ident}:phase:thinking"))
                window.render_messages()
                app.processEvents()
                self.assertIn("thinking text", window.transcript.toPlainText())
                self.assertNotIn("private trace", window.transcript.toPlainText())
                window.message_action(QUrl(f"scopeview:{ident}:technical:toggle"))
                window.render_messages()
                self.assertIn("Feature ID", window.transcript.toPlainText())
                self.assertEqual(window.scope_views[str(ident)]["phase"], "thinking")
            finally:
                window.close()
                service.jobs.shutdown(wait=True)
