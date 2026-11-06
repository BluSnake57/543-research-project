"""Import evaluation inputs from chat without training or running a model."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QPushButton

from sentinel_x.desktop import Window
from sentinel_x.service import Service

CASES = Path(__file__).resolve().parents[1] / "datasets/twitter16-model-cases"


def finish(service, ident):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = service.jobs.get(ident)
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.005)
    raise AssertionError("Import timed out")


class EvaluationImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = Service(Path(self.temp.name))
        self.window = Window(self.service, auto_setup=False)
        self.window.timer.stop()

    def tearDown(self):
        self.window.close()
        self.service.jobs.shutdown(wait=True)
        self.app.processEvents()
        self.temp.cleanup()

    def native_action(self, folder):
        with patch("sentinel_x.desktop.QFileDialog.getExistingDirectory", return_value=str(folder)):
            self.window.import_posts_button.menu().actions()[0].trigger()
        return next(iter(self.window.watched))

    def test_all_evaluation_input_folders_import_and_repeat_without_duplicates(self):
        ids = []
        for case in sorted(CASES.glob("case-*")):
            job = finish(self.service, self.service.import_twitter(case / "input"))
            self.assertEqual(job["status"], "completed", job.get("error"))
            self.assertEqual(job["result"]["errors"], [])
            self.assertEqual(job["result"]["imported"], 1)
            ids.extend(job["result"]["graph_ids"])
        self.assertEqual(len(set(ids)), 15)
        repeat = finish(self.service, self.service.import_twitter(CASES / "case-01"))
        self.assertEqual(repeat["result"]["graph_ids"], ids[:1])
        self.assertEqual(len(self.service.store.graphs()), 15)
        self.assertFalse(self.service.jobs.list()[0]["kind"].startswith("train"))

    def test_native_menu_import_creates_conversation_and_attaches_cascade(self):
        self.assertIsNone(self.window.conversation)
        ident = self.native_action(CASES / "case-01/input")
        job = finish(self.service, ident)
        self.window._poll()
        self.assertEqual(job["status"], "completed", job.get("error"))
        self.assertEqual(self.window.attached, set(job["result"]["graph_ids"]))
        self.assertFalse(self.window.attachments_panel.isHidden())
        self.assertIn("Imported 1", self.window.notice.text())
        graph = self.service.store.graph(next(iter(self.window.attached)))
        self.assertEqual(graph["cascade_id"], "643103859729166337")
        self.assertTrue(graph["native_twitter"])
        self.assertFalse(graph.get("label"))
        self.assertEqual(self.service.store.messages(self.window.conversation), [])

    def test_import_attaches_to_original_draft_after_navigation(self):
        w = self.window
        w.new_conversation()
        original = w.conversation
        w.message_input.setPlainText("Keep this request")
        w.thinking.setChecked(True)
        ident = self.native_action(CASES / "case-01/input")
        w.new_conversation()
        job = finish(self.service, ident)
        w._poll()
        self.assertEqual(w.attached, set())
        draft = self.service.store.setting("draft:" + original)
        self.assertEqual(draft["graph_ids"], job["result"]["graph_ids"])
        self.assertEqual(draft["text"], "Keep this request")
        self.assertTrue(draft["thinking"])

    def test_cancelled_folder_picker_does_not_create_a_conversation_or_job(self):
        with patch("sentinel_x.desktop.QFileDialog.getExistingDirectory", return_value=""):
            self.window.import_posts_button.menu().actions()[0].trigger()
        self.assertIsNone(self.window.conversation)
        self.assertEqual(self.service.jobs.list(), [])

    def test_invalid_folder_and_missing_native_files_are_actionable(self):
        invalid = Path(self.temp.name) / "invalid"
        invalid.mkdir()
        with self.assertRaisesRegex(ValueError, "input folder"):
            self.service.import_twitter(invalid)
        (invalid / "tree").mkdir()
        with self.assertRaisesRegex(ValueError, "source_tweets.txt, label.txt"):
            self.service.import_twitter(invalid)
        self.assertEqual(self.service.store.graphs(), [])

    def test_rejected_cascade_is_reported_without_attaching_it(self):
        with patch(
            "sentinel_x.service.native_twitter", return_value=iter([{"error": "Invalid tree"}])
        ):
            ident = self.native_action(CASES / "case-01/input")
            job = finish(self.service, ident)
        self.window._poll()
        self.assertEqual(job["result"]["graph_ids"], [])
        self.assertEqual(self.window.attached, set())
        self.assertIn("No cascades imported", self.window.notice.text())
        self.assertIn("1 rejected", self.window.notice.text())

    def test_json_menu_still_previews_imports_and_attaches_posts(self):
        source = Path(self.temp.name) / "posts.json"
        source.write_text(json.dumps([{"id": "one", "text": "Example post"}]))
        with patch(
            "sentinel_x.desktop.QFileDialog.getOpenFileName", return_value=(str(source), "")
        ):
            self.window.import_posts_button.menu().actions()[1].trigger()
        preview = next(iter(self.window.watched))
        self.assertEqual(finish(self.service, preview)["status"], "completed")
        self.window._poll()
        dialog = self.window.dialogs[-1]
        button = next(
            b for b in dialog.findChildren(QPushButton) if b.text() == "Import and attach"
        )
        button.click()
        imported = next(iter(self.window.watched))
        result = finish(self.service, imported)
        self.window._poll()
        self.assertEqual(result["status"], "completed", result.get("error"))
        self.assertEqual(self.window.attached, set(result["result"]))
        self.assertFalse(dialog.isVisible())


if __name__ == "__main__":
    unittest.main()
