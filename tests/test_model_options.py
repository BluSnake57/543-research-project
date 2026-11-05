import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QPushButton
from sentinel_x.config import model_profile
from sentinel_x.service import Service
from sentinel_x.runtime_ui import RuntimePanel
from sentinel_x.scope_worker import placement


class ModelOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_paired_downloads_and_independent_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            service = Service(Path(directory))
            panel = RuntimePanel(service)
            try:
                with patch.object(service, "download_pair") as download:
                    for size in ("9B", "2B"):
                        button = next(
                            b
                            for b in panel.findChildren(QPushButton)
                            if b.text().startswith("Download Qwen3.5-" + size)
                        )
                        button.click()
                    self.assertEqual([c.args[0] for c in download.call_args_list], ["9b", "2b"])
                with patch.object(service, "setup") as setup:
                    service.download_pair("2b")
                    setup.assert_called_once_with(keys=("qwen_2b", "scope_2b"))
                panel.model.setCurrentIndex(panel.model.findData("2b"))
                panel.offload.setCurrentIndex(panel.offload.findData("modules"))
                self.assertEqual(service.store.setting("scope_model"), "2b")
                self.assertEqual(service.store.setting("scope_offload"), "modules")
                self.assertTrue(service.runtime.configured()[0].endswith("qwen_2b"))
                panel.choice.setCurrentIndex(panel.choice.findData("cpu"))
                self.assertFalse(panel.offload.isEnabled())
                self.assertEqual(service.store.setting("scope_offload"), "modules")
                self.assertFalse(service.jobs.list())
                for key in ("qwen_2b", "scope_2b"):
                    catalog = service.models.catalog(key)
                    self.assertEqual(len(catalog["revision"]), 40)
                    self.assertTrue(catalog["files"])
            finally:
                panel.timer.stop()
                panel.close()
                service.close()
                service.jobs.shutdown(wait=True)

    def test_offload_placement_never_cross_pairs_sae(self):
        for name in ("9b", "2b"):
            profile = model_profile(name)
            devices, sae = placement("cuda", "modules", profile)
            self.assertEqual(sae, "cpu")
            self.assertEqual(devices["lm_head"], "cpu")
            self.assertEqual(devices["model.language_model.layers.16"], "cuda")
            self.assertEqual(len(devices), profile["layers"] + 4)
            self.assertEqual(placement("cuda", "sae", profile), ("cuda", "cpu"))
            self.assertEqual(placement("cpu", "modules", profile), ("cpu", "cpu"))
        with self.assertRaises(ValueError):
            model_profile("4b")
