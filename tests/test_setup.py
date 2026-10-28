import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from sentinel_x.service import Service
from sentinel_x.setup_dialog import SetupDialog, needs_prompt


class SetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.service = Service(Path(self.temp.name))

    def tearDown(self):
        self.service.close(); self.service.jobs.pool.shutdown(wait=True); self.temp.cleanup()

    def test_skip_starts_no_download_and_prompt_returns_even_after_prior_setup(self):
        self.service.store.configure('setup_started', True)
        dialog = SetupDialog(self.service); dialog.show(); self.app.processEvents()
        self.assertFalse(dialog.choices['fever'].isChecked())
        self.assertTrue(dialog.choices['qwen'].isChecked())
        dialog.skip.click(); self.app.processEvents()
        self.assertFalse(dialog.isVisible())
        self.assertEqual(self.service.jobs.list(), [])
        self.assertTrue(needs_prompt(self.service))

    def test_downloads_weights_only_never_runtime(self):
        def install(key, job):
            folder = self.service.store.root/'models'/key; folder.mkdir(exist_ok=True)
            (folder/'.ready').write_text('test')
            if key == 'qwen': (folder/'model.gguf').write_bytes(b'GGUF')
            return {'model': key}
        with patch.object(self.service.models, 'install', side_effect=install) as installer, \
             patch.object(self.service.models, 'install_runtime', side_effect=AssertionError('Runtime must be bundled')):
            dialog = SetupDialog(self.service); dialog.start()
            deadline = time.monotonic()+5
            while self.service.jobs.get(dialog.job_id)['status'] not in ('completed', 'failed') and time.monotonic() < deadline:
                time.sleep(.01)
            dialog.poll()
            self.assertEqual(self.service.jobs.get(dialog.job_id)['status'], 'completed')
            self.assertEqual([call.args[0] for call in installer.call_args_list], ['qwen', 'embedding'])
            self.assertFalse(dialog.choices['qwen'].isEnabled()); dialog.close()
        # Optional evidence is still offered on later launches until installed.
        self.assertTrue(needs_prompt(self.service))
        self.service.store.configure('fever_ready', True)
        self.assertFalse(needs_prompt(self.service))
        with self.assertRaises(ValueError): self.service.download('runtime')
