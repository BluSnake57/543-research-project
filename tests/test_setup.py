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
        self.service.close(); self.service.jobs.shutdown(wait=True); self.temp.cleanup()

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
            (folder/'.ready').write_text(self.service.models.manifest[key]['revision'])
            return {'model': key}
        with patch.object(self.service.models, 'install', side_effect=install) as installer:
            dialog = SetupDialog(self.service); dialog.start()
            deadline = time.monotonic()+5
            while self.service.jobs.get(dialog.job_id)['status'] not in ('completed', 'failed') and time.monotonic() < deadline:
                time.sleep(.01)
            dialog.poll()
            self.assertEqual(self.service.jobs.get(dialog.job_id)['status'], 'completed')
            self.assertEqual([call.args[0] for call in installer.call_args_list], ['qwen', 'scope', 'embedding'])
            self.assertFalse(dialog.choices['qwen'].isEnabled()); dialog.close()
        # Optional evidence is still offered on later launches until installed.
        self.assertTrue(needs_prompt(self.service))
        self.service.store.configure('fever_ready', True)
        self.assertFalse(needs_prompt(self.service))
        with self.assertRaises(ValueError): self.service.download('runtime')

    def test_opening_main_window_prompts_without_starting_downloads(self):
        from sentinel_x.desktop import Window
        self.service.store.configure('setup_started', True)
        window = Window(self.service)
        try:
            window.show(); self.app.processEvents()
            self.assertTrue(window.setup_guide.isVisible())
            self.assertEqual(self.service.jobs.list(), [])
            window.setup_guide.skip.click(); self.app.processEvents()
            self.assertFalse(window.setup_guide.isVisible())
            self.assertTrue(needs_prompt(self.service))
        finally:
            window.close(); self.app.processEvents()

    def test_hide_setup_is_persisted_and_manual_reopen_can_restore_startup_prompt(self):
        from sentinel_x.desktop import Window
        dialog = SetupDialog(self.service)
        dialog.hide_at_startup.setChecked(True); dialog.skip.click()
        self.assertFalse(needs_prompt(self.service))
        self.service.close(); self.service.jobs.shutdown(wait=True)
        self.service = Service(Path(self.temp.name))
        window = Window(self.service)
        try:
            window.show(); self.app.processEvents()
            self.assertFalse(hasattr(window, 'setup_guide'))
            window.offer_setup(force=True)
            self.assertTrue(window.setup_guide.isVisible())
            self.assertTrue(window.setup_guide.hide_at_startup.isChecked())
            window.setup_guide.hide_at_startup.setChecked(False)
            window.setup_guide.close()
            self.assertTrue(needs_prompt(self.service))
            self.assertEqual(self.service.jobs.list(), [])
        finally:
            window.close(); self.app.processEvents()

    def test_fever_setup_defaults_to_download_only_with_explicit_index_option(self):
        dialog = SetupDialog(self.service)
        try:
            dialog.choices['scope'].setChecked(False); dialog.choices['qwen'].setChecked(False); dialog.choices['embedding'].setChecked(False)
            dialog.choices['fever'].setChecked(True)
            self.assertTrue(dialog.index_fever.isEnabled()); self.assertFalse(dialog.index_fever.isChecked())
            with patch.object(self.service, 'setup', return_value='fixture') as setup:
                dialog.start()
            setup.assert_called_once_with([], True, False)
        finally:
            dialog.close()
