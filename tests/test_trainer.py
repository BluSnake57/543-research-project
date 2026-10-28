import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import threading
import time
import unittest
from PySide6.QtWidgets import QApplication, QPushButton
from sentinel_x.desktop import Window
from sentinel_x.service import Service
from sentinel_x.trainer import TrainerWindow
from test_core import graph


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_starting_trainer_does_not_interrupt_main_application_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); main = Service(root); release = threading.Event()
            trainer = None
            try:
                job = main.jobs.submit('chat:example', lambda j: release.wait(5))
                deadline = time.monotonic()+5
                while main.jobs.get(job)['status'] != 'running' and time.monotonic() < deadline: time.sleep(.01)
                trainer = Service(root, role='trainer')
                self.assertEqual(main.jobs.get(job)['status'], 'running')
                self.assertEqual(trainer.jobs.list(), [])
                trainer.close()
                self.assertFalse(main.jobs.live[job].cancelled.is_set())
            finally:
                release.set(); main.close(); main.jobs.pool.shutdown(wait=True)
                if trainer: trainer.jobs.pool.shutdown(wait=True)

    def test_separate_trainer_window_trains_and_main_app_sees_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); main = Service(root); training = Service(root, role='trainer')
            for i in range(6):
                main.store.save_graph(graph(f'n{i}', 2, 30)); main.store.save_graph(graph(f'p{i}', 8, 20))
            window = Window(main, auto_setup=False); trainer = TrainerWindow(training)
            try:
                window.show(); trainer.show(); self.app.processEvents()
                self.assertNotIn('Training', [window.navigation.item(i).text() for i in range(window.navigation.count())])
                self.assertIn('Sentinel-X Trainer', trainer.windowTitle())
                trainer.threshold.setValue(4); trainer.epochs.setValue(1)
                start = next(b for b in trainer.findChildren(QPushButton) if b.text() == 'Start training')
                start.click()
                deadline = time.monotonic()+15
                while time.monotonic() < deadline:
                    self.app.processEvents()
                    jobs = training.jobs.list()
                    if jobs and jobs[0]['status'] in ('completed', 'failed'): break
                    time.sleep(.02)
                self.assertEqual(jobs[0]['status'], 'completed', jobs[0].get('error'))
                trainer._poll(); window._poll()
                self.assertEqual(main.jobs.list(), [])
                self.assertEqual(trainer.checkpoints.count(), 1)
                self.assertEqual(window.checkpoints.count(), 1)
                self.assertEqual(window.service.store.setting('checkpoint'), jobs[0]['result']['id'])
            finally:
                trainer.close(); window.close()
                training.jobs.pool.shutdown(wait=True); main.jobs.pool.shutdown(wait=True)
                self.app.processEvents()
