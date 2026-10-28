"""Separate GraphSAGE training desktop; the assistant application only performs inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFormLayout, QSpinBox, QFileDialog, QDialog)
from .service import Service
from .state import data_directory
from .ui import CheckpointControls, LossChart, tree


class TrainerWindow(CheckpointControls, QMainWindow):
    def __init__(self, service=None):
        super().__init__()
        self.service = service or Service(role='trainer')
        self.setWindowTitle('Sentinel-X Trainer · GraphSAGE'); self.resize(980, 820)
        self.dialogs = []; self.notified = set()
        self.training_page(); self.refresh_checkpoints()
        self.timer = QTimer(self); self.timer.setInterval(600); self.timer.timeout.connect(self.poll); self.timer.start()

    def page(self, title, description):
        root = QWidget(); self.setCentralWidget(root); layout = QVBoxLayout(root)
        heading = QLabel(f'<h2>Sentinel-X Trainer</h2><p>{description}</p>'); heading.setWordWrap(True)
        layout.addWidget(heading)
        self.notice = QLabel('Separate trainer · Checkpoints are shared with Sentinel-X'); self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        return layout

    def button(self, text, callback):
        button = QPushButton(text); button.clicked.connect(lambda checked=False: self.run(callback)); return button

    def run(self, callback):
        try:
            return callback()
        except Exception as exc:
            self.notice.setText(str(exc))

    def watch(self, ident):
        self.notice.setText('Queued: '+self.service.jobs.get(ident)['kind']); return ident

    def dialog(self, title, content):
        dialog = QDialog(self); dialog.setWindowTitle(title); dialog.resize(850, 650)
        layout = QVBoxLayout(dialog); layout.addWidget(content); layout.addWidget(self.button('Close', dialog.close))
        self.dialogs.append(dialog); dialog.show()

    def training_page(self):
        layout = self.page('Training', 'GraphSAGE binary virality: observed early propagation → recorded cascade size.')
        layout.addWidget(self.button('Import native Twitter15 / Twitter16', self.import_twitter))
        form = QFormLayout()
        self.threshold = QSpinBox(); self.threshold.setRange(2, 1_000_000); self.threshold.setValue(100)
        self.window = QSpinBox(); self.window.setRange(1, 100_000); self.window.setValue(60)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 2000); self.epochs.setValue(80)
        form.addRow('Viral at ≥ recorded posts', self.threshold); form.addRow('Observation window (minutes)', self.window); form.addRow('Maximum epochs', self.epochs)
        layout.addLayout(form)
        layout.addWidget(QLabel('CPU training · Two GraphSAGE layers · Seed 42 · Event-level train/validation/test split'))
        layout.addWidget(self.button('Start training', lambda: self.watch(self.service.train(self.threshold.value(), self.window.value(), self.epochs.value()))))
        self.chart = LossChart(); layout.addWidget(self.chart)
        self.training_progress = QLabel('No training run yet'); layout.addWidget(self.training_progress)
        self.checkpoint_controls(layout)
        layout.addWidget(self.button('Cancel active work', self.cancel_training))
        layout.addWidget(self.button('View latest run details', self.latest_details))
        layout.addStretch()

    def import_twitter(self):
        path = QFileDialog.getExistingDirectory(self, 'Twitter15/16 directory or their parent')
        if path: self.watch(self.service.import_twitter(path))

    def poll(self):
        self.run(self._poll)

    def _poll(self):
        jobs = self.service.jobs.list()
        active = [j for j in jobs if j['status'] in ('queued', 'running')]
        if active:
            self.notice.setText(active[0]['kind']+' · '+active[0]['status'])
        for job in jobs:
            if job['status'] in ('completed', 'failed', 'cancelled', 'interrupted') and job['id'] not in self.notified:
                self.notified.add(job['id'])
                result = job.get('result') or {}
                self.notice.setText(job.get('error') or (f'Imported {result["imported"]} cascades; {len(result["errors"])} errors' if 'imported' in result else job['kind']+' · '+job['status']))
                self.refresh_checkpoints()
        training = next((j for j in jobs if j['kind'] == 'training'), None)
        if training:
            self.chart.history = [e for e in training['events'] if e['event'] == 'training']; self.chart.update()
            last = self.chart.history[-1] if self.chart.history else None
            self.training_progress.setText(f'{training["status"]} · Epoch {last["epoch"]} · Training {last["loss"]:.4f} · Validation {last["validation_loss"]:.4f}' if last else training['status'])

    def latest_details(self):
        jobs = self.service.jobs.list()
        if jobs: self.dialog('Latest training/import run', tree(jobs[0]))

    def cancel_training(self):
        for job in self.service.jobs.list():
            if job['status'] in ('queued', 'running'): self.service.cancel(job['id'])

    def closeEvent(self, event):
        self.timer.stop(); self.service.close(); event.accept()


def main(root=None):
    if root is None:
        parser = argparse.ArgumentParser(prog='sentinel-x-trainer')
        parser.add_argument('--data-dir', type=Path, default=data_directory())
        root = parser.parse_args().data_dir
    from .desktop import STYLE
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName('Sentinel-X Trainer'); app.setStyle('Fusion'); app.setStyleSheet(STYLE)
    window = TrainerWindow(Service(root, role='trainer')); window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
