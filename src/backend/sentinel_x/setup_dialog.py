"""Skippable startup guide: optional GPU engines, weights, and evidence."""
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)
from .activity_ui import job_summary
from .runtime_ui import RuntimePanel


def download_options(service):
    status = service.status()
    qwen = bool(status['qwen_path'] and Path(status['qwen_path']).is_file())
    return [
        {'key': 'qwen', 'name': 'Chat · Qwen3.5-9B (large download)', 'required': True, 'ready': qwen},
        {'key': 'embedding', 'name': 'Semantic search · BGE-small', 'required': True,
         'ready': status['models']['embedding']['ready']},
        {'key': 'fever', 'name': 'Optional evidence · FEVER Wikipedia (download only; index later)',
         'required': False, 'ready': status['fever']['downloaded'] or status['fever']['indexed']},
    ]


def needs_prompt(service):
    if service.store.setting('hide_setup_at_startup', False):
        return False
    return any(not option['ready'] for option in download_options(service))


class SetupDialog(QDialog):
    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service; self.job_id = None
        self.setWindowTitle('Set up Sentinel-X'); self.setModal(True); self.resize(760, 760)
        outer = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); layout = QVBoxLayout(content); scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        title = QLabel('<h2>Get Sentinel-X ready</h2><p>Choose what to download. You can skip for now and return later.</p>')
        title.setWordWrap(True); layout.addWidget(title)
        executable, _ = service.runtime.configured()
        ready = bool(executable and Path(executable).is_file())
        runtime = QLabel('Included engine is ready. GPU engines can be downloaded below.' if ready else
                        'No engine is available. Download a compatible engine below.')
        runtime.setWordWrap(True); layout.addWidget(runtime)
        self.runtime_panel = RuntimePanel(service, self); layout.addWidget(self.runtime_panel)
        self.choices = {}
        for option in download_options(service):
            checkbox = QCheckBox(option['name']+(' · Ready' if option['ready'] else ' · Not downloaded'))
            checkbox.setChecked(option['required'] and not option['ready'])
            checkbox.setEnabled(not option['ready']); self.choices[option['key']] = checkbox
            layout.addWidget(checkbox)
        self.index_fever = QCheckBox('Also build the FEVER search index now (can take a long time)')
        self.index_fever.setChecked(False); layout.addWidget(self.index_fever)
        self.index_fever.setEnabled(False)
        self.choices['fever'].toggled.connect(self.index_fever.setEnabled)
        note = QLabel('Saved FEVER files are kept without indexing. Build/resume its index from Evidence when needed. '
                      'Missing downloads will be offered again next time you open the app. '
                      'GraphSAGE training is available in the separate Sentinel-X Trainer application.')
        note.setWordWrap(True); layout.addWidget(note)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0); layout.addWidget(self.progress)
        self.message = QLabel('No downloads have started.'); self.message.setWordWrap(True); layout.addWidget(self.message)
        self.hide_at_startup = QCheckBox('Don’t show setup at startup again')
        self.hide_at_startup.setChecked(service.store.setting('hide_setup_at_startup', False))
        self.hide_at_startup.toggled.connect(lambda checked: service.store.configure('hide_setup_at_startup', checked))
        outer.addWidget(self.hide_at_startup)
        buttons = QHBoxLayout()
        self.skip = QPushButton('Skip for now'); self.skip.clicked.connect(self.reject); buttons.addWidget(self.skip)
        self.cancel = QPushButton('Cancel downloads'); self.cancel.setEnabled(False); self.cancel.clicked.connect(self.cancel_downloads); buttons.addWidget(self.cancel)
        self.download = QPushButton('Download selected'); self.download.clicked.connect(self.start); buttons.addWidget(self.download)
        outer.addLayout(buttons)
        self.timer = QTimer(self); self.timer.setInterval(400); self.timer.timeout.connect(self.poll)
        self.finished.connect(lambda result: self.timer.stop())
        self.finished.connect(lambda result: self.runtime_panel.timer.stop())

    def start(self):
        keys = [key for key in ('qwen', 'embedding') if self.choices[key].isChecked()]
        fever = self.choices['fever'].isChecked()
        if not keys and not fever:
            self.message.setText('Select at least one download, or continue into the app.'); return
        try:
            self.job_id = self.service.setup(keys, fever, self.index_fever.isChecked() and fever)
        except Exception as exc:
            self.message.setText(str(exc)); return
        self.download.setEnabled(False); self.cancel.setEnabled(True)
        for checkbox in self.choices.values(): checkbox.setEnabled(False)
        self.index_fever.setEnabled(False)
        self.skip.setText('Continue in workspace'); self.message.setText('Downloads queued. You can continue while they run.')
        self.progress.setRange(0, 0); self.timer.start()

    def cancel_downloads(self):
        if self.job_id:
            self.service.cancel(self.job_id); self.message.setText('Cancelling downloads…')

    def poll(self):
        if not self.job_id: return
        job = self.service.jobs.get(self.job_id)
        summary = job_summary(job)
        if summary['percent'] is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100); self.progress.setValue(summary['percent'])
        self.message.setText(summary['detail'])
        if job['status'] in ('completed', 'failed', 'cancelled', 'interrupted'):
            self.timer.stop(); self.cancel.setEnabled(False); self.progress.setRange(0, 100)
            options = {option['key']: option for option in download_options(self.service)}
            for key, checkbox in self.choices.items():
                checkbox.setText(options[key]['name']+(' · Ready' if options[key]['ready'] else ' · Not downloaded'))
                checkbox.setEnabled(not options[key]['ready'])
                if options[key]['ready']: checkbox.setChecked(False)
            self.download.setEnabled(True)
            self.index_fever.setEnabled(self.choices['fever'].isChecked())
            if job['status'] == 'completed':
                deferred = job.get('result', {}).get('fever', {}).get('index_deferred')
                self.progress.setValue(100); self.message.setText(
                    'Downloads saved. FEVER indexing is deferred; use Evidence → Build / resume FEVER index when needed.'
                    if deferred else 'Selected downloads are ready. You can continue.')
                self.skip.setText('Continue')
            else:
                self.message.setText(job.get('error', 'Downloads cancelled. Partial files are retained for retry.'))
                self.download.setText('Retry selected downloads')
