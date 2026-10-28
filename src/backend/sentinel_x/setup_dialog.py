"""Skippable startup guide: bundled runtime, downloadable weights and optional evidence."""
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout


def download_options(service):
    status = service.status()
    qwen = bool(status['qwen_path'] and Path(status['qwen_path']).is_file())
    return [
        {'key': 'qwen', 'name': 'Qwen3.5-9B — chat model (large download)', 'required': True, 'ready': qwen},
        {'key': 'embedding', 'name': 'BGE-small — local evidence search', 'required': True,
         'ready': status['models']['embedding']['ready']},
        {'key': 'fever', 'name': 'FEVER Wikipedia evidence — optional, very large download and index',
         'required': False, 'ready': bool(service.store.setting('fever_ready', False))},
    ]


def needs_prompt(service):
    return any(not option['ready'] for option in download_options(service))


class SetupDialog(QDialog):
    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service; self.job_id = None
        self.setWindowTitle('Set up Sentinel-X'); self.setModal(True); self.resize(660, 400)
        layout = QVBoxLayout(self)
        title = QLabel('<h2>Get Sentinel-X ready</h2><p>Choose what to download. You can skip for now and return later.</p>')
        title.setWordWrap(True); layout.addWidget(title)
        executable, _ = service.runtime.configured()
        ready = bool(executable and Path(executable).is_file())
        runtime = QLabel('llama.cpp: included and ready — no runtime download needed.' if ready else
                        'Bundled llama.cpp is missing from this installation. Reinstall the complete application bundle.')
        runtime.setWordWrap(True); layout.addWidget(runtime)
        self.choices = {}
        for option in download_options(service):
            checkbox = QCheckBox(option['name']+(' · already available' if option['ready'] else ''))
            checkbox.setChecked(option['required'] and not option['ready'])
            checkbox.setEnabled(not option['ready']); self.choices[option['key']] = checkbox
            layout.addWidget(checkbox)
        note = QLabel('Missing downloads will be offered again next time you open the app. '
                      'GraphSAGE training is available in the separate Sentinel-X Trainer application.')
        note.setWordWrap(True); layout.addWidget(note)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0); layout.addWidget(self.progress)
        self.message = QLabel('No downloads have started.'); self.message.setWordWrap(True); layout.addWidget(self.message)
        buttons = QHBoxLayout()
        self.skip = QPushButton('Skip for now'); self.skip.clicked.connect(self.reject); buttons.addWidget(self.skip)
        self.cancel = QPushButton('Cancel downloads'); self.cancel.setEnabled(False); self.cancel.clicked.connect(self.cancel_downloads); buttons.addWidget(self.cancel)
        self.download = QPushButton('Download selected'); self.download.clicked.connect(self.start); buttons.addWidget(self.download)
        layout.addLayout(buttons)
        self.timer = QTimer(self); self.timer.setInterval(400); self.timer.timeout.connect(self.poll)
        self.finished.connect(lambda result: self.timer.stop())

    def start(self):
        keys = [key for key in ('qwen', 'embedding') if self.choices[key].isChecked()]
        fever = self.choices['fever'].isChecked()
        if not keys and not fever:
            self.message.setText('Select at least one download, or continue into the app.'); return
        try:
            self.job_id = self.service.setup(keys, fever)
        except Exception as exc:
            self.message.setText(str(exc)); return
        self.download.setEnabled(False); self.cancel.setEnabled(True)
        for checkbox in self.choices.values(): checkbox.setEnabled(False)
        self.skip.setText('Continue in workspace'); self.message.setText('Downloads queued. You can continue while they run.')
        self.progress.setRange(0, 0); self.timer.start()

    def cancel_downloads(self):
        if self.job_id:
            self.service.cancel(self.job_id); self.message.setText('Cancelling downloads…')

    def poll(self):
        if not self.job_id: return
        job = self.service.jobs.get(self.job_id)
        event = job['events'][-1] if job['events'] else {}
        if event.get('event') == 'download':
            total = event.get('total')
            if total:
                self.progress.setRange(0, 100); self.progress.setValue(min(100, int(event['downloaded']/total*100)))
            self.message.setText(f'{event["file"]}: {event["downloaded"]/1024**2:.0f} MB downloaded')
        elif event.get('event') == 'indexing':
            self.progress.setRange(0, 0); self.message.setText('Indexing evidence: '+event.get('path', ''))
        if job['status'] in ('completed', 'failed', 'cancelled', 'interrupted'):
            self.timer.stop(); self.cancel.setEnabled(False); self.progress.setRange(0, 100)
            options = {option['key']: option for option in download_options(self.service)}
            for key, checkbox in self.choices.items():
                checkbox.setEnabled(not options[key]['ready'])
                if options[key]['ready']: checkbox.setChecked(False)
            self.download.setEnabled(True)
            if job['status'] == 'completed':
                self.progress.setValue(100); self.message.setText('Selected downloads are ready. You can continue.'); self.skip.setText('Continue')
            else:
                self.message.setText(job.get('error', 'Downloads cancelled. Partial files are retained for retry.'))
                self.download.setText('Retry selected downloads')
