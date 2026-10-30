"""Optional runtime downloads and activation, shared by setup and Models."""
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from .activity_ui import job_summary
from .runtime_packages import LABELS, HELP, runtime_choices


class RuntimePanel(QWidget):
    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service; self.job_id = None; self.busy = False
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel('llama.cpp engines · optional GPU downloads'); layout.addWidget(title)
        self.choice = QComboBox(); self.choice.setAccessibleName('Runtime download backend')
        for backend in runtime_choices():
            self.choice.addItem(LABELS[backend], backend)
        layout.addWidget(self.choice)
        self.description = QLabel(); self.description.setWordWrap(True); layout.addWidget(self.description)
        self.inventory = QLabel(); self.inventory.setWordWrap(True)
        self.inventory.setTextFormat(Qt.TextFormat.PlainText); layout.addWidget(self.inventory)
        self.use_after = QCheckBox('Use after download (waits for the current response)')
        self.use_after.setChecked(True); layout.addWidget(self.use_after)
        row = QHBoxLayout()
        self.download = QPushButton('Download engine'); self.download.clicked.connect(self.start_download)
        self.activate = QPushButton('Use installed engine'); self.activate.clicked.connect(self.use_installed)
        self.bundled = QPushButton('Use included engine'); self.bundled.clicked.connect(self.use_bundled)
        for button in (self.download, self.activate, self.bundled): row.addWidget(button)
        layout.addLayout(row)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.hide(); layout.addWidget(self.progress)
        self.message = QLabel('Downloads start only when you choose Download engine.')
        self.message.setTextFormat(Qt.TextFormat.PlainText); self.message.setWordWrap(True); layout.addWidget(self.message)
        self.choice.currentIndexChanged.connect(self.refresh)
        self.timer = QTimer(self); self.timer.setInterval(500); self.timer.timeout.connect(self.poll); self.timer.start()
        self.refresh()

    def refresh(self):
        backend = self.choice.currentData()
        self.description.setText(HELP.get(backend, 'No downloadable runtime is available for this platform.'))
        entries = self.service.store.setting('installed_runtimes', {})
        entry = entries.get(backend)
        selected, _ = self.service.runtime.configured()
        active_name = next((LABELS[k] for k, value in entries.items() if value['executable'] == selected),
                           'Included or locally selected engine')
        self.inventory.setText('Selected: '+active_name+'\n'+
            ('This download: installed · '+entry['release'] if entry else 'This download: not installed'))
        if self.service.runtime.active:
            active = self.service.runtime.active
            self.inventory.setText(self.inventory.text()+'\nRunning: '+
                ('CPU' if active['gpu_layers'] == '0' else 'automatic GPU offload requested'))
        self.choice.setEnabled(not self.busy); self.use_after.setEnabled(not self.busy)
        self.download.setEnabled(not self.busy and backend is not None)
        self.activate.setEnabled(not self.busy and entry is not None)
        self.bundled.setEnabled(not self.busy)

    def begin(self, callback):
        try:
            self.job_id = callback()
            self.busy = True; self.progress.setRange(0, 0); self.progress.show()
            self.message.setText('Queued. Follow progress here or in Background work.')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh()

    def start_download(self):
        self.begin(lambda: self.service.download_runtime(self.choice.currentData(), self.use_after.isChecked()))

    def use_installed(self):
        self.begin(lambda: self.service.activate_runtime(self.choice.currentData()))

    def use_bundled(self):
        self.begin(lambda: self.service.activate_runtime('bundled'))

    def poll(self):
        if getattr(self.window(), 'closed', False):
            self.timer.stop(); return
        try:
            if not self.job_id:
                return
            job = self.service.jobs.get(self.job_id); summary = job_summary(job)
            self.message.setText(summary['detail'])
            if summary['percent'] is None:
                self.progress.setRange(0, 0)
            else:
                self.progress.setRange(0, 100); self.progress.setValue(summary['percent'])
            if job['status'] not in ('completed', 'failed', 'cancelled', 'interrupted'):
                return
            result = job.get('result', {})
            if job['status'] == 'completed' and result.get('activation_job'):
                self.job_id = result['activation_job']
                self.message.setText('Engine installed. Waiting to activate after current model work.')
                return
            self.job_id = None; self.busy = False
            if job['status'] == 'completed':
                devices = result.get('devices', [])
                self.message.setText(('Engine selected.' if result.get('activated') else 'Engine installed.')+
                                     ('\nDetected: '+'; '.join(devices) if devices else ''))
            self.refresh()
        except Exception as exc:
            self.message.setText(str(exc))
