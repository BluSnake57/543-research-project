"""Device preferences and Scope model verification, shared by setup and Models."""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget


class RuntimePanel(QWidget):
    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Qwen3.5 + matching Qwen-Scope · required for every model response")
        )
        from .config import MODEL_PROFILES

        self.model = QComboBox()
        self.model.setAccessibleName("Scope model size")
        for key, profile in MODEL_PROFILES.items():
            self.model.addItem(profile["label"] + " + Scope", key)
        self.model.setCurrentIndex(self.model.findData(service.store.setting("scope_model", "9b")))
        self.model.currentIndexChanged.connect(self.select)
        layout.addWidget(self.model)
        for key, profile in MODEL_PROFILES.items():
            download = QPushButton("Download " + profile["label"] + " + matching Scope SAE")
            download.clicked.connect(lambda checked=False, name=key: service.download_pair(name))
            layout.addWidget(download)
        self.offload = QComboBox()
        self.offload.setAccessibleName("CPU offloading")
        self.offload.addItem("No CPU offloading", "none")
        self.offload.addItem("Keep Scope SAE on CPU", "sae")
        self.offload.addItem("Keep Scope, embeddings, output head and vision on CPU", "modules")
        self.offload.setCurrentIndex(
            self.offload.findData(service.store.setting("scope_offload", "none"))
        )
        self.offload.currentIndexChanged.connect(self.select)
        layout.addWidget(self.offload)
        self.choice = QComboBox()
        self.choice.setAccessibleName("Scope device")
        self.choice.addItem("CUDA · 4-bit NF4", "cuda")
        self.choice.addItem("CPU · 4-bit NF4, slower", "cpu")
        self.choice.setCurrentIndex(
            self.choice.findData(service.store.setting("scope_device", "cuda"))
        )
        self.choice.currentIndexChanged.connect(self.select)
        layout.addWidget(self.choice)
        self.inventory = QLabel()
        self.inventory.setWordWrap(True)
        layout.addWidget(self.inventory)
        button = QPushButton("Verify Qwen and Scope files")
        button.clicked.connect(lambda: service.verify_scope())
        layout.addWidget(button)
        note = QLabel(
            "Old llama.cpp engines and GGUF files are unused on this branch. "
            "They are preserved until you explicitly remove them or factory reset."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def select(self):
        self.service.configure(
            scope_device=self.choice.currentData(),
            scope_model=self.model.currentData(),
            scope_offload=self.offload.currentData(),
        )
        self.refresh()

    def refresh(self):
        import psutil

        from .config import model_profile

        ready = self.service.models.status()
        self.offload.setEnabled(self.choice.currentData() == "cuda")
        self.inventory.setText(
            f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GiB. "
            "CUDA availability and memory are checked in the worker before loading.\n"
            + "\n".join(
                f"{key}: {'downloaded' if ready[key]['ready'] else 'download required'}"
                for key in model_profile(self.service.store.setting("scope_model", "9b"))["keys"]
            )
        )
