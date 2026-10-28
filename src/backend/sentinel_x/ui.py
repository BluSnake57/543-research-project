"""Native widgets shared by the assistant and the separate trainer application."""
import html
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import (QWidget, QTreeWidget, QTreeWidgetItem, QTableWidget,
    QTableWidgetItem, QHeaderView, QLabel, QComboBox, QHBoxLayout)

def escape(value):
    return html.escape(str(value)).replace('\n', '<br>')


def tree(value):
    widget = QTreeWidget(); widget.setHeaderLabels(['Property', 'Value'])
    def append(parent, key, item):
        node = QTreeWidgetItem([str(key), '' if isinstance(item, (dict, list)) else str(item)])
        parent.addChild(node) if isinstance(parent, QTreeWidgetItem) else parent.addTopLevelItem(node)
        if isinstance(item, dict):
            for name, entry in item.items(): append(node, name, entry)
        elif isinstance(item, list):
            for index, entry in enumerate(item): append(node, index+1, entry)
    if isinstance(value, dict):
        for key, item in value.items(): append(widget, key, item)
    else:
        append(widget, 'Items', value)
    widget.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    return widget


def table(columns, rows):
    widget = QTableWidget(len(rows), len(columns)); widget.setHorizontalHeaderLabels(columns)
    widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    widget.setWordWrap(True)
    for i, row in enumerate(rows):
        for j, value in enumerate(row): widget.setItem(i, j, QTableWidgetItem(str(value)))
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    return widget


class LossChart(QWidget):
    def __init__(self):
        super().__init__(); self.history = []; self.setMinimumHeight(160)

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor('#17232d'))
        if not self.history:
            painter.setPen(QColor('#8da4b5'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Training and validation loss will appear here')
            return
        maximum = max(.01, max(max(r['loss'], r['validation_loss']) for r in self.history))
        for key, color in [('loss', '#83d9b7'), ('validation_loss', '#e6b779')]:
            painter.setPen(QPen(QColor(color), 2))
            points = [(20+i/max(len(self.history)-1, 1)*(self.width()-40),
                       self.height()-20-r[key]/maximum*(self.height()-40)) for i, r in enumerate(self.history)]
            for (a, b), (c, d) in zip(points, points[1:]):
                painter.drawLine(int(a), int(b), int(c), int(d))


class CheckpointControls:
    def checkpoint_controls(self, layout):
        layout.addWidget(QLabel('GRAPHSAGE INFERENCE CHECKPOINT'))
        self.checkpoints = QComboBox(); layout.addWidget(self.checkpoints)
        row = QHBoxLayout(); row.addWidget(self.button('Use checkpoint', self.select_checkpoint))
        row.addWidget(self.button('View evaluation metrics', self.checkpoint_details)); layout.addLayout(row)
        self.metrics = QLabel(); self.metrics.setWordWrap(True); layout.addWidget(self.metrics)

    def refresh_checkpoints(self):
        selected = self.service.store.setting('checkpoint'); previous = self.checkpoints.currentData()
        self.checkpoints.clear()
        for c in self.service.checkpoints():
            self.checkpoints.addItem(f'{c["id"][:8]} · ≥{c["config"]["threshold"]} posts · {c["config"]["window"]} min', c['id'])
        target = previous or selected
        index = self.checkpoints.findData(target)
        if index >= 0: self.checkpoints.setCurrentIndex(index)
        selected_model = next((c for c in self.service.checkpoints() if c['id'] == selected), None)
        if selected_model:
            report = selected_model['report']['test']
            self.metrics.setText(f'Selected: {selected[:8]} · Test F1 {report["f1"]:.3f} · Precision {report["precision"]:.3f} · Recall {report["recall"]:.3f} · PR-AUC {report["pr_auc"]:.3f}')

    def select_checkpoint(self):
        ident = self.checkpoints.currentData()
        if ident: self.service.configure(checkpoint=ident); self.refresh_checkpoints()

    def checkpoint_details(self):
        ident = self.checkpoints.currentData()
        checkpoint = next((c for c in self.service.checkpoints() if c['id'] == ident), None)
        if checkpoint: self.dialog('Checkpoint evaluation and provenance', tree(checkpoint))

