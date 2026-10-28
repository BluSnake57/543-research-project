"""Native Qt Widgets desktop. No JavaScript, browser engine, or local API server."""
from __future__ import annotations

import html
from pathlib import Path
import sys

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSpinBox, QSplitter, QStackedWidget,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextBrowser, QTextEdit, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .graphs import read_posts, suggest_mapping
from .service import Service

DONE = {'completed', 'failed', 'cancelled', 'interrupted'}
ROLE = Qt.ItemDataRole.UserRole


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


class Window(QMainWindow):
    def __init__(self, service=None, auto_setup=True):
        super().__init__()
        self.service = service or Service()
        self.conversation = None; self.attached = set(); self.watched = {}; self.reported = set()
        self.dialogs = []; self.latest_jobs = []; self.closed = False
        self.setWindowTitle('Veritas · Evidence & propagation'); self.resize(1320, 900)
        self.setMinimumSize(1000, 700)
        root = QWidget(); self.setCentralWidget(root); layout = QHBoxLayout(root)
        self.navigation = QListWidget(); self.navigation.setFixedWidth(190)
        self.navigation.addItems(['Conversations', 'Resources', 'Skills', 'Training', 'Models', 'Activity'])
        left = QVBoxLayout(); left.addWidget(self.heading('VERITAS', 'Local research workspace'))
        left.addWidget(self.navigation); left.addWidget(self.button('Open data directory', self.open_data))
        layout.addLayout(left)
        right = QVBoxLayout(); layout.addLayout(right, 1)
        self.pages = QStackedWidget(); right.addWidget(self.pages, 1)
        self.notice = QLabel('Local data · Local inference'); self.notice.setWordWrap(True)
        right.addWidget(self.notice)
        self.chat_page(); self.resources_page(); self.skills_page(); self.training_page(); self.models_page(); self.activity_page()
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex); self.navigation.setCurrentRow(0)
        self.timer = QTimer(self); self.timer.setInterval(800); self.timer.timeout.connect(self.poll); self.timer.start()
        self.refresh(); self.render_messages()
        if auto_setup and self.service.status()['first_run']:
            QTimer.singleShot(0, lambda: self.run(lambda: self.watch(self.service.setup())))

    def heading(self, title, description=''):
        label = QLabel(f'<h2>{escape(title)}</h2><p style="color:#93a9b9">{escape(description)}</p>')
        label.setWordWrap(True); return label

    def button(self, text, callback):
        button = QPushButton(text); button.clicked.connect(lambda checked=False: self.run(callback)); return button

    def run(self, callback):
        try:
            return callback()
        except Exception as exc:
            self.notice.setText('Error: '+str(exc)); self.notice.setStyleSheet('color:#e8b582')
            return None

    def page(self, title, description):
        widget = QWidget(); layout = QVBoxLayout(widget); layout.setSpacing(14)
        layout.addWidget(self.heading(title, description)); self.pages.addWidget(widget)
        return layout

    def watch(self, ident, callback=None):
        self.watched[ident] = callback
        self.notice.setText('Queued background work'); return ident

    def dialog(self, title, content):
        dialog = QDialog(self); dialog.setWindowTitle(title); dialog.resize(850, 650)
        layout = QVBoxLayout(dialog); layout.addWidget(content)
        layout.addWidget(self.button('Close', dialog.close)); self.dialogs.append(dialog); dialog.show()
        return dialog

    def chat_page(self):
        layout = self.page('Conversations', 'Assess factual support and potential reach as separate questions.')
        splitter = QSplitter(); layout.addWidget(splitter, 1)
        conversations = QWidget(); column = QVBoxLayout(conversations)
        self.conversation_search = QLineEdit(); self.conversation_search.setPlaceholderText('Search conversations…')
        self.conversation_search.textChanged.connect(lambda: self.run(self.refresh_conversations))
        column.addWidget(self.conversation_search)
        self.conversation_list = QListWidget(); self.conversation_list.currentItemChanged.connect(self.select_conversation)
        column.addWidget(self.conversation_list)
        row = QHBoxLayout(); row.addWidget(self.button('New', self.new_conversation)); row.addWidget(self.button('Rename', self.rename)); row.addWidget(self.button('Delete', self.delete))
        column.addLayout(row); splitter.addWidget(conversations)
        chat = QWidget(); column = QVBoxLayout(chat)
        self.transcript = QTextBrowser(); self.transcript.setOpenExternalLinks(False); column.addWidget(self.transcript, 1)
        self.live = QLabel(); self.live.setWordWrap(True); self.live.setMaximumHeight(70); column.addWidget(self.live)
        self.message_input = QTextEdit(); self.message_input.setPlaceholderText('Ask about claims, virality, or both…'); self.message_input.setMaximumHeight(115)
        column.addWidget(self.message_input)
        row = QHBoxLayout(); row.addWidget(self.button('Import JSON / JSONL posts', self.import_posts)); row.addStretch()
        row.addWidget(self.button('Stop response', self.stop_chat)); self.send_button = self.button('Send message ↑', self.send); row.addWidget(self.send_button)
        column.addLayout(row); splitter.addWidget(chat)
        attachments = QWidget(); column = QVBoxLayout(attachments)
        column.addWidget(QLabel('POST DATASETS · select attachments'))
        self.graph_filter = QLineEdit(); self.graph_filter.setPlaceholderText('Filter cascades…'); self.graph_filter.textChanged.connect(self.filter_graphs); column.addWidget(self.graph_filter)
        self.graph_list = QTreeWidget(); self.graph_list.setHeaderLabels(['Cascade', 'Posts', 'Metadata'])
        self.graph_list.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.graph_list.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.graph_list.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.graph_list.itemChanged.connect(self.toggle_attachment); column.addWidget(self.graph_list)
        column.addWidget(self.button('Inspect / review graph', self.inspect_selected))
        label = QLabel('Truth ≠ reach\n\nEstimated graphs require explicit review. Their predictions are experimental.'); label.setWordWrap(True); column.addWidget(label)
        splitter.addWidget(attachments); splitter.setSizes([210, 650, 290])

    def refresh_conversations(self):
        self.conversation_list.blockSignals(True); self.conversation_list.clear()
        for c in self.service.store.conversations(self.conversation_search.text()):
            item = QListWidgetItem(c['title']); item.setData(ROLE, c['id']); self.conversation_list.addItem(item)
            if c['id'] == self.conversation: self.conversation_list.setCurrentItem(item)
        self.conversation_list.blockSignals(False)

    def select_conversation(self, current, previous):
        self.conversation = current.data(ROLE) if current else None
        self.attached.clear(); self.run(self.refresh_graphs); self.render_messages()

    def new_conversation(self):
        self.conversation = self.service.store.create_conversation()['id']; self.attached.clear()
        self.refresh_conversations(); self.refresh_graphs(); self.render_messages()

    def rename(self):
        if not self.conversation: return
        title, ok = QInputDialog.getText(self, 'Rename conversation', 'Name:')
        if ok and title.strip(): self.service.rename(self.conversation, title); self.refresh_conversations()

    def delete(self):
        if not self.conversation: return
        if QMessageBox.question(self, 'Delete conversation', 'Delete this conversation and its messages?') != QMessageBox.StandardButton.Yes: return
        self.service.delete_conversation(self.conversation); self.conversation = None
        self.refresh_conversations(); self.render_messages()

    def send(self):
        text = self.message_input.toPlainText().strip()
        if not text: return
        if not self.conversation:
            self.conversation = self.service.store.create_conversation(text[:70])['id']
        ident = self.service.chat(self.conversation, text, sorted(self.attached))
        self.watch(ident); self.message_input.clear(); self.refresh_conversations()

    def stop_chat(self):
        for job in self.latest_jobs:
            if job['kind'] == f'chat:{self.conversation}' and job['status'] not in DONE:
                self.service.cancel(job['id'])

    def render_messages(self):
        if not self.conversation:
            self.transcript.setHtml('<h1>Understand the claim.<br>Explore its potential reach.</h1><p>Import posts, attach cascades, and ask a question. Resources ground factual judgments; GraphSAGE forecasts recorded cascade growth.</p>'); return
        pieces = []
        for message in self.service.store.messages(self.conversation):
            if message['role'] == 'tool': continue
            pieces.append(f'<h4>{"YOU" if message["role"] == "user" else "VERITAS"}</h4><p>{escape(message["content"])}</p>')
            data = message['data']
            for claim in data.get('claims', []):
                pieces.append(f'<blockquote><b>{escape(claim["verdict"].replace("_", " ").upper())}</b><p>{escape(claim["claim"])}</p><p>{escape(claim.get("explanation", ""))}</p><small>{escape(" · ".join(claim["citations"]))}</small></blockquote>')
            for tool in data.get('tools', []):
                if tool['name'] != 'graphsage_predict': continue
                result = tool['result']
                if isinstance(result, dict): pieces.append('<p>'+escape(result.get('error', result))+'</p>'); continue
                for p in result:
                    warning = '<p><b>Estimated graph — experimental prediction</b></p>' if p['estimated'] else ''
                    pieces.append(f'<blockquote><h4>GRAPHSAGE · {escape(p["cascade_id"])}</h4><b>{p["viral_probability"]:.1%} viral probability</b><p>≥{p["size_threshold"]} recorded posts · First {p["observation_minutes"]} minutes</p>{warning}</blockquote>')
            for e in data.get('evidence', []):
                source = e['provenance'].get('page', e['path'])
                pieces.append(f'<p><b>{escape(e["citation"])}</b> · {escape(source)} · {escape(e["provenance"])}</p><blockquote>{escape(e["text"])}</blockquote>')
            pieces.append('<hr>')
        self.transcript.setHtml(''.join(pieces)); self.transcript.verticalScrollBar().setValue(self.transcript.verticalScrollBar().maximum())

    def refresh_graphs(self):
        current = self.graph_list.currentItem(); ident = current.data(0, ROLE) if current else None
        self.graph_list.blockSignals(True); self.graph_list.clear()
        for g in self.service.store.graphs():
            state = 'Incomplete' if g.get('needs_estimation') else 'Estimated' if g.get('estimated') else 'Observed'
            item = QTreeWidgetItem([g['cascade_id'], str(len(g['nodes'])), state]); item.setData(0, ROLE, g['id'])
            item.setCheckState(0, Qt.CheckState.Checked if g['id'] in self.attached else Qt.CheckState.Unchecked)
            self.graph_list.addTopLevelItem(item)
            if g['id'] == ident: self.graph_list.setCurrentItem(item)
        self.graph_list.blockSignals(False); self.filter_graphs()

    def filter_graphs(self):
        for i in range(self.graph_list.topLevelItemCount()):
            item = self.graph_list.topLevelItem(i); item.setHidden(self.graph_filter.text().lower() not in item.text(0).lower())

    def toggle_attachment(self, item, column):
        ident = item.data(0, ROLE)
        self.attached.add(ident) if item.checkState(0) == Qt.CheckState.Checked else self.attached.discard(ident)

    def import_posts(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import posts', '', 'Post datasets (*.json *.jsonl)')
        if not path: return
        rows = read_posts(Path(path)); mapping = suggest_mapping(rows)
        dialog = QDialog(self); dialog.setWindowTitle('Map post fields'); dialog.resize(800, 520)
        layout = QVBoxLayout(dialog); layout.addWidget(QLabel(f'{len(rows)} posts · {path}'))
        form = QFormLayout(); choices = {}
        keys = sorted(set().union(*(r.keys() for r in rows[:100])))
        for field, value in mapping.items():
            combo = QComboBox(); combo.addItem('Missing', None)
            for key in keys: combo.addItem(key, key)
            combo.setCurrentIndex(combo.findData(value)); choices[field] = combo; form.addRow(field, combo)
        layout.addLayout(form)
        combine = QCheckBox('Group these records as one cascade for estimation'); layout.addWidget(combine)
        layout.addWidget(table(keys, [[r.get(k, '') for k in keys] for r in rows[:3]]))
        def apply(proposed):
            for key, value in proposed.items(): choices[key].setCurrentIndex(choices[key].findData(value))
        def commit():
            ids = self.service.import_posts(path, {key: combo.currentData() for key, combo in choices.items()}, combine.isChecked())
            self.attached.update(ids); dialog.accept(); self.refresh_graphs()
        row = QHBoxLayout(); row.addWidget(self.button('Ask Qwen to map fields', lambda: self.watch(self.service.map_posts(path), apply)))
        row.addWidget(self.button('Import', commit)); row.addWidget(self.button('Cancel', dialog.reject)); layout.addLayout(row)
        self.dialogs.append(dialog); dialog.show()

    def inspect_selected(self):
        item = self.graph_list.currentItem()
        if item: self.inspect_graph(item.data(0, ROLE))

    def inspect_graph(self, ident):
        graph = self.service.store.graph(ident)
        proposed = self.service.store.setting('proposal:'+ident)
        dialog = QDialog(self); dialog.setWindowTitle('Review cascade '+graph['cascade_id']); dialog.resize(900, 650)
        layout = QVBoxLayout(dialog); tabs = QTabWidget(); layout.addWidget(tabs)
        warning = QLabel(); warning.setWordWrap(True); layout.addWidget(warning)
        accept = self.button('Accept estimated graph', lambda: self.accept_graph(ident, dialog)); layout.addWidget(accept)
        def show(candidate):
            tabs.clear()
            tabs.addTab(table(['Node', 'Text', 'Delay (minutes)'], [[n['id'], n.get('text', ''), n.get('delay')] for n in candidate['nodes']]), 'Posts and timing')
            observed = {tuple(edge) for edge in graph['edges']}
            tabs.addTab(table(['Parent', 'Child', 'Provenance'], [[a, b, 'Observed' if (a, b) in observed else 'INFERRED'] for a, b in candidate['edges']]), 'Propagation links')
            tabs.addTab(tree({'assumptions': candidate.get('assumptions', ''), 'inferred': candidate.get('inferred', []), 'warnings': candidate.get('warnings', [])}), 'Assumptions & warnings')
            is_proposal = candidate is not graph
            accept.setVisible(is_proposal)
            warning.setText('ESTIMATED GRAPH — experimental prediction. Review all inferred links/times before acceptance.' if candidate.get('estimated') else '\n'.join(candidate.get('warnings', [])))
        show(proposed or graph)
        propose = self.button('Ask Qwen to propose missing metadata', lambda: self.watch(self.service.estimate(ident), show))
        propose.setEnabled(bool(graph.get('needs_estimation'))); layout.addWidget(propose)
        layout.addWidget(self.button('Close', dialog.close)); self.dialogs.append(dialog); dialog.show()

    def accept_graph(self, ident, dialog):
        self.service.accept(ident); dialog.accept(); self.refresh_graphs()

    def resources_page(self):
        layout = self.page('Resources', 'Local evidence from TXT, Markdown, JSON, JSONL and CSV. FEVER evidence is optional.')
        self.resource_path = QLineEdit(self.service.status()['resources']); layout.addWidget(self.resource_path)
        row = QHBoxLayout(); row.addWidget(self.button('Choose resources directory', self.choose_resources))
        row.addWidget(self.button('Index / update directory', lambda: self.watch(self.service.index(self.resource_path.text()))))
        row.addWidget(self.button('Download & index FEVER', lambda: self.watch(self.service.fever())))
        layout.addLayout(row)
        label = QLabel('FEVER is a large historical Wikipedia corpus. Claims/labels are kept outside the evidence index. Semantic search uses precompiled NumPy with memory-mapped vector files.'); label.setWordWrap(True); layout.addWidget(label)
        self.search_input = QLineEdit(); self.search_input.setPlaceholderText('Search for a claim or topic'); layout.addWidget(self.search_input)
        layout.addWidget(self.button('Search evidence', lambda: self.watch(self.service.search(self.search_input.text()), self.show_evidence)))
        self.evidence_view = QTextBrowser(); layout.addWidget(self.evidence_view, 1)

    def choose_resources(self):
        path = QFileDialog.getExistingDirectory(self, 'Resources directory')
        if path: self.resource_path.setText(path)

    def show_evidence(self, results):
        self.evidence_view.setHtml(''.join(f'<h4>{escape(e["citation"])} · {escape(e["retrieval"])}</h4><p>{escape(e["text"])}</p><small>{escape(e["path"])} · {escape(e["provenance"])}</small>' for e in results) or '<p>No matching evidence.</p>')

    def skills_page(self):
        layout = self.page('Skills', 'Local instructions bind Qwen to registered Python tools.')
        layout.addWidget(self.button('Import skill folder', self.import_skill))
        self.skill_list = QTreeWidget(); self.skill_list.setHeaderLabels(['Skill', 'Capability / error']); self.skill_list.itemChanged.connect(self.toggle_skill)
        layout.addWidget(self.skill_list)
        layout.addWidget(self.button('View skill instructions', self.skill_details))

    def refresh_skills(self):
        self.skill_list.blockSignals(True); self.skill_list.clear()
        for skill in self.service.skills.list():
            detail = skill.get('error') or ', '.join(t['handler'] for t in skill['tools'])
            item = QTreeWidgetItem([skill.get('name', skill['id']), detail]); item.setData(0, ROLE, skill)
            item.setCheckState(0, Qt.CheckState.Checked if skill['enabled'] else Qt.CheckState.Unchecked)
            if skill.get('error'): item.setDisabled(True)
            self.skill_list.addTopLevelItem(item)
        self.skill_list.blockSignals(False)

    def toggle_skill(self, item, column):
        skill = item.data(0, ROLE)
        self.run(lambda: self.service.store.configure('skill:'+skill['id'], item.checkState(0) == Qt.CheckState.Checked))

    def skill_details(self):
        item = self.skill_list.currentItem()
        if item:
            text = QTextBrowser(); text.setPlainText(item.data(0, ROLE).get('instructions', item.text(1)))
            self.dialog('Skill instructions', text)

    def import_skill(self):
        path = QFileDialog.getExistingDirectory(self, 'Skill directory')
        if path: self.service.import_skill(path); self.refresh_skills()

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
        self.checkpoints = QComboBox(); layout.addWidget(self.checkpoints)
        row = QHBoxLayout(); row.addWidget(self.button('Use checkpoint', self.select_checkpoint)); row.addWidget(self.button('View evaluation metrics', self.checkpoint_details)); layout.addLayout(row)
        self.metrics = QLabel(); self.metrics.setWordWrap(True); layout.addWidget(self.metrics)
        layout.addStretch()

    def import_twitter(self):
        path = QFileDialog.getExistingDirectory(self, 'Twitter15/16 directory or their parent')
        if path: self.watch(self.service.import_twitter(path))

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

    def models_page(self):
        layout = self.page('Models', 'Precompiled llama.cpp runtime and Python wheels. No compiler or Node.js is required for normal setup.')
        self.model_status = QLabel(); self.model_status.setWordWrap(True); layout.addWidget(self.model_status)
        row = QHBoxLayout()
        for key, label in [('runtime', 'Download precompiled runtime'), ('qwen', 'Download / verify Qwen'), ('embedding', 'Download / verify embeddings')]:
            row.addWidget(self.button(label, lambda name=key: self.watch(self.service.download(name))))
        layout.addLayout(row)
        row = QHBoxLayout(); row.addWidget(self.button('Import local GGUF', self.import_model)); row.addWidget(self.button('Choose llama-server', self.choose_runtime)); row.addWidget(self.button('Verify generation & tools', lambda: self.watch(self.service.smoke())))
        layout.addLayout(row)
        form = QFormLayout()
        self.context = QComboBox()
        for tokens in (2048, 4096, 8192, 16384, 32768): self.context.addItem(str(tokens), tokens)
        self.acceleration = QComboBox(); self.acceleration.addItem('Automatic GPU offload', 'auto'); self.acceleration.addItem('CPU only', '0')
        form.addRow('Context tokens', self.context); form.addRow('Acceleration', self.acceleration); layout.addLayout(form)
        layout.addWidget(self.button('Apply runtime settings', lambda: self.apply_settings()))
        label = QLabel('A quantized 9B model still requires several GB of memory; CPU operation can be slow. GPU acceleration depends on the downloaded runtime and installed drivers. Imports and inference stay on this device.'); label.setWordWrap(True); layout.addWidget(label); layout.addStretch()

    def import_model(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import Qwen GGUF', '', 'GGUF models (*.gguf)')
        if path: self.service.import_model(path); self.refresh_models()

    def choose_runtime(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose llama-server executable')
        if path: self.service.configure(llama_path=path); self.refresh_models()

    def apply_settings(self):
        self.service.configure(context=self.context.currentData(), gpu_layers=self.acceleration.currentData()); self.refresh_models()

    def refresh_models(self):
        status = self.service.status()
        self.model_status.setText(f'Runtime: {status["llama_path"] or "not installed"}\nQwen: {status["qwen_path"] or "not installed"}\nEmbeddings: {"downloaded" if status["models"]["embedding"]["ready"] else "not downloaded"}\nData: {status["root"]}')
        self.context.setCurrentIndex(self.context.findData(status['context']))
        self.acceleration.setCurrentIndex(self.acceleration.findData(status['gpu_layers']))

    def activity_page(self):
        layout = self.page('Activity', 'Downloads, imports, indexing, inference, and training run in the background.')
        self.activity = QTreeWidget(); self.activity.setHeaderLabels(['Job', 'Status', 'Progress / error']); layout.addWidget(self.activity)
        row = QHBoxLayout(); row.addWidget(self.button('Cancel selected job', self.cancel_selected)); row.addWidget(self.button('View job details', self.job_details)); layout.addLayout(row)

    def cancel_selected(self):
        item = self.activity.currentItem()
        if item: self.service.cancel(item.data(0, ROLE))

    def job_details(self):
        item = self.activity.currentItem()
        if item: self.dialog('Background job details', tree(self.service.jobs.get(item.data(0, ROLE))))

    def poll(self):
        if self.closed: return
        self.run(self._poll)

    def _poll(self):
        jobs = self.service.jobs.list(); self.latest_jobs = jobs
        current = self.activity.currentItem(); selected = current.data(0, ROLE) if current else None
        self.activity.clear(); changed = False
        for job in jobs:
            last = job['events'][-1] if job['events'] else {}
            progress = job.get('error', '')
            if last.get('event') == 'download':
                progress = f'{last["file"]}: {last["downloaded"]/1024**2:.0f} MB'
            item = QTreeWidgetItem([job['kind'], job['status'], progress]); item.setData(0, ROLE, job['id']); self.activity.addTopLevelItem(item)
            if job['id'] == selected: self.activity.setCurrentItem(item)
            if job['status'] in DONE and job['id'] not in self.reported:
                self.reported.add(job['id']); changed = True
                if job['status'] in ('failed', 'interrupted'): self.notice.setText(f'{job["kind"]}: {job.get("error", job["status"])}')
        for ident, callback in list(self.watched.items()):
            job = self.service.jobs.get(ident)
            if job['status'] in DONE:
                del self.watched[ident]
                if job['status'] == 'completed' and callback: self.run(lambda fn=callback, result=job['result']: fn(result))
        chat = next((j for j in jobs if j['kind'] == f'chat:{self.conversation}' and j['status'] not in DONE), None)
        self.send_button.setEnabled(chat is None)
        if chat:
            text = ''.join(e.get('text', '') for e in chat['events'] if e['event'] == 'token')
            self.live.setText('Working locally… '+text[-220:])
        else: self.live.clear()
        training = next((j for j in jobs if j['kind'] == 'training'), None)
        if training:
            self.chart.history = [e for e in training['events'] if e['event'] == 'training']; self.chart.update()
            last = self.chart.history[-1] if self.chart.history else None
            self.training_progress.setText(f'{training["status"]} · Epoch {last["epoch"]} · Training {last["loss"]:.4f} · Validation {last["validation_loss"]:.4f}' if last else training['status'])
        if changed: self.refresh(); self.render_messages()
        active = sum(j['status'] not in DONE for j in jobs)
        if active: self.notice.setText(f'{active} background job(s) · Activity shows progress and errors')

    def refresh(self):
        self.refresh_conversations(); self.refresh_graphs(); self.refresh_skills(); self.refresh_checkpoints(); self.refresh_models()

    def open_data(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.service.store.root)))

    def closeEvent(self, event):
        self.closed = True; self.timer.stop(); self.service.close(); event.accept()


STYLE = '''
QWidget {background:#121b24;color:#dce8ed;font-size:13px;}
QMainWindow {background:#121b24;}
QPushButton {background:#253b3d;border:1px solid #45615d;border-radius:6px;padding:10px;color:#b9eed9;}
QPushButton:hover {background:#34524e;} QPushButton:disabled {color:#6f8790;}
QLineEdit,QTextEdit,QTextBrowser,QListWidget,QTreeWidget,QTableWidget,QComboBox,QSpinBox {background:#192731;border:1px solid #344854;border-radius:5px;padding:7px;selection-background-color:#3b6f61;}
QHeaderView::section {background:#243540;color:#b8cbd5;border:0;padding:8px;}
QListWidget::item {padding:12px;} QListWidget::item:selected {background:#2a5148;color:#aeebd2;}
QTabBar::tab {background:#233743;padding:10px;} QTabBar::tab:selected {background:#365b51;}
QLabel {padding:3px;} QSplitter::handle {background:#30434d;}
'''


def main(root=None):
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName('Veritas'); application.setOrganizationName('Veritas')
    application.setStyle('Fusion'); application.setStyleSheet(STYLE)
    window = Window(Service(root)); window.show()
    return application.exec()


if __name__ == '__main__':
    sys.exit(main())
