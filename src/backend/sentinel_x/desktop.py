"""Native Qt Widgets desktop. No JavaScript, browser engine, or local API server."""
from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSpinBox, QSplitter, QStackedWidget,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextBrowser, QTextEdit, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget, QMenu, QScrollArea,
)

from .activity_ui import job_summary, update_job_rows
from .config import model_profile
from .runtime_ui import RuntimePanel
from .conversation_ui import MessageComposer, MessageEditor, GenerationStatus, transcript_html
from .service import Service
from .ui import escape, tree, table, CheckpointControls

DONE = {'completed', 'failed', 'cancelled', 'interrupted'}
ROLE = Qt.ItemDataRole.UserRole


class Window(CheckpointControls, QMainWindow):
    def __init__(self, service=None, auto_setup=True):
        super().__init__()
        self.service = service or Service()
        self.conversation = None; self.attached = set(); self.watched = {}; self.reported = set()
        self.dialogs = []; self.latest_jobs = []; self.closed = False
        self.rendered_conversation = None; self.rendered_signature = None
        self.thinking_expanded = {}
        self.scope_expanded = {}
        self.scope_views = {}
        self.loading_draft = False; self.import_feedback = {}
        self.setWindowTitle('Sentinel-X · Evidence & propagation'); self.resize(1320, 900)
        self.setMinimumSize(900, 650)
        root = QWidget(); self.setCentralWidget(root); layout = QHBoxLayout(root)
        self.navigation = QListWidget(); self.navigation.setFixedWidth(140)
        self.navigation.addItems(['Chat', 'Evidence', 'Settings'])
        left = QVBoxLayout(); left.addWidget(self.heading('Sentinel-X', 'Local research workspace'))
        left.addWidget(self.navigation); left.addWidget(self.button('Open data directory', self.open_data))
        self.activity_button = self.button('Background work', self.open_activity)
        left.addWidget(self.activity_button)
        layout.addLayout(left)
        right = QVBoxLayout(); layout.addLayout(right, 1)
        self.pages = QStackedWidget(); right.addWidget(self.pages, 1)
        self.notice = QLabel('Local data · Local inference'); self.notice.setWordWrap(True)
        footer = QHBoxLayout(); footer.addWidget(self.notice, 1)
        self.dismiss_notice = self.button('Dismiss message', lambda: self.notice.clear())
        footer.addWidget(self.dismiss_notice); right.addLayout(footer)
        self.chat_page(); self.resources_page(); self.skills_page(); self.models_page(); self.activity_page()
        # Nest existing capability pages within Settings, preserving all operations.
        skills, models, activity = [self.pages.widget(i) for i in (2, 3, 4)]
        for widget in (skills, models, activity):
            self.pages.removeWidget(widget)
        self.settings_tabs = QTabWidget()
        for widget, title in ((models, 'Models'), (skills, 'Skills')):
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(widget)
            self.settings_tabs.addTab(scroll, title)
        self.pages.addWidget(self.settings_tabs)
        self.activity_dialog = QDialog(self); self.activity_dialog.setWindowTitle('Background work')
        self.activity_dialog.resize(1080, 620)
        self.activity_dialog.setMinimumSize(760, 480)
        QVBoxLayout(self.activity_dialog).addWidget(activity)
        # Removing a page from QStackedWidget leaves it explicitly hidden.
        # Reparenting into a layout does not clear that state.
        activity.show()
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex); self.navigation.setCurrentRow(0)
        self.timer = QTimer(self); self.timer.setInterval(800); self.timer.timeout.connect(self.poll); self.timer.start()
        self.refresh(); self.render_messages()
        last = self.service.store.setting('ui:last_conversation')
        if any(c['id'] == last for c in self.service.store.conversations()):
            self.conversation = last
            self.refresh_conversations(); self.render_messages()
        self.load_draft()
        self.sidebar.setVisible(self.service.store.setting('ui:sidebar', True))
        self.attachments_panel.setVisible(self.service.store.setting('ui:attachments', False))
        if auto_setup:
            QTimer.singleShot(0, lambda: self.run(self.offer_setup))

    def offer_setup(self, force=False):
        from .setup_dialog import SetupDialog, needs_prompt
        if not force and not needs_prompt(self.service):
            return
        previous = getattr(self, 'setup_guide', None)
        if previous is not None and previous.isVisible():
            previous.raise_(); return
        self.setup_guide = SetupDialog(self.service, self)
        self.setup_guide.show()

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
        return ident

    def open_activity(self):
        self._poll()
        self.activity_dialog.show(); self.activity_dialog.raise_()
        self.activity_dialog.activateWindow()

    def dialog(self, title, content):
        dialog = QDialog(self); dialog.setWindowTitle(title); dialog.resize(850, 650)
        layout = QVBoxLayout(dialog); layout.addWidget(content)
        layout.addWidget(self.button('Close', dialog.close)); self.dialogs.append(dialog); dialog.show()
        return dialog

    def chat_page(self):
        layout = self.page('Chat', 'Explore factual support and potential reach.')
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.button('Conversations', lambda: self.toggle_panel(self.sidebar, 'ui:sidebar')))
        self.attachment_button = self.button('Datasets · 0 attached', lambda: self.toggle_panel(self.attachments_panel, 'ui:attachments'))
        toolbar.addWidget(self.attachment_button)
        toolbar.addStretch()
        menu_button = QPushButton('Conversation actions')
        menu = QMenu(menu_button)
        for label, callback in [('Rename', self.rename), ('Export Markdown', self.export_conversation), ('Delete', self.delete)]:
            menu.addAction(label, lambda checked=False, fn=callback: self.run(fn))
        menu_button.setMenu(menu); toolbar.addWidget(menu_button)
        layout.addLayout(toolbar)
        splitter = QSplitter(); layout.addWidget(splitter, 1)
        conversations = QWidget(); self.sidebar = conversations; column = QVBoxLayout(conversations)
        conversations.setMinimumWidth(150)
        self.conversation_search = QLineEdit(); self.conversation_search.setPlaceholderText('Search conversations…')
        self.conversation_search.textChanged.connect(lambda: self.run(self.refresh_conversations))
        column.addWidget(self.conversation_search)
        self.conversation_list = QListWidget(); self.conversation_list.currentItemChanged.connect(self.select_conversation)
        column.addWidget(self.conversation_list)
        column.addWidget(self.button('New conversation', self.new_conversation)); splitter.addWidget(conversations)
        chat = QWidget(); column = QVBoxLayout(chat)
        chat.setMinimumWidth(300)
        self.transcript = QTextBrowser(); self.transcript.setOpenExternalLinks(False)
        self.transcript.document().setDefaultStyleSheet('a {color:#83d9b7;} h3 {color:#b9eed9;}')
        self.transcript.setOpenLinks(False); self.transcript.anchorClicked.connect(lambda url: self.run(lambda: self.message_action(url)))
        column.addWidget(self.transcript, 1)
        self.editor = MessageEditor(); self.editor.saved.connect(self.save_edit)
        self.editor.cancelled.connect(self.close_editor)
        self.generation = GenerationStatus(); column.addWidget(self.generation)
        self.live = self.generation.label
        self.generation.restore_requested.connect(lambda: self.run(self.restore_request))
        self.generation.details_requested.connect(lambda: self.dialog('Generation details', tree(self.generation.job)))
        self.setup_hint = self.button('Chat setup · check models and downloads', lambda: self.offer_setup(force=True))
        column.addWidget(self.setup_hint)
        self.message_input = MessageComposer(); self.message_input.setPlaceholderText('Ask about claims, virality, or both…'); self.message_input.setMaximumHeight(115)
        self.message_input.submitted.connect(lambda: self.run(self.send))
        self.compose_stack = QStackedWidget()
        self.compose_stack.addWidget(self.message_input); self.compose_stack.addWidget(self.editor)
        self.compose_stack.setFixedHeight(115)
        column.addWidget(self.compose_stack)
        self.draft_timer = QTimer(self); self.draft_timer.setSingleShot(True); self.draft_timer.setInterval(350)
        self.draft_timer.timeout.connect(lambda: self.run(self.save_draft))
        self.message_input.textChanged.connect(self.draft_changed)
        row = QHBoxLayout()
        self.import_posts_button = QPushButton('Import posts')
        import_menu = QMenu(self.import_posts_button)
        import_menu.addAction('Twitter15 / Twitter16 input folder…', lambda: self.run(self.import_twitter_posts))
        import_menu.addAction('JSON / JSONL file…', lambda: self.run(self.import_posts))
        self.import_posts_button.setMenu(import_menu); row.addWidget(self.import_posts_button)
        self.thinking = QCheckBox('Thinking · Off'); self.thinking.setAccessibleName('Thinking')
        self.thinking.setObjectName('thinkingSwitch')
        self.thinking.setToolTip('Think before answering. Responses may take longer. Changes apply to the next request.')
        self.thinking.toggled.connect(self.thinking_changed); row.addWidget(self.thinking); row.addStretch()
        self.stop_button = self.button('Stop', self.stop_chat); self.stop_button.setEnabled(False); row.addWidget(self.stop_button)
        self.send_button = self.button('Send ↑', self.send); row.addWidget(self.send_button)
        column.addLayout(row); splitter.addWidget(chat)
        attachments = QWidget(); self.attachments_panel = attachments; column = QVBoxLayout(attachments)
        attachments.setMinimumWidth(190)
        datasets_label = QLabel('Post datasets · select attachments'); datasets_label.setWordWrap(True)
        column.addWidget(datasets_label)
        self.graph_filter = QLineEdit(); self.graph_filter.setPlaceholderText('Filter cascades…'); self.graph_filter.textChanged.connect(self.filter_graphs); column.addWidget(self.graph_filter)
        self.graph_list = QTreeWidget(); self.graph_list.setHeaderLabels(['Cascade', 'Posts', 'Metadata'])
        self.graph_list.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.graph_list.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.graph_list.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.graph_list.itemChanged.connect(self.toggle_attachment); column.addWidget(self.graph_list)
        column.addWidget(self.button('Inspect / review graph', self.inspect_selected))
        label = QLabel('Truth ≠ reach\n\nEstimated graphs require explicit review. Their predictions are experimental.'); label.setWordWrap(True); column.addWidget(label)
        splitter.addWidget(attachments); splitter.setSizes([180, 800, 240])

    def refresh_conversations(self):
        self.conversation_list.blockSignals(True); self.conversation_list.clear()
        for c in self.service.store.conversations(self.conversation_search.text()):
            item = QListWidgetItem(c['title']); item.setData(ROLE, c['id']); self.conversation_list.addItem(item)
            if c['id'] == self.conversation: self.conversation_list.setCurrentItem(item)
        self.conversation_list.blockSignals(False)

    def select_conversation(self, current, previous):
        self.save_draft()
        self.conversation = current.data(ROLE) if current else None
        self.load_draft(); self.render_messages()

    def new_conversation(self):
        self.save_draft()
        self.conversation = self.service.store.create_conversation()['id']; self.attached.clear()
        self.load_draft(); self.refresh_conversations(); self.render_messages()

    def rename(self):
        if not self.conversation: return
        title, ok = QInputDialog.getText(self, 'Rename conversation', 'Name:')
        if ok and title.strip(): self.service.rename(self.conversation, title); self.refresh_conversations()

    def export_conversation(self):
        if not self.conversation:
            raise ValueError('Select a conversation to export')
        dialog = QFileDialog(self, 'Export conversation as Markdown')
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter('Markdown report (*.md)'); dialog.setDefaultSuffix('md')
        dialog.selectFile(f'Sentinel-X-{self.conversation[:8]}.md')
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        path = self.service.export(self.conversation, dialog.selectedFiles()[0])
        self.notice.setText(f'Conversation exported to {path}')

    def delete(self):
        if not self.conversation: return
        if QMessageBox.question(self, 'Delete conversation', 'Delete this conversation and its messages?') != QMessageBox.StandardButton.Yes: return
        self.service.delete_conversation(self.conversation)
        self.service.store.configure('draft:'+self.conversation, {})
        self.conversation = None
        self.load_draft(); self.refresh_conversations(); self.render_messages()

    def send(self):
        text = self.message_input.toPlainText().strip()
        if not text: return
        if not self.conversation:
            self.conversation = self.service.store.create_conversation(text[:70])['id']
            self.service.store.configure('draft:None', {})
        ident = self.service.chat(self.conversation, text, sorted(self.attached), self.thinking.isChecked())
        self.send_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.watch(ident); self.message_input.clear(); self.save_draft(); self.refresh_conversations()
        self._poll()

    def stop_chat(self):
        for job in self.service.jobs.list():
            if job['kind'] == f'chat:{self.conversation}' and job['status'] not in DONE:
                self.cancelling_job = job['id']
                self.service.cancel(job['id'])
        self._poll()

    def toggle_panel(self, panel, key):
        visible = panel.isHidden()
        if visible and self.width() < 1100:
            other = self.sidebar if panel is self.attachments_panel else self.attachments_panel
            other.hide()
        panel.setVisible(visible)
        self.service.store.configure(key, visible)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'attachments_panel') and self.width() < 1100:
            if not self.attachments_panel.isHidden() and not self.sidebar.isHidden():
                self.sidebar.hide()

    def draft_changed(self):
        if not self.loading_draft:
            self.draft_timer.start()

    def thinking_changed(self, checked):
        self.thinking.setText('Thinking · On' if checked else 'Thinking · Off')
        self.draft_changed()

    def save_draft(self):
        self.draft_timer.stop()
        self.service.store.configure('draft:'+str(self.conversation), {
            'text': self.message_input.toPlainText(), 'graph_ids': sorted(self.attached),
            'thinking': self.thinking.isChecked()})
        self.service.store.configure('ui:last_conversation', self.conversation)

    def load_draft(self):
        self.draft_timer.stop()
        self.loading_draft = True
        draft = self.service.store.setting('draft:'+str(self.conversation), {})
        self.message_input.setPlainText(draft.get('text', ''))
        self.attached = set(draft.get('graph_ids', []))
        self.thinking.setChecked(draft.get('thinking', False))
        self.loading_draft = False
        self.close_editor()
        self.refresh_graphs()
        self._poll()

    def render_messages(self):
        messages = self.service.store.messages(self.conversation) if self.conversation else []
        live_job = next((j for j in reversed(list(self.service.jobs.live.values()))
            if j.data['kind'] == f'chat:{self.conversation}'), None)
        preview = live_job.preview() if live_job and live_job.data['status'] != 'completed' else None
        if preview:
            preview['key'] = 'live-'+live_job.data['id']
        signature = repr((messages, preview, self.thinking_expanded, self.scope_expanded, self.scope_views))
        changed_conversation = self.rendered_conversation != self.conversation
        if signature == self.rendered_signature and not changed_conversation:
            return
        scroll = self.transcript.verticalScrollBar()
        old = scroll.value()
        follow = changed_conversation or scroll.maximum()-old < 35
        self.transcript.setHtml(transcript_html(messages, preview, self.thinking_expanded, self.scope_expanded, self.scope_views))
        scroll.setValue(scroll.maximum() if follow else old)
        self.rendered_signature = signature
        self.rendered_conversation = self.conversation
        conversation = self.conversation
        def restore_scroll():
            if not self.closed and self.conversation == conversation and self.rendered_signature == signature:
                scroll.setValue(scroll.maximum() if follow else old)
        QTimer.singleShot(0, restore_scroll)

    def message_action(self, url):
        action, value = url.toString().split(':', 1)
        if action == 'action':
            if value == 'import': self.import_posts()
            elif value == 'evidence': self.navigation.setCurrentRow(1)
            return
        if action == 'scopeview':
            key, control, choice = value.split(':', 2)
            view = self.scope_views.setdefault(key, {})
            if control == 'phase' and choice in ('answer', 'thinking', 'prompt'):
                view['phase'] = choice
                view.pop('token', None)
            elif control in ('formatting', 'technical'):
                view[control] = not view.get(control, False)
                if control == 'formatting': view.pop('token', None)
            elif control in ('call', 'token') and choice.isdigit():
                view[control] = int(choice)
                if control == 'call': view.pop('token', None)
            self.render_messages()
            return
        if action == 'scope':
            self.scope_expanded[value] = not self.scope_expanded.get(value, False)
            self.render_messages()
            return
        if action == 'thinking':
            active = next((j for j in self.service.jobs.live.values() if 'live-'+j.data['id'] == value), None)
            default = active.preview()['active'] if active else False
            self.thinking_expanded[value] = not self.thinking_expanded.get(value, default)
            self.render_messages()
            return
        ident = int(value.split(':')[0])
        message = next((m for m in self.service.store.messages(self.conversation) if m['id'] == ident), None)
        if not message: raise ValueError('Message no longer exists')
        if action == 'edit':
            self.compose_stack.setCurrentWidget(self.editor); self.compose_stack.setFixedHeight(215)
            self.setup_hint.hide(); self.send_button.setEnabled(False)
            self.editor.open_message(message)
        elif action == 'regenerate':
            self.save_draft()
            conversation, job = self.service.regenerate(self.conversation, ident)
            self.conversation = conversation
            self.load_draft(); self.watch(job); self.refresh_conversations(); self.render_messages()
        elif action == 'history':
            self.dialog('Message revision history', tree(message['data'].get('revisions', [])))
        elif action == 'citation':
            evidence = message['data']['evidence'][int(value.split(':')[1])]
            view = QTextBrowser()
            view.setPlainText(evidence['citation']+' · '+evidence['path']+'\n\n'+
                              evidence['text']+'\n\n'+str(evidence.get('provenance', {})))
            self.dialog('Evidence source', view)
        elif action == 'details':
            self.dialog('Original research record', tree(message['data']))

    def save_edit(self, ident, text, revision):
        try:
            self.service.edit_message(self.conversation, ident, text, revision)
        except Exception as exc:
            self.editor.error.setText(str(exc))
            return
        self.close_editor(); self.render_messages(); self.refresh_conversations()

    def close_editor(self):
        self.compose_stack.setCurrentWidget(self.message_input); self.compose_stack.setFixedHeight(115)
        self.refresh_models(sync_controls=False)
        self.send_button.setEnabled(not any(j.data['kind'] == f'chat:{self.conversation}' and
            j.data['status'] not in DONE for j in self.service.jobs.live.values()))

    def restore_request(self):
        request = self.generation.job.get('request', {})
        if not request: return
        editing = self.compose_stack.currentWidget() is self.editor
        if self.message_input.toPlainText().strip() or editing:
            question = 'Replace the current draft with this request?'
            if editing: question += '\nThis also closes the unsaved message edit.'
            if QMessageBox.question(self, 'Restore request', question) != QMessageBox.StandardButton.Yes:
                return
        self.close_editor()
        self.message_input.setPlainText(request['text'])
        self.attached = set(request.get('graph_ids', []))
        self.thinking.setChecked(request.get('thinking_enabled', False))
        self.refresh_graphs(); self.save_draft(); self.message_input.setFocus()

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
        self.attachment_button.setText(f'Datasets · {len(self.attached)} attached')

    def filter_graphs(self):
        for i in range(self.graph_list.topLevelItemCount()):
            item = self.graph_list.topLevelItem(i); item.setHidden(self.graph_filter.text().lower() not in item.text(0).lower())

    def toggle_attachment(self, item, column):
        ident = item.data(0, ROLE)
        self.attached.add(ident) if item.checkState(0) == Qt.CheckState.Checked else self.attached.discard(ident)
        self.attachment_button.setText(f'Datasets · {len(self.attached)} attached'); self.draft_changed()

    def import_posts(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import posts', '', 'Post datasets (*.json *.jsonl)')
        if path:
            if not self.conversation: self.new_conversation()
            target = self.conversation
            self.watch(self.service.preview_posts(path), lambda preview: self.import_preview(path, preview, target))
            self.notice.setText('Reading dataset in the background…')

    def attach_imported_graphs(self, ids, target):
        # Complete against the originating conversation, even after navigation.
        if self.conversation == target:
            self.attached.update(ids); self.refresh_graphs(); self.save_draft()
            if ids and self.attachments_panel.isHidden():
                self.toggle_panel(self.attachments_panel, 'ui:attachments')
        else:
            draft = self.service.store.setting('draft:'+str(target), {})
            draft['graph_ids'] = sorted(set(draft.get('graph_ids', [])) | set(ids))
            self.service.store.configure('draft:'+str(target), draft)

    def import_twitter_posts(self):
        path = QFileDialog.getExistingDirectory(self, 'Import Twitter15/16 · choose the case input folder')
        if not path: return
        if not self.conversation: self.new_conversation()
        target = self.conversation
        ident = self.service.import_twitter(path)
        def done(result):
            self.attach_imported_graphs(result['graph_ids'], target)
            count, rejected = result['imported'], len(result['errors'])
            self.notice.setText(
                f'Imported {count} cascade(s). Attached to the originating conversation.' if count else
                'No cascades imported.'
            )
            if rejected:
                self.notice.setText(self.notice.text()+f' {rejected} rejected · Open Background work for details.')
        self.watch(ident, done)
        self.notice.setText('Reading native Twitter dataset in the background…')

    def import_preview(self, path, preview, target):
        dialog = QDialog(self); dialog.setWindowTitle('Import posts'); dialog.resize(720, 540)
        layout = QVBoxLayout(dialog)
        label = QLabel(f'{preview["count"]} posts · {Path(path).name}'); label.setWordWrap(True); layout.addWidget(label)
        form = QFormLayout(); choices = {}; keys = preview['keys']
        names = {'id': 'Post ID (required)', 'text': 'Text (required)', 'author_id': 'Author',
                 'timestamp': 'Timestamp', 'parent_id': 'Parent post', 'cascade_id': 'Cascade'}
        for field, value in preview['mapping'].items():
            combo = QComboBox(); combo.addItem('Not provided', None)
            for key in keys: combo.addItem(key, key)
            combo.setCurrentIndex(combo.findData(value)); choices[field] = combo
            form.addRow(names[field], combo)
        layout.addLayout(form)
        combine = QCheckBox('Group posts as one cascade for estimation'); layout.addWidget(combine)
        layout.addWidget(table(keys, [[r.get(k, '') for k in keys] for r in preview['samples']]))
        error = QLabel(); error.setWordWrap(True); layout.addWidget(error)
        def apply(proposed):
            if not dialog.isVisible(): return
            for key, value in proposed.items(): choices[key].setCurrentIndex(choices[key].findData(value))
        def commit():
            try:
                mapping = {key: combo.currentData() for key, combo in choices.items()}
                job_id = self.service.import_posts_background(path, mapping, combine.isChecked())
            except Exception as exc:
                error.setText(str(exc)); return
            import_button.setEnabled(False)
            error.setText('Validating and importing in the background…')
            def done(ids):
                self.attach_imported_graphs(ids, target)
                self.notice.setText(f'Imported {preview["count"]} posts in {len(ids)} cascade(s). Attached to the originating conversation.')
                dialog.accept()
            self.watch(job_id, done)
            self.import_feedback[job_id] = (error, import_button)
        row = QHBoxLayout()
        row.addWidget(self.button('Suggest field mapping', lambda: self.watch(self.service.map_posts(path), apply)))
        import_button = self.button('Import and attach', commit); row.addWidget(import_button)
        row.addWidget(self.button('Close', dialog.reject)); layout.addLayout(row)
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
        layout = self.page('Resources', 'Add a folder of local documents, then index it to ground answers in evidence.')
        self.resource_path = QLineEdit(self.service.status()['resources']); layout.addWidget(self.resource_path)
        row = QHBoxLayout(); row.addWidget(self.button('Choose resources directory', self.choose_resources))
        row.addWidget(self.button('Index / update directory', lambda: self.watch(self.service.index(self.resource_path.text()))))
        self.fever_status = QLabel(); self.fever_status.setWordWrap(True); layout.addWidget(self.fever_status)
        fever_row = QHBoxLayout()
        self.fever_download = self.button('Download FEVER only', lambda: self.watch(self.service.fever()))
        self.fever_index = self.button('Build / resume FEVER index', lambda: self.watch(self.service.build_fever_index()))
        fever_row.addWidget(self.fever_download); fever_row.addWidget(self.fever_index)
        layout.addLayout(fever_row)
        layout.addLayout(row)
        label = QLabel('Search your indexed documents here. FEVER is an optional, large historical Wikipedia corpus; its evaluation labels are excluded from evidence.'); label.setWordWrap(True); layout.addWidget(label)
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

    def models_page(self):
        layout = self.page('Models', 'Set up chat, evidence search, and virality prediction.')
        self.model_status = QLabel(); self.model_status.setWordWrap(True); layout.addWidget(self.model_status)
        layout.addWidget(self.button('Set up downloads…', lambda: self.offer_setup(force=True)))
        self.runtime_panel = RuntimePanel(self.service, self); layout.addWidget(self.runtime_panel)
        self.checkpoint_controls(layout)
        layout.addWidget(self.button('Open separate GraphSAGE trainer', self.open_trainer))
        layout.addWidget(self.button('Refresh trained checkpoints', self.refresh_checkpoints))
        advanced_button = QPushButton('Advanced settings ▸'); advanced_button.setCheckable(True)
        layout.addWidget(advanced_button)
        advanced = QWidget(); self.advanced = advanced; inner = QVBoxLayout(advanced)
        inner.addWidget(self.button('Verify generation and tools', lambda: self.watch(self.service.smoke())))
        form = QFormLayout()
        self.context = QComboBox()
        for tokens in (2048, 4096, 8192, 16384, 32768): self.context.addItem(str(tokens), tokens)
        form.addRow('Context tokens', self.context); inner.addLayout(form)
        self.thinking_budget = QSpinBox(); self.thinking_budget.setRange(64, 2048); self.thinking_budget.setSingleStep(64)
        form.addRow('Maximum thinking tokens', self.thinking_budget)
        inner.addWidget(self.button('Apply runtime settings', self.apply_settings))
        self.runtime_details = QLabel(); self.runtime_details.setWordWrap(True); inner.addWidget(self.runtime_details)
        layout.addWidget(advanced); advanced.hide()
        advanced_button.toggled.connect(advanced.setVisible)
        advanced_button.toggled.connect(lambda checked: advanced_button.setText('Advanced settings ▾' if checked else 'Advanced settings ▸'))
        note = QLabel('Chat runs on this device. Memory allocation failures are reported when loading or generating. Offloading and the 2B fallback are available in Models; actual fit requires validation.')
        note.setWordWrap(True); layout.addWidget(note); layout.addStretch()

    def open_trainer(self):
        import os
        import subprocess
        if getattr(sys, 'frozen', False):
            executable = Path(sys.executable)
            if sys.platform == 'darwin':
                target = executable.parents[3]/'Sentinel-X-Trainer.app/Contents/MacOS/Sentinel-X-Trainer'
            else:
                name = 'Sentinel-X-Trainer.exe' if sys.platform == 'win32' else 'Sentinel-X-Trainer'
                target = executable.parent.parent/'Sentinel-X-Trainer'/name
            if not target.is_file():
                raise ValueError('Install the companion Sentinel-X Trainer bundle next to this app')
            command = [str(target)]
        else:
            command = [sys.executable, '-m', 'sentinel_x.trainer']
        subprocess.Popen([*command, '--data-dir', str(self.service.store.root)], env=os.environ.copy(),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.notice.setText('Sentinel-X Trainer opened as a separate application')

    def apply_settings(self):
        self.service.configure(context=self.context.currentData(),
                               thinking_budget=self.thinking_budget.value()); self.refresh_models()

    def refresh_models(self, sync_controls=True):
        status = self.service.status()
        self.refresh_fever(status['fever'])
        self.runtime_panel.refresh()
        chat_ready = all(status['models'][key]['ready'] for key in model_profile(status['scope_model'])['keys'])
        self.setup_hint.setVisible(not chat_ready)
        self.model_status.setText('Chat: '+('Ready' if chat_ready else 'Setup needed')+
            '\nSemantic search: '+('Ready' if status['models']['embedding']['ready'] else 'Keyword search available · embeddings optional')+
            '\nVirality: '+('Checkpoint selected' if status['checkpoint'] else 'Choose a trained checkpoint'))
        self.runtime_details.setText(f'Transformers + Qwen-Scope · {status["scope_device"]}\nData: {status["root"]}')
        if sync_controls:
            self.context.setCurrentIndex(self.context.findData(status['context']))
            self.thinking_budget.setValue(status['thinking_budget'])

    def refresh_fever(self, fever=None):
        fever = fever or self.service.models.fever_status()
        active = next((j for j in list(self.service.jobs.live.values()) if
            (j.data['kind'] in ('fever', 'fever-index') or
             j.data['kind'] == 'setup' and j.data.get('request', {}).get('fever')) and
            j.data['status'] in ('running', 'queued')), None)
        self.fever_status.setText('FEVER: '+('Working · see Background work for progress or cancel to resume later.' if active else
            'Indexed · available for evidence search.' if fever['indexed'] else
            'Downloaded · indexing deferred or incomplete. Build/resume to use all of it in chat.' if fever['downloaded'] else
            'Not downloaded. You can save the archive now and index it later.'))
        self.fever_download.setEnabled(not fever['downloaded'] and not active)
        self.fever_index.setEnabled(fever['downloaded'] and not active)

    def activity_page(self):
        layout = self.page('Background work', 'See what is running, how far it has progressed, and what to do if it stops.')
        self.activity = QTreeWidget()
        self.activity.setHeaderLabels(['Task', 'Status', 'Current step / details', 'Progress', 'Time'])
        self.activity.setRootIsDecorated(False); self.activity.setWordWrap(True)
        self.activity.setColumnWidth(0, 200); self.activity.setColumnWidth(3, 140)
        self.activity.setColumnWidth(4, 65)
        self.activity.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.activity.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.activity.header().setStretchLastSection(False)
        self.activity.itemSelectionChanged.connect(self.update_activity_selection)
        layout.addWidget(self.activity, 1)
        self.activity_selection = QLabel('No background tasks yet. Downloads and research tasks will appear here.')
        self.activity_selection.setTextFormat(Qt.TextFormat.PlainText)
        self.activity_selection.setWordWrap(True); layout.addWidget(self.activity_selection)
        row = QHBoxLayout()
        self.cancel_job_button = self.button('Cancel task', self.cancel_selected)
        self.retry_job_button = self.button('Retry / resume task', self.retry_selected)
        self.job_details_button = self.button('View details', self.job_details)
        for button in (self.cancel_job_button, self.retry_job_button, self.job_details_button):
            button.setEnabled(False); row.addWidget(button)
        layout.addLayout(row)

    def update_activity_selection(self):
        item = self.activity.currentItem()
        job = next((j for j in self.latest_jobs if item and j['id'] == item.data(0, ROLE)), None)
        self.cancel_job_button.setEnabled(bool(job and job['status'] not in DONE and not job.get('cancelling')))
        self.job_details_button.setEnabled(job is not None)
        retry = bool(job and job['status'] in ('failed', 'cancelled', 'interrupted') and
                     (job['kind'] in ('fever', 'fever-index', 'download:qwen', 'download:scope', 'download:embedding') or
                      job['kind'].startswith('runtime-download:') or
                      (job['kind'] == 'setup' and job.get('request'))))
        self.retry_job_button.setEnabled(retry)
        if job:
            summary = job_summary(job, {c['id']: c['title'] for c in self.service.store.conversations()})
            self.activity_selection.setText(summary['name']+' · '+summary['status']+'\n'+summary['detail'])
        elif self.latest_jobs:
            self.activity_selection.setText('Select a task for more information. Progress bars describe the current file or phase.')
        else:
            self.activity_selection.setText('No background tasks yet. Downloads and research tasks will appear here.')

    def cancel_selected(self):
        item = self.activity.currentItem()
        if item:
            self.service.cancel(item.data(0, ROLE)); self._poll()

    def retry_selected(self):
        item = self.activity.currentItem()
        if item:
            self.watch(self.service.retry_download(item.data(0, ROLE))); self._poll()

    def job_details(self):
        item = self.activity.currentItem()
        if not item: return
        job = self.service.jobs.get(item.data(0, ROLE))
        summary = job_summary(job, {c['id']: c['title'] for c in self.service.store.conversations()})
        tabs = QTabWidget(); readable = QTextBrowser()
        readable.setPlainText(summary['name']+'\nStatus: '+summary['status']+
            '\nTime: '+summary['elapsed']+'\n\n'+summary['detail']+
            '\n\nProgress refers to the current file or phase. Verification and indexing may follow a download.')
        tabs.addTab(readable, 'Task summary'); tabs.addTab(tree(job), 'Diagnostics')
        self.dialog('Background job details', tabs)

    def poll(self):
        if self.closed: return
        self.run(self._poll)

    def _poll(self):
        jobs = self.service.jobs.list(); self.latest_jobs = jobs
        self.refresh_fever()
        titles = {c['id']: c['title'] for c in self.service.store.conversations()}
        update_job_rows(self.activity, jobs, titles)
        self.update_activity_selection()
        for job in jobs:
            if job['status'] in DONE and job['id'] not in self.reported:
                self.reported.add(job['id'])
                if job['kind'] in ('setup', 'fever', 'fever-index') or job['kind'].startswith(('download:', 'runtime-')):
                    self.refresh_models(sync_controls=False)
        for ident, callback in list(self.watched.items()):
            job = self.service.jobs.get(ident)
            if job['status'] not in DONE: continue
            del self.watched[ident]
            if job['status'] == 'completed' and callback:
                self.run(lambda fn=callback, result=job['result']: fn(result))
            elif job['status'] in ('failed', 'interrupted'):
                self.notice.setText(f'{job_summary(job, titles)["name"]}: {job.get("error", job["status"])} · Open Background work for details.')
            feedback = self.import_feedback.pop(ident, None)
            if feedback and job['status'] != 'completed':
                feedback[0].setText(job.get('error', job['status'])); feedback[1].setEnabled(True)
        chat = next((j for j in jobs if j['kind'] == f'chat:{self.conversation}'), None)
        if chat and chat['id'] == getattr(self, 'cancelling_job', None): chat['cancelling'] = True
        active_chat = chat is not None and chat['status'] not in DONE
        self.send_button.setEnabled(not active_chat and self.compose_stack.currentWidget() is not self.editor)
        self.stop_button.setEnabled(active_chat and not chat.get('cancelling'))
        self.generation.update_job(chat)
        self.render_messages()
        selected_checkpoint = self.service.store.setting('checkpoint')
        if selected_checkpoint != getattr(self, 'last_checkpoint', None):
            self.last_checkpoint = selected_checkpoint; self.refresh_checkpoints()
        active = sum(j['status'] not in DONE for j in jobs)
        self.activity_button.setText(f'Background work · {active}' if active else 'Background work')

    def refresh(self):
        self.refresh_conversations(); self.refresh_graphs(); self.refresh_skills(); self.refresh_checkpoints(); self.refresh_models()

    def open_data(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.service.store.root)))

    def closeEvent(self, event):
        self.save_draft(); self.closed = True
        for timer in self.findChildren(QTimer): timer.stop()
        self.service.close(); event.accept()


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
QPushButton:focus,QCheckBox:focus,QLineEdit:focus,QTextEdit:focus {border:2px solid #83d9b7;}
QCheckBox#thinkingSwitch::indicator {width:32px;height:20px;border:1px solid #69818d;border-radius:10px;background:#344854;}
QCheckBox#thinkingSwitch::indicator:checked {background:#83d9b7;border:2px solid #dce8ed;}
QTextBrowser {padding:18px;}
'''


def main(root=None):
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName('Sentinel-X'); application.setOrganizationName('Sentinel-X')
    application.setStyle('Fusion'); application.setStyleSheet(STYLE)
    window = Window(Service(root)); window.show()
    return application.exec()


if __name__ == '__main__':
    sys.exit(main())
