"""Conversation presentation, inline editing, and contextual job feedback."""
import time
import json
import re

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from .ui import escape
from .activity_ui import duration


class MessageComposer(QTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and event.modifiers() & (
                Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            self.submitted.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class MessageEditor(QWidget):
    saved = Signal(int, str, int)
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label = QLabel('Edit message')
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.text = QTextEdit()
        self.text.setAccessibleName('Message text')
        self.text.setMaximumHeight(180)
        self.text.setMinimumHeight(60)
        layout.addWidget(self.text)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        row = QHBoxLayout()
        self.save = QPushButton('Save changes')
        self.save.setMinimumHeight(36)
        self.save.clicked.connect(self.submit)
        row.addWidget(self.save)
        cancel = QPushButton('Cancel')
        cancel.setMinimumHeight(36)
        cancel.clicked.connect(self.cancelled.emit)
        row.addWidget(cancel)
        layout.addLayout(row)
        self.hide()

    def open_message(self, message):
        self.ident = message['id']
        self.revision = message['data'].get('revision', 0)
        self.label.setText('Edit your message' if message['role'] == 'user'
                           else 'Edit answer · original assessments and evidence are retained')
        self.text.setPlainText(message['content'])
        self.error.clear()
        self.show()
        self.text.setFocus()

    def submit(self):
        if not self.text.toPlainText().strip():
            self.error.setText('Message cannot be empty.')
            return
        self.saved.emit(self.ident, self.text.toPlainText(), self.revision)


class GenerationStatus(QWidget):
    restore_requested = Signal()
    details_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setAccessibleName('Generation status')
        layout.addWidget(self.label)
        row = QHBoxLayout()
        self.restore = QPushButton('Restore request')
        self.restore.clicked.connect(self.restore_requested.emit)
        row.addWidget(self.restore)
        self.details = QPushButton('Details')
        self.details.clicked.connect(self.details_requested.emit)
        row.addWidget(self.details)
        row.addStretch()
        layout.addLayout(row)
        self.update_job(None)

    def update_job(self, job):
        self.job = job
        self.setVisible(job is not None)
        if not job:
            return
        status = job['status']
        terminal = status in ('completed', 'failed', 'cancelled', 'interrupted')
        events = job.get('events', [])
        stage = next((e['stage'] for e in reversed(events) if e['event'] == 'stage'), 'Queued')
        elapsed = max(0, int((events[-1].get('time', time.time()) if terminal and events
                              else time.time()) - job['created']))
        text = status.capitalize() if terminal else stage
        if job.get('cancelling') and not terminal:
            text = 'Cancelling'
        mode = ' · Thinking enabled' if job.get('request', {}).get('thinking_enabled') else ''
        self.label.setText(f'{text}{mode} · {elapsed}s'+
                           (f'\n{job["error"]}' if job.get('error') else ''))
        warning = next((e['warning'] for e in reversed(events) if e.get('warning')), None)
        if warning:
            self.label.setText(self.label.text()+'\n'+warning)
        self.restore.setVisible(status in ('failed', 'cancelled', 'interrupted') and bool(job.get('request')))
        self.details.setVisible(terminal or bool(warning))


def draft_summary(content):
    """Decode only the summary string, including a safely decodable partial string."""
    match = re.search(r'"summary"\s*:\s*"', content)
    if not match:
        return ''
    encoded = content[match.end()-1:]
    try:
        summary, _ = json.JSONDecoder().raw_decode(encoded)
        return summary
    except ValueError:
        for trim in range(7):
            prefix = encoded if trim == 0 else encoded[:-trim]
            try:
                summary = json.loads(prefix+'"')
                return summary.encode('utf-8', 'ignore').decode('utf-8')
            except ValueError:
                continue
    return ''


def thinking_dropdown(key, label, text, expanded, truncated=False):
    arrow = '▼' if expanded else '▶'
    result = f'<p><a href="thinking:{key}">{arrow} {escape(label)}</a></p>'
    if expanded:
        note = '<p><i>Showing the most recent 64,000 characters.</i></p>' if truncated else ''
        result += note+'<blockquote>'+escape(text).replace('\n', '<br>')+'</blockquote>'
    return result


def scope_dropdown(key, records, expanded, edited=False):
    from .scope import explanation_text
    statuses = ', '.join(dict.fromkeys(r['status'] for r in records))
    label = 'Explanation · Scope · ' + statuses
    arrow = '▾' if expanded else '▸'
    result = f'<p><a href="scope:{key}">{arrow} {escape(label)}</a></p>'
    if edited:
        result += '<p><i>Scope analysis belongs to the original response.</i></p>'
    if expanded:
        result += '<pre>' + escape(explanation_text(records)) + '</pre>'
    return result


def transcript_html(messages, preview=None, thinking_expanded=None, scope_expanded=None):
    scope_expanded = scope_expanded or {}
    thinking_expanded = thinking_expanded or {}
    if not messages:
        return ('<h2>What would you like to investigate?</h2>'
                '<p>Ask a question below, <a href="action:import">import posts</a>, or '
                '<a href="action:evidence">add local evidence</a>.</p>'
                '<p>Factual support and potential reach are assessed separately.</p>')
    pieces = []
    for message in messages:
        if message['role'] == 'tool':
            continue
        ident = message['id']
        data = message['data']
        role = 'You' if message['role'] == 'user' else 'Sentinel-X'
        actions = f'<a href="edit:{ident}">Edit</a>'
        if message['role'] == 'user':
            actions += f' · <a href="regenerate:{ident}">Regenerate from here</a>'
        if data.get('revision'):
            actions += f' · <a href="history:{ident}">Edited · revision history</a>'
        pieces.append(f'<h3>{role}</h3><small>{actions}</small>')
        if data.get('outdated_by'):
            pieces.append('<p><b>Based on an earlier message version</b> · Changed message(s): '+
                          escape(', '.join(str(x) for x in data['outdated_by']))+'</p>')
        pieces.append('<p>'+escape(message['content'])+'</p>')
        if data.get('thinking_seconds', 0) > 0 and message['role'] == 'assistant':
            label = 'Thought for '+duration(max(1, round(data['thinking_seconds'])))
            if data.get('thinking_trace'):
                pieces.append(thinking_dropdown(str(ident), label, data['thinking_trace'],
                    thinking_expanded.get(str(ident), False), data.get('thinking_trace_truncated', False)))
            else:
                pieces.append('<p><i>'+label+' · Trace was not saved for this older response.</i></p>')
        elif message['role'] == 'assistant' and data.get('thinking_enabled') and data.get('thinking_seconds') == 0:
            pieces.append('<p><i>Thinking was enabled; the model did not emit a separate thinking trace.</i></p>')
        if data.get('scope'):
            pieces.append(scope_dropdown(str(ident), data['scope'],
                                         scope_expanded.get(str(ident), False), bool(data.get('revision'))))
        if data.get('revision') and message['role'] == 'assistant':
            pieces.append('<p><i>Edited answer. Assessments and evidence below belong '
                          'to the original generated response.</i></p>')
        for claim in data.get('claims', []):
            citations = ' '.join(f'<a href="citation:{ident}:{i}">{escape(e["citation"])}</a>'
                for i, e in enumerate(data.get('evidence', [])) if e['citation'] in claim['citations'])
            pieces.append(f'<blockquote><b>{escape(claim["verdict"].replace("_", " ").upper())}</b>'
                f'<p>{escape(claim["claim"])}</p><p>{escape(claim.get("explanation", ""))}</p>'
                f'<p>{citations}</p></blockquote>')
        for tool in data.get('tools', []):
            if tool['name'] != 'graphsage_predict':
                continue
            result = tool['result']
            if isinstance(result, dict):
                pieces.append('<p>'+escape(result.get('error', result))+'</p>')
                continue
            for prediction in result:
                warning = '<p><b>Estimated graph · experimental prediction</b></p>' if prediction['estimated'] else ''
                pieces.append(f'<blockquote><h4>Potential reach · {escape(prediction["cascade_id"])}</h4>'
                    f'<b>{prediction["viral_probability"]:.1%} viral probability</b>'
                    f'<p>≥{prediction["size_threshold"]} recorded posts · First '
                    f'{prediction["observation_minutes"]} minutes</p>{warning}</blockquote>')
        if data.get('evidence') or data.get('tools') or data.get('reasoning_summary'):
            pieces.append(f'<p><a href="details:{ident}">Evidence and original research details</a></p>')
        pieces.append('<hr>')
    if preview and preview.get('scope'):
        key = preview.get('key', 'live')
        pieces.append(scope_dropdown(key, preview['scope'], scope_expanded.get(key, False)))
    if preview and preview.get('active'):
        key = preview.get('key', 'live')
        pieces.append(thinking_dropdown(key, 'Thinking · '+duration(preview['seconds']),
            preview.get('trace') or preview['text'], thinking_expanded.get(key, True),
            preview.get('trace_truncated', False)))
    elif preview and preview.get('seconds', 0) > 0:
        label = 'Thought for '+duration(max(1, round(preview['seconds'])))
        if preview.get('trace'):
            key = preview.get('key', 'live')
            pieces.append(thinking_dropdown(key, label, preview['trace'], thinking_expanded.get(key, False),
                                            preview.get('trace_truncated', False)))
        else:
            pieces.append('<p><i>'+label+'</i></p>')
    if preview and not preview.get('active'):
        answer = draft_summary(preview.get('answer', ''))
        if answer:
            pieces.append('<h3>Sentinel-X · Writing response…</h3><p>'+escape(answer)+
                          '</p><small>Draft response · citations have not been checked yet.</small>')
    return ''.join(pieces)
