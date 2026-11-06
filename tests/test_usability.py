"""Deterministic persistence, runtime protocol, and Qt interaction regressions."""
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import Qt, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from sentinel_x.desktop import Window
from sentinel_x.exporting import render_conversation
from sentinel_x.service import Service
from sentinel_x.state import Store


def finish(service, ident):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        job = service.jobs.get(ident)
        if job['status'] in ('completed', 'failed', 'cancelled'):
            return job
        time.sleep(.005)
    raise AssertionError('Job timed out')


class EditingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = Service(Path(self.temp.name))
        self.store = self.service.store
        self.conversation = self.store.create_conversation('Research')['id']

    def tearDown(self):
        self.service.close(); self.service.jobs.shutdown(wait=True)
        self.temp.cleanup()

    def test_revisions_outdated_records_restart_and_export(self):
        user = self.store.message(self.conversation, 'user', 'Original', {'thinking_enabled': True})
        tool = self.store.message(self.conversation, 'tool', '{}', {'name': 'evidence_search', 'result': []})
        answer = self.store.message(self.conversation, 'assistant', 'Generated', {'claims': [], 'evidence': []})
        self.service.edit_message(self.conversation, user, 'Changed', 0)
        self.service.edit_message(self.conversation, user, 'Changed again', 1)
        self.service.edit_message(self.conversation, answer, 'My annotation', 0)
        rows = Store(self.store.root).messages(self.conversation)
        self.assertEqual([r['content'] for r in rows], ['Changed again', '{}', 'My annotation'])
        self.assertEqual(rows[0]['data']['revision'], 2)
        self.assertEqual(rows[0]['data']['revisions'][0]['content'], 'Original')
        self.assertEqual(rows[1]['id'], tool)
        self.assertEqual(rows[1]['data']['result'], [])
        self.assertEqual(rows[2]['data']['outdated_by'], [user])
        exported = render_conversation(self.store, self.conversation)
        for text in ('Changed again', 'Original', 'My annotation', 'Revision history',
                     'Based on an earlier message version', 'Thinking:** On', 'original generated answer'):
            self.assertIn(text, exported)

    def test_edit_rejects_empty_conflicts_wrong_conversation_and_tool(self):
        user = self.store.message(self.conversation, 'user', 'Question')
        tool = self.store.message(self.conversation, 'tool', '{}')
        other = self.store.create_conversation()['id']
        for conversation, ident, content, revision in (
                (self.conversation, user, ' ', 0), (other, user, 'edit', 0),
                (self.conversation, tool, 'edit', 0), (self.conversation, user, 'edit', 5)):
            with self.assertRaises(ValueError):
                self.service.edit_message(conversation, ident, content, revision)
        self.assertEqual(self.store.messages(self.conversation)[0]['content'], 'Question')

    def test_branch_remaps_dependencies_and_preserves_original(self):
        first = self.store.message(self.conversation, 'user', 'first')
        self.store.message(self.conversation, 'assistant', 'answer')
        target = self.store.message(self.conversation, 'user', 'next', {'thinking_enabled': True})
        self.store.edit_message(self.conversation, first, 'updated', 0)
        with patch.object(self.service.assistant, 'chat', return_value={'summary': 'done'}) as chat:
            branch, job = self.service.regenerate(self.conversation, target)
            self.assertEqual(finish(self.service, job)['status'], 'completed')
            self.assertTrue(chat.call_args.args[-1])
        copied = self.store.messages(branch)
        self.assertEqual(len(copied), 2)
        self.assertEqual(copied[1]['data']['outdated_by'], [copied[0]['id']])
        self.assertEqual(len(self.store.messages(self.conversation)), 3)

    def test_context_uses_edited_text_excludes_outdated_answers(self):
        first = self.store.message(self.conversation, 'user', 'Old question')
        self.store.message(self.conversation, 'assistant', 'Stale answer')
        self.store.message(self.conversation, 'tool', 'Stale tool')
        self.store.message(self.conversation, 'user', 'Follow-up')
        self.service.edit_message(self.conversation, first, 'New question')
        captured = []
        def stream(messages, tools, job):
            captured.extend(messages)
            return {'role': 'assistant', 'content': '{"summary":"ok","claims":[]}'}
        with patch.object(self.service.runtime, 'stream', side_effect=stream):
            ident = self.service.chat(self.conversation, 'Continue', [], True)
            job = finish(self.service, ident)
        self.assertEqual(job['status'], 'completed')
        content = [m['content'] for m in captured]
        self.assertIn('New question', content)
        self.assertIn('Follow-up', content)
        self.assertNotIn('Stale answer', content)
        self.assertNotIn('Stale tool', content)
        self.assertTrue(job['request']['thinking_enabled'])
        self.assertTrue(self.store.messages(self.conversation)[-2]['data']['audit']['generation']['thinking_enabled'])
        self.assertIn('Checking response and citations', [e.get('stage') for e in job['events']])

    def test_background_import_preview_and_failure_are_atomic(self):
        source = self.store.root/'posts.json'
        source.write_text(json.dumps([{'id': 'p1', 'text': 'one'}, {'id': 'p1', 'text': 'duplicate'}]))
        preview = finish(self.service, self.service.preview_posts(source))['result']
        self.assertEqual(preview['count'], 2)
        with self.assertRaises(ValueError):
            self.service.import_posts_background(source, {'id': 'id'})
        job = finish(self.service, self.service.import_posts_background(source, {'id': 'id', 'text': 'text'}))
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(self.store.graphs(), [])
        source.write_text(json.dumps([{'id': 'p1', 'text': 'one'}]))
        job = finish(self.service, self.service.import_posts_background(source, {'id': 'id', 'text': 'text'}))
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(len(job['result']), 1)

    def test_tool_rounds_report_real_stages(self):
        answers = [
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call1', 'type': 'function',
              'function': {'name': 'evidence_search', 'arguments': '{"query":"Mars"}'}}]},
            {'role': 'assistant', 'content': '{"summary":"No evidence","claims":[]}'},
        ]
        with patch.object(self.service.runtime, 'stream', side_effect=answers):
            job = finish(self.service, self.service.chat(self.conversation, 'Investigate Mars', []))
        stages = [e.get('stage') for e in job['events'] if e['event'] == 'stage']
        self.assertEqual(stages, ['Queued', 'Retrieving evidence', 'Searching evidence',
                                  'Checking response and citations'])
        self.assertEqual(job['status'], 'completed')


class DesktopUsabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = Service(Path(self.temp.name))
        self.window = Window(self.service, auto_setup=False)
        self.window.show(); self.app.processEvents()

    def tearDown(self):
        self.window.close(); self.service.jobs.shutdown(wait=True)
        self.app.processEvents(); self.temp.cleanup()

    def test_thinking_defaults_on_and_is_controlled_from_settings(self):
        w = self.window
        w.new_conversation()
        original = w.conversation
        self.assertTrue(w.thinking.isChecked())
        self.assertFalse(w.thinking.isVisible())
        w.navigation.setCurrentRow(2)
        self.app.processEvents()
        self.assertTrue(w.thinking.isVisible())
        w.thinking.setChecked(False)
        w.save_draft()
        w.new_conversation()
        self.assertTrue(w.thinking.isChecked())
        self.assertFalse(self.service.store.setting('draft:'+original)['thinking'])

    def test_drafts_thinking_attachments_and_restart(self):
        w = self.window
        w.new_conversation(); first = w.conversation
        w.message_input.setPlainText('Draft one'); w.thinking.setChecked(True)
        source = Path(self.temp.name)/'posts.json'; source.write_text('[{"id":"p","text":"post"}]')
        graph = self.service.import_posts(source)[0]; w.attached.add(graph)
        w.new_conversation(); second = w.conversation
        w.message_input.setPlainText('Draft two')
        item = next(w.conversation_list.item(i) for i in range(w.conversation_list.count())
                    if w.conversation_list.item(i).data(Qt.ItemDataRole.UserRole) == first)
        w.conversation_list.setCurrentItem(item)
        self.assertEqual(w.message_input.toPlainText(), 'Draft one')
        self.assertTrue(w.thinking.isChecked()); self.assertIn(graph, w.attached)
        self.assertEqual(self.service.store.setting('draft:'+second)['text'], 'Draft two')
        w.close(); self.service.jobs.shutdown(wait=True)
        self.service = Service(Path(self.temp.name)); self.window = Window(self.service, auto_setup=False)
        self.assertEqual(self.window.conversation, first)
        self.assertEqual(self.window.message_input.toPlainText(), 'Draft one')
        self.assertTrue(self.window.thinking.isChecked())

    def test_edit_buttons_editor_errors_and_history(self):
        w = self.window; w.new_conversation()
        ident = self.service.store.message(w.conversation, 'assistant', 'Generated', {'evidence': []})
        w.render_messages(); self.assertIn(f'edit:{ident}', w.transcript.toHtml())
        w.message_action(QUrl(f'edit:{ident}'))
        w.editor.text.setPlainText(' '); w.editor.submit()
        self.assertIn('empty', w.editor.error.text())
        w.editor.text.setPlainText('Edited answer'); w.editor.submit()
        self.assertIn('Edited answer', w.transcript.toPlainText())
        self.assertIn('original generated', w.transcript.toPlainText())
        w.message_action(QUrl(f'history:{ident}'))
        self.assertTrue(w.dialogs[-1].isVisible())

    def test_active_job_stage_stop_and_edit_guard(self):
        w = self.window; w.new_conversation()
        entered = threading.Event(); release = threading.Event()
        def work(conversation, text, graphs, job, thinking=False):
            job.emit('stage', stage='Thinking'); entered.set(); release.wait(3); job.check()
        try:
            with patch.object(self.service.assistant, 'chat', side_effect=work):
                w.message_input.setPlainText('Question'); w.thinking.setChecked(True); w.send()
                self.assertFalse(w.send_button.isEnabled())
                self.assertTrue(entered.wait(2)); w._poll()
                self.assertIn('Thinking', w.live.text())
                ident = self.service.store.message(w.conversation, 'user', 'Earlier')
                w.message_action(QUrl(f'edit:{ident}')); w.editor.text.setPlainText('Revised'); w.editor.submit()
                self.assertIn('Stop', w.editor.error.text())
                self.assertEqual(w.editor.text.toPlainText(), 'Revised')
                w.stop_chat(); self.assertIn('Cancelling', w.live.text())
                release.set()
                job = self.service.jobs.list()[0]; finish(self.service, job['id']); w._poll()
                self.assertFalse(w.send_button.isEnabled())  # Finish or cancel the open edit first.
                w.close_editor()
                self.assertTrue(w.send_button.isEnabled()); self.assertFalse(w.stop_button.isEnabled())
                w.restore_request(); self.assertEqual(w.message_input.toPlainText(), 'Question')
                self.assertTrue(w.thinking.isChecked())
        finally: release.set()

    def test_poll_preserves_scroll_settings_and_unrelated_status(self):
        w = self.window; w.new_conversation()
        for i in range(30): self.service.store.message(w.conversation, 'user', f'Paragraph {i}\n'*10)
        w.render_messages(); self.app.processEvents()
        scroll = w.transcript.verticalScrollBar(); scroll.setValue(20)
        w.context.setCurrentIndex(w.context.findData(16384))
        self.service.store.message(w.conversation, 'assistant', 'new answer')
        ident = self.service.jobs.submit('unrelated', lambda job: {'ok': True})
        finish(self.service, ident); w._poll()
        self.assertEqual(scroll.value(), 20)
        self.assertEqual(w.context.currentData(), 16384)
        self.assertFalse(w.generation.isVisible())

    def test_shortcut_citation_and_minimum_layout(self):
        w = self.window; w.new_conversation()
        evidence = {'citation': 'E1', 'path': '/local/source.txt', 'text': 'Evidence excerpt', 'provenance': {}}
        ident = self.service.store.message(w.conversation, 'assistant', 'answer', {'evidence': [evidence],
            'claims': [{'claim': 'claim', 'verdict': 'supported', 'citations': ['E1']}]})
        w.render_messages(); w.message_action(QUrl(f'citation:{ident}:0'))
        self.assertEqual(w.dialogs[-1].windowTitle(), 'Evidence source')
        w.dialogs[-1].close()
        w.resize(900, 650); self.app.processEvents()
        self.assertEqual(w.width(), 900); self.assertEqual(w.height(), 650)
        self.assertFalse(w.thinking.isVisible()); self.assertTrue(w.send_button.isVisible())
        with patch.object(self.service, 'chat', side_effect=ValueError('Test submission')) as chat:
            w.message_input.setFocus(); w.message_input.setPlainText('keyboard request')
            QTest.keyClick(w.message_input, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
            self.app.processEvents(); self.assertEqual(chat.call_count, 1)
            self.assertEqual(w.message_input.toPlainText(), 'keyboard request')

    def test_restore_does_not_overwrite_draft_without_confirmation(self):
        w = self.window
        w.generation.job = {'request': {'text': 'failed request'}}
        w.message_input.setPlainText('precious draft')
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
            w.restore_request()
        self.assertEqual(w.message_input.toPlainText(), 'precious draft')

    def test_background_work_contents_visible_on_open_and_reopen(self):
        w = self.window
        w.activity_button.click(); self.app.processEvents()
        self.assertTrue(w.activity_dialog.isVisible())
        self.assertTrue(w.activity.isVisible())
        self.assertGreater(w.activity.width(), 0)
        self.assertGreater(w.activity.height(), 0)
        w.activity_dialog.close(); self.app.processEvents()
        self.assertFalse(w.activity_dialog.isVisible())
        ident = self.service.jobs.submit('example-job', lambda job: {'result': 'ok'})
        finish(self.service, ident)
        w.activity_button.click(); self.app.processEvents()
        self.assertTrue(w.activity.isVisible())
        self.assertEqual(w.activity.topLevelItemCount(), 1)
        self.assertEqual(w.activity.topLevelItem(0).text(1), 'Complete')

    def test_background_work_progress_details_selection_and_cancel(self):
        w = self.window
        entered = threading.Event(); release = threading.Event()
        def work(job):
            job.emit('stage', stage='Reading dataset')
            entered.set(); release.wait(3); job.check()
        try:
            ident = self.service.jobs.submit('posts-preview', work)
            self.assertTrue(entered.wait(2))
            w.open_activity(); self.app.processEvents()
            item = w.activity.topLevelItem(0)
            self.assertEqual(item.text(2), 'Reading dataset')
            w.activity.setCurrentItem(item)
            w.job_details(); self.app.processEvents()
            self.assertTrue(w.dialogs[-1].isVisible())
            self.assertEqual(w.dialogs[-1].windowTitle(), 'Background job details')
            w.dialogs[-1].close()
            self.service.jobs.live[ident].emit('stage', stage='Preparing preview')
            w._poll()
            self.assertEqual(w.activity.currentItem().data(0, Qt.ItemDataRole.UserRole), ident)
            self.assertEqual(w.activity.currentItem().text(2), 'Preparing preview')
            w.cancel_selected(); release.set()
            self.assertEqual(finish(self.service, ident)['status'], 'cancelled')
            w._poll()
            self.assertEqual(w.activity.currentItem().text(1), 'Cancelled')
            self.assertEqual(w.activity.currentItem().text(2), 'Task cancelled.')
        finally:
            release.set()


if __name__ == '__main__':
    unittest.main()
