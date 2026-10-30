"""Chat must stay responsive while a large evidence corpus is being indexed."""
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from sentinel_x.conversation_ui import draft_summary
from sentinel_x.desktop import Window
from sentinel_x.jobs import Cancelled
from sentinel_x.retrieval import Retrieval, documents
from sentinel_x.service import Service


def finish(service, ident):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        result = service.jobs.get(ident)
        if result['status'] in ('completed', 'failed', 'cancelled'):
            return result
        time.sleep(.005)
    raise AssertionError('Job timed out')


class Response(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.status = 200; self.headers = {'Content-Length': str(len(payload))}


class InteractiveIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.service = Service(self.root/'app')
        self.directory = self.root/'evidence'; self.directory.mkdir()
        self.source = self.directory/'wiki.jsonl'
        self.source.write_text('\n'.join(json.dumps({'id': f'Page{i}',
            'lines': '0\tExample evidence about Mars.'}) for i in range(200)))

    def tearDown(self):
        self.service.close(); self.service.jobs.shutdown(wait=True); self.temp.cleanup()

    def test_chat_and_search_complete_while_index_writer_is_still_running(self):
        entered = threading.Event(); release = threading.Event(); generated = threading.Event()
        def gated(path):
            for number, record in enumerate(documents(path), 1):
                yield record
                if number == 64:
                    entered.set(); release.wait(4)
        def stream(messages, tools, job):
            self.assertIn('Example evidence about Mars.', messages[0]['content'])
            generated.set()
            return {'role': 'assistant', 'content': '{"summary":"Evidence is available","claims":[]}'}
        try:
            with patch('sentinel_x.retrieval.documents', side_effect=gated), \
                 patch.object(self.service.runtime, 'stream', side_effect=stream):
                background = self.service.jobs.submit('index', lambda job:
                    self.service.retrieval.index(self.directory, job, semantic=False, resumable=True, batch_size=64))
                self.assertTrue(entered.wait(2))
                conversation = self.service.store.create_conversation()['id']
                chat = self.service.chat(conversation, 'Mars', [])
                self.assertTrue(generated.wait(2), 'Chat waited for the indexing worker or lock')
                self.assertEqual(finish(self.service, chat)['status'], 'completed')
                self.assertEqual(self.service.jobs.get(background)['status'], 'running')
                release.set()
                self.assertEqual(finish(self.service, background)['status'], 'completed')
        finally: release.set()

    def test_cancelled_initial_index_resumes_committed_batches_without_duplicates(self):
        def interrupted(path):
            for number, record in enumerate(documents(path), 1):
                yield record
                if number == 80:
                    raise Cancelled()
        with patch('sentinel_x.retrieval.documents', side_effect=interrupted):
            with self.assertRaises(Cancelled):
                self.service.retrieval.index(self.directory, semantic=False, resumable=True, batch_size=64)
        with self.service.retrieval.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM chunks').fetchone()[0], 64)
            self.assertEqual(db.execute('SELECT chunks FROM index_progress').fetchone()[0], 64)
            self.assertEqual(db.execute('SELECT count(*) FROM files').fetchone()[0], 0)
        found = self.service.retrieval.search('Mars')
        self.assertTrue(found); self.assertIn('partial index', found[0]['retrieval'])
        reopened = Retrieval(self.service.store)
        with patch.object(reopened, 'embedding', side_effect=AssertionError('No full-corpus embeddings')):
            result = reopened.index(self.directory, semantic=False, resumable=True, batch_size=64)
        self.assertEqual(result['indexed'], 1)
        with reopened.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM chunks').fetchone()[0], 200)
            self.assertEqual(db.execute('SELECT count(*) FROM index_progress').fetchone()[0], 0)

    def test_official_archive_layout_ignores_licenses_and_macos_metadata(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as bundle:
            bundle.writestr('license.html', '<html>License</html>')
            bundle.writestr('__MACOSX/wiki-pages/._wiki-001.jsonl', b'\x00\x05metadata')
            bundle.writestr('wiki-pages/wiki-001.jsonl', '{"id":"Mars","lines":"0\\tMars has two moons."}')
        with patch('sentinel_x.downloads.request', side_effect=[Response(archive.getvalue()), Response(b'{}'), Response(b'{}')]), \
             patch.object(self.service.retrieval, 'embedding', side_effect=AssertionError('Keyword first')):
            result = finish(self.service, self.service.fever(index=True))
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertTrue(self.service.store.setting('fever_ready'))
        self.assertTrue(result['result']['evidence_directory'].endswith('wiki-pages'))
        self.assertEqual(result['result']['index']['errors'], [])
        self.assertEqual(result['result']['index']['mode'], 'keyword-only')

    def test_changed_partial_file_restarts_and_deleted_partial_file_is_removed(self):
        def interrupted(path):
            for number, record in enumerate(documents(path), 1):
                yield record
                if number == 80: raise Cancelled()
        for replacement in (False, True):
            with patch('sentinel_x.retrieval.documents', side_effect=interrupted):
                with self.assertRaises(Cancelled):
                    self.service.retrieval.index(self.directory, semantic=False, resumable=True, batch_size=64)
            if replacement:
                self.source.unlink()
            else:
                self.source.write_text(self.source.read_text().replace('Mars', 'Venus'))
            self.service.retrieval.index(self.directory, semantic=False, resumable=True, batch_size=64)
            with self.service.retrieval.connection() as db:
                self.assertEqual(db.execute('SELECT count(*) FROM chunks').fetchone()[0], 0 if replacement else 200)
            if not replacement:
                # Force another initial build to exercise deleting a partial file.
                with self.service.retrieval.connection() as db:
                    db.execute('DELETE FROM files'); db.execute('DELETE FROM chunks'); db.commit()


class DraftStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])

    def test_partial_json_summary_handles_escapes_and_incomplete_unicode(self):
        self.assertEqual(draft_summary('{"summary":"Hel'), 'Hel')
        self.assertEqual(draft_summary('{"summary":"Hello\\nworld\\u263'), 'Hello\nworld')
        self.assertEqual(draft_summary('{"summary":"\\ud83d\\ude00"}'), '😀')
        self.assertEqual(draft_summary('{"summary":"Quoted \\"text'), 'Quoted "text')

    def test_non_thinking_answer_is_visible_before_final_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            service = Service(Path(folder)); window = Window(service, auto_setup=False)
            window.show(); window.new_conversation()
            entered = threading.Event(); release = threading.Event()
            def stream(messages, tools, job):
                job.stream_piece('{"summary":"Hello from the model')
                entered.set(); release.wait(4)
                return {'role': 'assistant', 'content': '{"summary":"Hello from the model","claims":[]}'}
            try:
                with patch.object(service.runtime, 'stream', side_effect=stream):
                    ident = service.chat(window.conversation, 'hello', [], False)
                    self.assertTrue(entered.wait(2)); window._poll(); self.app.processEvents()
                    text = window.transcript.toPlainText()
                    self.assertIn('Hello from the model', text)
                    self.assertIn('Draft response', text)
                    self.assertNotIn('Hello from the model', json.dumps(service.jobs.get(ident)))
                    release.set(); finish(service, ident); window._poll(); self.app.processEvents()
                    self.assertIn('Hello from the model', window.transcript.toPlainText())
                    self.assertNotIn('Draft response', window.transcript.toPlainText())
            finally:
                release.set(); window.close(); service.jobs.shutdown(wait=True)


if __name__ == '__main__': unittest.main()
