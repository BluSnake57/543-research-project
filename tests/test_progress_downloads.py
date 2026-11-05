"""Download integrity, job presentation, and transient local-model thinking."""
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import urllib.error
import zipfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from sentinel_x.activity_ui import job_summary
from sentinel_x.desktop import Window
from sentinel_x.downloads import download, request, DownloadError
from sentinel_x.jobs import Cancelled
from sentinel_x.service import Service
from sentinel_x.state import Store
from scripts.check_downloads import check_source


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers if headers is not None else {'Content-Length': str(len(content))}


class Recorder:
    def __init__(self): self.events = []
    def check(self): pass
    def emit(self, event, **payload): self.events.append({'event': event, **payload})


def finish(service, ident):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        job = service.jobs.get(ident)
        if job['status'] in ('completed', 'failed', 'cancelled'):
            return job
        time.sleep(.005)
    raise AssertionError('Job timed out')


class DownloadProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def test_actual_request_transfer_checks_checksum_and_reports_stages(self):
        payload = b'Example downloaded config'
        job = Recorder(); target = self.root/'config.json'
        with patch('sentinel_x.downloads.urllib.request.urlopen', return_value=Response(payload)) as opened:
            download('https://example.com/config', target, hashlib.sha256(payload).hexdigest(), len(payload), job)
        self.assertEqual(target.read_bytes(), payload)
        self.assertEqual(opened.call_args.args[0].full_url, 'https://example.com/config')
        stages = [e['stage'] for e in job.events if e['event'] == 'stage']
        self.assertIn('Connecting to example.com', stages)
        self.assertEqual(stages[-2:], ['Verifying file integrity', 'File verified'])
        progress = [e for e in job.events if e['event'] == 'download']
        self.assertEqual(progress[0]['downloaded'], 0)
        self.assertEqual(progress[-1]['downloaded'], len(payload))
        self.assertGreaterEqual(progress[-1]['speed'], 0)

    def test_http_content_length_truncation_without_pinned_size_is_not_promoted(self):
        class Socket:
            def makefile(self, *args):
                return io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort')
        response = http.client.HTTPResponse(Socket()); response.begin()
        target = self.root/'archive.zip'
        with patch('sentinel_x.downloads.request', return_value=response):
            with self.assertRaisesRegex(DownloadError, 'mismatch|interrupted'):
                download('https://example.com/archive', target)
        self.assertFalse(target.exists())
        self.assertEqual(target.with_name('archive.zip.part').read_bytes(), b'short')
        self.assertFalse(target.with_name('archive.zip.checksum.json').exists())

    def test_unknown_size_without_content_length_stays_indeterminate(self):
        job = Recorder(); target = self.root/'unknown'
        with patch('sentinel_x.downloads.request', return_value=Response(b'payload', headers={})):
            download('https://example.com/unknown', target, job=job)
        self.assertTrue(all(e['total'] is None for e in job.events if e['event'] == 'download'))

    def test_cancel_mid_transfer_retains_bytes_and_resume_verifies(self):
        payload = b'a'*(128*1024); target = self.root/'weights'
        response = Response(payload)
        class Canceller(Recorder):
            def check(self):
                if response.tell() >= 64*1024: raise Cancelled()
        with patch('sentinel_x.downloads.request', return_value=response):
            with self.assertRaises(Cancelled):
                download('https://example.com/weights', target, job=Canceller())
        partial = target.with_name('weights.part')
        self.assertEqual(partial.stat().st_size, 64*1024)
        rest = payload[64*1024:]
        with patch('sentinel_x.downloads.request', return_value=Response(rest, 206, {
                'Content-Length': str(len(rest)), 'Content-Range': f'bytes 65536-{len(payload)-1}/{len(payload)}'})) as fetch:
            download('https://example.com/weights', target, hashlib.sha256(payload).hexdigest(), len(payload))
        self.assertEqual(fetch.call_args.args[1], {'Range': 'bytes=65536-'})
        self.assertEqual(target.read_bytes(), payload)

    def test_bad_range_is_rejected_and_oversized_partial_restarts(self):
        target = self.root/'weights'; partial = target.with_name('weights.part')
        partial.write_bytes(b'abc')
        with patch('sentinel_x.downloads.request', return_value=Response(b'def', 206, {
                'Content-Length': '3', 'Content-Range': 'bytes 3-5/999'})):
            with self.assertRaisesRegex(ValueError, 'size changed'):
                download('https://example.com/weights', target, size=6)
        self.assertEqual(partial.read_bytes(), b'abc')
        partial.write_bytes(b'old-oversized-file')
        error = urllib.error.HTTPError('https://example.com/weights', 416, '', {'Content-Range': 'bytes */6'}, None)
        with patch('sentinel_x.downloads.request', side_effect=[error, Response(b'abcdef')]) as fetch:
            download('https://example.com/weights', target, hashlib.sha256(b'abcdef').hexdigest(), 6)
        self.assertEqual(len(fetch.call_args_list), 2)
        self.assertEqual(target.read_bytes(), b'abcdef')

    def test_complete_partial_and_invalid_receipt_recover(self):
        target = self.root/'archive'; target.with_name('archive.part').write_bytes(b'abcdef')
        error = urllib.error.HTTPError('https://example.com/archive', 416, '', {'Content-Range': 'bytes */6'}, None)
        with patch('sentinel_x.downloads.request', side_effect=error):
            download('https://example.com/archive', target, hashlib.sha256(b'abcdef').hexdigest(), 6)
        target.with_name('archive.checksum.json').write_text('broken')
        with patch('sentinel_x.downloads.request', return_value=Response(b'fresh')):
            download('https://example.com/archive', target)
        self.assertEqual(target.read_bytes(), b'fresh')

    def test_network_http_and_https_errors_have_recovery_instructions(self):
        with patch('sentinel_x.downloads.urllib.request.urlopen', side_effect=urllib.error.URLError('DNS failed')):
            with self.assertRaisesRegex(DownloadError, 'DNS.*retry'):
                request('https://huggingface.co/test')
        for status in (401, 403, 404, 429, 503):
            with self.subTest(status=status), patch('sentinel_x.downloads.urllib.request.urlopen',
                    side_effect=urllib.error.HTTPError('https://example.com', status, '', {}, None)):
                with self.assertRaisesRegex(DownloadError, f'HTTP {status}'):
                    request('https://example.com/file')
        with self.assertRaisesRegex(ValueError, 'HTTPS'):
            request('http://example.com/file')

    def test_smoke_check_bounds_large_file_reads_and_verifies_small_config(self):
        from sentinel_x.downloads import Models
        model = Models(Store(self.root), {'fever': {'evidence': 'https://example.com/wiki.zip'}})
        payload = b'{"hidden_size":384}'
        item = {'name': 'config.json', 'url': 'https://example.com/config', 'algorithm': 'sha256',
                'checksum': hashlib.sha256(payload).hexdigest(), 'size': len(payload)}
        with patch.object(model, 'catalog', return_value={'files': [item]}), \
             patch('sentinel_x.downloads.request', return_value=Response(payload)):
            self.assertIn('verified', check_source(model, 'embedding', self.root))
        with patch('scripts.check_downloads.request', return_value=Response(b'PK\x03\x04')) as fetch:
            self.assertIn('full multi-gigabyte transfer not tested', check_source(model, 'fever', self.root))
            self.assertEqual(fetch.call_args.args[1], {'Range': 'bytes=0-3'})


class TaskPresentationTests(unittest.TestCase):
    def test_readable_names_bytes_speed_eta_and_phase_scoped_progress(self):
        job = {'kind': 'download:qwen', 'created': 10, 'status': 'running', 'events': [
            {'event': 'file', 'index': 1, 'count': 2},
            {'event': 'download', 'file': 'model.gguf', 'downloaded': 1024**3,
             'total': 2*1024**3, 'speed': 1024**2, 'eta': 1024}]}
        summary = job_summary(job, now=20)
        self.assertEqual(summary['name'], 'Download chat model')
        self.assertEqual(summary['percent'], 50)
        for text in ('1.0 GiB of 2.0 GiB', '1.0 MiB/s', '17m 4s left', 'File 1 of 2'):
            self.assertIn(text, summary['detail'])
        job['events'].append({'event': 'stage', 'stage': 'Verifying file integrity', 'file': 'model.gguf'})
        self.assertIsNone(job_summary(job)['percent'])
        job['kind'] = 'chat:abc'
        self.assertEqual(job_summary(job, {'abc': 'Mars study'})['name'], 'Generate response · Mars study')

    def test_unknown_size_and_terminal_states_do_not_claim_download_completion(self):
        job = {'kind': 'fever', 'created': 1, 'status': 'running', 'events': [
            {'event': 'download', 'file': 'wiki.zip', 'downloaded': 12, 'total': None}]}
        self.assertIsNone(job_summary(job)['percent'])
        self.assertIn('unknown', job_summary(job)['detail'])
        job.update(status='failed', error='Cannot connect. Retry.')
        self.assertEqual(job_summary(job)['percent'], 0)
        self.assertEqual(job_summary(job)['detail'], job['error'])


class ProgressIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.service = Service(Path(self.temp.name))

    def tearDown(self):
        self.service.close(); self.service.jobs.shutdown(wait=True); self.temp.cleanup()

    def test_model_ready_marker_waits_for_every_verified_file(self):
        payload = b'config'; item = {'name': 'config.json', 'url': 'https://example.com/config',
            'checksum': hashlib.sha256(payload).hexdigest(), 'size': len(payload), 'algorithm': 'sha256'}
        catalog = {'revision': 'a'*40, 'files': [item, {**item, 'name': 'weights', 'url': 'https://example.com/weights'}]}
        with patch.object(self.service.models, 'catalog', return_value=catalog), \
             patch('sentinel_x.downloads.request', side_effect=[Response(payload), DownloadError('offline')]):
            job = finish(self.service, self.service.download('embedding'))
        self.assertEqual(job['status'], 'failed')
        self.assertFalse((self.service.store.root/'models/embedding/.ready').exists())
        with patch.object(self.service.models, 'catalog', return_value=catalog), \
             patch('sentinel_x.downloads.request', return_value=Response(payload)):
            retried = finish(self.service, self.service.retry_download(job['id']))
        self.assertEqual(retried['status'], 'completed')
        self.assertTrue((self.service.store.root/'models/embedding/.ready').exists())

    def test_fever_download_extract_index_and_index_failure_never_marks_ready(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as bundle:
            bundle.writestr('wiki-001.jsonl', '{"id":"Mars","lines":"0\\tMars has two moons."}\n')
        with patch('sentinel_x.downloads.request', side_effect=[Response(archive.getvalue()), Response(b'{}'), Response(b'{}')]):
            job = finish(self.service, self.service.fever(index=True))
        self.assertEqual(job['status'], 'completed', job.get('error'))
        self.assertTrue(self.service.store.setting('fever_ready'))
        self.assertTrue(self.service.retrieval.search('Mars'))
        self.assertTrue(any(e['event'] == 'extraction' for e in job['events']))
        with patch.object(self.service.retrieval, 'index', return_value={'errors': [{'error': 'Disk full'}]}):
            retried = finish(self.service, self.service.fever(index=True))  # All downloads reused from receipts.
        self.assertEqual(retried['status'], 'failed')
        self.assertIn('could not be indexed', retried['error'])
        self.assertFalse(self.service.store.setting('fever_ready'))

    def test_invalid_wikipedia_archive_is_removed_so_retry_can_recover(self):
        with patch('sentinel_x.downloads.request', return_value=Response(b'not a ZIP')):
            job = finish(self.service, self.service.fever())
        self.assertEqual(job['status'], 'failed')
        self.assertIn('invalid archive was removed', job['error'])
        self.assertFalse((self.service.store.root/'resources/fever/wiki-pages.zip').exists())


    def test_setup_retry_preserves_selected_downloads(self):
        with patch.object(self.service.models, 'install', side_effect=DownloadError('offline')):
            job = finish(self.service, self.service.setup(keys=('embedding',), fever=False))
        self.assertEqual(job['status'], 'failed')
        with patch.object(self.service.models, 'install', return_value={'model': 'embedding'}) as install:
            retried = finish(self.service, self.service.retry_download(job['id']))
        self.assertEqual(retried['status'], 'completed')
        self.assertEqual(install.call_args.args[0], 'embedding')

    def test_live_thinking_spools_then_collapses_and_reopens_from_saved_message(self):
        w = Window(self.service, auto_setup=False); w.show(); w.new_conversation()
        entered = threading.Event(); release = threading.Event()
        def stream(messages, tools, job):
            job.emit('stage', stage='Thinking')
            job.stream_piece('Inspecting the example claim…', reasoning=True)
            entered.set(); release.wait(3)
            job.finish_thinking()
            return {'role': 'assistant', 'content': '{"summary":"Example answer","claims":[]}'}
        try:
            with patch.object(self.service.runtime, 'stream', side_effect=stream):
                ident = self.service.chat(w.conversation, 'Example question', [], True)
                self.assertTrue(entered.wait(2)); w._poll(); self.app.processEvents()
                self.assertIn('Inspecting the example claim', w.transcript.toPlainText())
                self.assertNotIn('Inspecting the example claim', json.dumps(self.service.jobs.get(ident)))
                release.set(); job = finish(self.service, ident); w._poll(); self.app.processEvents()
                text = w.transcript.toPlainText()
                self.assertIn('Thought for', text); self.assertIn('Example answer', text)
                self.assertNotIn('Inspecting the example claim', text)
                self.assertNotIn('Inspecting the example claim', json.dumps(job))
                self.assertGreater(job['thinking_seconds'], 0)
                message = self.service.store.messages(w.conversation)[-1]
                self.assertEqual(message['data']['thinking_trace'], 'Inspecting the example claim…')
                from PySide6.QtCore import QUrl
                w.message_action(QUrl('thinking:'+str(message['id'])))
                self.assertIn('Inspecting the example claim', w.transcript.toPlainText())
                w.message_action(QUrl('thinking:'+str(message['id'])))
                self.assertNotIn('Inspecting the example claim', w.transcript.toPlainText())
        finally:
            release.set(); w.close()

    def test_row_widgets_selection_and_indeterminate_to_measured_progress(self):
        w = Window(self.service, auto_setup=False); w.show()
        entered = threading.Event(); release = threading.Event()
        def work(job):
            job.emit('stage', stage='Connecting to example.com'); entered.set(); release.wait(3)
        try:
            ident = self.service.jobs.submit('download:qwen', work)
            self.assertTrue(entered.wait(2)); w.open_activity(); self.app.processEvents()
            item = w.activity.topLevelItem(0); w.activity.setCurrentItem(item)
            bar = w.activity.itemWidget(item, 3)
            self.assertEqual(bar.maximum(), 0)
            self.assertEqual(item.text(0), 'Download chat model')
            self.service.jobs.live[ident].emit('download', file='model.gguf', downloaded=50, total=100)
            w._poll()
            self.assertIs(w.activity.currentItem(), item); self.assertIs(w.activity.itemWidget(item, 3), bar)
            self.assertEqual(bar.value(), 50); self.assertEqual(bar.maximum(), 100)
            self.assertTrue(w.cancel_job_button.isEnabled())
            w.cancel_selected(); self.assertIn('Stopping', w.activity_selection.text())
            release.set(); finish(self.service, ident); w._poll()
            self.assertTrue(w.retry_job_button.isEnabled())
            self.assertFalse(w.cancel_job_button.isEnabled())
        finally:
            release.set(); w.close()



if __name__ == '__main__': unittest.main()
