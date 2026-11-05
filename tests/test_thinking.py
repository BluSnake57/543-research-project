"""Thinking-mode protocol and lifecycle checks, independent of model downloads."""
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from sentinel_x.jobs import Job
from sentinel_x.service import Service
from sentinel_x.conversation_ui import transcript_html
from scripts.check_thinking import check_mode


def wire(*deltas):
    packets = [{'choices': [{'delta': delta}]} for delta in deltas]
    packets.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
    return io.BytesIO((''.join('data: '+json.dumps(p)+'\n\n' for p in packets)+'data: [DONE]\n').encode())


def finish(service, ident):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        job = service.jobs.get(ident)
        if job['status'] in ('completed', 'failed', 'cancelled'):
            return job
        time.sleep(.005)
    raise AssertionError('Job timed out')


class ThinkingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = Service(Path(self.temp.name))
        self.runtime = self.service.runtime; self.runtime.port = 1234

    def tearDown(self):
        self.service.close(); self.service.jobs.shutdown(wait=True); self.temp.cleanup()





    def test_multiple_thinking_phases_accumulate_only_active_time(self):
        job = Job(self.service.jobs, 'test')
        with patch('sentinel_x.jobs.time.monotonic', return_value=10) as clock:
            job.stream_piece('first phase', reasoning=True)
            clock.return_value = 12; job.finish_thinking()
            clock.return_value = 30; job.stream_piece('second phase', reasoning=True)
            clock.return_value = 34; job.finish_thinking()
        self.assertEqual(job.data['thinking_seconds'], 6)
        self.assertEqual(job.preview(), {'text': '', 'active': False, 'seconds': 6, 'answer': '',
            'trace': 'first phase\n\nsecond phase', 'trace_truncated': False})
        self.assertNotIn('first phase', json.dumps(job.data))

    def test_failure_and_cancellation_clear_the_preview(self):
        def fail(job):
            job.stream_piece('temporary failure text', reasoning=True)
            raise ValueError('transport failed')
        ident = self.service.jobs.submit('failure', fail)
        failed = finish(self.service, ident)
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(self.service.jobs.live[ident].preview()['text'], '')
        self.assertNotIn('temporary failure text', json.dumps(failed))
        entered = threading.Event(); release = threading.Event()
        def work(job):
            job.stream_piece('temporary cancel text', reasoning=True)
            entered.set(); release.wait(3); job.check()
        try:
            ident = self.service.jobs.submit('cancel', work)
            self.assertTrue(entered.wait(2)); self.service.cancel(ident); release.set()
            cancelled = finish(self.service, ident)
            self.assertEqual(cancelled['status'], 'cancelled')
            self.assertEqual(self.service.jobs.live[ident].preview()['text'], '')
            self.assertNotIn('temporary cancel text', json.dumps(cancelled))
        finally: release.set()


    def test_empty_thinking_is_not_counted_or_presented_as_real_thinking(self):
        job = Job(self.service.jobs, 'test')
        job.stream_piece('\n \n', reasoning=True)
        self.assertFalse(job.preview()['active'])
        self.assertEqual(job.data.get('thinking_characters', 0), 0)
        text = transcript_html([{'id': 1, 'role': 'assistant', 'content': 'Answer',
            'data': {'thinking_enabled': True, 'thinking_seconds': 0}}])
        self.assertIn('did not emit a separate thinking trace', text)
        self.assertNotIn('Thought for', text)

    def test_retained_trace_is_collapsed_by_default_and_escaped_when_opened(self):
        message = {'id': 17, 'role': 'assistant', 'content': 'Answer', 'data': {
            'thinking_seconds': 2, 'thinking_trace': 'Model trace <img src="untrusted">\nNext line.'}}
        collapsed = transcript_html([message])
        self.assertIn('thinking:17', collapsed)
        self.assertNotIn('Model trace', collapsed)
        expanded = transcript_html([message], thinking_expanded={'17': True})
        self.assertIn('Model trace &lt;img', expanded)
        self.assertNotIn('<img', expanded)
        self.assertIn('Next line.', expanded)
        message['data'].pop('thinking_trace')
        self.assertIn('Trace was not saved', transcript_html([message]))

    def test_retained_trace_is_bounded_without_token_by_token_database_writes(self):
        job = Job(self.service.jobs, 'test')
        with patch.object(self.service.jobs, 'persist') as persist:
            job.stream_piece('a'*70000+'final segment', reasoning=True)
        persist.assert_not_called()
        self.assertEqual(len(job.thinking_trace), 64000)
        self.assertTrue(job.thinking_trace.endswith('final segment'))
        self.assertTrue(job.thinking_trace_truncated)






if __name__ == '__main__': unittest.main()
