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

    def test_explicit_budget_changes_per_request_without_restarting_runtime(self):
        self.service.store.configure('thinking_budget', 128)
        for enabled in (False, True, False):
            with self.subTest(enabled=enabled):
                job = Job(self.service.jobs, 'test'); job.thinking_enabled = enabled
                def template(path, body):
                    if path == '/tokenize': return {'tokens': [1]}
                    return {'prompt': 'on' if body['chat_template_kwargs']['enable_thinking'] else 'off'}
                with patch.object(self.runtime, 'start'), patch.object(self.runtime, 'call', side_effect=template) as calls, \
                     patch('sentinel_x.runtime.urllib.request.urlopen', return_value=wire({'content': 'answer'})) as opened:
                    self.runtime.stream([{'role': 'user', 'content': 'hello'}], [], job)
                payload = json.loads(opened.call_args.args[0].data)
                self.assertEqual(payload['reasoning_budget_tokens'], 128 if enabled else 0)
                self.assertEqual(payload['reasoning_format'], 'deepseek')
                first_template = calls.call_args_list[0].args[1]
                self.assertEqual(first_template['reasoning_budget_tokens'], payload['reasoning_budget_tokens'])
                self.assertEqual(first_template['chat_template_kwargs'], payload['chat_template_kwargs'])

    def test_off_rejects_runtime_that_ignores_the_switch(self):
        job = Job(self.service.jobs, 'test'); job.thinking_enabled = False
        with patch.object(self.runtime, 'start'), patch.object(self.runtime, 'call', side_effect=[
                {'prompt': 'off'}, {'tokens': [1]}]), \
             patch('sentinel_x.runtime.urllib.request.urlopen', return_value=wire({'reasoning_content': 'Unexpected thinking'})):
            with self.assertRaisesRegex(ValueError, 'Thinking Off'):
                self.runtime.stream([], [], job)
        self.assertFalse(job.preview()['active'])
        self.assertNotIn('Unexpected thinking', json.dumps(job.data))

    def test_off_ignores_empty_reasoning_fields(self):
        job = Job(self.service.jobs, 'test'); job.thinking_enabled = False
        with patch.object(self.runtime, 'start'), patch.object(self.runtime, 'call', side_effect=[
                {'prompt': 'off'}, {'tokens': [1]}]), \
             patch('sentinel_x.runtime.urllib.request.urlopen', return_value=wire(
                 {'reasoning_content': ' \n'}, {'content': 'Direct answer'})):
            answer = self.runtime.stream([], [], job)
        self.assertEqual(answer['content'], 'Direct answer')
        self.assertFalse(any(e.get('stage') == 'Thinking' for e in job.data['events']))

    def test_reasoning_alias_same_packet_answer_and_tool_transition(self):
        job = Job(self.service.jobs, 'test'); job.thinking_enabled = True
        with patch.object(self.runtime, 'start'), patch.object(self.runtime, 'call', side_effect=[
                {'prompt': 'on'}, {'prompt': 'off'}, {'tokens': [1]}]), \
             patch('sentinel_x.runtime.urllib.request.urlopen', return_value=wire(
                 {'reasoning': 'Example transient text'},
                 {'reasoning_content': ' more text', 'content': 'Answer'},
                 {'tool_calls': [{'index': 0, 'id': 'call1', 'function': {'name': 'lookup', 'arguments': '{}'}}]})):
            answer = self.runtime.stream([], [], job)
        self.assertEqual(answer['content'], 'Answer')
        self.assertEqual(answer['tool_calls'][0]['function']['name'], 'lookup')
        self.assertFalse(job.preview()['active']); self.assertEqual(job.preview()['text'], '')
        self.assertGreater(job.data['thinking_seconds'], 0)
        self.assertNotIn('transient text', json.dumps(job.data)+json.dumps(answer))

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

    def test_structured_helpers_explicitly_disable_thinking(self):
        job = Mock()
        with patch.object(self.runtime, 'start'), patch.object(self.runtime, 'call', return_value={
                'choices': [{'message': {'content': '{"ready":true}'}}]}) as call:
            self.assertEqual(self.runtime.structured('check', job), {'ready': True})
        self.assertEqual(call.call_args.args[1]['reasoning_budget_tokens'], 0)
        self.assertFalse(call.call_args.args[1]['chat_template_kwargs']['enable_thinking'])
        with patch.object(self.runtime, 'start'), patch.object(self.runtime, 'call', return_value={
                'choices': [{'message': {'content': '{}', 'reasoning_content': 'unexpected'}}]}):
            with self.assertRaisesRegex(ValueError, 'Thinking Off'):
                self.runtime.structured('check', job)

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

    def test_blocked_port_produces_actionable_error_before_model_load(self):
        dummy = self.service.store.root/'model'; dummy.touch()
        with patch.object(self.runtime, 'configured', return_value=(str(dummy), str(dummy))), \
             patch('sentinel_x.runtime.socket.socket', side_effect=PermissionError('blocked')), \
             patch('sentinel_x.runtime.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'loopback port'):
                self.runtime.start()
        spawn.assert_not_called()

    def test_live_check_reports_verified_modes_only_when_thinking_is_observed(self):
        def template(path, body):
            if path == '/tokenize': return {'tokens': [1]}
            return {'prompt': 'on' if body['chat_template_kwargs']['enable_thinking'] else 'off'}
        for enabled in (False, True):
            deltas = [{'content': '{"summary":"2/5","claims":[]}'}]
            if enabled:
                deltas.insert(0, {'reasoning_content': 'Example diagnostic reasoning'})
            with patch.object(self.runtime, 'start'), patch.object(self.runtime, 'call', side_effect=template), \
                 patch('sentinel_x.runtime.urllib.request.urlopen', return_value=wire(*deltas)):
                report = check_mode(self.service, enabled, 5)
            self.assertTrue(report['passed'], report)
            self.assertEqual(report['thinking_observed'], enabled)
            self.assertNotIn('diagnostic reasoning', json.dumps(report))

    def test_live_check_is_inconclusive_when_on_emits_no_thinking(self):
        with patch.object(self.runtime, 'start'), patch.object(self.runtime, 'call', side_effect=[
                {'prompt': 'on'}, {'prompt': 'off'}, {'tokens': [1]}]), \
             patch('sentinel_x.runtime.urllib.request.urlopen', return_value=wire(
                 {'content': '{"summary":"2/5","claims":[]}'})):
            report = check_mode(self.service, True, 5)
        self.assertTrue(report['inconclusive'])
        self.assertFalse(report['passed'])

    def test_live_check_fails_if_the_runtime_ignores_off(self):
        with patch.object(self.runtime, 'start'), patch.object(self.runtime, 'call', side_effect=[
                {'prompt': 'off'}, {'tokens': [1]}]), \
             patch('sentinel_x.runtime.urllib.request.urlopen', return_value=wire(
                 {'reasoning_content': 'Unexpected reasoning'})):
            report = check_mode(self.service, False, 5)
        self.assertFalse(report['passed'])
        self.assertIn('Thinking Off', report['detail'])


if __name__ == '__main__': unittest.main()
