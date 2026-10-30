"""Generation health checks, bounded format recovery, and citation preservation."""
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from sentinel_x.assistant import AnswerFormatError, validate_answer
from sentinel_x.exporting import render_conversation
from sentinel_x.jobs import Job
from sentinel_x.service import Service


def finish(service, ident):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        result = service.jobs.get(ident)
        if result['status'] in ('completed', 'failed', 'cancelled'):
            return result
        time.sleep(.005)
    raise AssertionError('Job did not finish')


class AnswerFormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = Service(Path(self.temp.name))
        self.conversation = self.service.store.create_conversation('Generation check')['id']

    def tearDown(self):
        self.service.close(); self.service.jobs.shutdown(wait=True); self.temp.cleanup()

    def test_test_returns_real_plain_reply_without_retrieval_tools_or_old_history(self):
        self.service.store.message(self.conversation, 'user', 'A large earlier question')
        self.service.store.message(self.conversation, 'assistant', 'Earlier answer')
        def stream(messages, tools, job):
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[-1]['content'], 'test')
            self.assertLess(len(messages[0]['content']), 300)
            self.assertEqual(tools, [])
            self.assertTrue(job.require_json)
            self.assertIn('Do not announce system status', messages[0]['content'])
            return {'role': 'assistant', 'content': 'I received your test message.'}
        with patch.object(self.service.retrieval, 'search') as search, \
             patch.object(self.service.runtime, 'stream', side_effect=stream) as model:
            result = finish(self.service, self.service.chat(self.conversation, 'test', []))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['result']['summary'], 'I received your test message.')
        self.assertEqual(result['result']['claims'], [])
        search.assert_not_called(); model.assert_called_once()

    def test_factual_format_retries_once_with_no_thinking_and_validates_citations(self):
        evidence = [{'citation': 'E1', 'text': 'Mars has two moons.'}]
        calls = []
        def stream(messages, tools, job):
            calls.append((job.thinking_enabled, job.require_json, tools))
            if len(calls) == 1:
                return {'role': 'assistant', 'content': 'Mars has two moons.'}
            self.assertIn('JSON object', messages[-1]['content'])
            return {'role': 'assistant', 'content': json.dumps({'summary': 'Verified!', 'claims': [
                {'claim': 'Mars has two moons.', 'verdict': 'supported', 'citations': ['E1']},
                {'claim': 'Another unsupported claim', 'verdict': 'supported', 'citations': ['invented']}]})}
        with patch.object(self.service.retrieval, 'search', return_value=evidence), \
             patch.object(self.service.runtime, 'stream', side_effect=stream):
            result = finish(self.service, self.service.chat(self.conversation, 'Check Mars moons', [], True))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([(c[0], c[1]) for c in calls], [(True, False), (False, True)])
        self.assertEqual(calls[-1][2], [])
        claims = result['result']['claims']
        self.assertEqual(claims[0]['verdict'], 'supported')
        self.assertEqual(claims[1]['verdict'], 'insufficient_evidence')
        self.assertEqual(claims[1]['citations'], [])
        self.assertIn('Formatting model response', [e.get('stage') for e in result['events']])
        self.assertIn('Response formatting retry', render_conversation(self.service.store, self.conversation))

    def test_empty_model_output_fails_clearly_after_one_retry_without_fake_acknowledgment(self):
        with patch.object(self.service.runtime, 'stream', return_value={'role': 'assistant', 'content': ''}) as model:
            result = finish(self.service, self.service.chat(self.conversation, 'test', []))
        self.assertEqual(model.call_count, 2)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('invalid or empty', result['error'])
        self.assertNotIn('Expecting value', result['error'])
        self.assertEqual([m['role'] for m in self.service.store.messages(self.conversation)], ['user'])

    def test_invalid_schema_and_markdown_json(self):
        for text in ('', 'plain text', '[]', '{}', '{"summary":"a","claims":{}}'):
            with self.subTest(text=text), self.assertRaises(AnswerFormatError):
                validate_answer(text, [])
        self.assertEqual(validate_answer('\ufeff```json\n{"summary":"ok","claims":[]}\n```', [])['summary'], 'ok')

    def test_json_constraint_and_empty_streaming_frames(self):
        runtime = self.service.runtime; runtime.port = 1234
        job = Job(self.service.jobs, 'test'); job.require_json = True
        packet = {'choices': [{'delta': {'content': '{"summary":"ok","claims":[]}'}}]}
        frames = io.BytesIO(('data: \n\n: keep-alive\n\ndata: '+json.dumps(packet)+'\n\ndata: [DONE]\n').encode())
        with patch.object(runtime, 'start'), patch.object(runtime, 'call', side_effect=[
                {'prompt': 'prompt'}, {'tokens': [1]}]), \
             patch('sentinel_x.runtime.urllib.request.urlopen', return_value=frames) as opened:
            result = runtime.stream([{'role': 'user', 'content': 'test'}], [], job)
        self.assertEqual(json.loads(result['content'])['summary'], 'ok')
        self.assertEqual(json.loads(opened.call_args.args[0].data)['response_format'], {'type': 'json_object'})

    def test_corrupt_streaming_packet_has_actionable_error(self):
        runtime = self.service.runtime
        with patch.object(runtime, 'start'), patch.object(runtime, 'call', side_effect=[
                {'prompt': 'prompt'}, {'tokens': [1]}]), \
             patch('sentinel_x.runtime.urllib.request.urlopen', return_value=io.BytesIO(b'data: broken\n')):
            with self.assertRaisesRegex(RuntimeError, 'invalid streaming response'):
                runtime.stream([], [], Job(self.service.jobs, 'test'))

    def test_empty_server_response_names_the_failed_operation(self):
        with patch('sentinel_x.runtime.urllib.request.urlopen', return_value=io.BytesIO(b'')):
            with self.assertRaisesRegex(ValueError, 'invalid or empty response for /apply-template'):
                self.service.runtime.call('/apply-template', {})
