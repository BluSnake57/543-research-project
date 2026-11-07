"""Generation health checks, bounded format recovery, and citation preservation."""
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from sentinel_x.assistant import AnswerFormatError, validate_answer
from sentinel_x.answer_contract import repair_messages
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

    def test_ambiguous_or_trailing_output_is_not_silently_discarded(self):
        for raw in (
            '{"summary":"first","summary":"second","claims":[]}',
            '{"summary":"first","claims":[]} {"analysis":{}}',
            '{"summary":"first","claims":[],"score":NaN}',
            '```json\n{"summary":"first","claims":[]}\n```\nMore claims here.',
        ):
            with self.subTest(raw=raw), self.assertRaises(AnswerFormatError) as error:
                validate_answer(raw, [])
            self.assertTrue(error.exception.detail)

    def test_insufficiency_explanation_is_preserved_without_fabricating_citations(self):
        claim = {
            'claim': 'The event happened.', 'verdict': 'insufficient_evidence',
            'explanation': 'The retrieved passage discusses a different event.', 'citations': [],
        }
        result = validate_answer(json.dumps({'summary': 'Unverified.', 'claims': [claim]}), [])
        self.assertEqual(result['claims'], [claim])

    def test_repair_preserves_followup_context_as_data_without_failed_assistant_continuation(self):
        context = [
            {'role': 'user', 'content': 'Assess the claim about Mars.'},
            {'role': 'assistant', 'content': 'The retrieved passages do not establish it.'},
            {'role': 'user', 'content': 'Explain why.'},
        ]
        messages = repair_messages('Evidence instructions', 'Explain why.', [], 'broken draft',
                                   AnswerFormatError('Invalid JSON'), context=context)
        self.assertEqual([m['role'] for m in messages], ['system', 'user'])
        data = json.loads(messages[-1]['content'].split('\n', 1)[1])
        self.assertEqual(data['conversation_context'], context[:-1])
        self.assertEqual(data['failed_draft'], 'broken draft')
        self.assertEqual(context[-1]['content'], 'Explain why.')

    def test_fresh_request_repair_has_no_redundant_history(self):
        messages = repair_messages('Evidence instructions', 'Check this.', [], 'broken',
                                   AnswerFormatError('Invalid JSON'),
                                   context=[{'role': 'user', 'content': 'Check this.'}])
        data = json.loads(messages[-1]['content'].split('\n', 1)[1])
        self.assertNotIn('conversation_context', data)
