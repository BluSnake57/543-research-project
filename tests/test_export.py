import json
from pathlib import Path
import tempfile
import unittest

from sentinel_x.assistant import Assistant, SYSTEM
from sentinel_x.config import ASSETS
from sentinel_x.exporting import export_conversation, render_conversation
from sentinel_x.retrieval import Retrieval
from sentinel_x.skills import Skills
from sentinel_x.state import Store


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.store = Store(self.root)
        self.conversation = self.store.create_conversation('Research: Mars')['id']

    def tearDown(self):
        self.temp.cleanup()

    def test_markdown_includes_actual_prompt_messages_reasoning_tools_and_filenames(self):
        resource = self.root/'resources/astronomy.md'; resource.write_text('Mars has two moons.', encoding='utf-8')
        retrieval = Retrieval(self.store); retrieval.index(resource.parent)
        class Runtime:
            def __init__(self): self.calls = 0
            def stream(self, messages, tools, job):
                self.calls += 1
                if self.calls == 1:
                    return {'role': 'assistant', 'content': '', 'tool_calls': [
                        {'id': 'lookup', 'function': {'name': 'evidence_search', 'arguments': '{"query":"Mars"}'}}]}
                return {'role': 'assistant', 'content': json.dumps({'summary': 'The statement is supported.',
                    'reasoning_summary': 'The retrieved astronomy source directly states the moon count.',
                    'claims': [{'claim': 'Mars has two moons.', 'verdict': 'supported',
                                'explanation': 'It matches the cited evidence.', 'citations': ['E1']}]})}
        class Job:
            def check(self): pass
            def emit(self, *args, **kwargs): pass
        assistant = Assistant(self.store, retrieval, Runtime(), Skills([ASSETS/'skills'], self.store))
        assistant.chat(self.conversation, 'Check the claim about Mars.', [], Job())
        snapshot = self.store.messages(self.conversation)[0]['data']['audit']
        self.assertIn(SYSTEM, snapshot['system_prompt'])
        report = render_conversation(self.store, self.conversation)
        for text in ('# Sentinel-X Conversation Export', 'Check the claim about Mars.',
                     'The statement is supported.', 'The retrieved astronomy source',
                     'astronomy.md', 'evidence_search', 'Core RAG evidence search',
                     'Request arguments', 'Application system prompt', SYSTEM,
                     'It matches the cited evidence.', 'E1'):
            self.assertIn(text, report)
        # A later settings/skill change must not replace the historical prompt snapshot.
        self.store.configure('skill:graphsage', False)
        self.assertIn(snapshot['system_prompt'], render_conversation(self.store, self.conversation))
        target = self.root/'conversation.md'
        export_conversation(self.store, self.conversation, target)
        self.assertTrue(target.read_text(encoding='utf-8').endswith('\n'))
        self.assertFalse(list(self.root.glob('.sentinel-x-export-*')))

    def test_historical_unknown_prompt_is_explicit_and_not_reconstructed(self):
        self.store.message(self.conversation, 'user', 'An old question')
        self.store.message(self.conversation, 'assistant', 'An old answer')
        report = render_conversation(self.store, self.conversation)
        self.assertIn('Not recorded for this historical turn', report)
        self.assertIn('No separate model-provided reasoning summary', report)
        self.assertNotIn(SYSTEM, report)

    def test_legacy_tool_wire_arguments_and_failure_are_readable(self):
        wire = {'function': {'name': 'graphsage_predict', 'arguments': '{"graph_ids":["g1"]}'}}
        self.store.message(self.conversation, 'tool', json.dumps({'name': 'graphsage_predict',
            'result': {'error': 'No trained checkpoint selected'}}), {'wire': wire})
        report = render_conversation(self.store, self.conversation)
        self.assertIn('GraphSAGE virality skill', report)
        self.assertIn('g1', report)
        self.assertIn('No trained checkpoint selected', report)

    def test_extension_validation_and_missing_conversation_leave_file_untouched(self):
        target = self.root/'report.txt'; target.write_text('existing')
        with self.assertRaises(ValueError): export_conversation(self.store, self.conversation, target)
        self.assertEqual(target.read_text(), 'existing')
        with self.assertRaises(ValueError): export_conversation(self.store, 'missing', self.root/'missing.md')
        self.assertFalse((self.root/'missing.md').exists())
