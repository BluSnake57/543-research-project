import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/backend'))
from sentinel_x.assistant import Assistant, validate_answer
from sentinel_x.downloads import Models, digest_file, download
from sentinel_x.graphs import (estimate_graph, native_twitter, node_features, normalize_posts,
                            observed_prefix, suggest_mapping, validate_graph)
from sentinel_x.jobs import Jobs
from sentinel_x.learning import best_cutoff, scores, split_graphs
from sentinel_x.retrieval import Retrieval
from sentinel_x.skills import Skills
from sentinel_x.state import Store

ROOT = Path(__file__).resolve().parents[1]


def graph(ident='event', size=3, delay_step=1):
    return validate_graph({'id': ident, 'cascade_id': ident, 'source': 'twitter15',
        'nodes': [{'id': str(i), 'delay': i*delay_step, 'text': ''} for i in range(size)],
        'edges': [[str(i-1), str(i)] for i in range(1, size)], 'accepted': True,
        'estimated': False, 'needs_estimation': False, 'warnings': []})


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root)

    def tearDown(self):
        self.temp.cleanup()


class GraphTests(TemporaryTest):
    def test_json_normalization_groups_and_time(self):
        rows = [{'tweet_id': 'a', 'content': 'claim', 'created_at': '2020-01-01T00:00:00Z'},
                {'tweet_id': 'b', 'reply_to': 'a', 'created_at': '2020-01-01T00:02:00Z'},
                {'tweet_id': 'c', 'created_at': '2020-01-01T01:00:00Z'}]
        result = normalize_posts(rows, suggest_mapping(rows))
        self.assertEqual(len(result), 2)
        first = result[0]
        self.assertEqual(first['edges'], [['a', 'b']])
        self.assertEqual(first['nodes'][1]['delay'], 2)
        self.assertFalse(first['needs_estimation'])

    def test_duplicate_and_cycle_rejected(self):
        for rows in ([{'id': 'a'}, {'id': 'a'}],
                     [{'id': 'a', 'parent_id': 'b'}, {'id': 'b', 'parent_id': 'a'}]):
            with self.assertRaises(ValueError):
                normalize_posts(rows, suggest_mapping(rows))

    def test_estimates_require_explicit_acceptance(self):
        rows = [{'id': 'a', 'text': 'a', 'cascade_id': 'bundle'},
                {'id': 'b', 'text': 'b', 'cascade_id': 'bundle'}]
        original = normalize_posts(rows, suggest_mapping(rows))[0]
        proposal = estimate_graph(original, {'edges': [['a', 'b']], 'delays': {'a': 0, 'b': 3},
                                            'assumptions': 'Text similarity; three-minute assumed gap'})
        self.assertTrue(proposal['estimated'])
        self.assertIsNone(original['nodes'][0]['delay'])
        with self.assertRaisesRegex(ValueError, 'accept'):
            observed_prefix(proposal, 60)
        proposal['accepted'] = True
        self.assertEqual(len(observed_prefix(proposal, 60)['nodes']), 2)

    def test_estimation_cannot_replace_observed_parent(self):
        original = graph(size=3)
        original['nodes'][2]['delay'] = None
        with self.assertRaisesRegex(ValueError, 'observed parent'):
            estimate_graph(original, {'delays': {'2': 2}, 'edges': [['0', '2']]})

    def test_cycles_unknown_nodes_and_nonfinite_times_rejected(self):
        for bad in ([['0', '1'], ['1', '0']], [['0', 'unknown']]):
            value = graph(); value['edges'] = bad
            with self.assertRaises(ValueError):
                validate_graph(value)
        value = graph(); value['nodes'][0]['delay'] = float('nan')
        with self.assertRaises(ValueError):
            validate_graph(value)

    def test_no_future_feature_leakage(self):
        first = graph(size=3, delay_step=10)
        second = graph(size=20, delay_step=10)
        a = observed_prefix(first, 15); b = observed_prefix(second, 15)
        self.assertEqual(node_features(a, 15), node_features(b, 15))
        self.assertEqual(len(a['edges']), 1)

    def test_native_parser_composite_ids_root_duplicates_bad_lines(self):
        directory = self.root/'twitter15'; (directory/'tree').mkdir(parents=True)
        (directory/'source_tweets.txt').write_text('100\tA source claim\n')
        (directory/'label.txt').write_text('false:100\n')
        (directory/'tree/100.txt').write_text(
            "['ROOT','ROOT',0]->['u1','100',0]\n"
            "['u1','100',0]->['u2','100',2]\n"
            "['u1','100',0]->['u2','100',2]\n"
            "['u2','100',2]->['u3','200',-1]\n"
            "this is malformed\n")
        value = list(native_twitter(directory))[0]
        self.assertNotIn('error', value)
        self.assertEqual(len(value['nodes']), 3)
        self.assertEqual(len(value['edges']), 2)
        self.assertEqual(value['rumor_label'], 'false')
        self.assertTrue(any('negative' in w for w in value['warnings']))
        self.assertEqual(value['nodes'][0]['text'], 'A source claim')

    def test_split_deduplicates_and_is_reproducible(self):
        values = [graph(f'n{i}', 2) for i in range(10)] + [graph(f'p{i}', 6) for i in range(10)]
        a = split_graphs(values+[values[0]], 4, 42)
        b = split_graphs(list(reversed(values)), 4, 42)
        groups = [{g['cascade_id'] for g in a[key]} for key in ('train', 'validation', 'test')]
        self.assertFalse(groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
        self.assertEqual(sum(map(len, groups)), 20)
        self.assertEqual([[g['id'] for g in a[k]] for k in a], [[g['id'] for g in b[k]] for k in b])

    def test_metrics_and_validation_cutoff(self):
        result = scores([0, 1], [.2, .8])
        self.assertEqual(result['f1'], 1)
        self.assertEqual(result['pr_auc'], 1)
        self.assertEqual(scores([0, 1], [.5, .5])['pr_auc'], .5)
        self.assertGreaterEqual(best_cutoff([0, 1], [.1, .3]), .1)


class RetrievalTests(TemporaryTest):
    def test_index_search_update_and_delete(self):
        retrieval = Retrieval(self.store)
        path = self.root/'resources/facts.txt'; path.write_text('Mars has two moons. Phobos and Deimos.')
        self.assertEqual(retrieval.index(path.parent)['indexed'], 1)
        result = retrieval.search('Mars moons')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['provenance'], {'chunk': 0})
        self.assertEqual(retrieval.index(path.parent)['unchanged'], 1)
        path.write_text('Venus has no moons.')
        retrieval.index(path.parent)
        self.assertFalse(retrieval.search('Mars'))
        path.unlink(); self.assertEqual(retrieval.index(path.parent)['deleted'], 1)
        self.assertFalse(retrieval.search('Venus'))

    def test_fever_evidence_without_claim_label_leakage(self):
        directory = self.root/'resources'
        (directory/'claims.jsonl').write_text(json.dumps({'id': 1, 'claim': 'Secret answer',
                                                       'label': 'REFUTES', 'evidence': []})+'\n')
        (directory/'wiki.jsonl').write_text(json.dumps({'id': 'Mars', 'lines': '0\tMars is a planet.\n1\tIt has two moons.'})+'\n')
        retrieval = Retrieval(self.store); retrieval.index(directory)
        self.assertFalse(retrieval.search('Secret REFUTES'))
        result = retrieval.search('Mars')[0]
        self.assertEqual(result['provenance'], {'page': 'Mars', 'sentence': '0'})

    def test_invalid_import_preserves_previous_index(self):
        path = self.root/'resources/facts.jsonl'
        path.write_text('{"text":"Kepler discovered planets"}\n')
        retrieval = Retrieval(self.store); retrieval.index(path.parent)
        path.write_text('invalid json')
        result = retrieval.index(path.parent)
        self.assertEqual(len(result['errors']), 1)
        self.assertTrue(retrieval.search('Kepler'))

    def test_unsupported_file_report(self):
        (self.root/'resources/image.pdf').write_bytes(b'%PDF')
        result = Retrieval(self.store).index(self.root/'resources')
        self.assertIn('Unsupported', result['errors'][0]['error'])


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data); self.status = status
        self.headers = headers or {'Content-Length': str(len(data))}


class DownloadTests(TemporaryTest):
    def test_resume_and_checksum(self):
        target = self.root/'models/example'
        target.with_name('example.part').write_bytes(b'abc')
        expected = hashlib.sha256(b'abcdef').hexdigest()
        with patch('sentinel_x.downloads.request', return_value=Response(b'def', 206,
                    {'Content-Range': 'bytes 3-5/6', 'Content-Length': '3'})) as fetch:
            download('https://example.com/model', target, expected, 6)
        self.assertEqual(target.read_bytes(), b'abcdef')
        self.assertEqual(fetch.call_args.args[1], {'Range': 'bytes=3-'})

    def test_server_ignoring_range_restarts(self):
        target = self.root/'models/example'; target.with_name('example.part').write_bytes(b'abc')
        with patch('sentinel_x.downloads.request', return_value=Response(b'abcdef')):
            download('https://example.com/model', target, hashlib.sha256(b'abcdef').hexdigest(), 6)
        self.assertEqual(target.read_bytes(), b'abcdef')

    def test_bad_checksum_removed(self):
        target = self.root/'models/example'
        with patch('sentinel_x.downloads.request', return_value=Response(b'bad')):
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                download('https://example.com/model', target, '0'*64, 3)
        self.assertFalse(target.exists())
        self.assertFalse(target.with_name('example.part').exists())

    def test_git_blob_checksum(self):
        path = self.root/'file'; path.write_bytes(b'hello')
        self.assertEqual(digest_file(path, 'git-sha1'), hashlib.sha1(b'blob 5\0hello').hexdigest())

    def test_cancel_retains_partial_and_prevents_ready(self):
        class Cancel:
            def check(self):
                raise ValueError('cancelled')
        target = self.root/'models/example'
        with patch('sentinel_x.downloads.request', return_value=Response(b'abcdef')):
            with self.assertRaisesRegex(ValueError, 'cancelled'):
                download('https://example.com/model', target, job=Cancel())
        self.assertFalse(target.exists())
        self.assertTrue(target.with_name('example.part').exists())

    def test_pinned_release_catalog_needs_no_metadata_network(self):
        config = {'qwen': {'repo': 'example/repo', 'revision': 'a'*40,
                           'artifacts': [{'name': 'model', 'checksum': 'b'*64}]}}
        models = Models(self.store, config)
        with patch('sentinel_x.downloads.get_json', side_effect=AssertionError('Network not allowed')):
            self.assertEqual(models.catalog('qwen')['revision'], 'a'*40)

    def test_completed_fever_style_download_is_reusable_offline(self):
        target = self.root/'models/archive'
        with patch('sentinel_x.downloads.request', return_value=Response(b'archive')):
            download('https://example.com/archive.zip', target)
        with patch('sentinel_x.downloads.request', side_effect=AssertionError('Offline')):
            self.assertEqual(download('https://example.com/archive.zip', target), target)


class AssistantTests(TemporaryTest):
    def test_invalid_citations_cannot_support_verdict(self):
        result = validate_answer(json.dumps({'summary': 'Check', 'claims': [
            {'claim': 'A claim', 'verdict': 'supported', 'citations': ['E999']}]}), [])
        self.assertEqual(result['claims'][0]['verdict'], 'insufficient_evidence')

    def test_skill_manifest_rejects_executable_handler(self):
        skill = self.root/'skills/unsafe'; skill.mkdir()
        (skill/'SKILL.md').write_text('Execute arbitrary things')
        (skill/'tools.json').write_text(json.dumps({'id': 'unsafe', 'tools': [
            {'handler': 'shell.execute', 'function': {'name': 'shell'}}]}))
        loaded = Skills([self.root/'skills'], self.store).list()
        self.assertFalse(loaded[0]['enabled'])
        self.assertIn('Unknown tool handler', loaded[0]['error'])

    def test_conversation_tool_loop_persistence_and_scope(self):
        path = self.root/'resources/facts.txt'; path.write_text('Mars has two moons.')
        retrieval = Retrieval(self.store); retrieval.index(path.parent)
        class Runtime:
            def __init__(self): self.calls = 0
            def stream(self, messages, tools, job):
                self.calls += 1
                if self.calls == 1:
                    return {'role': 'assistant', 'content': '', 'tool_calls': [
                        {'id': 'call1', 'type': 'function', 'function': {'name': 'evidence_search',
                         'arguments': '{"query":"Mars"}'}}]}
                return {'role': 'assistant', 'content': json.dumps({'summary': 'Supported by local evidence.',
                    'claims': [{'claim': 'Mars has two moons', 'verdict': 'supported',
                                'explanation': 'Matches evidence', 'citations': ['E1']}]})}
        class Job:
            def check(self): pass
            def emit(self, *args, **kwargs): pass
        skills = Skills([ROOT/'src/backend/sentinel_x/assets/skills'], self.store)
        assistant = Assistant(self.store, retrieval, Runtime(), skills)
        conversation = self.store.create_conversation()['id']
        result = assistant.chat(conversation, 'Does Mars have two moons?', [], Job())
        self.assertEqual(result['claims'][0]['verdict'], 'supported')
        roles = [m['role'] for m in self.store.messages(conversation)]
        self.assertEqual(roles, ['user', 'tool', 'assistant'])
        with self.assertRaisesRegex(ValueError, 'attached'):
            assistant.invoke('graphsage_predict', {'graph_ids': ['secret-other-conversation']}, [], [])
        reopened = Store(self.root)
        self.assertEqual(reopened.messages(conversation)[-1]['content'], result['summary'])


class JobTests(TemporaryTest):
    def test_cancelled_transport_is_not_reported_as_failure(self):
        jobs = Jobs(self.store)
        def interrupted(job):
            job.cancelled.set()
            raise OSError('Transport closed by cancellation')
        ident = jobs.submit('chat:test', interrupted)
        jobs.pool.shutdown()
        self.assertEqual(jobs.get(ident)['status'], 'cancelled')

    def test_failed_task_persists_error(self):
        jobs = Jobs(self.store)
        def failed(job):
            raise ValueError('Invalid graph')
        ident = jobs.submit('predict', failed)
        jobs.pool.shutdown()
        self.assertEqual(jobs.get(ident)['status'], 'failed')
        self.assertEqual(jobs.get(ident)['error'], 'Invalid graph')

    def test_completion_failure_and_restart(self):
        jobs = Jobs(self.store)
        ident = jobs.submit('smoke', lambda job: {'ok': True})
        for _ in range(100):
            data = jobs.get(ident)
            if data['status'] == 'completed': break
            time.sleep(.01)
        self.assertEqual(data['result'], {'ok': True})
        self.assertEqual([e['seq'] for e in data['events']], list(range(len(data['events']))))
        jobs.pool.shutdown()
        self.assertEqual(Jobs(self.store).get(ident)['status'], 'completed')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class RealGraphSAGESmoke(TemporaryTest):
    def test_train_checkpoint_and_batch_inference(self):
        from sentinel_x.learning import predict, train
        graphs = [graph(f'n{i}', 2, 30) for i in range(8)] + [graph(f'p{i}', 8, 20) for i in range(8)]
        result = train(graphs, self.root/'checkpoints', {'threshold': 4, 'window': 60, 'epochs': 3})
        outputs = predict(graphs[:2], self.root/'checkpoints'/f'{result["id"]}.json')
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(0 <= row['viral_probability'] <= 1 for row in outputs))
        self.assertEqual(result['feature_version'], 1)
        self.assertEqual(result['config']['threshold'], 4)
        for item in graphs[:2]:
            self.store.save_graph(item)
        self.store.configure('checkpoint', result['id'])
        assistant = Assistant(self.store, Retrieval(self.store), None, Skills([ROOT/'src/backend/sentinel_x/assets/skills'], self.store))
        via_skill = assistant.invoke('graphsage_predict', {'graph_ids': [g['id'] for g in graphs[:2]]},
                                     [g['id'] for g in graphs[:2]], [])
        self.assertEqual(via_skill, outputs)


if __name__ == '__main__':
    unittest.main()
