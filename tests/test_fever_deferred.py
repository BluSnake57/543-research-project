"""Saved FEVER archives are usable later without another transfer or full extraction."""
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from sentinel_x.jobs import Cancelled, Job
from sentinel_x.retrieval import documents
from sentinel_x.service import Service


def finish(service, ident):
    deadline = time.monotonic()+8
    while time.monotonic() < deadline:
        result = service.jobs.get(ident)
        if result['status'] in ('completed', 'failed', 'cancelled'):
            return result
        time.sleep(.005)
    raise AssertionError('Job did not finish')


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(data); self.status = 200
        self.headers = {'Content-Length': str(len(data))}


class DeferredFeverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.service = Service(self.root)
        self.evidence = '{"id":"Mars","lines":"0\\tMars has two moons."}\n'

    def tearDown(self):
        self.service.close(); self.service.jobs.shutdown(wait=True); self.temp.cleanup()

    def saved_download(self, evidence=None, setup=False):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as bundle:
            bundle.writestr('wiki-pages/wiki-001.jsonl', evidence or self.evidence)
            bundle.writestr('license.html', '<html>License</html>')
        with patch('sentinel_x.downloads.request', side_effect=[Response(data.getvalue()), Response(b'{}'), Response(b'{}')]), \
             patch.object(self.service.retrieval, 'index', side_effect=AssertionError('Download must not index')):
            ident = self.service.setup(keys=(), fever=True) if setup else self.service.fever()
            result = finish(self.service, ident)
        self.assertEqual(result['status'], 'completed', result.get('error'))
        return result

    def test_download_and_setup_default_to_archive_only(self):
        result = self.saved_download(setup=True)['result']['fever']
        self.assertTrue(Path(result['archive']).is_file())
        self.assertFalse(Path(result['evidence_directory']).exists())
        self.assertTrue(result['index_deferred'])
        status = self.service.status()['fever']
        self.assertTrue(status['downloaded']); self.assertFalse(status['indexed'])
        self.assertEqual(self.service.retrieval.search('Mars'), [])

    def test_restart_indexes_the_saved_download_offline_and_reuses_extracted_files(self):
        self.saved_download()
        self.service.close(); self.service.jobs.shutdown(wait=True)
        self.service = Service(self.root)
        self.assertTrue(self.service.status()['fever']['downloaded'])
        with patch('sentinel_x.downloads.request', side_effect=AssertionError('No network while indexing')), \
             patch.object(self.service.retrieval, 'embedding', side_effect=AssertionError('Keyword only')):
            result = finish(self.service, self.service.build_fever_index())
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertTrue(self.service.status()['fever']['indexed'])
        self.assertEqual(self.service.retrieval.search('Mars')[0]['provenance']['page'], 'Mars')
        evidence = Path(result['result']['evidence_directory'])/'wiki-001.jsonl'
        stamp = evidence.stat().st_mtime_ns
        with patch('sentinel_x.downloads.request', side_effect=AssertionError('No network')), \
             patch('zipfile.ZipFile.open', side_effect=AssertionError('Do not re-extract verified files')):
            repeated = finish(self.service, self.service.build_fever_index())
        self.assertEqual(repeated['status'], 'completed', repeated.get('error'))
        self.assertEqual(evidence.stat().st_mtime_ns, stamp)
        self.assertEqual(repeated['result']['index']['unchanged'], 1)

    def test_legacy_extraction_is_adopted_once_and_corrupt_cached_file_is_replaced(self):
        self.saved_download()
        evidence = self.root/'resources/fever/wiki-pages/wiki-001.jsonl'
        evidence.parent.mkdir(); evidence.write_text(self.evidence)
        stamp = evidence.stat().st_mtime_ns
        with patch('sentinel_x.downloads.request', side_effect=AssertionError('No network')), \
             patch('zipfile.ZipFile.open', side_effect=AssertionError('Already extracted')):
            result = finish(self.service, self.service.build_fever_index())
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(evidence.stat().st_mtime_ns, stamp)
        evidence.write_text(self.evidence.replace('Mars', 'Moon'))  # Same length, different CRC.
        with patch('sentinel_x.downloads.request', side_effect=AssertionError('No network')):
            repaired = finish(self.service, self.service.build_fever_index())
        self.assertEqual(repaired['status'], 'completed', repaired.get('error'))
        self.assertEqual(evidence.read_text(), self.evidence)

    def test_cancelled_index_keeps_archive_files_and_committed_batches_for_offline_resume(self):
        corpus = '\n'.join(json.dumps({'id': f'Page{i}', 'lines': '0\tMars has two moons.'}) for i in range(5000))
        self.saved_download(corpus)
        def interrupted(path):
            for number, row in enumerate(documents(path), 1):
                yield row
                if number == 2100: raise Cancelled()
        with patch('sentinel_x.downloads.request', side_effect=AssertionError('No network')), \
             patch('sentinel_x.retrieval.documents', side_effect=interrupted):
            stopped = finish(self.service, self.service.build_fever_index())
        self.assertEqual(stopped['status'], 'cancelled')
        self.assertTrue(self.service.status()['fever']['downloaded'])
        self.assertFalse(self.service.status()['fever']['indexed'])
        with self.service.retrieval.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM chunks').fetchone()[0], 2048)
        evidence = self.root/'resources/fever/wiki-pages/wiki-001.jsonl'
        stamp = evidence.stat().st_mtime_ns
        with patch('sentinel_x.downloads.request', side_effect=AssertionError('No network')):
            resumed = finish(self.service, self.service.retry_download(stopped['id']))
        self.assertEqual(resumed['status'], 'completed', resumed.get('error'))
        self.assertEqual(evidence.stat().st_mtime_ns, stamp)
        with self.service.retrieval.connection() as db:
            self.assertEqual(db.execute('SELECT count(*),count(DISTINCT key) FROM chunks').fetchone()[:], (5000, 5000))

    def test_missing_archive_has_actionable_error_without_a_download(self):
        with patch('sentinel_x.downloads.request') as request:
            result = finish(self.service, self.service.build_fever_index())
        self.assertEqual(result['status'], 'failed')
        self.assertIn('Download FEVER first', result['error']); request.assert_not_called()

    def test_existing_archive_without_old_download_settings_is_recognized_and_kept(self):
        archive = self.root/'resources/fever/wiki-pages.zip'; archive.parent.mkdir()
        with zipfile.ZipFile(archive, 'w') as bundle:
            bundle.writestr('wiki-pages/wiki-001.jsonl', self.evidence)
        stamp = archive.stat().st_mtime_ns
        self.assertTrue(self.service.status()['fever']['downloaded'])
        with patch('sentinel_x.downloads.request', side_effect=AssertionError('Keep existing archive')):
            result = finish(self.service, self.service.build_fever_index())
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(archive.stat().st_mtime_ns, stamp)

    def test_generic_resources_index_does_not_start_fever_indexing_or_embeddings(self):
        self.saved_download()
        fever = self.root/'resources/fever/wiki-pages'; fever.mkdir()
        (fever/'wiki-001.jsonl').write_text(self.evidence)
        (self.root/'resources/local.txt').write_text('Local Neptune document.')
        result = self.service.retrieval.index(self.root/'resources', semantic=False)
        self.assertEqual(result['indexed'], 1)
        self.assertTrue(self.service.retrieval.search('Neptune'))
        self.assertEqual(self.service.retrieval.search('Mars'), [])

    def test_keyword_bulk_import_avoids_per_file_full_chunk_scans_and_tiny_commits(self):
        directory = self.root/'sample'; directory.mkdir()
        corpus = '\n'.join(json.dumps({'id': f'Page{i}', 'lines': '0\tMars evidence.'}) for i in range(3000))
        for name in ('first.jsonl', 'second.jsonl'):
            (directory/name).write_text(corpus)
        retrieval = self.service.retrieval; connection = retrieval.connection
        statements = []
        def traced():
            db = connection()
            db.set_trace_callback(lambda sql: statements.append(sql) if
                sql == 'COMMIT' or 'FROM chunks WHERE path=' in sql else None)
            return db
        with patch.object(retrieval, 'connection', side_effect=traced):
            result = retrieval.index(directory, semantic=False, resumable=True)
        self.assertEqual(result['indexed'], 2)
        self.assertLess(statements.count('COMMIT'), 12)
        self.assertFalse(any('FROM chunks WHERE path=' in sql for sql in statements))
        with connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM chunks').fetchone()[0], 6000)
        for size in (0, -1, 10000):
            with self.assertRaises(ValueError): retrieval.index(directory, batch_size=size)

    def test_long_index_progress_replaces_samples_and_keeps_stage_and_file_completions(self):
        job = Job(self.service.jobs, 'fever-index')
        job.emit('stage', stage='Indexing Wikipedia')
        for number in range(500):
            job.emit('indexing', path='wiki-001.jsonl', chunks=number*2048, file_complete=False)
        job.emit('indexing', path='wiki-001.jsonl', chunks=500*2048, file_complete=True)
        job.emit('indexing', path='wiki-002.jsonl', chunks=2048, file_complete=False)
        events = self.service.jobs.get(job.data['id'])['events']
        self.assertEqual(len(events), 4)
        self.assertEqual(events[1]['chunks'], 499*2048)
        self.assertTrue(events[2]['file_complete'])
        self.assertEqual([e['seq'] for e in events], list(range(4)))
