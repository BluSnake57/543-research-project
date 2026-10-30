"""Actual storage migration, stable evidence identities, Git recovery and ignore rules."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.restore_git import restore
from sentinel_x.jobs import Cancelled
from sentinel_x.migration import migrate_storage
from sentinel_x.retrieval import Retrieval, documents
from sentinel_x.state import Store, data_directory

ROOT = Path(__file__).resolve().parents[1]


class StorageMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.legacy = self.root/'veritas'; self.current = self.root/'sentinel-x'
        self.store = Store(self.legacy)
        self.conversation = self.store.create_conversation('Existing conversation')['id']
        self.store.message(self.conversation, 'user', 'Original question', {'audit': {'model_path': str(self.legacy/'models/old.gguf')}})

    def tearDown(self):
        self.temp.cleanup()

    def test_directory_database_paths_downloads_and_evidence_migrate_without_reindexing(self):
        model = self.legacy/'models/model.gguf'; model.write_bytes(b'GGUF existing weights')
        executable = self.legacy/'runtime/llama-server'; executable.touch()
        evidence = self.legacy/'resources/fact.txt'; evidence.write_text('Mars has two moons.')
        inode = model.stat().st_ino
        self.store.configure('qwen_path', str(model)); self.store.configure('llama_path', str(executable))
        self.store.configure('resources', str(evidence.parent))
        self.store.configure('installed_runtimes', {'cpu': {'executable': str(executable), 'library_dirs': [str(executable.parent)]}})
        retrieval = Retrieval(self.store); retrieval.index(evidence.parent, semantic=False, resumable=True)
        citation = retrieval.search('Mars')[0]['citation']
        (self.legacy/'sentinel-x.sqlite').rename(self.legacy/'veritas.sqlite')
        with patch.dict(os.environ, {'XDG_DATA_HOME': str(self.root)}, clear=True), patch('sys.platform', 'linux'):
            self.assertEqual(data_directory(), self.current)
        migrated = Store(self.current)
        self.assertFalse(self.legacy.exists())
        self.assertEqual((self.current/'models/model.gguf').stat().st_ino, inode)
        self.assertEqual(migrated.setting('qwen_path'), str(self.current/'models/model.gguf'))
        self.assertEqual(migrated.setting('installed_runtimes')['cpu']['library_dirs'], [str(self.current/'runtime')])
        self.assertEqual(migrated.messages(self.conversation)[0]['data']['audit']['model_path'], str(self.legacy/'models/old.gguf'))
        current = Retrieval(migrated)
        found = current.search('Mars')[0]
        self.assertEqual(found['citation'], citation)
        self.assertEqual(found['path'], str(self.current/'resources/fact.txt'))
        result = current.index(self.current/'resources', semantic=False, resumable=True)
        self.assertEqual(result['unchanged'], 1)
        with current.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM chunks').fetchone()[0], 1)
        (self.current/'resources/fact.txt').unlink()
        self.assertEqual(current.index(self.current/'resources', semantic=False)['deleted'], 1)

    def test_explicit_data_directory_renames_database_and_preserves_original(self):
        (self.legacy/'sentinel-x.sqlite').rename(self.legacy/'veritas.sqlite')
        reopened = Store(self.legacy)
        self.assertTrue((self.legacy/'sentinel-x.sqlite').is_file())
        self.assertTrue((self.legacy/'veritas.sqlite').is_file())
        self.assertEqual(reopened.conversations()[0]['id'], self.conversation)

    def test_existing_current_database_is_not_merged_or_overwritten(self):
        current = Store(self.current); conversation = current.create_conversation('Current')['id']
        migrate_storage(self.current, self.legacy)
        self.assertTrue(self.legacy.is_dir())
        self.assertEqual(Store(self.current).conversations()[0]['id'], conversation)

    def test_interrupted_migration_retries_from_marker_without_losing_data(self):
        with patch('sentinel_x.migration.sqlite3.connect', side_effect=sqlite3.OperationalError('temporarily unavailable')):
            with self.assertRaises(sqlite3.OperationalError): migrate_storage(self.current, self.legacy)
        self.assertTrue((self.current/'.migration-origin.json').exists())
        reopened = Store(self.current)
        self.assertEqual(reopened.conversations()[0]['id'], self.conversation)
        self.assertFalse((self.current/'.migration-origin.json').exists())
        self.assertEqual(reopened.setting('legacy_data_roots'), [str(self.legacy)])

    def test_nonempty_destination_is_preserved(self):
        self.current.mkdir(); (self.current/'keep.txt').write_text('Keep this file')
        with self.assertRaisesRegex(ValueError, 'already contains files'):
            migrate_storage(self.current, self.legacy)
        self.assertEqual((self.current/'keep.txt').read_text(), 'Keep this file')
        self.assertTrue((self.legacy/'sentinel-x.sqlite').exists())

    def test_committed_wal_records_are_included_in_database_upgrade(self):
        source = self.legacy/'veritas.sqlite'
        (self.legacy/'sentinel-x.sqlite').rename(source)
        with sqlite3.connect(source) as writer:
            writer.execute('PRAGMA journal_mode=WAL')
            writer.execute('INSERT INTO settings VALUES (?,?)', ('wal-example', json.dumps('committed')))
            writer.commit()
            self.assertTrue(source.with_name(source.name+'-wal').exists())
            migrated = Store(self.legacy)
            self.assertEqual(migrated.setting('wal-example'), 'committed')

    def test_partial_keyword_index_resumes_after_storage_move_without_duplicate_passages(self):
        evidence = self.legacy/'resources/wiki.jsonl'
        evidence.write_text('\n'.join(json.dumps({'id': f'Page{i}', 'lines': '0\tMars evidence.'}) for i in range(200)))
        retrieval = Retrieval(self.store)
        def interrupted(path):
            for number, item in enumerate(documents(path), 1):
                yield item
                if number == 80: raise Cancelled()
        with patch('sentinel_x.retrieval.documents', side_effect=interrupted), self.assertRaises(Cancelled):
            retrieval.index(evidence.parent, semantic=False, resumable=True, batch_size=64)
        migrate_storage(self.current, self.legacy)
        current = Retrieval(Store(self.current))
        result = current.index(self.current/'resources', semantic=False, resumable=True)
        self.assertEqual(result['indexed'], 1)
        with current.connection() as db:
            self.assertEqual(db.execute('SELECT count(*),count(DISTINCT key) FROM chunks').fetchone()[:], (200, 200))
            self.assertEqual(db.execute('SELECT count(*) FROM index_progress').fetchone()[0], 0)


class GitRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.source = self.root/'source'; self.source.mkdir()
        self.command(self.source, 'init', '-q', '--initial-branch=main')
        self.command(self.source, 'config', 'user.name', 'Test user')
        self.command(self.source, 'config', 'user.email', 'test@example.com')
        (self.source/'app.py').write_text('original content\n')
        self.command(self.source, 'add', 'app.py'); self.command(self.source, 'commit', '-qm', 'Initial')
        self.bundle = self.root/'recovery.bundle'
        self.command(self.source, 'bundle', 'create', str(self.bundle), '--all')

    def tearDown(self):
        self.temp.cleanup()

    def command(self, root, *args, input=None):
        return subprocess.run(['git', '-C', str(root), *args], input=input,
                              capture_output=True, text=True, check=True).stdout.strip()

    def test_empty_metadata_restores_history_and_plain_git_commands_preserving_edits(self):
        checkout = self.root/'checkout'; checkout.mkdir(); (checkout/'.git').mkdir()
        (checkout/'app.py').write_text('my edited content\n')
        (checkout/'local.txt').write_text('local notes')
        restore(checkout, self.bundle)
        self.assertEqual(self.command(checkout, 'log', '-1', '--format=%s'), 'Initial')
        self.assertIn('github.com/BluSnake57/543-research-project', self.command(checkout, 'remote', 'get-url', 'origin'))
        self.assertEqual((checkout/'app.py').read_text(), 'my edited content\n')
        self.assertEqual((checkout/'local.txt').read_text(), 'local notes')
        self.assertIn('M app.py', self.command(checkout, 'status', '--short'))

    def test_working_repository_and_unknown_metadata_are_not_overwritten(self):
        self.assertIn('nothing changed', restore(self.source, self.bundle))
        checkout = self.root/'broken'; (checkout/'.git').mkdir(parents=True)
        (checkout/'.git/unknown').write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'left untouched'): restore(checkout, self.bundle)
        self.assertEqual((checkout/'.git/unknown').read_text(), 'preserve')

    def test_ignore_personal_settings_downloads_and_state_but_keep_source_assets(self):
        shutil.copyfile(ROOT/'.gitignore', self.source/'.gitignore')
        ignored = ['.data/settings.json', '.env.local', '.codex/config.toml', '.aws/credentials',
                   'settings.json', 'models/qwen/model.gguf', 'datasets/fever/train.jsonl',
                   'resources/fever/wiki-pages.zip', 'checkpoints/weights.pt', 'index/evidence.sqlite-wal',
                   'runtime/llama-server', 'graphs/import.json', 'skills/local/SKILL.md',
                   'sentinel-x/sentinel-x.sqlite', 'veritas/veritas.sqlite', 'weights.safetensors']
        kept = ['.env.example', 'resources/.gitkeep', 'src/backend/sentinel_x/config.py',
                'src/backend/sentinel_x/assets/runtime-manifest.json',
                'src/backend/sentinel_x/assets/skills/graphsage/SKILL.md', 'tests/test_core.py']
        found = self.command(self.source, 'check-ignore', '--stdin', input='\n'.join(ignored+kept)+'\n').splitlines()
        self.assertEqual(set(found), set(ignored))
