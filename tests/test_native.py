import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from subprocess import CompletedProcess
import zipfile

import numpy as np

from sentinel_x.config import ASSETS
from sentinel_x.retrieval import Retrieval
from sentinel_x.service import Service

ROOT = Path(__file__).resolve().parents[1]


def finish(service, ident):
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        result = service.jobs.get(ident)
        if result['status'] in ('completed', 'failed', 'cancelled'):
            if result['status'] != 'completed':
                raise AssertionError(result.get('error', result['status']))
            return result['result']
        time.sleep(.01)
    raise AssertionError('Job did not finish')


class NativeServices(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.service = Service(self.root)

    def tearDown(self):
        self.service.close(); self.service.jobs.shutdown(wait=True)
        self.temp.cleanup()

    def test_packaged_assets_and_no_javascript_stack(self):
        self.assertTrue((ASSETS/'skills/graphsage/SKILL.md').exists())
        self.assertTrue(self.service.skills.tools())
        self.assertFalse((ROOT/'package.json').exists())
        self.assertFalse((ROOT/'src/renderer').exists())
        self.assertFalse((ROOT/'src/backend/sentinel_x/api.py').exists())
        self.assertIn('--only-binary=:all:', (ROOT/'requirements.txt').read_text())

    def test_service_import_conversations_and_estimate_review(self):
        posts = self.root/'posts.json'; posts.write_text(json.dumps([{'id': '1', 'text': 'Mars has two moons.'}]))
        ids = self.service.import_posts(posts)
        graph = self.service.store.graph(ids[0]); self.assertTrue(graph['needs_estimation'])
        from sentinel_x.graphs import estimate_graph
        proposal = estimate_graph(graph, {'delays': {'1': 0}, 'edges': [], 'assumptions': 'Assumed root time'})
        self.service.store.configure('proposal:'+ids[0], proposal)
        self.assertFalse(self.service.store.graph(ids[0])['accepted'])
        self.service.accept(ids[0]); self.assertTrue(self.service.store.graph(ids[0])['accepted'])
        c = self.service.store.create_conversation('Research'); self.service.rename(c['id'], 'Renamed')
        self.assertEqual(self.service.store.conversations()[0]['title'], 'Renamed')
        self.service.delete_conversation(c['id']); self.assertEqual(self.service.store.conversations(), [])
        with self.assertRaises(ValueError): self.service.configure(context=1)

    def test_legacy_database_migrates_and_remains_accessible_after_rename(self):
        from sentinel_x.state import Store, data_directory
        root = self.root/'veritas'
        old = Store(root)
        conversation = old.create_conversation('Existing research')
        (root/'sentinel-x.sqlite').rename(root/'veritas.sqlite')
        reopened = Store(root)
        self.assertEqual(reopened.conversations()[0]['id'], conversation['id'])
        self.assertTrue((root/'sentinel-x.sqlite').exists())
        with patch.dict(os.environ, {'XDG_DATA_HOME': str(self.root)}, clear=True), patch('sys.platform', 'linux'):
            current = data_directory()
            self.assertEqual(current, self.root/'sentinel-x')
            self.assertEqual(Store(current).conversations()[0]['id'], conversation['id'])

    def test_numpy_semantic_index_and_migration(self):
        class Embedding:
            def encode(self, texts, **kwargs):
                values = np.zeros((len(texts), 384), dtype=np.float32)
                for i, text in enumerate(texts):
                    values[i, 0 if any(w in text.lower() for w in ('mars', 'crimson')) else 1] = 1
                return values
        retrieval = self.service.retrieval; retrieval.embedding = lambda: Embedding()
        directory = self.root/'resources'
        (directory/'mars.txt').write_text('Mars is a planet.')
        (directory/'water.txt').write_text('Water is blue.')
        result = retrieval.index(directory); self.assertEqual(result['mode'], 'hybrid')
        found = retrieval.search('crimson')
        self.assertIn('Mars', found[0]['text']); self.assertEqual(found[0]['retrieval'], 'hybrid')
        manifest = json.loads((self.root/'index/vectors.json').read_text())
        self.assertEqual(manifest['format'], 'numpy-v1')
        shard = self.root/'index'/manifest['shards'][0]
        self.assertIsInstance(np.load(shard, mmap_mode='r'), np.memmap)
        self.assertEqual(np.load(shard).shape, (2, 384))
        (directory/'mars.txt').unlink(); retrieval.index(directory)
        self.assertFalse(shard.exists())
        self.assertTrue(all('Mars' not in r['text'] for r in retrieval.search('crimson')))
        # Existing HNSW catalogs can be replaced by reindexing, with no native extension installed.
        (self.root/'index/vectors.json').write_text(json.dumps({'shards': []}))
        retrieval.index(directory)
        self.assertEqual(json.loads((self.root/'index/vectors.json').read_text())['format'], 'numpy-v1')

    def test_resource_index_service(self):
        (self.root/'resources/fact.txt').write_text('Mars has two moons.')
        finish(self.service, self.service.index(self.root/'resources'))
        result = finish(self.service, self.service.search('Mars'))
        self.assertIn('Mars', result[0]['text'])

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from PySide6.QtWidgets import QApplication
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, 'PySide6 Essentials is not installed')
class DesktopSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_native_window_pages_conversations_and_graph_attachments(self):
        from sentinel_x.desktop import Window
        with tempfile.TemporaryDirectory() as temporary:
            service = Service(Path(temporary)); window = Window(service, auto_setup=False)
            try:
                window.show(); self.app.processEvents()
                self.assertEqual(window.pages.count(), 3)
                window.new_conversation()
                self.assertEqual(window.conversation_list.count(), 1)
                self.assertIsNotNone(window.conversation)
                path = Path(temporary)/'posts.json'
                path.write_text(json.dumps([{'id': 'post', 'text': 'Mars has two moons.'}]))
                ident = service.import_posts(path)[0]
                window.attached.add(ident); window.refresh_graphs()
                self.assertEqual(window.graph_list.topLevelItemCount(), 1)
                self.assertIn(ident, window.attached)
                for i in range(3):
                    window.navigation.setCurrentRow(i); self.app.processEvents()
                    self.assertEqual(window.pages.currentIndex(), i)
                window._poll()
            finally:
                window.close(); service.jobs.shutdown(wait=True); self.app.processEvents()
