import importlib.util
from pathlib import Path
import tempfile
import unittest

READY = all(importlib.util.find_spec(name) for name in ('fastapi', 'httpx'))


@unittest.skipUnless(READY, 'FastAPI and httpx are not installed')
class APITests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from veritas.api import create_app
        self.temp = tempfile.TemporaryDirectory()
        root = Path(__file__).resolve().parents[1]
        self.client = TestClient(create_app(Path(self.temp.name), 'a'*32,
                                            root/'runtime-manifest.json', root/'skills'))
        self.headers = {'Authorization': 'Bearer '+'a'*32}

    def tearDown(self):
        self.client.app.state.jobs.pool.shutdown()
        self.temp.cleanup()

    def test_auth_and_conversations(self):
        self.assertEqual(self.client.get('/v1/health').status_code, 401)
        response = self.client.post('/v1/conversations', json={'title': 'Research'}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        ident = response.json()['id']
        self.assertEqual(self.client.get('/v1/conversations', headers=self.headers).json()[0]['id'], ident)
        self.client.delete('/v1/conversations/'+ident, headers=self.headers)
        self.assertEqual(self.client.get('/v1/conversations', headers=self.headers).json(), [])

    def test_bad_settings_and_invalid_ids(self):
        for settings in ({'context': 0}, {'checkpoint': '../escape'}, {'gpu_layers': 'bad'}, {'arbitrary': 1}):
            response = self.client.patch('/v1/settings', json=settings, headers=self.headers)
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get('/v1/graphs/missing', headers=self.headers).status_code, 400)

    def test_estimate_acceptance_boundary(self):
        response = self.client.post('/v1/graphs/missing/accept', json={'accept': False}, headers=self.headers)
        self.assertEqual(response.status_code, 400)
