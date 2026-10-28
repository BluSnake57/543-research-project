from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_requirements_match_project_metadata(self):
        project = tomllib.loads((ROOT/'pyproject.toml').read_text())['project']
        requirements = {line.strip() for line in (ROOT/'requirements.txt').read_text().splitlines()
                        if line.strip() and not line.startswith(('#', '--'))}
        self.assertIn('-e .', requirements)
        self.assertEqual(requirements - {'-e .'}, set(project['dependencies']))
        development = {line.strip() for line in (ROOT/'requirements-dev.txt').read_text().splitlines()
                       if line.strip() and not line.startswith(('#', '--'))}
        self.assertEqual(development - {'-r requirements.txt'},
                         set(project['optional-dependencies']['dev']))
