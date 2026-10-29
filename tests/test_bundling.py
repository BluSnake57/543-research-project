from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from sentinel_x.bundling import bundle_runtime, ensure_runtime, verify_runtime
from sentinel_x.config import manifest


class BundlingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.source = self.root/'runtime'; self.source.mkdir()
        self.server = self.source/'llama-server'; self.server.write_text('precompiled executable')
        (self.source/'libllama.so').write_text('shared library')
        (self.source/'LICENSE').write_text('runtime license')
        self.version = subprocess.CompletedProcess([], 0, 'version: commit '+manifest()['llama_cpp']['commit'][:9], '')

    def tearDown(self):
        self.temp.cleanup()

    def test_runtime_is_required_and_libraries_licenses_are_included(self):
        with patch('sentinel_x.bundling.subprocess.run', return_value=self.version):
            included = bundle_runtime(self.source, self.root/'application')
            self.assertTrue(included.is_file())
            self.assertTrue((included.parent/'libllama.so').is_file())
            self.assertTrue((included.parent/'LICENSE').is_file())
            self.assertTrue((included.parent/'bundled-runtime.json').is_file())
        with self.assertRaises(ValueError): bundle_runtime(self.root/'missing', self.root/'bad')
        bad = subprocess.CompletedProcess([], 0, 'different commit abc123', '')
        with patch('sentinel_x.bundling.subprocess.run', return_value=bad):
            with self.assertRaises(ValueError): verify_runtime(self.source)

    def test_existing_runtime_is_used_without_a_build_time_download(self):
        with patch('sentinel_x.bundling.subprocess.run', return_value=self.version), \
             patch('sentinel_x.bundling.Models.install_runtime', side_effect=AssertionError('No download')):
            self.assertEqual(ensure_runtime(self.source), self.source)

    def test_release_archive_contains_both_apps_and_assistant_runtime(self):
        import scripts.package_app as builder
        release = self.root/'release'; release.mkdir(); (release/'build').mkdir()
        def freeze_or_version(command, **kwargs):
            if 'PyInstaller' in command:
                name = command[command.index('--name')+1]
                output = release/'dist'/name
                if builder.sys.platform == 'darwin': output = release/'dist'/f'{name}.app/Contents/MacOS'
                output.mkdir(parents=True)
                (output/name).write_text('frozen application test fixture')
            return self.version
        with patch.object(builder, 'ROOT', release), patch('subprocess.run', side_effect=freeze_or_version):
            archive = Path(builder.package('both', self.source))
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as bundle: names = bundle.namelist()
        else:
            with tarfile.open(archive) as bundle: names = bundle.getnames()
        self.assertTrue(any('Sentinel-X/' in name or 'Sentinel-X.app/' in name for name in names))
        self.assertTrue(any('Sentinel-X-Trainer' in name for name in names))
        self.assertTrue(any(name.endswith('/llama/llama-server') for name in names))
        self.assertTrue(any(name.endswith('/llama/libllama.so') for name in names))
        self.assertFalse(any('Sentinel-X-Trainer/llama/' in name for name in names))
