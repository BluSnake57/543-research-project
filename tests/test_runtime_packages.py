"""Optional GPU package selection, library installation, activation and recovery."""
import hashlib
import io
import os
from pathlib import Path
from subprocess import CompletedProcess
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import zipfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication

from sentinel_x.config import manifest
from sentinel_x.jobs import Job, Cancelled
from sentinel_x.runtime_packages import (asset_for, gpu_devices, package_catalog, probe_runtime,
                                        runtime_environment)
from sentinel_x.runtime_ui import RuntimePanel
from sentinel_x.service import Service


def finish(service, ident):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        result = service.jobs.get(ident)
        if result['status'] in ('completed', 'failed', 'cancelled'):
            return result
        time.sleep(.005)
    raise AssertionError('Job did not finish')


def asset(name, payload=b'package'):
    return {'name': name, 'size': len(payload), 'digest': 'sha256:'+hashlib.sha256(payload).hexdigest(),
            'browser_download_url': 'https://example.com/'+name}


def archive(**files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    return stream.getvalue()


class RuntimePackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.service = Service(self.root)
        self.commit = manifest()['llama_cpp']['commit']

    def tearDown(self):
        self.service.close(); self.service.jobs.shutdown(wait=True); self.temp.cleanup()

    def test_platform_backend_selection_never_silently_substitutes_cpu(self):
        names = ['llama-b11960-bin-ubuntu-x64.tar.gz', 'llama-b11960-bin-ubuntu-vulkan-x64.tar.gz',
                 'llama-b11960-bin-ubuntu-rocm-10.0-x64.tar.gz', 'llama-b11960-bin-ubuntu-cuda-13.4-x64.tar.gz',
                 'llama-b11960-bin-ubuntu-cuda-12.8-x64.tar.gz', 'cudart-llama-b11960-bin-ubuntu-cuda-12.8-x64.tar.gz',
                 'llama-b11960-bin-win-vulkan-x64.zip', 'llama-b11960-bin-macos-arm64.tar.gz']
        assets = [asset(name) for name in names]
        for backend, index in [('cpu', 0), ('vulkan', 1), ('rocm', 2), ('cuda', 4)]:
            self.assertEqual(asset_for(assets, 'Linux', 'x86_64', backend)['name'], names[index])
        self.assertEqual(asset_for(assets, 'Windows', 'AMD64', 'vulkan')['name'], names[6])
        self.assertEqual(asset_for(assets, 'Darwin', 'arm64')['name'], names[7])
        for system, machine, backend in [('Linux', 'arm64', 'rocm'), ('Darwin', 'arm64', 'vulkan'),
                                          ('Windows', 'AMD64', 'cuda'), ('Linux', 'i686', 'cpu')]:
            with self.subTest(system=system, backend=backend), self.assertRaises(ValueError):
                asset_for(assets, system, machine, backend)

    def test_catalog_pins_release_separates_backends_and_downloads_matching_cuda_libraries(self):
        assets = [asset('llama-b11960-bin-ubuntu-vulkan-x64.zip'),
                  asset('llama-b11960-bin-ubuntu-cuda-12.8-x64.zip'),
                  asset('cudart-llama-b11960-bin-ubuntu-cuda-13.4-x64.zip'),
                  asset('cudart-llama-b11960-bin-ubuntu-cuda-12.8-x64.zip')]
        with patch('sentinel_x.downloads.get_json', return_value={'assets': assets}) as metadata, \
             patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='x86_64'):
            vulkan = package_catalog(self.service.models, 'vulkan')
            cuda = package_catalog(self.service.models, 'cuda')
            self.assertEqual(package_catalog(self.service.models, 'vulkan'), vulkan)
        self.assertEqual(metadata.call_count, 2)
        self.assertTrue(metadata.call_args.args[0].endswith('/b11960'))
        self.assertEqual(len(vulkan['files']), 1)
        self.assertEqual([f['name'] for f in cuda['files']], [assets[1]['name'], assets[3]['name']])
        self.assertTrue(all(len(f['checksum']) == 64 for f in cuda['files']))

    def test_missing_digest_or_cuda_companion_is_rejected(self):
        for backend, assets in [('vulkan', [{**asset('llama-bin-ubuntu-vulkan-x64.zip'), 'digest': ''}]),
                                ('cuda', [asset('llama-bin-ubuntu-cuda-12.8-x64.zip')])]:
            with patch('sentinel_x.downloads.get_json', return_value={'assets': assets}), \
                 patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='x86_64'), \
                 self.assertRaisesRegex(ValueError, 'digest|libraries'):
                package_catalog(self.service.models, backend)

    def test_install_verifies_both_cuda_archives_keeps_cpu_and_passes_library_environment(self):
        payloads = {
            'llama-b11960-bin-ubuntu-cuda-12.8-x64.zip': archive(**{'bin/llama-server': 'fixture server'}),
            'cudart-llama-b11960-bin-ubuntu-cuda-12.8-x64.zip': archive(**{'cuda/lib/libcudart.so.12': 'fixture library'})}
        assets = [asset(name, data) for name, data in payloads.items()]
        def request(url, headers=None):
            data = payloads[url.rsplit('/', 1)[-1]]
            response = io.BytesIO(data); response.status = 200; response.headers = {'Content-Length': str(len(data))}
            return response
        self.service.store.configure('llama_path', 'old-engine')
        self.service.store.configure('installed_runtimes', {'cpu': {'executable': 'old-engine'}})
        with patch('sentinel_x.downloads.get_json', return_value={'assets': assets}), \
             patch('sentinel_x.downloads.request', side_effect=request), \
             patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='x86_64'), \
             patch('sentinel_x.runtime_packages.subprocess.run', side_effect=[
                 CompletedProcess([], 0, 'version: '+self.commit[:9], ''),
                 CompletedProcess([], 0, 'Available devices:\n  CUDA0: NVIDIA fixture (8192 MiB)', '')]) as probe:
            result = self.service.models.install_runtime(backend='cuda', activate=False)
        self.assertTrue(Path(result['executable']).is_file())
        self.assertTrue(any((Path(d)/'libcudart.so.12').is_file() for d in result['library_dirs']))
        self.assertIn('cuda/lib', probe.call_args.kwargs['env']['LD_LIBRARY_PATH'])
        self.assertIn('cuda/lib', runtime_environment(self.service.store, result['executable'])['LD_LIBRARY_PATH'])
        self.assertEqual(result['devices'], ['CUDA0: NVIDIA fixture (8192 MiB)'])
        self.assertEqual(self.service.store.setting('llama_path'), 'old-engine')
        self.assertIn('cpu', self.service.store.setting('installed_runtimes'))
        self.assertFalse(list((self.root/'runtime').glob('install-*')))

    def test_probe_rejects_incompatible_version_and_ignores_cpu_device(self):
        with patch('sentinel_x.runtime_packages.subprocess.run', return_value=CompletedProcess([], 0, 'wrong commit', '')):
            with self.assertRaisesRegex(ValueError, 'pinned llama.cpp'):
                probe_runtime('/fixture/server', self.commit, [], 'vulkan')
        self.assertEqual(gpu_devices('Available devices:\n  OPENVINO0: Intel OpenVINO device =CPU\n', 'vulkan'), [])
        self.assertEqual(gpu_devices('  Vulkan0: AMD Radeon (8192 MiB)\n', 'vulkan'), ['Vulkan0: AMD Radeon (8192 MiB)'])
        self.assertEqual(gpu_devices('ggml_vulkan: Found 0 Vulkan devices\n  Vulkan0: llvmpipe (CPU)\n'), [])

    def test_cancelled_install_does_not_change_the_selected_engine(self):
        self.service.store.configure('llama_path', 'previous-engine')
        job = Job(self.service.jobs, 'runtime-download:vulkan')
        job.cancelled.set()
        catalog = {'backend': 'vulkan', 'release': 'b11960', 'commit': self.commit,
                   'files': [{'name': 'engine.zip', 'url': 'https://example.com/engine.zip',
                              'checksum': '0'*64, 'size': 10}]}
        with patch('sentinel_x.runtime_packages.package_catalog', return_value=catalog), \
             patch('sentinel_x.downloads.request') as request, self.assertRaises(Cancelled):
            self.service.models.install_runtime(job, backend='vulkan', activate=False)
        request.assert_not_called()
        self.assertEqual(self.service.store.setting('llama_path'), 'previous-engine')
        self.assertEqual(self.service.store.setting('installed_runtimes', {}), {})

    def installed(self, backend):
        executable = self.root/(backend+'-server'); executable.touch()
        entry = {'executable': str(executable), 'library_dirs': [str(self.root)], 'backend': backend, 'release': 'b11960'}
        self.service.store.configure('installed_runtimes', {backend: entry})
        self.service.store.configure('llama_path', 'previous-engine')
        return entry

    def test_failed_gpu_activation_leaves_current_engine_running(self):
        self.installed('vulkan')
        with patch('sentinel_x.service.probe_runtime', return_value={'devices': []}), \
             patch.object(self.service.runtime, 'stop') as stop:
            job = finish(self.service, self.service.activate_runtime('vulkan'))
            stop.assert_not_called()
        self.assertEqual(job['status'], 'failed')
        self.assertIn('no compatible GPU', job['error'])
        self.assertEqual(self.service.store.setting('llama_path'), 'previous-engine')

    def test_download_does_not_interrupt_chat_and_activation_waits_for_it(self):
        entry = self.installed('vulkan')
        entered = threading.Event(); release = threading.Event()
        def busy(job):
            entered.set(); release.wait(4)
            return {'summary': 'finished'}
        chat = self.service.jobs.submit('chat:fixture', busy)
        self.assertTrue(entered.wait(2))
        try:
            with patch.object(self.service.models, 'install_runtime', return_value=entry) as install, \
                 patch('sentinel_x.service.probe_runtime', return_value={'devices': ['Vulkan0: fixture']}), \
                 patch.object(self.service.runtime, 'stop') as stop:
                download = finish(self.service, self.service.download_runtime('vulkan'))
                self.assertEqual(download['status'], 'completed')
                activation = download['result']['activation_job']
                self.assertEqual(self.service.jobs.get(activation)['status'], 'queued')
                install.assert_called_once_with(self.service.jobs.live[download['id']], backend='vulkan', activate=False)
                stop.assert_not_called()
                self.assertEqual(self.service.store.setting('llama_path'), 'previous-engine')
                release.set()
                self.assertEqual(finish(self.service, chat)['status'], 'completed')
                self.assertEqual(finish(self.service, activation)['status'], 'completed')
            self.assertEqual(self.service.store.setting('llama_path'), entry['executable'])
            self.assertEqual(self.service.store.setting('gpu_layers'), 'auto')
        finally:
            release.set()

    def test_missing_gpu_loader_falls_back_to_separate_cpu_binary(self):
        entry = self.installed('cuda')
        self.service.store.configure('llama_path', entry['executable'])
        model = self.root/'model.gguf'; model.touch(); self.service.store.configure('qwen_path', str(model))
        cpu = self.root/'cpu'; cpu.mkdir(); (cpu/'llama-server').touch()
        runtime = self.service.runtime; runtime.bundled = cpu
        process = Mock(); process.poll.return_value = None
        socket = Mock(); socket.getsockname.return_value = ('127.0.0.1', 1234)
        with patch('sentinel_x.runtime.socket.socket') as network, \
             patch.object(runtime, 'select_acceleration', return_value='auto'), \
             patch('sentinel_x.runtime.subprocess.Popen', side_effect=[OSError('missing CUDA'), process]) as start, \
             patch.object(runtime, 'call', return_value={'status': 'ok'}):
            network.return_value.__enter__.return_value = socket
            runtime.start(Job(self.service.jobs, 'test'))
        self.assertEqual(start.call_args_list[0].args[0][0], entry['executable'])
        self.assertEqual(start.call_args_list[1].args[0][0], str(cpu/'llama-server'))
        self.assertIn('none', start.call_args_list[1].args[0])
        self.assertEqual(runtime.active['gpu_layers'], '0')


class RuntimePanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_controls_submit_backend_and_follow_download_to_activation(self):
        with tempfile.TemporaryDirectory() as temporary:
            service = Service(Path(temporary)); panel = RuntimePanel(service)
            try:
                self.assertGreaterEqual(panel.choice.count(), 1)
                panel.choice.setCurrentIndex(panel.choice.findData('vulkan'))
                with patch.object(service, 'download_runtime', return_value='download') as download:
                    panel.download.click()
                download.assert_called_once_with('vulkan', True)
                self.assertTrue(panel.busy)
                self.assertFalse(panel.activate.isEnabled())
                with patch.object(service.jobs, 'get', return_value={'kind': 'runtime-download:vulkan',
                        'status': 'completed', 'created': time.time(), 'events': [], 'result': {'activation_job': 'activation'}}):
                    panel.poll()
                self.assertEqual(panel.job_id, 'activation')
                with patch.object(service.jobs, 'get', return_value={'kind': 'runtime-activate:vulkan',
                        'status': 'completed', 'created': time.time(), 'events': [], 'result': {'activated': True, 'devices': ['Vulkan0: AMD fixture']}}):
                    panel.poll()
                self.assertFalse(panel.busy)
                self.assertIn('AMD fixture', panel.message.text())
            finally:
                panel.timer.stop(); panel.close(); service.close(); service.jobs.shutdown(wait=True)
