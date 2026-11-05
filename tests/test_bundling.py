"""Release structure and worker inclusion without downloading model weights."""

from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile


class BundlingTests(unittest.TestCase):
    def test_release_includes_worker_dependencies_and_both_apps(self):
        import scripts.package_app as builder

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build").mkdir()
            commands = []

            def freeze(command, **kwargs):
                commands.append(command)
                name = command[command.index("--name") + 1]
                output = root / "dist" / name
                if builder.sys.platform == "darwin":
                    output = root / "dist" / f"{name}.app/Contents/MacOS"
                output.mkdir(parents=True)
                (output / name).write_text("frozen application fixture")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(builder, "ROOT", root), patch("subprocess.run", side_effect=freeze):
                archive = Path(builder.package("both"))
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as bundle:
                    names = bundle.namelist()
            else:
                with tarfile.open(archive) as bundle:
                    names = bundle.getnames()
            self.assertTrue(any("Sentinel-X-Trainer" in name for name in names))
            self.assertFalse(any("llama" in name or "safetensors" in name for name in names))
            for command in commands:
                for dependency in (
                    "transformers",
                    "accelerate",
                    "safetensors",
                    "sentinel_x.scope_worker",
                ):
                    self.assertIn(dependency, command)
