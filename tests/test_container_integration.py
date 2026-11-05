"""Offline checks for the Scope/container boundary; never touch user application data."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

from packaging.requirements import Requirement

from docker.check_persistence import check_storage, MARKER
from docker.check_runtime import check_worker_cleanup
from sentinel_x.state import Store

ROOT = Path(__file__).resolve().parents[1]


class ContainerIntegrationTests(unittest.TestCase):
    def test_container_constraints_satisfy_every_application_dependency(self):
        constraints = {
            requirement.name.lower().replace("_", "-"): requirement
            for line in (ROOT / "docker/requirements.lock").read_text().splitlines()
            if line and not line.startswith("#")
            for requirement in [Requirement(line)]
        }
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        for dependency in project["project"]["dependencies"]:
            requirement = Requirement(dependency)
            with self.subTest(dependency=dependency):
                constraint = constraints[requirement.name.lower().replace("_", "-")]
                pin = next(iter(constraint.specifier))
                self.assertEqual(pin.operator, "==")
                self.assertIn(pin.version, requirement.specifier)

    def test_startup_preserves_settings_and_both_model_directories(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                home = Path(directory)
                root = home / ".local/share/sentinel-x"
                check_storage(root, home, seed=True, device=device)
                # Run the same setup command as entrypoint in a fresh process twice.
                for _ in range(2):
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "scripts/setup_environment.py"),
                            "--configure-only",
                            "--data-dir",
                            str(root),
                        ],
                        cwd=ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    check_storage(root, home, device=device)

    def test_persistence_probe_rejects_reset_preferences_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / "data"
            check_storage(root, home, seed=True)
            store = Store(root)
            store.configure("checkpoint", "test-checkpoint")
            check_storage(root, home)
            store.configure("scope_offload", "none")
            with self.assertRaisesRegex(AssertionError, "scope_offload"):
                check_storage(root, home)
            check_storage(root, home, seed=True)
            (root / "models/scope_2b" / MARKER).write_text("corrupt")
            with self.assertRaisesRegex(AssertionError, "File did not persist"):
                check_storage(root, home)

    def test_scope_worker_cleanup_without_models(self):
        with redirect_stdout(io.StringIO()):
            check_worker_cleanup()


if __name__ == "__main__":
    unittest.main()
