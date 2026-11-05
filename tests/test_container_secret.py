"""The local credential must be usable by VNC and must survive repeated setup."""

import importlib.util
from pathlib import Path
import stat
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/create_container_secret.py"
SPEC = importlib.util.spec_from_file_location("container_secret", SCRIPT)
secret = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(secret)


class ContainerSecretTests(unittest.TestCase):
    def test_vnc_compatible_secret_in_private_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / ".secrets"
            path = secret.create_secret(directory)
            self.assertRegex(path.read_text(), r"^[A-Za-z0-9]{8}\n$")
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

    def test_existing_password_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "vnc_password"
            path.write_text("existing secret")
            with self.assertRaises(FileExistsError):
                secret.create_secret(directory)
            self.assertEqual(path.read_text(), "existing secret")


if __name__ == "__main__":
    unittest.main()
