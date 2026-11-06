"""Run the native Qt stress regression in a subprocess so aborts fail the test."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class QtStabilityTests(unittest.TestCase):
    def test_long_download_progress_refresh_does_not_abort(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] / "docker/check_qt.py")],
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("5,000 progress refreshes", result.stdout)
