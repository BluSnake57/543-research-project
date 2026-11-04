"""Installer tests use temporary storage; no network or user settings are changed."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from sentinel_x.graphs import FEATURE_VERSION
from sentinel_x.learning import torch_modules
from sentinel_x.state import Store

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/setup_environment.py"
SPEC = importlib.util.spec_from_file_location("setup_environment", SCRIPT)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class EnvironmentSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "model weights"
        self.source.mkdir()
        self.store = Store(self.root / "data")

    def pair(self, ident="example", hybrid=False):
        torch, model = torch_modules(int(hybrid))
        torch.save(model(32).state_dict(), self.source / f"{ident}.pt")
        metadata = {
            "id": ident,
            "feature_version": FEATURE_VERSION,
            "config": {"hidden": 32, "threshold": 464, "window": 60},
            "mean": [0] * 6,
            "scale": [1] * 6,
            "cutoff": 0.7,
            "llm_feature_names": ["emotional_language"] if hybrid else [],
            "llm_mean": [0] if hybrid else None,
            "llm_scale": [1] if hybrid else None,
            "report": {"test": dict.fromkeys(("f1", "precision", "recall", "pr_auc"), 0.5)},
        }
        (self.source / f"{ident}.json").write_text(json.dumps(metadata))
        return metadata

    def test_matching_pair_installed_selected_and_repeatable(self):
        self.pair(hybrid=True)
        for _ in range(2):
            self.assertEqual(setup.install_checkpoints(self.source, self.store), (["example"], []))
        self.assertEqual(self.store.setting("checkpoint"), "example")
        self.assertEqual(
            (self.source / "example.pt").read_bytes(),
            (self.store.root / "checkpoints/example.pt").read_bytes(),
        )

    def test_missing_metadata_is_not_substituted_by_mapping(self):
        self.pair()
        (self.source / "example.json").unlink()
        (self.source / "checkpoint_map.json").write_text(
            '{"twitter15_emotional_language": "example"}'
        )
        installed, problems = setup.install_checkpoints(self.source, self.store)
        self.assertEqual(installed, [])
        self.assertIn("Missing example.json", problems[0])
        self.assertEqual(list((self.store.root / "checkpoints").iterdir()), [])

    def test_bad_normalization_and_weights_are_rejected(self):
        metadata = self.pair()
        metadata["scale"][0] = 0
        (self.source / "example.json").write_text(json.dumps(metadata))
        self.assertIn("scale", setup.install_checkpoints(self.source, self.store)[1][0])
        self.pair()
        (self.source / "example.pt").write_bytes(b"invalid weights")
        self.assertTrue(setup.install_checkpoints(self.source, self.store)[1])
        self.assertIsNone(self.store.setting("checkpoint"))

    def test_conflicting_destination_is_preserved(self):
        self.pair()
        target = self.store.root / "checkpoints/example.pt"
        target.write_bytes(b"previous model")
        self.assertIn(
            "Refusing to overwrite", setup.install_checkpoints(self.source, self.store)[1][0]
        )
        self.assertEqual(target.read_bytes(), b"previous model")
        self.assertFalse(target.with_suffix(".json").exists())

    def test_map_selects_demo_and_preserves_user_selection(self):
        self.pair("other")
        self.pair("demo", hybrid=True)
        (self.source / "checkpoint_map.json").write_text('{"twitter15_emotional_language": "demo"}')
        setup.install_checkpoints(self.source, self.store)
        self.assertEqual(self.store.setting("checkpoint"), "demo")
        self.store.configure("checkpoint", "other")
        setup.install_checkpoints(self.source, self.store)
        self.assertEqual(self.store.setting("checkpoint"), "other")

    def test_environment_paths(self):
        self.assertEqual(
            setup.environment_python(Path("env"), windows=True), Path("env/Scripts/python.exe")
        )
        self.assertEqual(
            setup.environment_python(Path("env"), windows=False), Path("env/bin/python")
        )

    def test_repository_map_selects_demo_without_duplicate_map(self):
        mapping = json.loads((setup.ROOT / "results/final/checkpoint_map.json").read_text())
        ident = mapping["twitter15_emotional_language"]
        self.pair(ident, hybrid=True)
        self.pair("other")
        setup.install_checkpoints(self.source, self.store)
        self.assertEqual(self.store.setting("checkpoint"), ident)

    def test_shipped_demo_weights_match_metadata(self):
        path = setup.ROOT / "model weights/fe44637dfc4d49fe9cd402c4ce6cc21d.json"
        metadata = setup.validate_checkpoint(path)
        self.assertEqual(metadata["llm_feature_names"], ["emotional_language"])
        self.assertEqual(metadata["config"]["threshold"], 464)


if __name__ == "__main__":
    unittest.main()
