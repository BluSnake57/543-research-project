"""Installer tests use temporary storage; no network or user settings are changed."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

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

    def test_settings_reset_keeps_downloads_chats_and_operational_metadata(self):
        self.store.configure("context", 16384)
        self.store.configure("ui:last_conversation", "old")
        self.store.configure("installed_runtimes", {"cpu": {"executable": "somewhere"}})
        self.store.configure("fever_ready", True)
        self.store.configure("draft:example", {"text": "Keep this"})
        self.store.create_conversation("Keep me")
        model = self.store.root / "models/example.gguf"
        model.write_bytes(b"weights")
        self.assertEqual(setup.main(["--reset-settings", "--data-dir", str(self.store.root)]), 0)
        self.assertIsNone(self.store.setting("context"))
        self.assertIsNone(self.store.setting("ui:last_conversation"))
        self.assertTrue(self.store.setting("installed_runtimes"))
        self.assertTrue(self.store.setting("fever_ready"))
        self.assertEqual(self.store.setting("draft:example")["text"], "Keep this")
        self.assertEqual(len(self.store.conversations()), 1)
        self.assertEqual(model.read_bytes(), b"weights")

    def test_factory_reset_removes_downloads_and_data_but_preserves_source(self):
        self.store.configure("context", 16384)
        self.store.create_conversation("Delete me")
        (self.store.root / "datasets").mkdir()
        for folder in ("models", "resources", "datasets", "runtime", "index", "checkpoints"):
            (self.store.root / folder / "download").write_bytes(b"data")
        original = self.source / "original.pt"
        original.write_bytes(b"preserved")
        self.assertEqual(
            setup.main(
                [
                    "--factory-reset",
                    "--data-dir",
                    str(self.store.root),
                    "--weights-dir",
                    str(self.source),
                ]
            ),
            0,
        )
        self.assertEqual(self.store.conversations(), [])
        self.assertIsNone(self.store.setting("context"))
        self.assertFalse(list(self.store.root.rglob("download")))
        self.assertEqual(original.read_bytes(), b"preserved")
        self.assertEqual(list((self.store.root / "checkpoints").iterdir()), [])

    def test_reset_rejects_unrecognized_storage_and_protected_paths(self):
        important = self.source / "important.txt"
        important.write_text("keep")
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            setup.main(["--factory-reset", "--data-dir", str(self.source)])
        for target in (setup.ROOT, Path.home(), setup.ROOT.parent):
            with self.assertRaisesRegex(ValueError, "protected"):
                setup.validate_reset_path(target)
        with self.assertRaisesRegex(ValueError, "protected"):
            setup.main(
                [
                    "--factory-reset",
                    "--data-dir",
                    str(self.store.root),
                    "--weights-dir",
                    str(self.store.root / "checkpoints"),
                ]
            )
        self.assertEqual(important.read_text(), "keep")

    def test_factory_reset_does_not_follow_external_directory_links(self):
        external = self.root / "external"
        external.mkdir()
        (external / "keep").write_text("keep")
        link = self.store.root / "models/linked"
        try:
            link.symlink_to(external, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks unavailable")
        setup.main(["--factory-reset", "--data-dir", str(self.store.root)])
        self.assertTrue((external / "keep").exists())
        link.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            setup.main(["--factory-reset", "--data-dir", str(link)])

    def test_reset_venv_recreates_environment_and_reinstalls(self):
        environment = self.root / "venv"
        environment.mkdir()
        (environment / "pyvenv.cfg").write_text("home = example")
        old = environment / "old-package"
        old.write_text("stale")

        def create(path):
            self.assertFalse(old.exists())
            path.mkdir()
            (path / "pyvenv.cfg").write_text("home = example")
            python = setup.environment_python(path)
            python.parent.mkdir()
            python.touch()

        with (
            patch.object(setup.venv, "EnvBuilder") as builder,
            patch.object(setup, "run") as run,
            patch.object(setup.subprocess, "run", return_value=Mock(returncode=0)),
        ):
            builder.return_value.create.side_effect = create
            self.assertEqual(
                setup.main(
                    ["--reset-venv", "--venv", str(environment), "--data-dir", str(self.store.root)]
                ),
                0,
            )
        self.assertEqual(run.call_count, 3)
        self.assertIn("install", run.call_args_list[1].args[0])

    def test_reset_venv_rejects_active_environment_and_non_venv(self):
        with self.assertRaisesRegex(ValueError, "running Python"):
            setup.validate_environment_reset(Path(setup.sys.prefix))
        with self.assertRaisesRegex(ValueError, "pyvenv.cfg"):
            setup.validate_environment_reset(self.source)

    def test_conflicting_flags_fail_before_changing_data(self):
        self.store.configure("context", 16384)
        for flags in (
            ["--reset-venv", "--skip-install"],
            ["--factory-reset", "--download-models"],
            ["--reset-settings", "--runtime", "cpu"],
        ):
            with self.assertRaises(SystemExit) as result:
                setup.main([*flags, "--data-dir", str(self.store.root)])
            self.assertEqual(result.exception.code, 2)
        self.assertEqual(self.store.setting("context"), 16384)

    def test_skip_install_still_checks_and_configures(self):
        environment = self.root / "test-venv"
        setup.venv.EnvBuilder(with_pip=False).create(environment)
        with (
            patch.object(setup, "run") as run,
            patch.object(setup.subprocess, "run", return_value=Mock(returncode=0)) as child,
            patch.object(setup.sys, "prefix", setup.sys.base_prefix),
        ):
            setup.main(["--skip-install", "--venv", str(environment)])
        self.assertEqual(run.call_args.args[0][-2:], ["pip", "check"])
        self.assertEqual(run.call_count, 1)
        self.assertIn("--configure-only", child.call_args.args[0])


class ScopeSetupOptionsTests(unittest.TestCase):
    def test_rerun_preserves_selections_and_download_both_does_not_activate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "weights"
            source.mkdir()
            store = Store(root / "data")
            store.configure("scope_model", "2b")
            store.configure("scope_offload", "modules")
            args = ["--configure-only", "--data-dir", str(store.root), "--weights-dir", str(source)]
            with patch("sentinel_x.downloads.Models.install") as install:
                self.assertEqual(setup.main(args), 0)
                install.assert_not_called()
                self.assertEqual(setup.main([*args, "--download-both"]), 0)
                self.assertEqual(
                    [call.args[0] for call in install.call_args_list],
                    ["qwen", "scope", "qwen_2b", "scope_2b", "embedding"],
                )
            self.assertEqual(store.setting("scope_model"), "2b")
            self.assertEqual(store.setting("scope_offload"), "modules")
            with patch("sentinel_x.downloads.Models.install") as install:
                self.assertEqual(
                    setup.main(
                        [*args, "--model", "9b", "--cpu-offload", "sae", "--download-models"]
                    ),
                    0,
                )
                self.assertEqual(
                    [call.args[0] for call in install.call_args_list],
                    ["qwen", "scope", "embedding"],
                )
            self.assertEqual(store.setting("scope_model"), "9b")
            self.assertEqual(store.setting("scope_offload"), "sae")

    def test_model_flags_conflict_with_resets(self):
        for extra in (["--model", "2b"], ["--cpu-offload", "sae"], ["--download-both"]):
            with self.assertRaises(SystemExit):
                setup.main(["--reset-settings", *extra])

    def test_launcher_setup_without_existing_environment(self):
        import run as launcher

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(launcher.subprocess, "call", return_value=0) as call,
        ):
            environment = Path(directory) / "new-environment"
            self.assertEqual(
                launcher.main(
                    ["setup", "--venv", str(environment), "--model", "2b", "--download-both"]
                ),
                0,
            )
            command = call.call_args.args[0]
            self.assertEqual(command[0], launcher.sys.executable)
            self.assertTrue(command[1].endswith("scripts/setup_environment.py"))
            self.assertEqual(
                command[2:], ["--venv", str(environment), "--model", "2b", "--download-both"]
            )

    def test_launcher_passes_application_arguments_and_exit_code(self):
        import run as launcher

        with tempfile.TemporaryDirectory() as directory:
            environment = Path(directory)
            setup.venv.EnvBuilder(with_pip=False).create(environment)
            with patch.object(launcher.subprocess, "call", return_value=7) as call:
                self.assertEqual(
                    launcher.main(["--venv", directory, "--data-dir", "data", "trainer"]), 7
                )
                self.assertEqual(call.call_args.args[0][-3:], ["--data-dir", "data", "trainer"])


if __name__ == "__main__":
    unittest.main()
