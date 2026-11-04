# Local model files

Drop trained GraphSAGE checkpoint pairs here, retaining the original filenames:

```text
fe44637dfc4d49fe9cd402c4ce6cc21d.pt
fe44637dfc4d49fe9cd402c4ce6cc21d.json
```

Run `python scripts/setup_environment.py` from the repository root. After the
initial dependency installation, use `--skip-install` to install newly added pairs.
Files are validated and copied to the application's checkpoints directory.
Identical existing files are reused; different files with the same name are rejected.
Close the application before running setup.

Each `.json` must be the original metadata produced by training alongside its `.pt`.
`checkpoint_map.json` maps experiment names to IDs; `model_configuration.json`
summarizes the experiments. Neither replaces checkpoint metadata. Normalization
values cannot be recovered from a weights-only state dictionary.

The map in `results/final/checkpoint_map.json` selects the `twitter15_emotional_language`
checkpoint if installed and no checkpoint is selected already. With only one valid
pair, that checkpoint is selected automatically. Otherwise select one in the app.
A local `checkpoint_map.json` can override the default map.

An optional `Qwen3.5-9B-Q4_K_M.gguf` is configured in place; keep it here afterward.
Other files, including `twitter15_llm_features.json`, are not automatically imported.
Feature tables are inputs to training/evaluation, not model checkpoints.

Small GraphSAGE `.pt` files and their `.json` metadata in this directory are included
in Git. Large Qwen GGUF and other downloaded artifacts remain ignored. Keep feature
tables in `results/` and experiment summaries in `results/final/`.
See [setup.md](../setup.md). Launch with `python run.py` after setup.
