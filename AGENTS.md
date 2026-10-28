# Repository Guidelines

## Project Structure & Module Organization

This workspace currently contains no application source, tests, assets, or build manifests. Treat it as an uninitialized project; do not assume a language, framework, or runtime.

When adding the first implementation, organize application code in `src/`, automated tests in `tests/`, and supporting documentation in `docs/`, unless the chosen framework requires another layout. Keep assets with their owning module and document the actual structure in a root `README.md`.

## Build, Test, and Development Commands

No build, test, or local execution commands are currently configured. When introducing tooling, document exact setup and execution commands in `README.md` and commit the required manifests and lockfiles. Include commands for:

- Installing dependencies from a clean checkout.
- Running the application or research workflow locally.
- Executing automated tests.
- Checking formatting and lint rules.

Do not describe a command as supported until it has been verified in this repository.

## Coding Style & Naming Conventions

No language-specific style or formatter is established. Use consistent indentation within each file, descriptive names, and small modules with clear responsibilities. When selecting a language, adopt its standard formatter and lint tooling, and commit their configuration. Avoid unrelated formatting changes in functional patches.

## Testing Guidelines

No testing framework or coverage threshold is configured. Add tests alongside new behavior using the selected language’s conventional naming scheme. Cover expected results, boundary conditions, and failure paths. Keep tests deterministic and document required fixtures or external services.

## Commit & Pull Request Guidelines

Git history is unavailable in this workspace, so no existing commit convention can be inferred. Use concise, imperative commit subjects such as `Add data validation tests`.

Pull requests should explain the change, its purpose, and how it was verified. Link relevant issues, include screenshots for visual changes, and identify any setup or configuration changes.

## Security & Configuration

Never commit credentials, access tokens, or private datasets. Document required configuration with placeholder values and exclude local secrets and generated outputs through `.gitignore`.
