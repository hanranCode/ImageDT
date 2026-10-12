# Contributing to ImageDT

ImageDT is an Apache-2.0 Python utility package. Bug reports, regression fixtures and focused pull requests are welcome. Contributions use the existing project license; preserve copyright and license notices when modifying vendored code.

## Set up a development environment

Use Python 3.10 or newer. Create an isolated environment and install the project with the capabilities you will change:

```bash
python3 -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev,plot,excel,pdf,caffe]'
# For model conversion / TensorFlow integration:
python -m pip install -e '.[dev,plot,excel,pdf,tensorflow]'
```

Work on a feature branch based on `main`. Keep each commit focused and pair behavior changes with tests. Use commit subjects such as `fix: handle empty annotations` or `docs: clarify optional backends`. A PR should explain the problem, changed behavior, checks executed, and any compatibility limits. Do not commit generated distributions, virtual environments, credentials, personal datasets or machine-specific paths.

## Python code conventions

- Follow [PEP 8](https://peps.python.org/pep-0008/) and [PEP 257](https://peps.python.org/pep-0257/), with Ruff formatting at 88 columns and four-space indentation. The formatter's line length is a target; do not contort URLs, documentation or generated schema bytes to fit it.
- Run Ruff's error, import, common-bug and Python modernization rules (`E4/E7/E9`, `F`, `I`, `B`, `UP`). Import groups follow standard library, third-party and project-local order. Explicit re-exports use `as` or `__all__`; avoid wildcard imports.
- Keep historical public names, import paths and parameter names, including misspellings. New APIs use `snake_case` functions, `CapWords` classes and descriptive argument names. Add useful type annotations to new interfaces; do not require a heavyweight backend to resolve annotations for core tools.
- Document public behavior, units/shape conventions, return values and meaningful errors. Prefer a short docstring over comments that merely restate a line. Do not silently change behavior to satisfy a style rule.
- Prefer explicit validation and scoped exceptions. Preserve exception causes with `raise ... from error`. Do not use bare `except`, swallow failures, or rely on `assert` for public input validation.
- Keep backend imports optional and expose a helpful missing-extra error. Importing `imagedt` must not change eager execution, GPU settings, plotting settings or launch workers.
- Close sessions, figures, files and pools using context managers or explicit `close()`. File cleanup/rename tests must operate only in temporary fixtures. Avoid overwriting an existing destination without a documented contract.
- Generated `caffe_pb2.py` is excluded from formatting/lint because it embeds schema bytes. TensorFlow test import ordering has one documented `E402` exception so the optional backend can skip before dependent imports. Add no broad suppressions merely to make CI pass.

## Before submitting

Run the commands in [TESTING.md](TESTING.md). Formatting and lint checks are enforced by CI; optional TensorFlow tests run in a separate integration job. A pass with skipped backends must be reported with its skip count rather than as full backend validation. Explain any changed metric values or checkpoint naming in [docs/migration.md](docs/migration.md). Update README/CHANGELOG when installation or user-visible behavior changes.

On systems with GNU/BSD Make, `make PYTHON=python lint`, `make PYTHON=python test` and `make PYTHON=python build` wrap the same commands. Native Python commands remain the portable interface, including on Windows.

## Compatibility and release policy

The legacy API fixture is a baseline, not an authorization to remove methods. Tests must check behavior as well as name/argument preservation. Numerical bug fixes should state the old and new semantics. Keep optional capability and unsupported-operation boundaries explicit.

Support is Python 3.10+. Core CI covers 3.10, 3.12 and 3.14 across Linux/macOS/Windows; TensorFlow integration runs on 3.12. Intermediate Python versions are within the support range but are not separate matrix lanes. If a new dependency requires a higher Python version, discuss that compatibility change before merging.

Do not upload PyPI as part of an ordinary PR. The maintainer reviews the candidate, runs release validation, then explicitly publishes. Existing PyPI versions cannot be overwritten. `bash upload_pip.sh` builds/checks; `--upload` is the explicit publication step. Never include API tokens in a PR or committed config.

## Community conduct

Be respectful, specific and evidence-based in issues and reviews. Critique code and behavior rather than people. Harassment, discriminatory remarks and disclosure of personal information are not acceptable. Maintainers may edit or remove abusive content and limit participation. Report sensitive security issues through [SECURITY.md](SECURITY.md), not a public issue with exploit details.
