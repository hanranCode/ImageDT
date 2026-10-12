# Testing and development checks

Use the isolated environment described in [CONTRIBUTING.md](CONTRIBUTING.md). Tests use pytest and temporary self-contained fixtures; they do not need a GPU or personal `/data` directories.

```bash
python -m ruff check imagedt tests scripts test
python -m ruff format --check imagedt tests scripts test setup.py
python -m pytest -q
python -m pytest --cov=imagedt --cov-report=term-missing --cov-fail-under=88
python -m build
python -m twine check dist/*
python scripts/check_release.py
```

The coverage gate applies to the full optional-backend environment. Installing only core/plot/excel/pdf/caffe extras intentionally skips TensorFlow-specific tests; do not apply a whole-package coverage gate to that smaller environment. Skips remain visible (`-ra`). `caffe_pb2.py` and the duplicate standalone LeNet test are omitted from coverage, while protobuf roundtrips and LeNet behavior are tested in the main suite.

- `tests/test_core.py`: file/CSV/JSON/XML, directories, image processing, import isolation and timing.
- `tests/test_metrics_data_text.py`: F1/ROC semantics, batching, synthetic text and plotting.
- `tests/test_tensorflow.py`: actual graph/LiteRT conversion and inference, TFRecord write/read, losses and tensor transforms.
- `tests/test_caffe_backends.py`: protobuf/graph/transform behavior and Caffe conversion.
- `tests/test_contracts.py`: original public names/argument names and module import paths.
- `tests/test_remaining_utilities.py`: negative paths, file rename failure injection, PDF rendering and explicit unsupported behavior.

Use `tmp_path` for files, meaningful numerical/shape assertions for transformations, and mocked network access where the behavior under test is networking error handling. Test both the ordinary result and the relevant failure/boundary. Keep tests deterministic where the API promises deterministic behavior; for stochastic images, check contract properties rather than arbitrary exact pixels. Avoid tests that duplicate implementation details without observing behavior.

For optional integration fixtures, use `pytest.importorskip` and a declared marker. The intentionally delayed imports in `test_tensorflow.py` are necessary for backend-less environments, not a general exemption from import ordering.

`python scripts/benchmark_utils.py` checks ROC numerical equivalence and prints timing; timing is informational and is not a flaky CI speed threshold. `python -m imagedt.tensorflow.network.test_lenet` runs the older standalone test entrance. The files under `test/` remain launchers for corresponding maintained tests.

Before publication, run tests against an installed wheel outside the checkout as well as source tests. The source distribution must contain `tests/fixtures/legacy_api.json` and migration documentation; `scripts/check_release.py` verifies those entries, metadata and wheel compilation. Build in a clean output directory if old versions are present: the checker deliberately rejects ambiguous artifact sets.
