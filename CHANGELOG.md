# Changelog

## 0.6.0rc1 — 2026-10-12

Python 3 migration candidate. Requires Python 3.10+ and retains historical public names, module paths and argument names.

- Add modern package metadata, optional dependency groups, portable PDF rendering, CI and isolated release builds.
- Make imports lazy and remove global GPU/eager-mode changes.
- Fix image cleanup, padding, hashing, XML/CSV I/O, filename handling and paired rename failures.
- Prevent annotation overwrites during MD5 pair renaming; preserve split-specific corrupt-image reports and use zero-based converter class IDs.
- Correct BatchNorm protobuf normalization and terminal Caffe output aliases; reject numerically non-equivalent padding.
- Add contributor/testing/security guidance, editor conventions, strict Ruff/format CI checks and PR/issue templates.
- Correct detection/classification F1 and train/validation record leakage; accelerate ROC and Caffe DAG ordering.
- Migrate TensorFlow graph execution, Caffe protobuf, LeNet and Lite inference to supported Python 3 backends.
- Modernize synthetic text, manage pools/sessions/figures, and add self-contained regression, API compatibility and backend integration tests.
- Retain removed CoreML/handwriting/legacy TOCO entrances with explicit unsupported errors.

Read `docs/migration.md` for deliberate behavioral changes and checkpoint compatibility limits.
